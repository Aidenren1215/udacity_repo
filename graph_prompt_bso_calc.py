
# -*- coding: utf-8 -*-
"""
graph_prompt_bso_calc.py

Purpose
-------
- Adds an **LLM tool‑calling Calculator agent** that operates on CSV stored on QueryState.
- Keeps your original BSO flow (retrieval -> bso_init -> bso_agent) intact.
- Auto‑route: if CSV present (or query looks like math), go calculator; otherwise go BSO.

Requirements
------------
- langgraph, langchain-core, langchain-openai, httpx, pandas, numpy, pydantic
- your local modules: src.state, src.configuration, src.utils.config, src.utils.utils, src.utils.utils_macro
"""
from __future__ import annotations

import os
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import httpx
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, add_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# ---- Project-local imports (align with your repo) ----
from src.state import QueryState   # QueryState now includes csv_* fields
from src.configuration import Configuration
from src.utils.config import Config
from src.utils.utils import get_config_value, chroma_search
from src.utils.utils_macro import remove_think_tags

# -------------------------------------------------------------------------
# Working directory & logging
# -------------------------------------------------------------------------
# Align CWD to repo root (one level above this file's directory)
os.chdir(Path(__file__).resolve().parent.parent)

conf = Config()

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    llm_headers = getattr(conf, "vllm_payload_to_dict", lambda: {})()
except Exception:
    llm_headers = {}
http_client = httpx.Client(verify=False, headers=llm_headers)

# -------------------------------------------------------------------------
# CSV helpers (load/filters/fuzzy-match)
# -------------------------------------------------------------------------
def _ensure_df(state: Dict[str, Any]) -> pd.DataFrame:
    """Materialize a DataFrame from QueryState's csv_* fields."""
    df = state.get("csv_df")
    if isinstance(df, pd.DataFrame):
        return df

    if state.get("csv_bytes"):
        return pd.read_csv(io.BytesIO(state["csv_bytes"]))

    if state.get("csv_text"):
        return pd.read_csv(io.StringIO(state["csv_text"]))

    if state.get("csv_path"):
        return pd.read_csv(state["csv_path"])

    uf = state.get("uploaded_file")
    if uf is not None:
        if isinstance(uf, dict) and "bytes" in uf:
            return pd.read_csv(io.BytesIO(uf["bytes"]))
        if hasattr(uf, "read"):
            return pd.read_csv(io.BytesIO(uf.read()))

    raise ValueError("No CSV found on state (need csv_df/csv_bytes/csv_text/csv_path/uploaded_file).")

def _apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Apply simple equality/inequality and numeric comparisons. Each filter: {column, op, value}."""
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col = f.get("column")
        op = f.get("op", "==")
        val = f.get("value")
        if col not in out.columns:
            raise ValueError(f"Column not found: {col}")
        series = out[col]
        # Try numeric comparison; if all-NaN mask then fall back to string-compare
        try:
            left = pd.to_numeric(series, errors="coerce")
            right = pd.to_numeric(pd.Series([val] * len(out)), errors="coerce")
            if op == "==": mask = (left == right)
            elif op == "!=": mask = (left != right)
            elif op == ">": mask = (left > right)
            elif op == "<": mask = (left < right)
            elif op == ">=": mask = (left >= right)
            elif op == "<=": mask = (left <= right)
            else: raise ValueError("op must be one of ==, !=, >, <, >=, <=")
            if mask.isna().all():
                raise Exception("numeric compare failed")
        except Exception:
            s = series.astype(str).str.lower()
            rv = str(val).lower()
            if op == "==": mask = (s == rv)
            elif op == "!=": mask = (s != rv)
            else: raise ValueError("Non-equality ops require numeric-like values.")
        out = out[mask]
    return out

# Fuzzy column mapping (difflib fallback to avoid extra deps)
from difflib import get_close_matches

def _normalize_col(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")

def _most_similar_column(user_col: str, columns: List[str], min_ratio: float = 0.6) -> Tuple[Optional[str], List[str]]:
    """Return (best_match or None, candidate_list)."""
    if not user_col:
        return None, []
    norm_user = _normalize_col(user_col)
    norm_map = {c: _normalize_col(c) for c in columns}
    # exact (normalized) match
    for real, norm in norm_map.items():
        if norm == norm_user:
            return real, []
    # fuzzy via difflib
    cands_norm = get_close_matches(norm_user, list(norm_map.values()), n=5, cutoff=min_ratio)
    rev = {v: k for k, v in norm_map.items()}
    cands = [rev[nc] for nc in cands_norm if nc in rev]
    best = cands[0] if cands else None
    return best, cands

def _md_preview(df: pd.DataFrame, n: int = 5) -> str:
    head = df.head(n)
    return f"Rows={len(df)}, Cols={len(df.columns)}\n\n" + head.to_markdown(index=False)

# -------------------------------------------------------------------------
# Tools (Pydantic schemas + @tool)
# -------------------------------------------------------------------------
class _PreviewArg(BaseModel):
    n: int = Field(5, ge=1, le=50)

class _ResolveArg(BaseModel):
    fuzzy_name: str = Field(..., description="User-provided (possibly partial) column name.")
    min_ratio: float = Field(0.6, ge=0.0, le=1.0)
    topk: int = Field(5, ge=1, le=20)

class _AggArg(BaseModel):
    column: str = Field(..., description="Exact (or fuzzy) column name to aggregate.")
    agg: Literal["sum", "mean", "min", "max", "count", "median", "std"] = "sum"
    filters: Optional[List[Dict[str, Any]]] = None

class _GroupAggArg(BaseModel):
    by: str
    target: str
    agg: Literal["sum", "mean", "min", "max", "count", "median", "std"] = "sum"
    filters: Optional[List[Dict[str, Any]]] = None

class _ArithArg(BaseModel):
    column: str
    op: Literal["add", "sub", "mul", "div"]
    scalar: float
    in_place: bool = True
    new_column: Optional[str] = None
    filters: Optional[List[Dict[str, Any]]] = None

@tool("list_columns")
def list_columns(state: Dict[str, Any] = None) -> List[str]:
    """List all DataFrame column names."""
    df = _ensure_df(state)
    cols = list(df.columns)
    state["csv_columns"] = cols  # cache on state
    return cols

@tool("preview_csv", args_schema=_PreviewArg)
def preview_csv(n: int = 5, state: Dict[str, Any] = None) -> str:
    """Preview first N rows + shape."""
    df = _ensure_df(state)
    return _md_preview(df, n=n)

@tool("resolve_column", args_schema=_ResolveArg)
def resolve_column(fuzzy_name: str, min_ratio: float = 0.6, topk: int = 5, state: Dict[str, Any] = None) -> Dict[str, Any]:
    """Resolve a fuzzy column name to the closest actual column. Returns {best, candidates, exact}."""
    df = _ensure_df(state)
    best, cands = _most_similar_column(fuzzy_name, list(df.columns), min_ratio=min_ratio)
    return {"best": best, "candidates": cands[:topk], "exact": best is not None and _normalize_col(best)==_normalize_col(fuzzy_name)}

@tool("aggregate", args_schema=_AggArg)
def aggregate(column: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None, state: Dict[str, Any] = None) -> str:
    """Aggregate one column with an operation; accepts fuzzy column name and optional filters."""
    df = _ensure_df(state)
    real_col = column if column in df.columns else _most_similar_column(column, list(df.columns))[0]
    if real_col is None:
        raise ValueError(f"Column not found: {column}")
    work = _apply_filters(df, filters)
    s = pd.to_numeric(work[real_col], errors="coerce")
    val = getattr(s, agg)()
    note = "" if real_col == column else f" (interpreted as '{real_col}')"
    state["calc_result"] = f"{agg.upper()}({column}){note} = {val}"
    state["csv_last_action"] = "aggregate"
    return state["calc_result"]

@tool("group_aggregate", args_schema=_GroupAggArg)
def group_aggregate(by: str, target: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None, state: Dict[str, Any] = None) -> str:
    """Group by a key column and aggregate a value column; fuzzy names accepted."""
    df = _ensure_df(state)
    by_real = by if by in df.columns else _most_similar_column(by, list(df.columns))[0]
    tgt_real = target if target in df.columns else _most_similar_column(target, list(df.columns))[0]
    if by_real is None or tgt_real is None:
        raise ValueError(f"Column not found: by='{by}', target='{target}'")
    work = _apply_filters(df, filters)
    res = getattr(work.groupby(by_real)[tgt_real], agg)().reset_index()
    text = f"GROUP BY {by_real} {agg}({tgt_real})\n\n" + res.to_markdown(index=False)
    state["calc_result"] = text
    state["csv_last_action"] = "group_aggregate"
    return text

@tool("arith_column", args_schema=_ArithArg)
def arith_column(column: str, op: str, scalar: float, in_place: bool = True, new_column: Optional[str] = None,
                 filters: Optional[List[Dict[str, Any]]] = None, state: Dict[str, Any] = None) -> str:
    """Column-wise arithmetic with a scalar (add/sub/mul/div); fuzzy name accepted."""
    df = _ensure_df(state)
    real_col = column if column in df.columns else _most_similar_column(column, list(df.columns))[0]
    if real_col is None:
        raise ValueError(f"Column not found: {column}")
    work_idx = _apply_filters(df, filters).index
    series = pd.to_numeric(df.loc[work_idx, real_col], errors="coerce")
    if op == "add": res = series + scalar
    elif op == "sub": res = series - scalar
    elif op == "mul": res = series * scalar
    elif op == "div": res = series / (scalar if scalar != 0 else np.nan)
    else: raise ValueError("op must be one of add/sub/mul/div")

    target_col = real_col if in_place else (new_column or f"{real_col}_{op}{scalar}")
    df.loc[work_idx, target_col] = res  # write back
    state["csv_df"] = df
    state["csv_last_action"] = "arith_column"
    preview = _md_preview(df, n=5)
    result = f"{op.upper()} {real_col} by {scalar} -> wrote '{target_col}'\n\n{preview}"
    state["calc_result"] = result
    return result

@tool("download_csv")
def download_csv(state: Dict[str, Any] = None) -> bytes:
    """Return current CSV bytes after any transformations."""
    df = _ensure_df(state)
    return df.to_csv(index=False).encode("utf-8")

# -------------------------------------------------------------------------
# Calculator (LLM tool-calling) agent
# -------------------------------------------------------------------------
_SYS_PROMPT = (
    "You are a strict data tool orchestrator.\n"
    "Use the CSV tools to answer. If column names are fuzzy or ambiguous, call list_columns and resolve_column first.\n"
    "Never hallucinate numbers; always compute with a tool. Prefer concise answers with markdown tables for tabular results."
)

def calc_llm_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    model = get_config_value(cfg.planner_model)
    base_url = get_config_value(cfg.base_url)
    temperature = get_config_value(cfg.model_temp)

    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=getattr(conf, "vllm_api_key", None),
        http_client=http_client,
        temperature=temperature,
        max_tokens=2048,
        seed=42,
    )

    # Bind tools (inject state into each tool)
    tools = [
        list_columns.bind({"state": state}),
        preview_csv.bind({"state": state}),
        resolve_column.bind({"state": state}),
        aggregate.bind({"state": state}),
        group_aggregate.bind({"state": state}),
        arith_column.bind({"state": state}),
        download_csv.bind({"state": state}),
    ]
    llm = llm.bind_tools(tools)

    # Conversation assembly
    history = trim_messages({
        "state": {"messages": state.get("messages", [])},
        "strategy": "last",
        "token_counter": count_tokens_approximately,
        "max_tokens": 4096,
    })
    messages = add_messages(history, [SystemMessage(_SYS_PROMPT), HumanMessage(state["query"])])

    # Tool loop
    resp = llm.invoke(messages)
    tool_steps: List[Tuple[str, Dict[str, Any], Any]] = []
    name_to_tool = {t.name: t for t in tools}

    while getattr(resp, "tool_calls", None):
        for tc in resp.tool_calls:
            tname = tc["name"]
            targs = tc["args"]
            try:
                result = name_to_tool[tname].invoke(targs)
            except Exception as e:
                result = f"ToolError: {e}"
            # record to transcript for better follow-ups
            messages += [AIMessage(content=None, tool_calls=[tc]), SystemMessage(content=f"[{tname}] result:\n{result}")]
            tool_steps.append((tname, targs, result))
        resp = llm.invoke(messages)

    final_text = remove_think_tags(getattr(resp, "content", "") or "")
    # write back to state-like output
    out: Dict[str, Any] = {
        "messages": [AIMessage(final_text)],
        "calc_result": state.get("calc_result"),
        "calc_tool_steps": tool_steps,
    }
    # If df exists, expose it so UI can allow download
    if isinstance(state.get("csv_df"), pd.DataFrame):
        out["csv_df"] = state["csv_df"]
    return out

# -------------------------------------------------------------------------
# Original BSO flow (kept minimal and intact)
# -------------------------------------------------------------------------
async def retrieval_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    vector_store = get_config_value(cfg.vector_store)
    q = state["query"]
    ctx = chroma_search(query=q, vector_store=vector_store)
    return {"rag_context": ctx, "messages": [HumanMessage(q), AIMessage(ctx)]}

def bso_init(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    bsm_file = "./data/BSO_prompt_CT.txt"
    bso_prompt = Path(bsm_file).read_text(encoding="utf-8")
    table_path = Path("./src/prompts/sample_tables.md")
    bso_table = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
    return {"bso_prompt": bso_prompt, "bso_table": bso_table}

def bso_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    planner_model = get_config_value(cfg.planner_model)
    temperature = get_config_value(cfg.model_temp)
    base_url = get_config_value(cfg.base_url)

    llm = ChatOpenAI(
        base_url=base_url,
        model=planner_model,
        api_key=getattr(conf, "vllm_api_key", None),
        http_client=http_client,
        temperature=temperature,
        max_tokens=4096,
        seed=42,
    )
    context = f"[FD overview table]\n\n{state['bso_table']}\n\nThis is the provided [FD overview table] for your analysis."
    history = trim_messages({
        "state": {"messages": state.get("messages", [])},
        "strategy": "last",
        "token_counter": count_tokens_approximately,
        "max_tokens": 64096,
    })
    msgs = add_messages(history, [
        SystemMessage(content=state["bso_prompt"]),
        AIMessage(content=context),
        HumanMessage(content=state["query"]),
    ])
    stream = llm.stream(msgs, stream_usage=True)
    try:
        first = next(stream); full = getattr(first, "content", "") or ""
    except StopIteration:
        full = ""
    for ch in stream:
        full += getattr(ch, "content", "") or ""
    text = remove_think_tags(full)
    return {"bso_messages": [AIMessage(text)], "messages": [AIMessage(text)]}

# -------------------------------------------------------------------------
# Router & Graph wiring
# -------------------------------------------------------------------------
def _looks_like_math(q: str) -> bool:
    ql = q.lower()
    hints = ["sum", "total", "average", "mean", "min", "max", "count", "median", "std",
             "multiply", "divide", "add", "subtract", "group by", "filter "]
    return any(h in ql for h in hints)

def router(state: QueryState) -> str:
    has_csv = any(k in state and state[k] for k in ("csv_df", "csv_bytes", "csv_text", "csv_path", "uploaded_file"))
    return "calc_llm_agent" if has_csv or _looks_like_math(state.get("query", "")) else "retrieval_agent"

graph_builder = StateGraph(QueryState, config_schema=Configuration)
graph_builder.add_node("router", router)
graph_builder.add_node("calc_llm_agent", calc_llm_agent)
graph_builder.add_node("retrieval_agent", retrieval_agent)
graph_builder.add_node("bso_init", bso_init)
graph_builder.add_node("bso_agent", bso_agent)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", lambda s: router(s), {
    "calc_llm_agent": "calc_llm_agent",
    "retrieval_agent": "retrieval_agent",
})
graph_builder.add_edge("calc_llm_agent", END)
graph_builder.add_edge("retrieval_agent", "bso_init")
graph_builder.add_edge("bso_init", "bso_agent")
graph_builder.add_edge("bso_agent", END)

app = graph_builder.compile()
