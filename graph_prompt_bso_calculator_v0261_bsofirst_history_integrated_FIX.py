# -*- coding: utf-8 -*-
"""
graph_prompt_bso_calculator_v0261_bsofirst_history_integrated_FIX.py
- LangGraph 0.2.61 style
- BSO-first
- History window
- **Explicit SYNTHESIZE branch** feeding CALC_RESULT into BSO and returning a user-facing answer
NOTE: This file references your project modules under src.*; adjust imports to your tree as needed.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import httpx
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.state import QueryState
from src.configuration import Configuration
from src.utils.config import Config
from src.utils.utils import get_config_value, chroma_search
from src.utils.utils_macro import remove_think_tags

# calculator helpers
from src.utils.calculator import (
    set_global_state, get_global_state, ensure_df, apply_filters,
    most_similar_column, md_preview
)

# ----- env & logging -----
os.chdir(Path(__file__).resolve().parent.parent)
conf = Config()

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

llm_headers = conf.vllm_payload_to_dict()
langchain_client = httpx.Client(verify=False, headers=llm_headers)
async_langchain_client = httpx.AsyncClient(verify=False, headers=llm_headers)

# guards
MAX_LOOPS = 5
MAX_TOOL_HOPS = 6
MAX_SAME_CALLS = 2
MAX_HISTORY_TOKENS = 6000

# ===== history utils =====
def _ensure_history(state: Dict[str, Any]) -> None:
    if "chat_history" not in state or not isinstance(state["chat_history"], list):
        state["chat_history"] = []

def _append_history(state: Dict[str, Any], role: str, content: str) -> None:
    _ensure_history(state)
    state["chat_history"].append({"role": role, "content": content or ""})

def _reconstruct_history_msgs(state: Dict[str, Any]) -> List[Any]:
    _ensure_history(state)
    msgs: List[Any] = []
    for m in state["chat_history"]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content))
        elif role == "assistant":
            msgs.append(AIMessage(content))
        elif role == "system":
            msgs.append(SystemMessage(content))
        else:
            msgs.append(SystemMessage(content))
    # rolling window
    tokens = count_tokens_approximately(msgs)
    if tokens > MAX_HISTORY_TOKENS:
        i = 0
        while i < len(msgs) and tokens > MAX_HISTORY_TOKENS:
            tokens -= count_tokens_approximately([msgs[i]])
            i += 1
        msgs = msgs[i:]
    return msgs

# ===== CSV tools =====
class _PreviewArg(BaseModel):
    n: int = Field(5, ge=1, le=50)

class _ResolveArg(BaseModel):
    fuzzy_name: str
    min_ratio: float = Field(0.6, ge=0.0, le=1.0)
    topk: int = Field(5, ge=1, le=20)

class _AggArg(BaseModel):
    column: str
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
    operand: str
    in_place: bool = True
    new_column: Optional[str] = None
    filters: Optional[List[Dict[str, Any]]] = None

@tool("list_columns")
def list_columns() -> List[str]:
    df = ensure_df()
    cols = list(df.columns)
    st = get_global_state()
    if isinstance(st, dict):
        st["csv_columns"] = cols
    return cols

@tool("preview_csv", args_schema=_PreviewArg)
def preview_csv(n: int = 5) -> str:
    df = ensure_df()
    return md_preview(df, n=n)

@tool("resolve_column", args_schema=_ResolveArg)
def resolve_column(fuzzy_name: str, min_ratio: float = 0.6, topk: int = 5) -> Dict[str, Any]:
    df = ensure_df()
    best, cands = most_similar_column(fuzzy_name, list(df.columns), min_ratio=min_ratio)
    return {"best": best, "candidates": cands[:topk], "exact": best is not None}

@tool("aggregate", args_schema=_AggArg)
def aggregate(column: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None) -> str:
    df = ensure_df()
    real_col = column if column in df.columns else most_similar_column(column, list(df.columns))[0]
    if real_col is None:
        raise ValueError(f"Column not found: {column}")
    work = apply_filters(df, filters)
    s = pd.to_numeric(work[real_col], errors="coerce")
    val = getattr(s, agg)()
    msg = f"{agg.upper()}({real_col}) = {val}"
    st = get_global_state()
    if isinstance(st, dict):
        st["calc_result"] = msg
    return msg

@tool("group_aggregate", args_schema=_GroupAggArg)
def group_aggregate(by: str, target: str, agg: str = "sum",
                    filters: Optional[List[Dict[str, Any]]] = None) -> str:
    df = ensure_df()
    by_real  = by     if by     in df.columns else most_similar_column(by,     list(df.columns))[0]
    tgt_real = target if target in df.columns else most_similar_column(target, list(df.columns))[0]
    if by_real is None or tgt_real is None:
        raise ValueError("Column not found.")
    work = apply_filters(df, filters)
    res = getattr(work.groupby(by_real)[tgt_real], agg)().reset_index()
    return res.to_markdown(index=False)

@tool("arith_column", args_schema=_ArithArg)
def arith_column(column: str, op: str, operand: str, in_place: bool = True,
                 new_column: Optional[str] = None,
                 filters: Optional[List[Dict[str, Any]]] = None) -> str:
    df = ensure_df()
    real_col = column if column in df.columns else most_similar_column(column, list(df.columns))[0]
    if real_col is None:
        raise ValueError(f"Column not found: {column}")
    if operand in df.columns:
        op_series = pd.to_numeric(df[operand], errors="coerce")
        operand_desc = f"column '{operand}'"
    else:
        try:
            op_value = float(operand)
            op_series = op_value
            operand_desc = str(op_value)
        except ValueError:
            fuzzy_col, _ = most_similar_column(operand, list(df.columns))
            if fuzzy_col:
                op_series = pd.to_numeric(df[fuzzy_col], errors="coerce")
                operand_desc = f"column '{fuzzy_col}' (fuzzy matched)"
            else:
                raise ValueError(f"Operand '{operand}' is neither a column nor a numeric value")
    work_idx = apply_filters(df, filters).index
    left = pd.to_numeric(df.loc[work_idx, real_col], errors="coerce")
    if   op == "add": res = left + op_series
    elif op == "sub": res = left - op_series
    elif op == "mul": res = left * op_series
    elif op == "div": res = left / (op_series if not (isinstance(op_series, (int, float)) and op_series == 0) else np.nan)
    else: raise ValueError("op must be one of add/sub/mul/div")
    target_col = real_col if in_place else (new_column or f"{real_col}_{op}_{operand}")
    df.loc[work_idx, target_col] = res
    st = get_global_state()
    if isinstance(st, dict):
        st["csv_df"] = df
        st["calc_result"] = f"{real_col} {op} {operand_desc} -> '{target_col}'"
    preview = md_preview(df, n=5)
    return f"{real_col} {op} {operand_desc} -> '{target_col}'\\n\\n{preview}"

# ===== Retrieval =====
async def retrieval_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    ctx = chroma_search(query=state["query"], vector_store=get_config_value(cfg.vector_store))
    return {"rag_context": ctx, "messages": [AIMessage(ctx)]}

# ===== BSO init =====
def bso_init(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    if state.get("query"):
        _append_history(state, "user", state["query"])
    bso_prompt = Path("./data/BSO_prompt_CT.txt").read_text(encoding="utf-8")
    bso_table  = Path("./src/prompts/sample_tables.md").read_text(encoding="utf-8") \
                 if Path("./src/prompts/sample_tables.md").exists() else ""
    # CSV schema for smarter decisions
    csv_schema = ""
    try:
        df = ensure_df()
        preview_cols = [f"{c}({str(df[c].dtype)})" for c in df.columns[:40]]
        csv_schema = "Columns: " + ", ".join(preview_cols)
        sample = df.head(3).to_markdown(index=False)
        csv_schema += f"\\nSample(3rows):\\n{sample}"
    except Exception:
        pass
    return {"bso_prompt": bso_prompt, "bso_table": bso_table, "csv_schema": csv_schema,
            "loop_count": state.get("loop_count", 0)}

# ===== BSO agent =====
_BSO_SYSTEM = (
    "You are a banking business analyst. Use RAG context and domain prompts.\\n"
    "There are two modes:\\n"
    "  - DECIDE: Decide if answering needs computations on the uploaded CSV.\\n"
    "            Base your decision on QUESTION semantics + CSV_SCHEMA + CSV availability.\\n"
    "            Emit ONLY ONE JSON line:\\n"
    "            {\\\"need_calc\\\": true|false, \\\"calc_plan\\\": \\\"one-line instruction or empty\\\"}\\n"
    "  - SYNTHESIZE: CALC_RESULT is provided. PRODUCE a FINAL user-facing answer that INTEGRATES:\\n"
    "       * The user's question intent\\n"
    "       * CALC_RESULT (numbers/tables) — extract key values explicitly\\n"
    "       * RAG_CONTEXT / DOMAIN prompts (if relevant)\\n"
    "     Output should be concise, numeric where needed, and avoid any JSON.\\n"
)

def _has_csv_on_state(st: Dict[str, Any]) -> bool:
    return any(st.get(k) is not None for k in ("csv_df","csv_bytes","csv_text","csv_path","uploaded_file"))

def bso_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        base_url=get_config_value(cfg.base_url),
        model=get_config_value(cfg.planner_model),
        api_key=conf.vllm_api_key,
        http_client=async_langchain_client,
        temperature=get_config_value(cfg.model_temp),
        max_tokens=4096,
        seed=42,
    )
    has_csv = _has_csv_on_state(state)
    calc_result = state.get("calc_result") or ""
    need_synthesis = bool(calc_result) and not bool(state.get("bso_summarized"))
    hist_msgs = _reconstruct_history_msgs(state)

    if need_synthesis:
        # ---------- SYNTHESIZE branch (explicit integration) ----------
        msgs: List = [
            SystemMessage(_BSO_SYSTEM),
            SystemMessage("MODE=SYNTHESIZE"),
            SystemMessage(f"CSV_AVAILABLE={has_csv}"),
            SystemMessage(f"CSV_SCHEMA:\\n{state.get('csv_schema','')}"),
            SystemMessage(f"RAG_CONTEXT:\\n{state.get('rag_context','')}"),
            SystemMessage(state.get("bso_prompt","")),
            AIMessage(f"[FD overview table]\\n\\n{state.get('bso_table','')}"),
        ] + hist_msgs + [
            SystemMessage(f"CALC_RESULT:\\n{calc_result}"),   # <<<<< feed calculator output here
        ]
        out = llm.invoke(msgs)
        text = remove_think_tags(getattr(out, "content", "")).strip()
        _append_history(state, "assistant", text)
        # optional: clear calc_result after integration to avoid duplicate synthesis
        state["calc_result"] = ""
        return {
            "need_calc": False,
            "bso_summarized": True,
            "messages": [AIMessage(text)],
        }

    # ---------- DECIDE branch ----------
    msgs: List = [
        SystemMessage(_BSO_SYSTEM),
        SystemMessage("MODE=DECIDE"),
        SystemMessage(f"CSV_AVAILABLE={has_csv}"),
        SystemMessage(f"CSV_SCHEMA:\\n{state.get('csv_schema','')}"),
        SystemMessage(f"RAG_CONTEXT:\\n{state.get('rag_context','')}"),
        SystemMessage(state.get("bso_prompt","")),
        AIMessage(f"[FD overview table]\\n\\n{state.get('bso_table','')}"),
    ] + hist_msgs
    out = llm.invoke(msgs)
    text = remove_think_tags(getattr(out, "content", "")).strip()

    need_calc_flag, calc_plan = False, ""
    try:
        first = text.splitlines()[0]
        obj = json.loads(first)
        if isinstance(obj, dict):
            need_calc_flag = bool(obj.get("need_calc", False))
            calc_plan = str(obj.get("calc_plan", "")).strip()
    except Exception:
        need_calc_flag, calc_plan = False, ""

    if need_calc_flag and has_csv:
        return {
            "need_calc": True,
            "calc_plan": calc_plan or state.get("query",""),
            "bso_summarized": False,
            "messages": [AIMessage(f"[BSO] Need calculation: {calc_plan or state.get('query','')}")],
        }

    _append_history(state, "assistant", text if not need_calc_flag else f"(No CSV available) {text}")
    return {
        "need_calc": False,
        "bso_summarized": True,
        "messages": [AIMessage(text if not need_calc_flag else f"(No CSV available) {text}")],
    }

# ===== Calculator agent =====
_CALC_SYS = (
    "You are a strict data tool orchestrator.\\n"
    "Use ONLY the CSV tools to compute the requested result.\\n"
    "Never hallucinate numbers. If a column name is fuzzy, resolve it first.\\n"
)

def _tc_fp(tc: dict) -> str:
    import hashlib, json as _json
    return hashlib.sha1(_json.dumps(tc, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def calc_llm_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    set_global_state(state)
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        base_url=get_config_value(cfg.base_url),
        model=get_config_value(cfg.planner_model),
        api_key=conf.vllm_api_key,
        http_client=langchain_client,
        temperature=get_config_value(cfg.model_temp),
        max_tokens=4096,
        seed=42,
    )
    tools = [list_columns, preview_csv, resolve_column, aggregate, group_aggregate, arith_column]
    tool_map = {t.name: t for t in tools}
    llm = llm.bind_tools(tools)

    plan = state.get("calc_plan") or state.get("query","")
    msgs: List = [SystemMessage(_CALC_SYS), HumanMessage(plan)]
    hops = 0
    seen: Dict[str, int] = {}

    resp = llm.invoke(msgs)
    while True:
        tcs = getattr(resp, "tool_calls", None)
        if not tcs:
            break
        hops += 1
        if hops > MAX_TOOL_HOPS:
            msgs.append(AIMessage(f"[Guard] MAX_TOOL_HOPS={MAX_TOOL_HOPS} reached."))
            break
        for tc in tcs:
            fp = _tc_fp(tc)
            seen[fp] = seen.get(fp, 0) + 1
            msgs.append(AIMessage(content="", tool_calls=[tc]))
            if seen[fp] > MAX_SAME_CALLS:
                msgs.append(ToolMessage(tool_call_id=tc["id"], content=f"[Guard] repeated call skipped: {tc['name']}"))
                continue
            tool = tool_map.get(tc["name"])
            if tool is None:
                msgs.append(ToolMessage(tool_call_id=tc["id"], content=f"[ToolNotFound] {tc['name']}"))
                continue
            try:
                result = tool.invoke(tc.get("args", {}))
            except Exception as e:
                result = f"[ToolError] {type(e).__name__}: {e}"
            msgs.append(ToolMessage(tool_call_id=tc["id"], content=str(result)))
        resp = llm.invoke(msgs)

    final_text = remove_think_tags(getattr(resp, "content", "")) or "[Guard] No content from model."
    new_loop = int(state.get("loop_count", 0)) + 1
    st = get_global_state()
    if isinstance(st, dict):
        st["calc_result"] = final_text
        st["loop_count"] = new_loop
    _append_history(state, "system", f"[CALC_RESULT]\\n{final_text}")
    return {
        "need_calc": False,
        "calc_result": final_text,
        "loop_count": new_loop,
        "messages": [AIMessage(f"[CALC DONE] {final_text}")],
    }

# ===== Branching (after BSO only; BSO-first) =====
def branch_after_bso(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    return {}

def decide_after_bso(state: QueryState) -> str:
    if state.get("need_calc"):
        return "calc_llm_agent"
    if state.get("calc_result") and not state.get("bso_summarized") and int(state.get("loop_count", 0)) < MAX_LOOPS:
        return "bso_agent"
    return "__end__"

# ===== Build graph =====
graph = StateGraph(QueryState, config_schema=Configuration)
graph.add_node("retrieval_agent", retrieval_agent)
graph.add_node("bso_init", bso_init)
graph.add_node("bso_agent", bso_agent)
graph.add_node("calc_llm_agent", calc_llm_agent)
graph.add_node("branch_after_bso", branch_after_bso)

graph.add_edge(START, "retrieval_agent")
graph.add_edge("retrieval_agent", "bso_init")
graph.add_edge("bso_init", "bso_agent")

graph.add_edge("bso_agent", "branch_after_bso")
graph.add_conditional_edges(
    "branch_after_bso",
    decide_after_bso,
    {
        "calc_llm_agent": "calc_llm_agent",
        "bso_agent": "bso_agent",
        "__end__": END,
    },
)
graph.add_edge("calc_llm_agent", "bso_agent")

app = graph.compile()
