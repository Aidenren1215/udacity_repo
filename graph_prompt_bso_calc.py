# -*- coding: utf-8 -*-
"""
graph_prompt_bso_calc.py

Calculator agent + BSO agent 集成版本
- Calculator：CSV 计算（聚合、分组、算术、列名解析）
- BSO：原有业务流程
- Router：自动判断走计算流还是 BSO 流
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

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

from src.state import QueryState
from src.configuration import Configuration
from src.utils.config import Config
from src.utils.utils import get_config_value, chroma_search
from src.utils.utils_macro import remove_think_tags

# 导入计算辅助函数
from src.utils.calculator_utils import ensure_df, apply_filters, most_similar_column, md_preview

# -------------------------------------------------------------------------
# 初始化
# -------------------------------------------------------------------------
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
# 工具定义
# -------------------------------------------------------------------------
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
def list_columns(state: Dict[str, Any] = None) -> List[str]:
    df = ensure_df(state)
    cols = list(df.columns)
    state["csv_columns"] = cols
    return cols


@tool("preview_csv", args_schema=_PreviewArg)
def preview_csv(n: int = 5, state: Dict[str, Any] = None) -> str:
    df = ensure_df(state)
    return md_preview(df, n=n)


@tool("resolve_column", args_schema=_ResolveArg)
def resolve_column(fuzzy_name: str, min_ratio: float = 0.6, topk: int = 5, state: Dict[str, Any] = None) -> Dict[str, Any]:
    df = ensure_df(state)
    best, cands = most_similar_column(fuzzy_name, list(df.columns), min_ratio=min_ratio)
    return {"best": best, "candidates": cands[:topk], "exact": best is not None}


@tool("aggregate", args_schema=_AggArg)
def aggregate(column: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None, state: Dict[str, Any] = None) -> str:
    df = ensure_df(state)
    real_col = column if column in df.columns else most_similar_column(column, list(df.columns))[0]
    if real_col is None:
        raise ValueError(f"Column not found: {column}")
    work = apply_filters(df, filters)
    s = pd.to_numeric(work[real_col], errors="coerce")
    val = getattr(s, agg)()
    state["calc_result"] = f"{agg.upper()}({real_col}) = {val}"
    return state["calc_result"]


@tool("group_aggregate", args_schema=_GroupAggArg)
def group_aggregate(by: str, target: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None, state: Dict[str, Any] = None) -> str:
    df = ensure_df(state)
    by_real = by if by in df.columns else most_similar_column(by, list(df.columns))[0]
    tgt_real = target if target in df.columns else most_similar_column(target, list(df.columns))[0]
    work = apply_filters(df, filters)
    res = getattr(work.groupby(by_real)[tgt_real], agg)().reset_index()
    return res.to_markdown(index=False)


@tool("arith_column", args_schema=_ArithArg)
def arith_column(column: str, op: str, operand: str, in_place: bool = True,
                 new_column: Optional[str] = None, filters: Optional[List[Dict[str, Any]]] = None,
                 state: Dict[str, Any] = None) -> str:
    """对列执行加减乘除运算，可为列 vs 数字或列 vs 列。"""
    df = ensure_df(state)
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

    if op == "add":
        res = left + op_series
    elif op == "sub":
        res = left - op_series
    elif op == "mul":
        res = left * op_series
    elif op == "div":
        res = left / (op_series if not (isinstance(op_series, (int, float)) and op_series == 0) else np.nan)
    else:
        raise ValueError("op must be one of add/sub/mul/div")

    target_col = real_col if in_place else (new_column or f"{real_col}_{op}_{operand}")
    df.loc[work_idx, target_col] = res
    state["csv_df"] = df

    result_summary = f"{real_col} {op} {operand_desc} -> 写入列 '{target_col}'"
    preview = md_preview(df, n=5)
    state["calc_result"] = result_summary
    return result_summary + "\n\n" + preview


# -------------------------------------------------------------------------
# Calculator Agent
# -------------------------------------------------------------------------
_SYS_PROMPT = (
    "You are a strict data tool orchestrator.\n"
    "Use the CSV tools to answer. If column names are fuzzy, call list_columns/resolve_column first.\n"
    "Never hallucinate numbers."
)


def calc_llm_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        base_url=get_config_value(cfg.base_url),
        model=get_config_value(cfg.planner_model),
        api_key=getattr(conf, "vllm_api_key", None),
        http_client=http_client,
        temperature=get_config_value(cfg.model_temp),
    )
    tools = [
        list_columns.bind({"state": state}),
        preview_csv.bind({"state": state}),
        resolve_column.bind({"state": state}),
        aggregate.bind({"state": state}),
        group_aggregate.bind({"state": state}),
        arith_column.bind({"state": state}),
    ]
    llm = llm.bind_tools(tools)

    messages = add_messages(
        trim_messages({"state": {"messages": state.get("messages", [])}, "strategy": "last", "token_counter": count_tokens_approximately}),
        [SystemMessage(_SYS_PROMPT), HumanMessage(state["query"])]
    )
    resp = llm.invoke(messages)
    while getattr(resp, "tool_calls", None):
        for tc in resp.tool_calls:
            result = tools[[t.name for t in tools].index(tc["name"])].invoke(tc["args"])
            messages += [AIMessage(content=None, tool_calls=[tc]), SystemMessage(content=str(result))]
        resp = llm.invoke(messages)
    return {"messages": [AIMessage(remove_think_tags(getattr(resp, "content", "")))]}


# -------------------------------------------------------------------------
# BSO 流程 & Router
# -------------------------------------------------------------------------
async def retrieval_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    ctx = chroma_search(query=state["query"], vector_store=get_config_value(Configuration.from_runnable_config(config).vector_store))
    return {"rag_context": ctx, "messages": [HumanMessage(state["query"]), AIMessage(ctx)]}


def bso_init(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    return {
        "bso_prompt": Path("./data/BSO_prompt_CT.txt").read_text(encoding="utf-8"),
        "bso_table": Path("./src/prompts/sample_tables.md").read_text(encoding="utf-8") if Path("./src/prompts/sample_tables.md").exists() else ""
    }


def bso_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    llm = ChatOpenAI(
        base_url=get_config_value(Configuration.from_runnable_config(config).base_url),
        model=get_config_value(Configuration.from_runnable_config(config).planner_model),
        http_client=http_client,
    )
    msgs = [
        SystemMessage(state["bso_prompt"]),
        AIMessage(f"[FD overview table]\n\n{state['bso_table']}"),
        HumanMessage(state["query"]),
    ]
    result = llm.invoke(msgs)
    return {"bso_messages": [AIMessage(remove_think_tags(result.content))]}


def _looks_like_math(q: str) -> bool:
    return any(k in q.lower() for k in ["sum", "mean", "max", "min", "count", "group by", "filter"])


def router(state: QueryState) -> str:
    return "calc_llm_agent" if any(state.get(k) for k in ("csv_df", "csv_bytes", "csv_text")) or _looks_like_math(state["query"]) else "retrieval_agent"


# -------------------------------------------------------------------------
# 构建 LangGraph
# -------------------------------------------------------------------------
graph = StateGraph(QueryState, config_schema=Configuration)
graph.add_node("router", router)
graph.add_node("calc_llm_agent", calc_llm_agent)
graph.add_node("retrieval_agent", retrieval_agent)
graph.add_node("bso_init", bso_init)
graph.add_node("bso_agent", bso_agent)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", lambda s: router(s), {
    "calc_llm_agent": "calc_llm_agent",
    "retrieval_agent": "retrieval_agent",
})
graph.add_edge("calc_llm_agent", END)
graph.add_edge("retrieval_agent", "bso_init")
graph.add_edge("bso_init", "bso_agent")
graph.add_edge("bso_agent", END)

app = graph.compile()
