# -*- coding: utf-8 -*-
"""
graph_prompt_bso_calculator_v0261_prebranch_history.py — 适配 langgraph 0.2.61（先分流→再 BSO）+ 历史对话支持
改动点：
  - 新增 chat history：state["chat_history"] 以 [{"role":"user"/"assistant","content":str}, ...] 存储
  - 在 bso_agent 中将历史消息重建为 LangChain Messages，并做 token 截断（rolling window）
  - 每轮：进入 BSO 前把本轮用户 query 追加到历史；BSO 输出后把回答也写回历史
  - Calculator 的“完成提示/要点”也会以系统注记写入历史，便于 BSO 统一整合
  - 仍保留：Calculator 只产出数据，BSO 统一整合并出声；多轮通过 MAX_LOOPS 防死循环
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import httpx
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.state import QueryState
from src.configuration import Configuration
from src.utils.config import Config  # 你项目里的 Config（保持你原来的用法）
from src.utils.utils import get_config_value, chroma_search
from src.utils.utils_macro import remove_think_tags

# 计算相关：全局 state + 工具底层函数（确保 ensure_df 为“无参”版本）
from src.utils.calculator import (
    set_global_state, get_global_state, ensure_df, apply_filters,
    most_similar_column, md_preview
)

# ==================================
# 基础环境（保持你的 headers / client 写法）
# ==================================
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

# 保护上限
MAX_LOOPS = 5         # BSO↔CALC 回环最大次数
MAX_TOOL_HOPS = 6     # calculator 内部最多工具跳数
MAX_SAME_CALLS = 2    # 相同 tool_call 的最大重复次数
MAX_HISTORY_TOKENS = 6000  # BSO 的历史滚动窗口 token 上限（近似计数）

# ===========================
# 历史消息工具
# ===========================
def _ensure_history(state: Dict[str, Any]) -> None:
    if "chat_history" not in state or not isinstance(state["chat_history"], list):
        state["chat_history"] = []

def _append_history(state: Dict[str, Any], role: str, content: str) -> None:
    _ensure_history(state)
    state["chat_history"].append({"role": role, "content": content or ""})

def _reconstruct_history_msgs(state: Dict[str, Any]) -> List[Any]:
    """将 state['chat_history'] 重建为 LangChain Message 列表，并做 token 窗口截断"""
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
            # 兜底当作 system 记录
            msgs.append(SystemMessage(content))

    # 近似 token 计数，超过窗口则从头部裁剪（保留最近的）
    tokens = count_tokens_approximately(msgs)
    if tokens > MAX_HISTORY_TOKENS:
        # 简单滚动窗口：去掉最早的消息直到满足阈值
        i = 0
        while i < len(msgs) and tokens > MAX_HISTORY_TOKENS:
            tokens -= count_tokens_approximately([msgs[i]])
            i += 1
        msgs = msgs[i:]
    return msgs

# ===========================
# CSV 工具
# ===========================
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
    operand: str                         # 数字或列名（支持模糊）
    in_place: bool = True
    new_column: Optional[str] = None
    filters: Optional[List[Dict[str, Any]]] = None

@tool("list_columns")
def list_columns() -> List[str]:
    """Return the list of column names in the uploaded CSV file"""
    df = ensure_df()
    cols = list(df.columns)
    st = get_global_state()
    if isinstance(st, dict):
        st["csv_columns"] = cols
    return cols

@tool("preview_csv", args_schema=_PreviewArg)
def preview_csv(n: int = 5) -> str:
    """Return the first n rows of the CSV as a Markdown table preview"""
    df = ensure_df()
    return md_preview(df, n=n)

@tool("resolve_column", args_schema=_ResolveArg)
def resolve_column(fuzzy_name: str, min_ratio: float = 0.6, topk: int = 5) -> Dict[str, Any]:
    """Resolve a fuzzy column name into the most similar real columns in the CSV"""
    df = ensure_df()
    best, cands = most_similar_column(fuzzy_name, list(df.columns), min_ratio=min_ratio)
    return {"best": best, "candidates": cands[:topk], "exact": best is not None}

@tool("aggregate", args_schema=_AggArg)
def aggregate(column: str, agg: str = "sum", filters: Optional[List[Dict[str, Any]]] = None) -> str:
    """Compute an aggregation (sum, mean, max, etc.) over a column, with optional filters."""
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
    """Group the CSV by one column and aggregate another column (e.g., sum, mean)"""
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
    """Perform arithmetic (add/sub/mul/div). Operand can be a number or another column (fuzzy allowed)."""
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
    return f"{real_col} {op} {operand_desc} -> '{target_col}'\n\n{preview}"


# ===========================
# Retrieval
# ===========================
async def retrieval_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    ctx = chroma_search(query=state["query"], vector_store=get_config_value(cfg.vector_store))
    return {"rag_context": ctx, "messages": [AIMessage(ctx)]}

# ===========================
# BSO 初始化
# ===========================
def bso_init(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    # 将本轮用户 query 追加进历史（若没加过）
    if state.get("query"):
        _append_history(state, "user", state["query"])

    bso_prompt = Path("./data/BSO_prompt_CT.txt").read_text(encoding="utf-8")
    bso_table  = Path("./src/prompts/sample_tables.md").read_text(encoding="utf-8") \
                 if Path("./src/prompts/sample_tables.md").exists() else ""
    return {"bso_prompt": bso_prompt, "bso_table": bso_table, "loop_count": state.get("loop_count", 0)}

# ===========================
# BSO Agent（判定 + 整合 + 历史）
# ===========================
_BSO_SYSTEM = (
    "You are a banking business analyst. Use RAG context and domain prompts.\n"
    "Modes:\n"
    "  - DECIDE: Decide if answering needs computations on the uploaded CSV.\n"
    "            Emit ONLY ONE JSON line:\n"
    "            {\"need_calc\": true|false, \"calc_plan\": \"one-line instruction or empty\"}\n"
    "  - SYNTHESIZE: A calculation result (CALC_RESULT) is provided. Synthesize a final, user-facing answer,\n"
    "                integrating CALC_RESULT, RAG_CONTEXT, DOMAIN_PROMPTS and recent CHAT HISTORY. DO NOT output JSON here.\n"
)

def _has_csv_on_state(st: Dict[str, Any]) -> bool:
    return any(st.get(k) is not None for k in ("csv_df","csv_bytes","csv_text","csv_path","uploaded_file"))

def bso_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        base_url=get_config_value(cfg.base_url),
        model=get_config_value(cfg.planner_model),
        api_key=conf.vllm_api_key,
        http_client=async_langchain_client,         # 保持你的 async client
        temperature=get_config_value(cfg.model_temp),
        max_tokens=4096,
        seed=42,
    )

    has_csv = _has_csv_on_state(state)
    calc_result = state.get("calc_result") or ""
    need_synthesis = bool(calc_result) and not bool(state.get("bso_summarized"))

    # 历史消息（滚动窗口）
    hist_msgs = _reconstruct_history_msgs(state)

    # 组装上下文
    msgs: List = [
        SystemMessage(_BSO_SYSTEM),
        SystemMessage(f"MODE={'SYNTHESIZE' if need_synthesis else 'DECIDE'}"),
        SystemMessage(f"CSV_AVAILABLE={has_csv}"),
        SystemMessage(f"RAG_CONTEXT:\n{state.get('rag_context','')}"),
        SystemMessage(state.get("bso_prompt","")),
        AIMessage(f"[FD overview table]\n\n{state.get('bso_table','')}"),
    ] + hist_msgs  # 把历史插在业务提示后、当前轮输入前

    # 当前轮输入（来自 bso_init 写入的历史中最后一条 user，也可冗余再加一遍）
    if need_synthesis:
        msgs.append(SystemMessage(f"CALC_RESULT:\n{calc_result}"))

    out = llm.invoke(msgs)
    text = remove_think_tags(getattr(out, "content", "")).strip()

    # 整合模式：由 BSO 给出最终/阶段性业务答复
    if need_synthesis:
        _append_history(state, "assistant", text)
        return {
            "need_calc": False,
            "bso_summarized": True,   # 已完成整合
            "messages": [AIMessage(text)],
        }

    # 判定模式：首行 JSON 判定是否需要计算
    need_calc, calc_plan = False, ""
    try:
        first = text.splitlines()[0]
        obj = json.loads(first)
        if isinstance(obj, dict):
            need_calc = bool(obj.get("need_calc", False))
            calc_plan = str(obj.get("calc_plan", "")).strip()
    except Exception:
        # 模型未按 JSON 给出判定，视为直接回答
        need_calc, calc_plan = False, ""

    if need_calc and has_csv:
        # 这里不追加 assistant 历史，因为只是内部路由提示
        return {
            "need_calc": True,
            "calc_plan": calc_plan or state.get("query",""),
            "bso_summarized": False,
            "messages": [AIMessage(f"[BSO] Need calculation: {calc_plan or state.get('query','')}")],
        }

    # 不需要计算 / 无 CSV：作为回答写回历史
    _append_history(state, "assistant", text if not need_calc else f"(No CSV available) {text}")
    return {
        "need_calc": False,
        "bso_summarized": True,
        "messages": [AIMessage(text if not need_calc else f"(No CSV available) {text}")],
    }

# ===========================
# Calculator Agent（工具编排 + 防死循环 + 写入历史注记）
# ===========================
_CALC_SYS = (
    "You are a strict data tool orchestrator.\n"
    "Use ONLY the CSV tools to compute the requested result.\n"
    "Never hallucinate numbers. If a column name is fuzzy, resolve it first.\n"
)

def _tc_fp(tc: dict) -> str:
    import hashlib, json as _json
    return hashlib.sha1(_json.dumps(tc, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def calc_llm_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    # 让工具访问当前 state
    set_global_state(state)

    cfg = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        base_url=get_config_value(cfg.base_url),
        model=get_config_value(cfg.planner_model),
        api_key=conf.vllm_api_key,
        http_client=langchain_client,              # 同步 client
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

            # 回放模型的工具调用（content 不能为 None）
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
        st["calc_result"] = final_text   # 只写数据产物，交由 BSO 整合
        st["loop_count"] = new_loop

    # 把“关键产出”以 system 注记进历史，供 BSO 整合参考
    _append_history(state, "system", f"[CALC_RESULT]\n{final_text}")

    return {
        "need_calc": False,
        "calc_result": final_text,
        "loop_count": new_loop,
        "messages": [AIMessage(f"[CALC DONE] {final_text}")],
    }

# ===========================
# 轻量启发式：是否“明显是计算类问题”？（非 LLM，不算 Router）
# ===========================
_CALC_KEYWORDS = [
    r"\bsum\b", r"\btotal\b", r"\bmean\b", r"\bavg\b", r"\baverage\b",
    r"\bmedian\b", r"\bmin\b", r"\bmax\b", r"\bstd\b", r"\bcount\b",
    r"\bgroup\s+by\b", r"\baggregate\b", r"\bagg\b",
    r"\badd\b", r"\bsub(tract)?\b", r"\bmul(tiply)?\b", r"\bdiv(ide)?\b",
    r"求和", r"平均", r"总(和|数|计)", r"分组", r"聚合", r"加法", r"减法", r"乘法", r"除法",
]

def _looks_like_calc(query: str) -> bool:
    q = (query or "").lower()
    for pat in _CALC_KEYWORDS:
        if re.search(pat, q):
            return True
    return False

# ===========================
# 分流节点 & 条件函数（0.2.61 兼容写法）
# ===========================
def branch_before_bso(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    # 纯 pass-through，不改业务字段
    return {}

def decide_before_bso(state: QueryState) -> str:
    """先分流：如果 query 明显是计算类且有 CSV → 先去 calculator，否则 → 先去 BSO"""
    has_csv = _has_csv_on_state(state)
    if has_csv and _looks_like_calc(state.get("query", "")):
        if not state.get("calc_plan"):
            state["calc_plan"] = state.get("query", "")
        return "calc_llm_agent"
    return "bso_agent"

def branch_after_bso(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    # 纯 pass-through，不改业务字段
    return {}

def decide_after_bso(state: QueryState) -> str:
    """BSO 后续分流：如果需要计算 → calc；如果有 calc_result 且尚未整合 → 回 BSO；否则结束"""
    if state.get("need_calc"):
        return "calc_llm_agent"
    if state.get("calc_result") and not state.get("bso_summarized") and int(state.get("loop_count", 0)) < MAX_LOOPS:
        return "bso_agent"
    return "__end__"  # 在 mapping 中用 "__end__": END

# ===========================
# 装配 Graph（先分流再 BSO，0.2.61 风格 + 历史）
# ===========================
graph = StateGraph(QueryState, config_schema=Configuration)

graph.add_node("retrieval_agent", retrieval_agent)
graph.add_node("bso_init", bso_init)
graph.add_node("bso_agent", bso_agent)
graph.add_node("calc_llm_agent", calc_llm_agent)
graph.add_node("branch_before_bso", branch_before_bso)
graph.add_node("branch_after_bso", branch_after_bso)

graph.add_edge(START, "retrieval_agent")
graph.add_edge("retrieval_agent", "bso_init")

# 先进入“前置分流”（启发式判断是否先算）
graph.add_edge("bso_init", "branch_before_bso")

graph.add_conditional_edges(
    "branch_before_bso",
    decide_before_bso,   # callable（非字符串），0.2.61 可用
    {
        "calc_llm_agent": "calc_llm_agent",
        "bso_agent": "bso_agent",
    },
)

# calculator 完成后固定回 BSO，由 BSO 进行整合（并把回答追加进历史）
graph.add_edge("calc_llm_agent", "bso_agent")

# 每次经过 BSO，都再进入“后置分流”，决定是否继续算/回 BSO/结束
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

app = graph.compile()
