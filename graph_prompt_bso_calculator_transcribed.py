import os
import json
import re
import logging
import httpx
import pandas as pd
import numpy as np

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig

# >>> CHANGED/ADDED: 修正原始笔误的 import（之前是 "from src.state import import QueryState"）
from src.state import QueryState  # >>> CHANGED/ADDED

from src.configuration import Configuration
from src.utils.config import Config
from src.utils.utils import get_config_value, chroma_search
from src.utils.llm_macro import remove_think_tags, remove_channel_header
from src.utils.calculator import list_columns, preview_csv, resolve_column, aggregate, arith_column
# from src.utils.calculator import set_global_state, get_global_state, ensure_df, apply_filters, most_sim

# >>> CHANGED/ADDED: 引入 Command / interrupt 以支持审批中断与跳转
from langgraph.types import Command, interrupt  # >>> CHANGED/ADDED

os.chdir(Path(__file__).resolve().parent.parent)
conf = Config()

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

llm_headers = conf.vllm.payload.to_dict()
langchain_client = httpx.Client(verify=False, headers=llm_headers)
async_langchain_client = httpx.AsyncClient(verify=False, headers=llm_headers)

def BSO_validator(df_fd: pd.DataFrame, top_k: int = 4):
    """validate fixed deposit dataframe

    Args:
        df_fd (pd.DataFrame): FD proposal dataframe

    Returns:
        pd.DataFrame: dataframe after validation
    """
    # verify column names
    default_cols = ['FD Tenor Bucket', 'Current FD Volume ($Sm)', 'Proposed FD Volume ($Sm)', 'Change ($Sm)', '% Change']
    for col in df_fd.columns:
        if col not in default_cols:
            return "The FD Proposal Table is not generated correctly."

    # columns validation:
    # sum of change as indicator
    change_res = sum(df_fd['Change ($Sm)'])
    top_k_items = df_fd.nlargest(top_k, 'Change ($Sm)') if change_res > 0 else df_fd.nsmallest(top_k, 'Change ($Sm)')

    # scenario 1: proposed FD volume remained the same as current FD volume
    if sum(df_fd['Proposed FD Volume ($Sm)']) == sum(df_fd['Current FD Volume ($Sm)']):
        if change_res == 0:
            # pass
            pass
        else:
            # since proposed fd volume correct, recompute the change column
            df_fd['Change ($Sm)'] = df_fd['Proposed FD Volume ($Sm)'] - df_fd['Current FD Volume ($Sm)']
            df_fd['% Change'] = df_fd['Change ($Sm)'] / df_fd['Current FD Volume ($Sm)']
            df_fd['% Change'] = df_fd['% Change'].apply(lambda x: "{:.2%}".format(x))

        total_row = df_fd.select_dtypes(include='number').sum()
        total_row['FD Tenor Bucket'] = 'total'
        total_row['% Change'] = 0.0
        df_fd = pd.concat([df_fd, pd.DataFrame([total_row])], ignore_index=True)

        return df_fd.to_markdown(index=False)
    else:
        # scenario 2: proposed FD volume not equal to the current FD volume
        if change_res <= 0:
            add_on, residual = change_res*(-1)/top_k, change_res*(-1)%top_k
            df_fd.loc[top_k_items.index, 'Change ($Sm)'] = df_fd.loc[top_k_items.index, 'Change ($Sm)'] + add_on
            df_fd.loc[top_k_items.index[:1], 'Change ($Sm)'] = df_fd.loc[top_k_items.index[:1], 'Change ($Sm)'] + residual
        else:
            add_on, residual = change_res/top_k, change_res%top_k
            df_fd.loc[top_k_items.index, 'Change ($Sm)'] = df_fd.loc[top_k_items.index, 'Change ($Sm)'] + add_on
            df_fd.loc[top_k_items.index[:1], 'Change ($Sm)'] = df_fd.loc[top_k_items.index[:1], 'Change ($Sm)'] + residual

        df_fd['Proposed FD Volume ($Sm)'] = df_fd['Change ($Sm)'] + df_fd['Current FD Volume ($Sm)']
        df_fd['% Change'] = df_fd['Change ($Sm)'] / df_fd['Current FD Volume ($Sm)']
        df_fd['% Change'] = df_fd['% Change'].apply(lambda x: "{:.2%}".format(x))

        total_row = df_fd.select_dtypes(include='number').sum()
        total_row['FD Tenor Bucket'] = 'total'
        total_row['% Change'] = 0.0
        df_fd = pd.concat([df_fd, pd.DataFrame([total_row])], ignore_index=True)

        return df_fd.to_markdown(index=False)

# Calculator Agent
_SYS_PROMPT = (
    "You are a strict data tool orchestrator.\n"
    "Use the CSV tools to answer. If column names are fuzzy, call list_columns/resolve_column first.\n"
    "Never hallucinate numbers."
)

def calc_llm_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    # set_global_state(state)
    cfg = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(
        base_url=get_config_value(cfg, 'base_url'),
        model='gpt-oss-120b',
        api_key=conf.vllm.api_key,
        http_client=langchain_client,
        http_async_client=async_langchain_client,
        temperature=get_config_value(cfg, 'model_temp'),
        max_tokens=4096,
        seed=42,
    )

    tools = [list_columns, preview_csv, resolve_column, aggregate, arith_column]
    tool_map = {t.__name__: t for t in tools}
    llm = llm.bind_tools(tools)

    msgs: List = []
    msgs.append(SystemMessage(_SYS_PROMPT))
    msgs.append(HumanMessage(state['query']))

    resp = llm.invoke(msgs)
    while getattr(resp, 'tool_calls', None):
        for tc in resp.tool_calls:
            name = tc['name']
            args = tc.get('args', {}) or {}
            tool = tool_map.get(name)
            logger.info(f"calling tool {name} - args {args}")
            if tool is None:
                msgs.append(ToolMessage(tool_call_id=tc['id'], content=f'[ToolNotFound] {name}'))
                continue
            try:
                result = tool(**args)
            except Exception as e:
                result = f"[ToolError] {type(e).__name__}: {e}"
            msgs.append(ToolMessage(tool_call_id=tc['id'], content=str(result)))
        resp = llm.invoke(msgs)

    final_text = remove_channel_header(getattr(resp, 'content', ''))
    logger.info(f"result from calculator: {final_text}")

    return {
        "need_calc": False,
        "bso_summarized": False,
        "calc_result": final_text,
        "messages": [AIMessage(f"[CALC DONE] {final_text}")],
    }

# BSO workflow
async def retrieval_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    """chatbot with chroma search context"""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    vector_store = get_config_value(configurable, 'vector_store')
    context = chroma_search(query=state['query'], vector_store=vector_store)
    logger.info(context)
    return {"rag_context": context, "messages": [HumanMessage(state['query'])]}

def bso_init(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    return {
        "bso_prompt": Path("./data/BSO_prompt_CT.txt").read_text(encoding="utf-8"),
        "bso_json_schema": Path("./data/BSO_json_schema.txt").read_text(encoding="utf-8"),
    }

def _has_csv_on_state(st: Dict[str, Any]) -> bool:
    return any(st.get(k) is not None for k in ("csv_df", "csv_bytes", "csv_text"))

def bso_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    configurable = Configuration.from_runnable_config(config)
    planner_model = get_config_value(configurable, 'planner_model')
    temperature = get_config_value(configurable, 'model_temp')
    url = get_config_value(configurable, 'base_url')
    logger.info(f"Initiate BSO model ({planner_model}) with temp {temperature} ...")

    query = state['query'].split("\n")[0]
    logger.info(f"query from user displayed in BSO agent: {query}")
    logger.info(f"Found len({len(state['messages'])}) messages in the memory")

    llm = ChatOpenAI(
        base_url=url,
        model=planner_model,
        api_key=conf.vllm.api_key,
        http_client=langchain_client,
        http_async_client=async_langchain_client,
        temperature=temperature,
        max_tokens=4096,
        seed=42,
        response_format={"type": "json_object"},
    )

    # has_csv = _has_csv_on_state(state)
    calc_result = state.get('calc_result')
    logger.info(f"answer from calculator: {calc_result}")
    need_synthesis = bool(calc_result) and not bool(state.get('bso_summarized'))
    logger.info(f"whether query need to be integrated: {need_synthesis}")

    if need_synthesis:
        return {
            "need_calc": False,
            "bso_summarized": True,
            "bso_messages": [AIMessage(content=calc_result)],
        }

    # >>> CHANGED/ADDED: 注入用户对上轮 BSO 的评论（如果存在）
    user_review_comment = state.get("bso_review_comment", "").strip()
    extra_feedback_block = ""
    if user_review_comment:
        extra_feedback_block = (
            "\n\n### 用户反馈（请严格遵循）:\n"
            f"{user_review_comment}\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{state[bso_prompt]}"),
        ("system", "return ONLY a JSON object, json schema is defined as below: \n{state[bso_json_schema]}"),
        ("user", "{state[rag_context]}\n{query}" + extra_feedback_block),  # >>> CHANGED/ADDED
    ])

    result = (prompt | llm).invoke({"state": state, "query": query}).content
    res_dict = json.loads(result)
    df_fd = pd.DataFrame(res_dict['table'])
    rationale, qualitative_changes = res_dict['rationale'], res_dict['qualitative_changes']
    df_fd_md = BSO_validator(df_fd=df_fd)

    refl = (
        f"**Rationale:**\n\n{rationale}\n\n"
        f"**Qualitative changes:**\n{qualitative_changes}\n\n"
        f"{df_fd_md}"
    )

    # >>> CHANGED/ADDED: 本轮已消费评论，清空，避免污染后续轮次
    if "bso_review_comment" in state:
        state["bso_review_comment"] = ""

    return {
        "need_calc": False,
        "bso_summarized": True,
        "messages": [AIMessage(refl)],
        "bso_messages": [AIMessage(refl)],
        "bso_table": df_fd_md,
        "bso_general_strategy": refl,
    }

def fd_proposer(state: QueryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    temperature = get_config_value(configurable, 'model_temp')
    url = get_config_value(configurable, 'base_url')

    llm_fd_proposer = ChatOpenAI(
        base_url=url,
        model='gpt-oss-120b',
        api_key=conf.vllm.api_key,
        http_client=langchain_client,
        http_async_client=async_langchain_client,
        temperature=temperature,
        max_tokens=4096,
        seed=42,
    )

    fd_planner_message = Path("./src/prompts/bso_fd_proposer_instructions.md").read_text(encoding="utf-8")
    input_message = [
        SystemMessage(fd_planner_message),
        AIMessage(content=f"{state['bso_table']}"),
        HumanMessage(content='Please give me a complete reallocation plan according to the FD table you have now.'),
    ]

    stream = llm_fd_proposer.stream(input_message, stream_usage=True)
    full = next(stream)
    for chunk in stream:
        full += chunk

    refl = (
        f"{state['bso_general_strategy']}\n\n"
        f"{state.get('fd_shift_strategy','')}\n\n"
        f"{full.content}"
    )

    return {
        "messages": [AIMessage(full.content)],
        "bso_messages": [AIMessage(refl)],
        "fd_shift_strategy": full.content,
    }

def fd_extractor(state: QueryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    temperature = get_config_value(configurable, 'model_temp')
    url = get_config_value(configurable, 'base_url')

    llm_fd_proposer = ChatOpenAI(
        base_url=url,
        model='gpt-oss-120b',
        api_key=conf.vllm.api_key,
        http_client=langchain_client,
        http_async_client=async_langchain_client,
        temperature=temperature,
        max_tokens=4096,
        seed=42,
    )

    extract_prompt = Path("./src/prompts/bso_fd_extract_instructions.md").read_text(encoding="utf-8")
    input_message = [
        SystemMessage(extract_prompt),
        AIMessage(content=f"{state['fd_shift_strategy']}"),
        HumanMessage(content='Extract the reallocation table.'),
    ]

    stream = llm_fd_proposer.stream(input_message, stream_usage=True)
    full = next(stream)
    for chunk in stream:
        full += chunk

    return {"fd_shift_table": full.content}

def fd_monthly_agent(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    configurable = Configuration.from_runnable_config(config)
    temperature = get_config_value(configurable, 'model_temp')
    url = get_config_value(configurable, 'base_url')

    llm_fd_proposer = ChatOpenAI(
        base_url=url,
        model='gpt-oss-120b',
        api_key=conf.vllm.api_key,
        http_client=langchain_client,
        http_async_client=async_langchain_client,
        temperature=temperature,
        max_tokens=4096,
        seed=42,
    )

    # general shift table extracted from fd proposer
    shift_table = state['fd_shift_table']
    df_reallocation = pd.read_csv(StringIO((shift_table.strip())), sep='|')
    df_reallocation.columns = df_reallocation.columns.str.strip()
    df_reallocation['from bucket'] = df_reallocation['from bucket'].str.strip()
    df_reallocation['to bucket'] = df_reallocation['to bucket'].str.strip()
    df_reallocation['amount'] = df_reallocation['amount'].astype(float)
    df_reallocation['ratio_of_current_volume'] = (df_reallocation['ratio_of_current_volume'].astype(float) / 100)

    # maturity ladder data reading
    df_maturity = pd.read_csv(BytesIO(state['maturity_bytes']), encoding='utf-8')
    df_maturity = df_maturity.fillna(0)

    df_reallocate_monthly_plan = pd.merge(df_maturity, df_reallocation, left_on='Tenor', right_on='from bucket', how='inner')
    df_reallocate_monthly_plan.drop(columns=['Gross Rate %', 'Gross Rate % (bps)', 'amount'], inplace=True)
    df_reallocate_monthly_plan['balance_shift'] = (df_reallocate_monthly_plan['Balance $m']*df_reallocate_monthly_plan['ratio_of_current_volume']).round(2)
    df_reallocate_monthly_plan_md = df_reallocate_monthly_plan.to_markdown(index=False)

    query = state['query'].split("\n")[0]
    input_message = [
        AIMessage(content=f"{df_reallocate_monthly_plan_md}"),
        HumanMessage(content=query)
    ]

    stream = llm_fd_proposer.stream(input_message, stream_usage=True)
    full = next(stream)
    for chunk in stream:
        full += chunk

    refl = (
        f"{state['bso_general_strategy']}\n\n"
        f"{state['fd_shift_strategy']}\n\n"
        f"{full.content}"
    )

    return {
        "fd_monthly_plan": AIMessage(df_reallocate_monthly_plan_md),
        "bso_messages": [AIMessage(refl)],
    }

# >>> CHANGED/ADDED: 两个“人工审批闸门”节点（其中 BSO 审核支持 comment）
def _normalize_decision(decision: Any, default: str = "approve") -> str:
    if decision is None:
        return default
    if isinstance(decision, dict):
        val = str(decision.get("decision", default)).strip().lower()
    else:
        val = str(decision).strip().lower()
    if val in ("approve", "approved", "ok", "okay", "yes", "y", "继续", "满意"):
        return "approve"
    if val in ("retry", "redo", "reject", "no", "n", "不满意", "重跑", "重新"):
        return "retry"
    return default

def review_bso_decision(state: QueryState, config: RunnableConfig) -> Command:
    """BSO 之后的人审闸门：满意 → fd_proposer；不满意可附 comment 并重跑 bso_agent"""
    payload = {
        "type": "review_bso",
        "prompt": "请审阅 BSO 输出（含表格）。满意点“approve”，不满意点“retry”并填写修改意见。",
        # >>> CHANGED/ADDED: 发完整答案 + 表格，供前端整段展示
        "full_answer": state.get("bso_general_strategy", ""),
        "table": state.get("bso_table", ""),
        "note": "支持 approve/ok/yes 或 retry/no/reject；不满意可附带 comment。",
    }
    user_input = interrupt(payload)               # >>> CHANGED/ADDED
    # 允许 resume 传 {"decision": "...", "comment": "..."}
    decision = _normalize_decision(user_input)
    comment = ""
    if isinstance(user_input, dict):
        comment = str(user_input.get("comment", "") or "").strip()
    if comment:
        state["bso_review_comment"] = comment     # >>> CHANGED/ADDED

    if decision == "retry":
        logger.info("User rejected BSO result. Re-running bso_agent with user comment (if any).")
        return Command(jump="bso_agent")          # >>> CHANGED/ADDED
    logger.info("User approved BSO result. Proceeding to fd_proposer.")
    return Command(jump="fd_proposer")            # >>> CHANGED/ADDED

def review_fd_proposer_decision(state: QueryState, config: RunnableConfig) -> Command:
    """FD proposer 之后的人审闸门：满意 → fd_extractor；不满意 → 重跑 fd_proposer"""
    payload = {
        "type": "review_fd_proposer",
        "prompt": "请审阅 FD 重配方案。满意点“approve”，不满意点“retry”。",
        "plan_preview": state.get("fd_shift_strategy", ""),
        "note": "支持 approve/ok/yes 或 retry/no/reject。",
    }
    user_input = interrupt(payload)               # >>> CHANGED/ADDED
    decision = _normalize_decision(user_input)    # >>> CHANGED/ADDED
    if decision == "retry":
        logger.info("User rejected FD proposer output. Jumping back to fd_proposer.")
        return Command(jump="fd_proposer")        # >>> CHANGED/ADDED
    logger.info("User approved FD proposer output. Proceeding to fd_extractor.")
    return Command(jump="fd_extractor")           # >>> CHANGED/ADDED

_CALC_KEYWORDS = [
    r"\bsum\b", r"\btotal\b", r"\bmean\b", r"\bavg\b", r"\baverage\b",
    r"\bmedian\b", r"\bmin\b", r"\bmax\b", r"\bstd\b", r"\bcount\b",
    r"\bgroup\b", r"\baggregate\b", r"\bagg\b", r"\bfilter\b",
    r"\badd\b", r"\bsub(tract)?\b", r"\bmul(tiply)?\b", r"\bdiv(ide)?\b"
]

def _looks_like_calc(query: str) -> bool:
    q = (query or "").lower()
    for pat in _CALC_KEYWORDS:
        if re.search(pat, q):
            return True
    return False

def branch_before_bso(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    return {}

def decide_before_bso(state: QueryState) -> str:
    """routing first: query calculation related and has CSV + calculator; otherwise = BSO"""
    has_csv = _has_csv_on_state(state)
    if has_csv and _looks_like_calc(state.get('query', '')):
        if not state.get('calc_plan'):
            state['calc_plan'] = state.get('query', '')
        return 'calc_llm_agent'
    return 'bso_agent'

def branch_after_bso(state: QueryState, config: RunnableConfig) -> Dict[str, Any]:
    return {}

def decide_after_bso(state: QueryState) -> str:
    """after BSO routing: after calc related + calc; if calc_result not summarized -> BSO; otherwise terminate"""
    if state.get('need_calc'):
        return 'calc_llm_agent'
    if state.get('calc_result') and not state.get('bso_summarized'):
        return 'bso_agent'
    return '__end__'  # in mapping '__end__' = END

# construct LangGraph
graph_builder = StateGraph(QueryState, config_schema=Configuration)
graph_builder.add_node("retrieval_agent", retrieval_agent)
graph_builder.add_node("bso_init", bso_init)
graph_builder.add_node("bso_agent", bso_agent)
graph_builder.add_node("fd_proposer", fd_proposer)
graph_builder.add_node("fd_extractor", fd_extractor)
graph_builder.add_node("fd_monthly_agent", fd_monthly_agent)
graph_builder.add_node("calc_llm_agent", calc_llm_agent)
graph_builder.add_node("branch_before_bso", branch_before_bso)
# graph_builder.add_node("branch_after_bso", branch_after_bso)

# >>> CHANGED/ADDED: 新增两个审批闸门节点
graph_builder.add_node("review_bso_decision", review_bso_decision)                 # >>> CHANGED/ADDED
graph_builder.add_node("review_fd_proposer_decision", review_fd_proposer_decision) # >>> CHANGED/ADDED

graph_builder.add_edge(START, "retrieval_agent")
graph_builder.add_edge("retrieval_agent", "bso_init")

# routing before BSO
graph_builder.add_edge("bso_init", "branch_before_bso")
graph_builder.add_conditional_edges(
    "branch_before_bso",
    decide_before_bso,
    {
        "calc_llm_agent": "calc_llm_agent",
        "bso_agent": "bso_agent",
    },
)

# calculator back to BSO with calc result
graph_builder.add_edge("calc_llm_agent", "bso_agent")

# >>> CHANGED/ADDED: 在 bso_agent 之后插入审批闸门
graph_builder.add_edge("bso_agent", "review_bso_decision")                         # >>> CHANGED/ADDED

# >>> CHANGED/ADDED: 在 fd_proposer 之后插入审批闸门
graph_builder.add_edge("fd_proposer", "review_fd_proposer_decision")               # >>> CHANGED/ADDED

# 审批通过后保持后续流水线
graph_builder.add_edge("fd_extractor", "fd_monthly_agent")
graph_builder.add_edge("fd_monthly_agent", END)

graph = graph_builder.compile()
