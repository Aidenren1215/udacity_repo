import os
import glob
import uuid
import shutil
import logging
import mimetypes

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph_prompt_bso_calculator import graph_builder as builder
from src.vectorstore.file_loader import FileLoader
from src.vectorstore.vector_store import VectorStore
from src.streamlit_persist import persist, load_widget_state
from auth import login_gate, do_logout, is_widget_valid, SESSION_TTL_MIN, _get_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# >>> CHANGED/ADDED: 统一补齐会话级默认变量，避免未初始化导致的报错
def _ensure_defaults():
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "last_resume_decision" not in st.session_state:
        st.session_state.last_resume_decision = None
    if "model_dict" not in st.session_state:
        st.session_state.model_dict = {
            "gemma-3-27b-it": "https://ocbc-llm-coordinator.ml-3ab7488-2a6.apps.apps.prod7.ocbc.com",
            "gpt-oss-120b": "https://ocbc-llm-coordinator.ml-3ab7488-2a6.apps.apps.prod7.ocbc.com",
        }
    if "planner_model" not in st.session_state:
        st.session_state.planner_model = "gemma-3-27b-it"
    if "temp" not in st.session_state:
        st.session_state.temp = 0.0
    if "maturity_bytes" not in st.session_state:
        st.session_state.maturity_bytes = None


# >>> CHANGED/ADDED: 统一的事件流读取（支持 __interrupt__ 暂停 + 常规文本输出）
async def _stream_graph_once(input_dict: dict):
    async for event in st.session_state.graph.astream(
        input_dict,
        st.session_state.thread,
        stream_mode="updates"
    ):
        # 1) 捕获 LangGraph 的中断（来自 review_bso_decision/review_fd_proposer_decision）
        if "__interrupt__" in event:  # 后端节点 interrupt(payload) 时会发这个
            st.session_state.pending_interrupt = event["__interrupt__"]
            yield ":pause_button: 等待您的审批/备注…"
            return  # 立刻暂停，渲染审批 UI

        # 2) 常规节点输出：优先 bso_messages，其次 messages（保持“整段答案”）
        for node_name, payload in event.items():
            if node_name.startswith("__"):
                continue
            if not isinstance(payload, dict):
                continue

            display = None
            if "bso_messages" in payload and payload["bso_messages"]:
                display = payload["bso_messages"][-1]["content"]
            elif "messages" in payload and payload["messages"]:
                display = payload["messages"][-1]["content"]

            if display:
                st.session_state.thread["configurable"]["agent_messages"].append(display)
                yield display


# >>> CHANGED/ADDED: 首次/常规对话（封装）
async def run_chatbot(user_input: str):
    # 从 session_state 取 UI 可调参数，避免引用未定义的局部变量
    st.session_state.thread['configurable']['planner_model'] = st.session_state.planner_model
    st.session_state.thread['configurable']['base_url'] = st.session_state.model_dict[st.session_state.planner_model]
    st.session_state.thread['configurable']['model_temp'] = st.session_state.temp
    st.session_state.thread['configurable']['vector_store'] = st.session_state.vector_store

    async for chunk in _stream_graph_once(
        {"query": user_input, "maturity_bytes": st.session_state.maturity_bytes}
    ):
        yield chunk


# >>> CHANGED/ADDED: 针对中断的恢复（approve/ retry + 可选 comment）
async def resume_with_decision(decision: str, comment: str = ""):
    payload = {"__interrupt__": {"decision": decision}}
    if comment:
        payload["__interrupt__"]["comment"] = comment
    async for chunk in _stream_graph_once(payload):
        yield chunk


def save_messages():
    output_str = ""
    if len(st.session_state.messages) > 0:
        msg_list = [f"{{'role': '{msg['role']}', 'content': '{msg['content']}'}}" for msg in st.session_state.messages]
        output_str = "\\n\\n".join(msg_list)
    return output_str

def delete_uploaded_files():
    upload_dir = os.path.join("/home/cdsw/ct-alco-agent", "data", "files_bso")
    all_files = glob.glob(os.path.join(upload_dir, "*.*"))
    for file in all_files:
        os.remove(file)
    st.session_state.uploaded_files = None
    # >>> CHANGED/ADDED: 用会话内的 vector_store，避免未定义引用
    st.session_state.vector_store.client.delete_collection(st.session_state.vector_store.collection_name)
    st.success("All files have been deleted from the data/files directory")

def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_initial = True
    # >>> CHANGED/ADDED: 用会话内的 memory
    st.session_state.graph = builder.compile(checkpointer=st.session_state.memory, debug=True)

    # >>> CHANGED/ADDED: 线程可配置项明确从 session_state 取
    st.session_state.thread = {"configurable": {
        "thread_id": st.session_state.thread_id,
        "search_api": "cohesearch",
        "search_mode": "docsearch",
        "project": "bso_index",
        "planner_provider": "openai",
        "planner_model": st.session_state.planner_model,
        "writer_provider": "openai",
        "writer_model": "Meta-Llama-3.1-70B-Instruct-No-Guards",
        "max_search_depth": 2,
        "report_structure": "NA",
        "agent_messages": [],
        "vector_store": st.session_state.vector_store,
        "llm_cache": "y",
        "base_url": st.session_state.model_dict[st.session_state.planner_model],
        "model_temp": st.session_state.temp,
    }}
    st.session_state.uploaded_files = None
    # >>> CHANGED/ADDED: 初始化中断状态
    st.session_state.pending_interrupt = None
    st.session_state.last_resume_decision = None

if __name__ == "__main__":
    # login feature
    # if not login_gate:
    #     st.stop()

    with st.sidebar:
        # auth = _get_auth() or {}
        # st.write(f"Welcome, {auth.get('name', '')}!")
        # st.write(f"Lan id: {auth.get('lan_id', '')}")
        # st.write(f"Session period: {SESSION_TTL_MIN/60} hours")
        # if st.button("log out"):
        #     do_logout()
        #     st.rerun()
        pass

    st.title("ALCO Prompt BSO @OCBC AI Lab")

    # >>> CHANGED/ADDED: 先确保默认值存在
    _ensure_defaults()

    # Initialize button for reset session
    st.button("New Chat", on_click=reset_conversation)

    col1, _ = st.columns([1, 3])
    with col1:
        # >>> CHANGED/ADDED: 温度滑条直接写入 session_state
        st.session_state.temp = st.slider(
            label="Model Temperature",
            min_value=0.0, max_value=2.0, value=st.session_state.temp, step=0.1, format="%.1f"
        )

    # >>> CHANGED/ADDED: 如需切换 planner 模型，可加一个下拉；如果不需要，保持默认即可
    # st.session_state.planner_model = st.selectbox("Planner Model", list(st.session_state.model_dict.keys()))

    upload_dir = os.path.join("/home/cdsw/ct-alco-agent", "data", "files_bso")
    os.makedirs(upload_dir, exist_ok=True)

    st.write("---")

    # Document uploader section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.uploaded_files = st.file_uploader(
            "Upload pdf, docx, txt, csv, ppt, xlsx documents",
            type=None,
            accept_multiple_files=True,
            key=persist("file_uploader")
        )

    # Handle file upload
    # >>> CHANGED/ADDED: maturity_bytes、vector_store 放入 session_state，避免后面函数找不到变量
    st.session_state.maturity_bytes = None
    if st.session_state.uploaded_files:
        for uploaded_file in st.session_state.uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # detect file type from files uploaded
            file_name = uploaded_file.name
            file_type, _ = mimetypes.guess_type(file_name)

            # test for calculator agent
            if "maturity" in file_name.lower():
                st.session_state.maturity_bytes = file_bytes  # >>> CHANGED/ADDED
                logger.info("maturity data loaded")
                continue

            # index files
            logger.info(f"processing file {file_name}")
            file_loader = FileLoader()
            chunked_documents = file_loader.chunk_files(file_bytes, file_name, file_type)
            document_ids = [str(uuid.uuid4()) for _ in range(len(chunked_documents))]
            st.session_state.vector_store.add(documents=chunked_documents, ids=document_ids)  # >>> CHANGED/ADDED

        st.success(f"Uploaded {len(st.session_state.uploaded_files)} and indexed file(s) to data/files directory")
    else:
        st.session_state.maturity_bytes = None  # >>> CHANGED/ADDED

    uploaded_files_list = glob.glob(os.path.join(upload_dir, "*.*"))
    st.warning(f"Files uploaded will be used for answering questions. ({len(uploaded_files_list)} file(s) ready)")

    col1, col2 = st.columns([3, 1])
    with col1:
        if uploaded_files_list:
            st.write([os.path.basename(p) for p in uploaded_files_list])
        else:
            st.write("No Files uploaded yet")
    with col2:
        st.button("red-background[Delete Files]", on_click=delete_uploaded_files)
    st.write("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_conversation()
    if "graph" not in st.session_state:
        reset_conversation()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # >>> CHANGED/ADDED: 审批中断 UI（关键！）
    if st.session_state.pending_interrupt:
        intr = st.session_state.pending_interrupt
        intr_type = intr.get("type", "review")
        prompt_text = intr.get("prompt", "请审阅并做出选择：")
        st.info(f"**{prompt_text}**")

        # BSO 审核点：后端已把“完整答案”放在 full_answer
        if intr_type == "review_bso":
            full_answer = intr.get("full_answer", "")
            with st.expander("BSO 完整输出（含表格）", expanded=True):
                st.markdown(full_answer if full_answer else "_暂无内容_")

            retry_comment = st.text_area(
                "如果不满意，请写下修改意见（将用于重跑 BSO）",
                placeholder="例如：12M 下调幅度更小；3/6/9M 增量上限；总量保持不变；优先缩短久期…",
                key="bso_retry_comment",
                height=120,
            )
        # FD proposer 审核点（保持不强制备注）
        elif intr_type == "review_fd_proposer":
            full_plan = intr.get("full_plan", "") or intr.get("plan_preview", "")
            with st.expander("FD 重配方案（完整）", expanded=True):
                st.markdown(full_plan if full_plan else "_暂无内容_")
            retry_comment = st.text_area(
                "不满意？写下你的修改意见（可选）",
                key="fd_retry_comment",
                height=120,
            )
        else:
            retry_comment = ""

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ 满意，继续", key="approve_btn"):
                st.session_state.last_resume_decision = "approve"
                st.session_state.pending_interrupt = None
                with st.chat_message("assistant"):
                    response = st.write_stream(resume_with_decision("approve"))
                st.session_state.messages.append({"role": "assistant", "content": response})

        with col_b:
            if st.button("♻️ 不满意，重跑（可携带备注）", key="retry_btn"):
                st.session_state.last_resume_decision = "retry"
                st.session_state.pending_interrupt = None
                with st.chat_message("assistant"):
                    response = st.write_stream(resume_with_decision("retry", (retry_comment or "").strip()))
                st.session_state.messages.append({"role": "assistant", "content": response})

        # 中断未决时阻止继续输入
        st.stop()

    # Accept user input
    prompt = st.chat_input("Ask me something about BSO!")

    # Display user message in chat message container
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant messages
        with st.chat_message("assistant"):
            response = st.write_stream(run_chatbot(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Download chat history button
    chat_history = save_messages()
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat.txt",
        mime="text/csv",
        icon=":material/download:",
    )
