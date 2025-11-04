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

async def run_chatbot(user_input: str):
    st.session_state.thread['configurable']['planner_model'] = planner_model
    st.session_state.thread['configurable']['base_url'] = model_dict[planner_model]
    st.session_state.thread['configurable']['model_temp'] = temp
    st.session_state.thread['configurable']['vector_store'] = vector_store

    async for event in st.session_state.graph.astream(
        {"query": user_input, "maturity_bytes": maturity_bytes},
        st.session_state.thread,
        stream_mode="updates"
    ):
        # If our graph pauses for user confirmation, capture the interrupt payload and stop streaming.
        if "__interrupt__" in event:
            # Store latest payload; different langgraph versions shape this slightly differently.
            intr = event["__interrupt__"][0] if isinstance(event["__interrupt__"], (list, tuple)) else event["__interrupt__"]
            payload = intr.get("value") or intr.get("data") or intr
            st.session_state.pending_interrupt = intr
            st.session_state.pending_payload = payload
            # Try to surface the BSO preview for the user
            preview = None
            if isinstance(payload, dict) and payload.get("type") == "bso_review":
                parts = []
                if payload.get("bso_general_strategy"):
                    parts.append(str(payload["bso_general_strategy"]))
                if payload.get("fd_shift_strategy"):
                    parts.append(str(payload["fd_shift_strategy"]))
                if payload.get("bso_table"):
                    parts.append(str(payload["bso_table"]))
                preview = "\n\n".join(parts)
            if preview:
                yield preview
            else:
                yield "Review the BSO plan above. Choose to proceed or re-run."
            return  # stop here; UI below will render buttons to resume

        # Normal streaming of final monthly output
        if "fd_monthly_agent" in event:
            display = event["fd_monthly_agent"]["bso_messages"][-1].content
            st.session_state.thread["configurable"]["agent_messages"].append(display)
            yield display

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
    vector_store.client.delete_collection(vector_store.collection_name)
    st.success("All files have been deleted from the data/files directory")

def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_initial = True
    st.session_state.graph = builder.compile(checkpointer=memory, debug=True)

    st.session_state.thread = {"configurable": {
        "thread_id": st.session_state.thread_id,
        "search_api": "cohesearch",
        "search_mode": "docsearch",
        "project": "bso_index",
        "planner_provider": "openai",
        "planner_model": "gemma-3-27b-it",
        "writer_provider": "openai",
        "writer_model": "Meta-Llama-3.1-70B-Instruct-No-Guards",
        "max_search_depth": 2,
        "report_structure": "NA",
        "agent_messages": [],
        "vector_store": VectorStore(),
        "llm_cache": "y",
        "base_url": model_dict["gemma-3-27b-it"],
        "model_temp": temp,
    }}
    st.session_state.uploaded_files = None

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
    memory = MemorySaver()
    vector_store = VectorStore()

    # Initialize button for reset session
    st.button("New Chat", on_click=reset_conversation)

    col1, _ = st.columns([1, 3])
    with col1:
        temp = st.slider(label="Model Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1, format="%.1f")

    model_dict = {
        "gemma-3-27b-it": "https://ocbc-llm-coordinator.ml-3ab7488-2a6.apps.apps.prod7.ocbc.com",
        "gpt-oss-120b": "https://ocbc-llm-coordinator.ml-3ab7488-2a6.apps.apps.prod7.ocbc.com",
    }

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
    csv_bytes, maturity_bytes = None, None
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
                maturity_bytes = file_bytes
                logger.info("maturity data loaded")
                continue

            # index files
            logger.info(f"processing file {file_name}")
            file_loader = FileLoader()
            chunked_documents = file_loader.chunk_files(file_bytes, file_name, file_type)
            document_ids = [str(uuid.uuid4()) for _ in range(len(chunked_documents))]
            vector_store.add(documents=chunked_documents, ids=document_ids)

        st.success(f"Uploaded {len(st.session_state.uploaded_files)} and indexed file(s) to data/files directory")
    else:
        csv_bytes, maturity_bytes = None, None

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

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message.content)

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


    # If the graph paused for BSO review, show action buttons
    if st.session_state.get("pending_interrupt"):
        with st.chat_message("assistant"):
            st.info("Please review the BSO plan. Proceed to FD proposer or re-run BSO?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Proceed to FD proposer"):
                    _resume_after_bso_choice("proceed")
            with c2:
                if st.button("🔁 Re-run BSO"):
                    _resume_after_bso_choice("redo")

    # Download chat history button
    chat_history = save_messages()
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat.txt",
        mime="text/csv",
        icon=":material/download:",
    )


def _resume_after_bso_choice(choice: str):
    """
    Resume the paused graph by providing a preferred decision back into the state.
    We leverage MemorySaver's checkpointing and feed the choice via `bso_user_choice_pref`.
    """
    if not st.session_state.get("pending_interrupt"):
        st.warning("No pending review to resume.")
        return

    # Continue the same thread; the graph will pick up from the interrupt boundary.
    async def _run():
        async for event in st.session_state.graph.astream(
            {"bso_user_choice_pref": choice},
            st.session_state.thread,
            stream_mode="updates"
        ):
            if "fd_monthly_agent" in event:
                display = event["fd_monthly_agent"]["bso_messages"][-1].content
                st.session_state.thread["configurable"]["agent_messages"].append(display)
                st.write(display)

    import asyncio
    asyncio.run(_run())
    # Clear the interrupt
    st.session_state.pop("pending_interrupt", None)
    st.session_state.pop("pending_payload", None)
