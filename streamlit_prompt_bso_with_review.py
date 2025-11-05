async def run_chatbot(user_input: str):
    # 保留你的配置
    st.session_state.thread['configurable']['planner_model'] = planner_model
    st.session_state.thread['configurable']['base_url'] = model_dict[planner_model]
    st.session_state.thread['configurable']['model_temp'] = temp
    st.session_state.thread['configurable']['vector_store'] = vector_store

    async for event in st.session_state.graph.astream(
        {"query": user_input, "maturity_bytes": maturity_bytes},
        st.session_state.thread,
        stream_mode="updates",
    ):
        # ---------- 捕获 interrupt ----------
        if "__interrupt__" in event:
            # 你的环境返回 Interrupt 对象，用 .value 没问题
            msg = event["__interrupt__"][-1].value
            st.session_state.pending_interrupt = True
            st.session_state.interrupt_message = msg
            return  # 不 yield，交给按钮区展示
        # ---------- 正常消息 ----------
        display = None
        if "fd_monthly_agent" in event and event["fd_monthly_agent"].get("bso_messages"):
            display = event["fd_monthly_agent"]["bso_messages"][-1].content
        elif "fd_proposer" in event and event["fd_proposer"].get("messages"):
            display = event["fd_proposer"]["messages"][-1].content
        elif "bso_agent" in event and event["bso_agent"].get("bso_messages"):
            display = event["bso_agent"]["bso_messages"][-1].content
        elif "calc_llm_agent" in event and event["calc_llm_agent"].get("messages"):
            display = event["calc_llm_agent"]["messages"][-1].content

        if display is not None:
            yield display



async def resume_chatbot(decision: str):
    """恢复 bso_confirm 的中断。decision ∈ {'proceed','redo'}"""
    from langgraph.types import Command

    async for event in st.session_state.graph.astream(
        Command(resume={"value": decision}),
        st.session_state.thread,
        stream_mode="updates",
    ):
        # ---------- 再次中断 ----------
        if "__interrupt__" in event:
            msg = event["__interrupt__"][-1].value
            st.session_state.pending_interrupt = True
            st.session_state.interrupt_message = msg
            return
        # ---------- 正常消息 ----------
        display = None
        if "fd_monthly_agent" in event and event["fd_monthly_agent"].get("bso_messages"):
            display = event["fd_monthly_agent"]["bso_messages"][-1].content
        elif "fd_proposer" in event and event["fd_proposer"].get("messages"):
            display = event["fd_proposer"]["messages"][-1].content
        elif "bso_agent" in event and event["bso_agent"].get("bso_messages"):
            display = event["bso_agent"]["bso_messages"][-1].content
        elif "calc_llm_agent" in event and event["calc_llm_agent"].get("messages"):
            display = event["calc_llm_agent"]["messages"][-1].content

        if display is not None:
            yield display

    # 收尾清理中断状态
    st.session_state.pending_interrupt = False
    st.session_state.interrupt_message = ""




# === BSO 确认 UI ===
if st.session_state.get("pending_interrupt"):
    st.markdown(st.session_state.get("interrupt_message", ""))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ proceed", use_container_width=True):
            with st.chat_message("assistant"):
                st.write_stream(resume_chatbot("proceed"))
    with col2:
        if st.button("🔁 retry", use_container_width=True):
            with st.chat_message("assistant"):
                st.write_stream(resume_chatbot("redo"))