for k, v in {
    "pending_interrupt": None,   # 是否处于中断态（由 run/resume 填充）
    "interrupt_message": "",     # 中断提示文案
    "resume_decision": None,     # "proceed" | "redo" | None
    "resume_feedback": "",       # redo 的反馈
    "retry_mode": False,         # 是否进入填写反馈表单
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
        
@st.fragment
def interrupt_panel():
    if not st.session_state.pending_interrupt:
        return

    # 中断文案
    st.markdown(st.session_state.interrupt_message or "The graph is waiting for your decision.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ proceed", use_container_width=True, key="btn_proceed"):
            st.session_state.resume_decision = "proceed"   # 只设状态
            st.session_state.resume_feedback = ""          # 清空旧反馈

    with c2:
        if st.button("🔁 retry", use_container_width=True, key="btn_retry"):
            st.session_state.retry_mode = True            # 进入反馈模式

    # 反馈表单（retry 模式）
    if st.session_state.retry_mode:
        st.markdown("please provide feedback to the retry of BSO agent.")
        with st.form(key="retry_form", clear_on_submit=True):
            fb = st.text_area(
                "your feedback",
                key="retry_feedback",
                placeholder="Please decrease FD volume of 1Y by more than 80%"
            )
            c3, c4 = st.columns(2)
            submit = c3.form_submit_button("submit to regenerate", use_container_width=True)
            cancel = c4.form_submit_button("cancel", use_container_width=True)

        if submit and fb:
            st.session_state.resume_decision = "redo"
            st.session_state.resume_feedback = fb
            st.session_state.retry_mode = False

        if cancel:
            st.session_state.retry_mode = False

interrupt_panel()


@st.fragment
def resume_runner():
    decision = st.session_state.resume_decision
    if not decision:
        return

    # 开始继续跑；注意你的签名是 resume_chatbot(decision: str, feedback: str = "")
    with st.chat_message("assistant"):
        st.write_stream(resume_chatbot(decision, feedback=st.session_state.resume_feedback))

    # 跑完（或二次 interrupt）之后，清理/维持状态
    if st.session_state.pending_interrupt:
        # 二次中断：只清 resume 意图，保留中断态与文案，让面板继续出现
        st.session_state.resume_decision = None
        st.session_state.resume_feedback = ""
        # interrupt_message 已由你的函数再次写入
    else:
        # 已完成：清空所有中断/恢复意图
        st.session_state.resume_decision = None
        st.session_state.resume_feedback = ""
        st.session_state.pending_interrupt = None
        st.session_state.interrupt_message = ""

resume_runner()
