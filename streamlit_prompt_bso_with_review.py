@st.fragment
def interrupt_and_resume():
    """在同一 fragment 内：显示中断面板 + 捕捉点击后立即 resume"""

    # 没有中断就不显示
    if not st.session_state.get("pending_interrupt"):
        return

    # 4.1 中断提示
    msg = st.session_state.get("interrupt_message") or "The graph is waiting for your decision."
    st.info(msg)

    # 4.2 操作按钮
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ proceed", use_container_width=True, key="btn_proceed"):
            st.session_state.resume_decision = "proceed"
            st.session_state.resume_feedback = ""   # 清掉旧反馈

    with c2:
        if st.button("🔁 retry", use_container_width=True, key="btn_retry"):
            st.session_state.retry_mode = True

    # 4.3 反馈表单（仅在 retry 模式）
    if st.session_state.retry_mode:
        st.markdown("please provide feedback to the retry of BSO agent.")
        with st.form(key="retry_form", clear_on_submit=True):
            fb = st.text_area(
                "your feedback",
                key="retry_feedback",
                placeholder="Please decrease FD volume of 1Y by more than 80%",
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

    # 4.4 在同一 fragment 内立刻 resume（如果设置了决策）
    decision = st.session_state.get("resume_decision")
    if decision:
        with st.chat_message("assistant"):
            st.write_stream(resume_chatbot(decision, feedback=st.session_state.get("resume_feedback", "")))

        # 说明：
        # - 你的 resume_chatbot 一旦再次遇到 interrupt，会再次设置：
        #     st.session_state.pending_interrupt = True
        #     st.session_state.interrupt_message = <new msg>
        #   这样本 fragment 下一次重跑时会继续展示中断 UI。
        # - 如果没有再次中断，代表已完成，清空中断状态。

        if st.session_state.pending_interrupt:
            # 二次中断：清除 resume 意图，保留中断态
            st.session_state.resume_decision = None
            st.session_state.resume_feedback = ""
        else:
            # 已完成：清空所有中断/意图
            st.session_state.resume_decision = None
            st.session_state.resume_feedback = ""
            st.session_state.pending_interrupt = None
            st.session_state.interrupt_message = ""

# 渲染 fragment（在每轮渲染末尾调用）
interrupt_and_resume()