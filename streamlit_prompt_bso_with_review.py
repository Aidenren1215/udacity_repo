import streamlit as st

# ---------- 状态初始化（顶部只做一次） ----------
for k, v in {
    "history": [],
    "pending_interrupt": None,   # run/resume 内部会设 True/False
    "interrupt_message": "",
    "resume_decision": None,     # "proceed" | "redo"
    "resume_feedback": "",
    "retry_mode": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- （保留你已有的聊天输入与 run_chatbot 调用） ----------
# 示例：
# user_msg = st.chat_input("Ask me something about BSO!")
# if user_msg:
#     st.session_state.history.append(("user", user_msg))
#     with st.chat_message("user"):
#         st.markdown(user_msg)
#     with st.chat_message("assistant"):
#         st.write_stream(run_chatbot(user_msg))
# ↑↑↑ 这一段你已有，就别动 ↑↑↑


# ---------- 一个 fragment：中断面板 + 继续执行 ----------
@st.fragment
def interrupt_and_resume():
    # 让你看得到 fragment 是否在跑（调试时可注释）
    # st.write("DEBUG fragment running; pending_interrupt=", st.session_state.pending_interrupt)

    if not st.session_state.get("pending_interrupt"):
        return

    # 1) 中断提示
    msg = st.session_state.get("interrupt_message") or "The graph is waiting for your decision."
    st.info(msg)

    # 2) 操作按钮（只改状态；为了立刻生效，设完状态后 st.rerun() 触发本 fragment 重跑）
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ proceed", use_container_width=True, key="btn_proceed"):
            st.session_state.resume_decision = "proceed"
            st.session_state.resume_feedback = ""
            st.rerun()  # 关键：立刻重跑本 fragment，从而触发下面的 resume

    with c2:
        if st.button("🔁 retry", use_container_width=True, key="btn_retry"):
            st.session_state.retry_mode = True

    # 3) 反馈表单（只在 retry 模式显示）
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
            st.rerun()  # 同理：提交后立刻触发 resume

        if cancel:
            st.session_state.retry_mode = False
            st.rerun()

    # 4) 同一 fragment 内：检测到 resume_decision 就继续执行
    decision = st.session_state.get("resume_decision")
    if decision:
        # st.write(f"DEBUG resume: {decision}, fb={st.session_state.resume_feedback}")  # 调试可开

        with st.chat_message("assistant"):
            # 直接用你的签名：resume_chatbot(decision: str, feedback: str = "")
            st.write_stream(resume_chatbot(decision, feedback=st.session_state.get("resume_feedback", "")))

        # resume_chatbot 内部如果再次 interrupt，会把：
        #   st.session_state.pending_interrupt = True
        #   st.session_state.interrupt_message = <new msg>
        # 若没有再次中断，下面清空状态；有的话保留中断态，这个 fragment 会再次显示按钮区

        if st.session_state.pending_interrupt:
            # 二次中断：清 resume 意图，保留中断态
            st.session_state.resume_decision = None
            st.session_state.resume_feedback = ""
            # 不 rerun；交给用户下一次点击
        else:
            # 已完成：清空所有中断相关状态
            st.session_state.resume_decision = None
            st.session_state.resume_feedback = ""
            st.session_state.pending_interrupt = None
            st.session_state.interrupt_message = ""
            # 可选：st.rerun() 让 UI 立刻回到正常态（一般不必）

# 在页面底部调用 fragment（必须放在逻辑最后，保证随时可渲染）
interrupt_and_resume()
