# === 仅用于验证 interrupt 的最小 UI ===
if st.session_state.get("pending_interrupt"):
    # 显示 bso_confirm 抛出的提示文案（来自 interrupt(prompt_text)）
    st.info(st.session_state.get("interrupt_message", ""))

    col1, col2 = st.columns(2)

    # 继续到 FD proposer
    with col1:
        if st.button("✅ proceed", use_container_width=True):
            with st.chat_message("assistant"):
                # 直接把 'proceed' 作为恢复值传回图里
                st.write_stream(
                    st.session_state.graph.astream(
                        "proceed",
                        st.session_state.thread,
                        stream_mode="updates",
                    )
                )
            # 恢复后清理中断标记
            st.session_state.pending_interrupt = False
            st.session_state.interrupt_message = ""

    # 重跑 BSO
    with col2:
        if st.button("🔁 redo", use_container_width=True):
            with st.chat_message("assistant"):
                st.write_stream(
                    st.session_state.graph.astream(
                        "redo",
                        st.session_state.thread,
                        stream_mode="updates",
                    )
                )
            st.session_state.pending_interrupt = False
            st.session_state.interrupt_message = ""
# === 结束：仅用于验证 interrupt 的最小 UI ===
