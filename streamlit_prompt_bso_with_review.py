# ---- Accept user input ----
if st.session_state.get("pending_interrupt"):
    prompt_text = (
        st.session_state.get("interrupt_message")
        or "Agent paused for your decision. Type 'proceed' or 'retry[: your comment]'"
    )
else:
    prompt_text = "Ask me something about BSO!"

user_input = st.chat_input(prompt_text)

if user_input:
    # Add user message to history
    st.session_state.history.append(("user", user_input))
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Case 1: Normal input ----
    if not st.session_state.get("pending_interrupt"):
        with st.chat_message("assistant"):
            response = st.write_stream(run_chatbot(user_input))
        st.session_state.messages.append({"role": "assistant", "content": response})

    # ---- Case 2: Interrupt input (resume path) ----
    else:
        reply_lower = user_input.lower().strip()
        feedback = ""

        if reply_lower.startswith("retry"):
            parts = user_input.split(":", 1)
            if len(parts) > 1:
                feedback = parts[1].strip()
            with st.chat_message("assistant"):
                st.write_stream(resume_chatbot("retry", feedback))

        elif reply_lower.startswith("proceed"):
            with st.chat_message("assistant"):
                st.write_stream(resume_chatbot("proceed"))

        else:
            st.warning("Please type either 'proceed' or 'retry[: your comment]'")
