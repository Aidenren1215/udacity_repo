import asyncio

async def run_chatbot(user_input: str):
    # 你原来就有的这几行配置写回 thread，不动
    st.session_state.thread["configurable"]["planner_model"] = planner_model
    st.session_state.thread["configurable"]["base_url"] = model_dict[planner_model]
    st.session_state.thread["configurable"]["model_temp"] = temp
    st.session_state.thread["configurable"]["vector_store"] = vector_store

    agen = st.session_state.graph.astream(
        {"query": user_input, "maturity_bytes": maturity_bytes},
        st.session_state.thread,
        stream_mode="updates",
    )

    try:
        async for event in agen:
            # -------- 1) 本轮可显示的内容（先渲染） --------
            display = None
            if "fd_monthly_proposer" in event and event["fd_monthly_proposer"].get("fd_monthly_messages"):
                display = event["fd_monthly_proposer"]["fd_monthly_messages"][-1].content
            elif "fd_proposer" in event and event["fd_proposer"].get("fd_messages"):
                display = event["fd_proposer"]["fd_messages"][-1].content
            elif "bso_agent" in event and event["bso_agent"].get("bso_messages"):
                display = event["bso_agent"]["bso_messages"][-1].content

            if display is not None:
                st.session_state.chatbot_mode = True
                st.session_state.thread["configurable"]["agent_messages"].append(display)
                st.session_state.history.append(("assistant", display))
                yield display

            # -------- 2) 中断（后处理，改 return 为 break） --------
            if "__interrupt__" in event:
                msg = event["__interrupt__"][-1].value  # 你现在环境下 .value 可用
                st.session_state.pending_interrupt = True
                st.session_state.interrupt_message = msg
                break  # 不要 return，给 finally 机会 aclose()

        else:
            # 没有新的中断，清理挂起态
            st.session_state.pending_interrupt = False
            st.session_state.interrupt_message = ""

    except asyncio.CancelledError:
        raise
    finally:
        try:
            await agen.aclose()
        except Exception:
            pass


import asyncio
from langgraph.types import Command

async def resume_chatbot(decision: str, feedback: str = ""):
    """从 bso_confirm / fd_confirm 恢复。decision ∈ {'proceed','retry','retry_bso','retry_fd'...}"""
    payload = {"value": decision}
    if feedback:
        payload["feedback"] = feedback

    agen = st.session_state.graph.astream(
        Command(resume=payload),
        st.session_state.thread,
        stream_mode="updates",
    )

    try:
        async for event in agen:
            # -------- 1) 本轮输出（先渲染） --------
            display = None
            if "fd_monthly_proposer" in event and event["fd_monthly_proposer"].get("fd_monthly_messages"):
                display = event["fd_monthly_proposer"]["fd_monthly_messages"][-1].content
            elif "fd_proposer" in event and event["fd_proposer"].get("fd_messages"):
                display = event["fd_proposer"]["fd_messages"][-1].content
            elif "bso_agent" in event and event["bso_agent"].get("bso_messages"):
                display = event["bso_agent"]["bso_messages"][-1].content

            if display is not None:
                st.session_state.chatbot_mode = True
                st.session_state.thread["configurable"]["agent_messages"].append(display)
                st.session_state.history.append(("assistant", display))
                yield display

            # -------- 2) 新的中断（后处理，break 不 return） --------
            if "__interrupt__" in event:
                msg = event["__interrupt__"][-1].value  # 只读 .value
                st.session_state.pending_interrupt = True
                st.session_state.interrupt_message = msg
                break

        else:
            # 整段恢复没有中断，清理挂起态
            st.session_state.pending_interrupt = False
            st.session_state.interrupt_message = ""

    except asyncio.CancelledError:
        raise
    finally:
        try:
            await agen.aclose()
        except Exception:
            pass
