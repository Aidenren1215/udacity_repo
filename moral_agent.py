import json
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# 1. Pydantic Output Schema
# ============================================================

class MoralAgentOutput(BaseModel):
    """
    Structured output for the Moral Agent.

    Routing ONLY depends on `block`.
    Other fields are for audit/debug.
    """

    block: bool = Field(
        ...,
        description="True if the query should be blocked for a normal user."
    )

    categories: List[Literal[
        "ATTRIBUTION",
        "OPINION",
        "PERFORMANCE",
        "COMPARISON",
        "SENIOR_EVAL"
    ]] = Field(
        default_factory=list,
        description="Violation categories triggered by the query."
    )

    reason: str = Field(
        default="",
        description="Short generic rationale. Do NOT include names."
    )


# ============================================================
# 2. System Prompt (plain text, wrapper-friendly)
# ============================================================

MORAL_SYS_PROMPT = """
You are a compliance screening classifier in a regulated bank.

You are called ONLY for normal users, and only AFTER a deterministic
name screening step did NOT find attendee names in the query.

Task:
- Decide whether the QUESTION TYPE should be blocked.
- Do NOT answer the user’s question.
- Do NOT guess identities.
- Do NOT ask follow-up questions.
- Do NOT provide bypass advice.

BLOCK if the query asks for any of:
1) ATTRIBUTION: who said/raised/proposed/opposed/agreed; identify speakers.
2) OPINION: personal views/stance/feelings of individuals or identifiable roles.
3) PERFORMANCE: evaluation of an individual's or leadership performance.
4) COMPARISON: comparing or ranking individuals or roles.
5) SENIOR_EVAL: evaluation of senior management (CEO/CFO/Head) even if unnamed.

ALLOW if the query requests meeting-level topics, decisions, risks,
action items, or summaries WITHOUT attributing to individuals.

Return JSON strictly matching the provided schema.
""".strip()


# ============================================================
# 3. Instantiate LLM with guided_json
#    ⚠️ Replace YourLLMWrapper with your internal wrapper
# ============================================================

def build_moral_llm(YourLLMWrapper):
    guided_schema = MoralAgentOutput.model_json_schema()

    llm = YourLLMWrapper(
        model="your-model-name",  # ← 替换
        temperature=0,
        extra_body={
            "guided_json": guided_schema
        },
    )
    return llm


# ============================================================
# 4. Moral Agent Node
# ============================================================

def moral_agent_node(state: dict, llm_moral) -> dict:
    """
    Expected state input:
        state["query"]: str

    Writes back:
        state["block"]: bool
        state["moral_categories"]: List[str]
        state["moral_reason"]: str
    """

    query = state["query"]

    messages = [
        {"role": "system", "content": MORAL_SYS_PROMPT},
        {"role": "user", "content": query},
    ]

    # ---------------- LLM call ----------------
    resp = llm_moral.invoke(messages)

    # ---------------- Parse JSON ----------------
    try:
        data = json.loads(resp.content)
        result = MoralAgentOutput.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as e:
        # 🔴 FAIL-CLOSED (recommended for banking)
        state["block"] = True
        state["moral_categories"] = ["ATTRIBUTION"]
        state["moral_reason"] = f"Malformed classifier output: {type(e).__name__}"
        return state

    # ---------------- Consistency Auto-Heal ----------------
    # ⭐ 极其推荐：修复模型偶发漂移
    if not result.block:
        result.categories = []

    # ---------------- Write back to state ----------------
    state["block"] = result.block
    state["moral_categories"] = result.categories or []
    state["moral_reason"] = result.reason or ""

    return state