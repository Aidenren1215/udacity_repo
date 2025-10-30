# pip install "langchain>=0.2" "pydantic<2"
from typing import List, Dict, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# ==================== 1) 严格 Schema（与之前约定一致+新增比例字段） ====================
class ReallocationRecord(BaseModel):
    from_bucket: str
    to_bucket: str
    amount: float
    ratio_of_current_volume: str  # e.g. "25.0%"
    reason: str

class BucketCheck(BaseModel):
    inflow_expected: float = 0
    inflow_received: float = 0
    outflow_expected: float = 0
    outflow_allocated: float = 0
    status: Literal["OK","MISMATCH","N/A"]
    ratio_check: Optional[str] = None  # (total_outflow/current_volume)*100%

class GlobalValidation(BaseModel):
    total_inflow: float
    total_outflow: float
    status: Literal["Balanced","Unbalanced"]

class FDPlan(BaseModel):
    # Markdown 概述：一段话 + bullet points；禁止思考过程
    summary_markdown: str = Field(..., description="Markdown summary only; no chain-of-thought.")
    reallocation_plan: List[ReallocationRecord]
    bucket_validation: Dict[str, BucketCheck]
    global_validation: GlobalValidation

# ==================== 2) LLM 配置（与你现有环境一致） ====================
llm = ChatOpenAI(
    base_url=conf.vllm.url,
    model="gpt-oss-120b",
    api_key=conf.vllm.api_key,
    temperature=0,
    max_tokens=2048,  # 保守，避免截断；如仍不足，请精简输入表
)

# 关键：结构化输出，直接返回 FDPlan 对象
sllm = llm.with_structured_output(FDPlan)

# ==================== 3) Prompt（基于你之前那版，补充比例与校验的硬性要求） ====================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial optimization assistant. "
     "Return a single object that conforms to the given schema. "
     "Never include analysis, internal reasoning, or chain-of-thought. "
     "All numeric balances must reconcile exactly."),
    ("user",
     # 你把表格文本塞进 {fd_text}，建议只保留必要列以节省 tokens
     "Below is a table of FD (fixed deposit) volumes by tenor bucket with CURRENT, PROPOSED, and Change (S$ million). "
     "Positive Change = inflow; negative Change = outflow. Units: S$ million.\n\n"
     "{fd_text}\n\n"
     "TASKS:\n"
     "1) Construct a complete reallocation plan mapping outflows (negative Change) to inflows (positive Change). "
     "   Each movement must include fields: from_bucket, to_bucket, amount, reason, and ratio_of_current_volume.\n"
     "2) ratio_of_current_volume must be computed as: "
     "   (amount / current_FD_volume_of_from_bucket) * 100%, formatted as a percentage string with one decimal place "
     "   (e.g., \"25.0%\"). Example: if 1Y current volume=1000 and reallocates 200→1M, 400→6M, 200→3M, "
     "   the ratios must be 20.0%, 40.0%, 20.0% respectively; their sum equals (total_outflow/current_volume)*100%.\n"
     "3) Validation rules:\n"
     "   - Global: sum of all inflows must equal sum of all outflows.\n"
     "   - Per outflow bucket: sum of allocated amounts equals its absolute Change.\n"
     "   - Per inflow bucket: sum of received amounts equals its positive Change.\n"
     "   - Per outflow bucket: sum of ratio_of_current_volume across its reallocations must equal "
     "     (total_outflow/current_FD_volume) * 100% (within rounding tolerance of ±0.1%).\n"
     "4) Output must fit the schema fields only. "
     "   summary_markdown should be concise with a heading and bullet points (no reasoning text).")
])

# ==================== 4) 调用 ====================
# 示例：建议把表整理为只含必要列的纯文本或CSV片段，以降低 token 占用
FD_table_min = """\
FD Tenor Bucket | Current FD Volume | Proposed FD Volume | Change
1M | 2113 | 4567 | 2454
2M | 1991 | 2500 | 509

"""

result: FDPlan = (prompt | sllm).invoke({"fd_text": FD_table_min})

# 现在 result 就是一个已校验的 “受控 JSON 对象”
# 直接使用：
# result.reallocation_plan  -> List[ReallocationRecord]
# result.bucket_validation  -> Dict[str, BucketCheck]
# result.global_validation  -> GlobalValidation
# 如需字典/JSON：
fdplan_dict = result.dict()  # pydantic v1: .dict()；如果是 v2 是 .model_dump()
