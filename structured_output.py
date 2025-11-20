from typing import List
from pydantic import BaseModel, Field


class FDTableRow(BaseModel):
    fd_tenor_bucket: str = Field(..., alias="FD Tenor Bucket")
    current_fd_volume: float = Field(..., alias="Current FD Volume (S$m)")
    proposed_fd_volume: float = Field(..., alias="Proposed FD Volume (S$m)")
    change: float = Field(..., alias="Change (S$m)")
    pct_change: float = Field(..., alias="% Change")

    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True


class BSOJsonSchema(BaseModel):
    rationale: str
    qualitative_changes: str
    table: List[FDTableRow]

    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

from langchain_core.output_parsers import JsonOutputParser
bso_parser = JsonOutputParser(pydantic_object=BSOJsonSchema)

from langchain_core.prompts import ChatPromptTemplate

my_own_system_msg = (
    "You are a precise financial modeling assistant. "
    "Follow all rules exactly and produce consistent structured outputs."
)

bso_user_msg = """<把上面那整段 RAW prompt 粘过来>"""

bso_prompt = ChatPromptTemplate.from_messages([
    ("system", my_own_system_msg),
    ("user", bso_user_msg),
]).partial(format_instructions=bso_parser.get_format_instructions())


result = (
    bso_prompt
    | llm_gemma3
    | bso_parser
).invoke({
    "CURRENT_FD_TABLE": current_fd_markdown,  # 当前FD表，markdown
    "USER_INPUT": user_input_text,           # 用户最初的rate outlook+问题
    "FEEDBACK": feedback_text or ""          # 用户对上一版答案的反馈，没有就传空字符串
})

final_json = result.dict(by_alias=True)
