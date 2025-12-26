class ReallocationRecord(BaseModel):
    from_bucket: str
    to_bucket: str
    amount: float
    ratio_of_current_volume: str = Field(
        ...,
        description='Percentage string like "25.0%" or "N/A" for EXTERNAL'
    )
    movement_type: Literal["internal", "external_in", "external_out"]


class BucketCheck(BaseModel):
    current: float
    proposed: float
    change: float
    inflow: float
    outflow: float
    residual: float
    status: Literal["OK", "MISMATCH", "N/A"]


class GlobalValidation(BaseModel):
    total_current: float
    total_proposed: float
    net_change: float
    external_in_total: float
    external_out_total: float
    status: Literal["OK", "MISMATCH"]


class FDPlan(BaseModel):
    summary_markdown: str = Field(
        ...,
        description="Markdown summary only; no chain-of-thought."
    )
    reallocation_plan: List[ReallocationRecord]
    bucket_validation: Dict[str, BucketCheck]
    global_validation: GlobalValidation
    
    
    
    
    
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

MovementType = Literal["internal", "external_in", "external_out"]

class ReallocationRecord(BaseModel):
    month: str = Field(..., description='Month in "Mon-YY" format, e.g. Jan-24')
    from_bucket: str
    to_bucket: str
    amount: float
    movement_type: MovementType

class ExternalNetCheck(BaseModel):
    net_external_general: float
    net_external_monthly: float

class GlobalValidation(BaseModel):
    status: Literal["OK", "MISMATCH"]
    bucket_mismatches: List[str] = Field(default_factory=list)
    external_net_check: ExternalNetCheck

class MonthlyFDTableRow(BaseModel):
    month: str = Field(..., description='Month in "Mon-YY" format, e.g. Jan-24')
    tenor_bucket: str
    current_balance: float
    proposed_balance: float

class MonthlyShiftWithFDTableResult(BaseModel):
    summary_markdown: str = Field(..., description="Markdown bullets only. No chain-of-thought.")
    reallocation_plan: List[ReallocationRecord]
    global_validation: GlobalValidation
    monthly_fd_table: List[MonthlyFDTableRow]

parser = JsonOutputParser(pydantic_object=MonthlyShiftWithFDTableResult)
format_instructions = parser.get_format_instructions()