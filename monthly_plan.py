from typing import List, Literal
from pydantic import BaseModel, Field


class MonthlyReallocationRecord(BaseModel):
    month: str = Field(..., description='Format: "YYYY-MM"')
    from_bucket: str
    to_bucket: str
    amount: float
    movement_type: Literal["internal", "external_in", "external_out"]


class BucketLevelCheck(BaseModel):
    bucket: str

    general_outflow: float
    monthly_outflow: float
    outflow_diff: float

    general_inflow: float
    monthly_inflow: float
    inflow_diff: float

    status: Literal["OK", "MISMATCH"]


class ExternalNetCheck(BaseModel):
    general_net: float
    monthly_net: float
    diff: float


class MonthlyGlobalValidation(BaseModel):
    status: Literal["OK", "MISMATCH"]
    bucket_checks: List[BucketLevelCheck]
    external_net_check: ExternalNetCheck


class MonthlyFDPlan(BaseModel):
    summary_markdown: str
    reallocation_plan: List[MonthlyReallocationRecord]
    global_validation: MonthlyGlobalValidation
    
    
# just for github new ssh test
