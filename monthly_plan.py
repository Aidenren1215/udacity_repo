from typing import List, Dict, Literal
from pydantic import BaseModel, Field


class MonthlyReallocationRecord(BaseModel):
    month: str = Field(..., description='Format: "YYYY-MM"')
    from_bucket: str
    to_bucket: str
    amount: float
    movement_type: Literal["internal", "external_in", "external_out"]


class MonthlyBucketCheck(BaseModel):
    capacity: float
    outflow: float
    slack: float
    status: Literal["OK", "MISMATCH"]


class MonthlyGlobalValidation(BaseModel):
    status: Literal["OK", "MISMATCH"]
    edge_total_mismatches: List[str]
    external_in_total: float
    external_out_total: float


class MonthlyFDPlan(BaseModel):
    summary_markdown: str
    reallocation_plan: List[MonthlyReallocationRecord]
    bucket_validation: Dict[str, MonthlyBucketCheck]
    global_validation: MonthlyGlobalValidation