from __future__ import annotations

from enum import Enum
from typing import List, Literal
from pydantic import BaseModel, Field, confloat


class MovementType(str, Enum):
    internal = "internal"
    external_in = "external_in"
    external_out = "external_out"


class MonthlyMovement(BaseModel):
    month: str = Field(..., description='Month in "Mon-YY" format, e.g. "Jan-24"')
    from_bucket: str = Field(..., description="FD tenor bucket or EXTERNAL")
    to_bucket: str = Field(..., description="FD tenor bucket or EXTERNAL")
    movement_type: MovementType = Field(..., description="internal | external_in | external_out")
    amount: confloat(ge=0) = Field(..., description="S$ million, non-negative")

    class Config:
        extra = "forbid"  # forbid any extra fields to reduce hallucination


class MonthlyShiftPlanAdjustResponse(BaseModel):
    status: Literal["ADJUSTED_OK", "UNCHANGED_INFEASIBLE_FEEDBACK"] = Field(
        ..., description="Whether adjusted plan is produced or baseline is returned unchanged"
    )
    adjusted_monthly_shift_plan: List[MonthlyMovement] = Field(
        default_factory=list,
        description="Adjusted plan; if infeasible must match baseline input plan records (order may differ)"
    )
    notes: str = Field(..., description="Plain text notes (2–6 lines)")

    class Config:
        extra = "forbid"