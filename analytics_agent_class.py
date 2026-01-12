from __future__ import annotations

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

class TimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["as_of", "range", "monthly_series"]
    start: Optional[str] = None
    end: Optional[str] = None
    relative: Optional[
        Literal["last_month","mtd","qtd","ytd","last_quarter","last_year"]
    ] = None
    granularity: Optional[Literal["day","month"]] = None

Scalar = Union[str, int, float]
Value = Union[Scalar, List[Scalar]]  # for "between": [lower, upper]

class WhereCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    op: Literal["=", "in", ">", ">=", "<", "<=", "between"]
    value: Value

class WhereSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["and", "or"] = "and"
    clauses: List[WhereCondition] = Field(default_factory=list)

class PlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table_type: Literal["daily", "monthly"]
    metric: Literal["balance", "volume", "count"]
    aggregation: Literal["sum", "count"]
    time: TimeSpec
    group_by: List[str] = Field(default_factory=list)
    where: WhereSpec = Field(default_factory=WhereSpec)
