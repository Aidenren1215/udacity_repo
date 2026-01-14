# planner_schema.py
# Pydantic v2.12.4
# Aligned with the latest Planner sys prompt (v1.3.2, JOIN FIRST-CLASS)
#
# Required top-level fields:
# status, unsupported_reason, table_type, metric, aggregation, time, group_by, where, joins
#
# Notes:
# - This is POC-friendly: minimal cross-field validators (stability > strictness).
# - extra="forbid" prevents the LLM from adding random fields.

from __future__ import annotations

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# -------------------------
# Time
# -------------------------
class TimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # sys prompt: as_of | range | monthly_series
    type: Literal["as_of", "range", "monthly_series"]

    # sys prompt: "YYYY-MM-DD or null"
    start: Optional[str] = None
    end: Optional[str] = None

    # sys prompt: relative time tokens
    relative: Optional[
        Literal[
            "last_month",
            "mtd",
            "qtd",
            "ytd",
            "last_quarter",
            "last_year",
        ]
    ] = None

    # sys prompt: day | month | null
    granularity: Optional[Literal["day", "month"]] = None


# -------------------------
# Where (flat, v1.x)
# -------------------------
Scalar = Union[str, int, float]
Value = Union[Scalar, List[Scalar]]  # "between" uses a 2-element list


class WhereCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: Literal["=", "in", ">", ">=", "<", "<=", "between"]
    value: Value


class WhereSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["and", "or"] = "and"
    clauses: List[WhereCondition] = Field(default_factory=list)


# -------------------------
# Joins (mapping tables)
# -------------------------
class JoinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    reason: str
    from_fact_key: str
    to_mapping_key: str
    join_type: Literal["left"] = "left"


# -------------------------
# Planner Output (v1.3.2)
# -------------------------
class PlannerPlan(BaseModel):
    """
    Planner output schema aligned with:
    - sys prompt v1.3.2 (JOIN FIRST-CLASS)
    - "joins" is mandatory (empty list allowed)
    """
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unsupported"]
    unsupported_reason: Optional[str] = None

    table_type: Literal["daily", "monthly"]

    metric: Literal["balance", "interest_rate", "count"]
    aggregation: Literal["sum", "average", "count"]

    time: TimeSpec

    # sys prompt: at most 2 (enforce in orchestrator if you prefer)
    group_by: List[str] = Field(default_factory=list)

    where: WhereSpec = Field(default_factory=WhereSpec)

    # sys prompt: always present
    joins: List[JoinSpec] = Field(default_factory=list)


# -------------------------
# Helper
# -------------------------
def parse_plan(obj: Union[str, dict]) -> PlannerPlan:
    """
    Parse Planner output from:
    - raw JSON string
    - already-parsed dict
    """
    if isinstance(obj, str):
        return PlannerPlan.model_validate_json(obj)
    return PlannerPlan.model_validate(obj)