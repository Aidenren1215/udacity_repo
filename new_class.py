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