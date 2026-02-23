from pydantic import BaseModel, Field


class TimeRangeOutput(BaseModel):
    start_date: str = Field(
        ...,
        description="Inclusive start date in ISO format (YYYY-MM-DD)."
    )

    end_date: str = Field(
        ...,
        description="Inclusive end date in ISO format (YYYY-MM-DD)."
    )

    normalized_query: str = Field(
        ...,
        description="User query rewritten to include explicit start_date and end_date."
    )

    note: str = Field(
        ...,
        description="Short explanation of how the time range was determined."
    )