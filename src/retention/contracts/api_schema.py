from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Annotated, Literal


class UsagePoint(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    day: Annotated[int, Field(ge=-59, le=0)]
    daily_usage: Annotated[float, Field(ge=0)]


class CustomerInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal["1.0"]
    customer_id: str
    company_size: Annotated[int, Field(ge=1)]
    package: str
    tenure_months: Annotated[int, Field(ge=0)]
    subscription_type: str
    monthly_revenue: Annotated[float, Field(ge=0)]
    support_tickets_30d: Annotated[int, Field(ge=0)]
    email_engagement: Annotated[float, Field(ge=0, le=1)]
    usage: list[UsagePoint]

    @field_validator("usage")
    @classmethod
    def no_duplicate_days(cls, v):
        days = [p.day for p in v]
        if len(days) != len(set(days)):
            raise ValueError("duplicate_usage_days")
        return v


class CustomerOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    customer_id: str
    p_churn_no_contact: float
    p_churn_contact: float
    estimated_uplift: float
    uplift_uncertainty: float
    ranking_score: float
    recommend_contact: bool
    reason_codes: list[str]
    data_quality_status: str
    model_version: str
    feature_version: str


class RankingOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    campaign_id: str
    ranked_customers: list[CustomerOutput]
    total_eligible: int
    top_k: int
    model_version: str
    policy_version: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None
    severity: str = "error"
    retryable: bool = False


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
