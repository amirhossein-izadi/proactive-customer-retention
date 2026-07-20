"""Data validation — Pandera schemas with hard domain bounds."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

from retention.monitoring.metrics import SCHEMA_VALIDATION_FAILURES


class CustomerSchema(pa.DataFrameModel):
    customer_id: int = pa.Field(ge=1, unique=True)
    company_size: int = pa.Field(ge=1)
    package: str = pa.Field(isin=["Basic", "Pro", "Enterprise"])
    tenure_months: int = pa.Field(ge=0)
    subscription_type: str = pa.Field(isin=["Monthly", "Annual"])
    monthly_revenue: float = pa.Field(ge=0)
    support_tickets_30d: int = pa.Field(ge=0)
    email_engagement: float = pa.Field(ge=0, le=1)
    contacted: int = pa.Field(isin=[0, 1])
    churned: int = pa.Field(isin=[0, 1])


class UsageSchema(pa.DataFrameModel):
    customer_id: int = pa.Field(ge=1)
    day: int = pa.Field(ge=-59, le=0)
    daily_usage: float = pa.Field(ge=0)


def validate_customers(df: pd.DataFrame) -> DataFrame[CustomerSchema]:
    try:
        return DataFrame[CustomerSchema](df)
    except Exception:
        try:
            SCHEMA_VALIDATION_FAILURES.labels(error_type="customer").inc()
        except Exception:
            pass
        raise


def validate_usage(df: pd.DataFrame) -> DataFrame[UsageSchema]:
    try:
        return DataFrame[UsageSchema](df)
    except Exception:
        try:
            SCHEMA_VALIDATION_FAILURES.labels(error_type="usage").inc()
        except Exception:
            pass
        raise


def check_join_integrity(customers: pd.DataFrame, usage: pd.DataFrame) -> dict:
    cust_ids = set(customers["customer_id"])
    usage_ids = set(usage["customer_id"])
    missing_in_usage = cust_ids - usage_ids
    missing_in_customers = usage_ids - cust_ids
    return {
        "customers_without_usage": len(missing_in_usage),
        "usage_without_customers": len(missing_in_customers),
        "join_coverage_pct": round(len(cust_ids & usage_ids) / len(cust_ids) * 100, 2),
    }


def check_usage_completeness(usage: pd.DataFrame) -> dict:
    counts = usage.groupby("customer_id").size()
    return {
        "min_days": int(counts.min()),
        "max_days": int(counts.max()),
        "median_days": int(counts.median()),
        "customers_with_full_history": int((counts == 60).sum()),
        "customers_with_partial_history": int((counts < 60).sum()),
    }
