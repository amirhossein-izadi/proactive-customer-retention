"""Raw data column contracts — single source of truth for column semantics."""

from __future__ import annotations

CUSTOMER_ID = "customer_id"
TREATMENT = "contacted"
OUTCOME = "churned"

STATIC_NUMERIC = [
    "company_size",
    "monthly_revenue",
    "tenure_months",
    "email_engagement",
    "support_tickets_30d",
]
STATIC_CATEGORICAL = ["package", "subscription_type"]
STATIC_INPUT = STATIC_NUMERIC + STATIC_CATEGORICAL

# Columns that must never be used as model features (labels / identifiers).
PROTECTED_COLUMNS = [CUSTOMER_ID, TREATMENT, OUTCOME]

DAY_MIN, DAY_MAX = -59, 0
EXPECTED_USAGE_DAYS = DAY_MAX - DAY_MIN + 1  # 60


def usage_columns() -> list[str]:
    """Chronologically ordered wide usage columns: usage_day_-59 ... usage_day_0."""
    return [f"usage_day_{d}" for d in range(DAY_MIN, DAY_MAX + 1)]
