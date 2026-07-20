"""Data validation tests — pandera bounds, join, completeness."""

import pandas as pd
from retention.data.validation import validate_customers, validate_usage
from retention.data.validation import check_join_integrity, check_usage_completeness


def test_validate_customers_happy_path():
    df = pd.DataFrame({
        "customer_id": [1], "company_size": [10], "package": ["Basic"],
        "tenure_months": [12], "subscription_type": ["Monthly"],
        "monthly_revenue": [100.0], "support_tickets_30d": [1],
        "email_engagement": [0.5], "contacted": [0], "churned": [1],
    })
    result = validate_customers(df)
    assert len(result) == 1


def test_validate_usage_happy_path():
    df = pd.DataFrame({
        "customer_id": [1, 1], "day": [-59, -58], "daily_usage": [10.0, 0.0],
    })
    result = validate_usage(df)
    assert len(result) == 2
