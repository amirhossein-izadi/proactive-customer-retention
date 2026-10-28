"""Input schema — Pydantic validation for CustomerInput."""

import json
from retention.contracts.api_schema import CustomerInput


def _valid_input():
    return {
        "schema_version": "1.0",
        "customer_id": "1001",
        "company_size": 50,
        "package": "Pro",
        "tenure_months": 24,
        "subscription_type": "Monthly",
        "monthly_revenue": 500.0,
        "support_tickets_30d": 2,
        "email_engagement": 0.5,
        "usage": [{"day": d, "daily_usage": 10.0} for d in range(-59, 1)],
    }


def test_valid_input():
    inp = CustomerInput(**_valid_input())
    assert inp.customer_id == "1001"


def test_missing_schema_version():
    d = _valid_input()
    d.pop("schema_version")
    import pytest
    with pytest.raises(Exception):
        CustomerInput(**d)


def test_negative_usage_rejected():
    d = _valid_input()
    d["usage"][0]["daily_usage"] = -1
    import pytest
    with pytest.raises(Exception):
        CustomerInput(**d)


def test_duplicate_days():
    d = _valid_input()
    d["usage"].append({"day": 0, "daily_usage": 5.0})
    import pytest
    with pytest.raises(Exception):
        CustomerInput(**d)


def test_customer_id_string():
    inp = CustomerInput(**_valid_input())
    assert isinstance(inp.customer_id, str)
