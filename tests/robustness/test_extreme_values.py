"""Extreme values — valid extreme inputs should not crash."""

import json

import pytest
from fastapi.testclient import TestClient

from retention.serving.app import app
from retention.serving.dependencies import get_bundle

pytestmark = pytest.mark.skipif(
    get_bundle() is None,
    reason="champion bundle not built — run `retention bundle` first",
)

client = TestClient(app)


def _payload(**overrides):
    p = {
        "schema_version": "1.0",
        "customer_id": "9999",
        "company_size": 50,
        "package": "Pro",
        "tenure_months": 24,
        "subscription_type": "Monthly",
        "monthly_revenue": 500.0,
        "support_tickets_30d": 2,
        "email_engagement": 0.5,
        "usage": [{"day": d, "daily_usage": 10.0} for d in range(-59, 1)],
    }
    p.update(overrides)
    return p


def test_zero_revenue():
    r = client.post("/v1/score", json=_payload(monthly_revenue=0.0))
    assert r.status_code in (200, 422)


def test_high_revenue():
    r = client.post("/v1/score", json=_payload(monthly_revenue=1e7))
    assert r.status_code in (200, 422)


def test_zero_company_size():
    r = client.post("/v1/score", json=_payload(company_size=1))
    assert r.status_code in (200, 422)


def test_max_usage():
    r = client.post("/v1/score", json=_payload(**{"usage": [{"day": d, "daily_usage": 1e6} for d in range(-59, 1)]}))
    assert r.status_code in (200, 422)


def test_minimal_history():
    r = client.post("/v1/score", json=_payload(**{"usage": [{"day": d, "daily_usage": 0.0} for d in range(-59, 1)]}))
    assert r.status_code in (200, 422)
