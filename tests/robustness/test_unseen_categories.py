"""Unseen categories — unknown package/subscription values should not crash."""

from fastapi.testclient import TestClient
from retention.serving.app import app

client = TestClient(app)


def _payload(**overrides):
    p = {
        "schema_version": "1.0",
        "customer_id": "3003",
        "company_size": 100,
        "package": "Pro",
        "tenure_months": 36,
        "subscription_type": "Monthly",
        "monthly_revenue": 1000.0,
        "support_tickets_30d": 1,
        "email_engagement": 0.8,
        "usage": [{"day": d, "daily_usage": 10.0} for d in range(-59, 1)],
    }
    p.update(overrides)
    return p


def test_unknown_package():
    r = client.post("/v1/score", json=_payload(package="Premium"))
    assert r.status_code == 200


def test_unknown_subscription():
    r = client.post("/v1/score", json=_payload(subscription_type="BiYearly"))
    assert r.status_code == 200


def test_both_unknown():
    r = client.post("/v1/score", json=_payload(package="Ultra", subscription_type="Quarterly"))
    assert r.status_code == 200


def test_empty_string_package():
    r = client.post("/v1/score", json=_payload(package=""))
    assert r.status_code in (200, 422)
