"""Missing days — partial usage history should not crash the pipeline."""

from fastapi.testclient import TestClient
from retention.serving.app import app

client = TestClient(app)


def _payload(usage_days: int):
    return {
        "schema_version": "1.0",
        "customer_id": "5000",
        "company_size": 50,
        "package": "Basic",
        "tenure_months": 12,
        "subscription_type": "Monthly",
        "monthly_revenue": 200.0,
        "support_tickets_30d": 0,
        "email_engagement": 0.3,
        "usage": [{"day": d, "daily_usage": 5.0} for d in range(-usage_days + 1, 1)],
    }


def test_full_history_60_days():
    r = client.post("/v1/score", json=_payload(60))
    assert r.status_code == 200


def test_partial_history_30_days():
    r = client.post("/v1/score", json=_payload(30))
    assert r.status_code in (200, 422), f"Expected 200 or 422, got {r.status_code}"


def test_short_history_7_days():
    r = client.post("/v1/score", json=_payload(7))
    assert r.status_code in (200, 422), f"Expected 200 or 422, got {r.status_code}"


def test_single_day():
    r = client.post("/v1/score", json=_payload(1))
    assert r.status_code in (200, 422), f"Expected 200 or 422, got {r.status_code}"


def test_no_usage():
    p = _payload(1)
    p["usage"] = []
    r = client.post("/v1/score", json=p)
    assert r.status_code in (200, 422)
