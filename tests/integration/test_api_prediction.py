"""API prediction — full integration test of the scoring API."""

import json
from fastapi.testclient import TestClient
from retention.serving.app import app

client = TestClient(app)


def _valid_payload():
    return {
        "schema_version": "1.0",
        "customer_id": "5555",
        "company_size": 100,
        "package": "Enterprise",
        "tenure_months": 48,
        "subscription_type": "Annual",
        "monthly_revenue": 2000.0,
        "support_tickets_30d": 0,
        "email_engagement": 0.9,
        "usage": [{"day": d, "daily_usage": 50.0} for d in range(-59, 1)],
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_ready():
    r = client.get("/ready")
    assert r.status_code == 200


def test_score_returns_expected_fields():
    r = client.post("/v1/score", json=_valid_payload())
    assert r.status_code == 200
    j = r.json()
    required = {"customer_id", "p_churn_no_contact", "p_churn_contact",
                "estimated_uplift", "uplift_uncertainty", "ranking_score",
                "recommend_contact", "reason_codes", "data_quality_status",
                "model_version", "feature_version"}
    assert required.issubset(j.keys()), f"Missing: {required - j.keys()}"


def test_score_probabilities_in_bounds():
    r = client.post("/v1/score", json=_valid_payload())
    j = r.json()
    assert 0 <= j["p_churn_no_contact"] <= 1
    assert 0 <= j["p_churn_contact"] <= 1


def test_rank_returns_list():
    r = client.post("/v1/rank", json=[_valid_payload(), _valid_payload()])
    assert r.status_code == 200
    j = r.json()
    assert "ranked_customers" in j
    assert len(j["ranked_customers"]) > 0
