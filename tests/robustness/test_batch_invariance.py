"""Batch invariance — same customer scored individually vs in batch must match."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from retention.serving.app import app
from retention.serving.dependencies import get_bundle

pytestmark = pytest.mark.skipif(
    get_bundle() is None,
    reason="champion bundle not built — run `retention bundle` first",
)

client = TestClient(app)


def _make_payload(customer_id: str, usage_val: float = 10.0):
    return {
        "schema_version": "1.0",
        "customer_id": customer_id,
        "company_size": 50,
        "package": "Pro",
        "tenure_months": 24,
        "subscription_type": "Monthly",
        "monthly_revenue": 500.0,
        "support_tickets_30d": 2,
        "email_engagement": 0.5,
        "usage": [{"day": d, "daily_usage": usage_val} for d in range(-59, 1)],
    }


def test_single_vs_batch_equal():
    p1 = _make_payload("1001", 10.0)
    p2 = _make_payload("1002", 20.0)

    r1 = client.post("/v1/score", json=p1)
    r2 = client.post("/v1/score", json=p2)
    assert r1.status_code == 200
    assert r2.status_code == 200

    r_batch = client.post("/v1/rank", json=[p1, p2])
    assert r_batch.status_code == 200
    batch_data = r_batch.json()
    batch_map = {c["customer_id"]: c for c in batch_data["ranked_customers"]}

    single = r1.json()
    batch = batch_map["1001"]
    assert abs(single["estimated_uplift"] - batch["estimated_uplift"]) < 1e-5
    assert abs(single["p_churn_no_contact"] - batch["p_churn_no_contact"]) < 1e-5
    assert abs(single["p_churn_contact"] - batch["p_churn_contact"]) < 1e-5
