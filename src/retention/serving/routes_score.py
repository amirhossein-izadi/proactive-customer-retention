"""Customer scoring endpoint — /v1/score."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from retention.data.joining import pivot_usage
from retention.monitoring.audit import audit_event
from retention.monitoring.metrics import (
    ACTIVE_DAYS_PCT,
    INFERENCE_COMMON_SUPPORT,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    NEGATIVE_UPLIFT_RATE,
    UPLIFT_SCORES,
    ZERO_DAYS_RATIO,
)
from retention.serving.dependencies import get_bundle, get_request_id
from retention.settings.config import Settings, get_settings
from retention.contracts.api_schema import CustomerInput, CustomerOutput
from retention.contracts.inference_policy import classify_input
from retention.bundle.model_bundle import RetentionPolicyBundle

router = APIRouter(prefix="/v1", tags=["score"])


@router.post("/score", response_model=CustomerOutput)
async def score_customer(
    inp: CustomerInput,
    settings: Settings = Depends(get_settings),
    request_id: str = Depends(get_request_id),
    bundle: RetentionPolicyBundle | None = Depends(get_bundle),
):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try /health or /ready.")

    # Input policy check (REJECT out-of-bounds)
    policy = classify_input(inp.model_dump())
    if policy["action"] == "REJECT":
        raise HTTPException(
            status_code=422, detail={"code": "REJECTED", "reason": policy["reason"]}
        )

    start = time.perf_counter()
    # Build wide DataFrame from the usage points
    usage_df = pd.DataFrame(
        {
            "customer_id": [inp.customer_id] * len(inp.usage),
            "day": [p.day for p in inp.usage],
            "daily_usage": [p.daily_usage for p in inp.usage],
        }
    )
    wide = pivot_usage(usage_df)
    static = pd.DataFrame([inp.model_dump(exclude={"usage", "schema_version"})])
    X_raw = static.merge(wide, on="customer_id", how="left")

    scores = bundle.score(X_raw)
    infer_time = time.perf_counter() - start
    INFERENCE_LATENCY.labels(endpoint="/v1/score").observe(infer_time)
    INFERENCE_REQUESTS.labels(endpoint="/v1/score", status="200").inc()
    uplift = float(scores["estimated_uplift"][0])
    UPLIFT_SCORES.observe(uplift)
    if uplift < 0:
        NEGATIVE_UPLIFT_RATE.inc()
    ACTIVE_DAYS_PCT.set(len(inp.usage) / 60.0)
    zero_frac = sum(1 for p in inp.usage if p.daily_usage == 0) / max(len(inp.usage), 1)
    ZERO_DAYS_RATIO.set(zero_frac)
    INFERENCE_COMMON_SUPPORT.set(1.0 if bool(scores["in_common_support"][0]) else 0.0)
    audit_event(
        "score",
        customer_id=inp.customer_id,
        uplift=float(scores["estimated_uplift"][0]),
        inference_ms=round(infer_time * 1000),
    )

    uncertainty = float(scores.get("uplift_uncertainty", np.array([0.0]))[0])

    return CustomerOutput(
        customer_id=inp.customer_id,
        p_churn_no_contact=float(scores["p_churn_no_contact"][0]),
        p_churn_contact=float(scores["p_churn_contact"][0]),
        estimated_uplift=float(scores["estimated_uplift"][0]),
        uplift_uncertainty=uncertainty,
        ranking_score=float(scores["estimated_uplift"][0]),
        recommend_contact=bool(
            scores["estimated_uplift"][0] > 0.01 and scores["in_common_support"][0]
        ),
        reason_codes=[],
        data_quality_status=policy["status"],
        model_version=settings.model_version,
        feature_version=settings.feature_version,
    )
