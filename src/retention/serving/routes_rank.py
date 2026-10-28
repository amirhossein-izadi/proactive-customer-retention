"""Batch ranking endpoint — /v1/rank."""

from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from retention.data.joining import pivot_usage
from retention.monitoring.metrics import (
    ACTIVE_DAYS_PCT,
    INFERENCE_COMMON_SUPPORT,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    NEGATIVE_UPLIFT_RATE,
    UPLIFT_SCORES,
    ZERO_DAYS_RATIO,
)
from retention.serving.dependencies import get_bundle
from retention.settings.config import Settings, get_settings
from retention.contracts.api_schema import CustomerInput, CustomerOutput, RankingOutput
from retention.bundle.model_bundle import RetentionPolicyBundle

router = APIRouter(prefix="/v1", tags=["rank"])


@router.post("/rank", response_model=RankingOutput)
async def rank_batch(
    inputs: list[CustomerInput],
    bundle: RetentionPolicyBundle | None = Depends(get_bundle),
    settings: Settings = Depends(get_settings),
):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try /health or /ready.")

    start = time.perf_counter()
    # Assemble wide frame from batch inputs
    rows = []
    for inp in inputs:
        usage_df = pd.DataFrame(
            {
                "customer_id": [inp.customer_id] * len(inp.usage),
                "day": [p.day for p in inp.usage],
                "daily_usage": [p.daily_usage for p in inp.usage],
            }
        )
        wide = pivot_usage(usage_df)
        static = pd.DataFrame([inp.model_dump(exclude={"usage", "schema_version"})])
        row = static.merge(wide, on="customer_id", how="left")
        rows.append(row)
    X_raw = pd.concat(rows, ignore_index=True)

    rank_df = bundle.rank(X_raw, k=min(settings.top_k, len(inputs)))
    infer_time = time.perf_counter() - start
    INFERENCE_LATENCY.labels(endpoint="/v1/rank").observe(infer_time)
    INFERENCE_REQUESTS.labels(endpoint="/v1/rank", status="200").inc()

    mean_uplift = rank_df["estimated_uplift"].mean()
    UPLIFT_SCORES.observe(mean_uplift)
    if mean_uplift < 0:
        NEGATIVE_UPLIFT_RATE.inc()
    n_usage = sum(len(inp.usage) for inp in inputs)
    n_zero = sum(1 for inp in inputs for p in inp.usage if p.daily_usage == 0)
    ACTIVE_DAYS_PCT.set(n_usage / (60.0 * max(len(inputs), 1)))
    ZERO_DAYS_RATIO.set(n_zero / max(n_usage, 1))
    INFERENCE_COMMON_SUPPORT.set(float(rank_df["in_common_support"].mean()))

    ranked = [
        CustomerOutput(
            customer_id=str(r["customer_id"]),
            p_churn_no_contact=float(r["p_churn_no_contact"]),
            p_churn_contact=float(r["p_churn_contact"]),
            estimated_uplift=float(r["estimated_uplift"]),
            uplift_uncertainty=float(r.get("uplift_uncertainty", 0.0)),
            ranking_score=float(r["estimated_uplift"]),
            recommend_contact=bool(r["estimated_uplift"] > 0.01 and r["in_common_support"]),
            reason_codes=[],
            data_quality_status="NORMAL",
            model_version=settings.model_version,
            feature_version=settings.feature_version,
        )
        for _, r in rank_df.iterrows()
    ]
    return RankingOutput(
        campaign_id="batch_rank",
        ranked_customers=ranked,
        total_eligible=len(rank_df),
        top_k=settings.top_k,
        model_version=settings.model_version,
        policy_version=settings.policy_version,
    )
