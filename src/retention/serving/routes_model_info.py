"""Model information endpoint — /v1/model-info."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from retention.bundle.model_bundle import RetentionPolicyBundle
from retention.serving.dependencies import get_bundle

router = APIRouter(prefix="/v1", tags=["model-info"])


@router.get("/model-info")
async def model_info(bundle: RetentionPolicyBundle | None = Depends(get_bundle)):  # noqa: B008
    info = {
        "model_version": "unknown",
        "feature_version": "unknown",
        "model_family": "unknown",
        "model_loaded": bundle is not None,
    }
    if bundle is not None:
        info.update(bundle.meta)
    return info
