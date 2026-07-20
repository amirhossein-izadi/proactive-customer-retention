"""Health-check routes."""

from __future__ import annotations

from fastapi import APIRouter

from retention.serving.dependencies import _load_bundle, clear_bundle_cache, get_bundle
from retention.monitoring.metrics import MODEL_LOADED

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "alive"}


@router.get("/ready")
async def ready():
    bundle = get_bundle()
    if bundle is None:
        clear_bundle_cache()
        bundle = _load_bundle()
    loaded = bundle is not None
    MODEL_LOADED.set(1 if loaded else 0)
    return {"status": "ready" if loaded else "loading", "model_loaded": loaded}
