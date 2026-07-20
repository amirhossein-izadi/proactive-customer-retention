"""Dependency injection — bundle loader, request ID, settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Request

from retention.bundle.model_bundle import RetentionPolicyBundle
from retention.settings.config import Settings, get_settings
from retention.utils.paths import CHAMPION_DIR


def _load_bundle(path: str | Path) -> RetentionPolicyBundle | None:
    """Load bundle from disk — no caching. Used by health checks to bypass cache on miss."""
    p = Path(path)
    if not (p / "calibrator_0.pkl").exists():
        return None
    try:
        return RetentionPolicyBundle().load(str(p))
    except Exception:
        return None


@lru_cache(maxsize=4)
def get_bundle(path: str | Path = CHAMPION_DIR) -> RetentionPolicyBundle | None:
    return _load_bundle(path)


def clear_bundle_cache() -> None:
    """Force next get_bundle call to reload from disk."""
    get_bundle.cache_clear()


def get_request_id(request: Request) -> str:
    return request.headers.get("x-request-id", "")


__all__ = ["clear_bundle_cache", "get_bundle", "get_request_id", "get_settings", "Settings", "_load_bundle"]
