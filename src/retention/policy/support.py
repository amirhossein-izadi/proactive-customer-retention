"""Propensity overlap and common support checks."""

import numpy as np


def overlap_stats(e: np.ndarray) -> dict:
    return {
        "min": float(e.min()),
        "max": float(e.max()),
        "p01": float(np.percentile(e, 1)),
        "p99": float(np.percentile(e, 99)),
        "below_005": float((e < 0.05).mean()),
        "above_095": float((e > 0.95).mean()),
        "common_support_pct": float(((e >= 0.05) & (e <= 0.95)).mean()) * 100,
    }


def overlap_support_mask(e: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Boolean mask: True for customers within propensity common-support region."""
    return (e >= lo) & (e <= hi)
