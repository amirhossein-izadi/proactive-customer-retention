"""Missingness vs zero-usage features — missing is never conflated with zero."""

from __future__ import annotations

import numpy as np


def missingness_features(arr: np.ndarray, expected_days: int = 60) -> dict[str, np.ndarray]:
    """Compute missing/zero structure of a wide usage matrix (n_samples, n_days).

    - ``missing_days``: NaN day count (data absent — NOT the same as zero usage).
    - ``zero_days``: observed days with exactly zero usage.
    - ``coverage_ratio``: fraction of the expected window actually observed.
    - ``days_since_activity``: days since the last observed positive usage
      (expected_days when never active).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    missing = np.isnan(arr)
    zero = ~missing & (arr == 0)
    active = ~missing & (arr > 0)

    n_cols = arr.shape[1]
    days_since = np.full(arr.shape[0], expected_days, dtype=float)
    for i in range(arr.shape[0]):
        idx = np.where(active[i])[0]
        if len(idx):
            days_since[i] = n_cols - 1 - idx[-1]

    return {
        "missing_days": missing.sum(axis=1).astype(float),
        "zero_days": zero.sum(axis=1).astype(float),
        "coverage_ratio": (arr.shape[1] - missing.sum(axis=1)) / expected_days,
        "days_since_activity": days_since,
    }
