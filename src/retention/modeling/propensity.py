"""Propensity model — P(contacted | X) with overlap diagnostics.

**Important**: No ``class_weight`` is used. The propensity is an unbiased
estimate of P(T=1 | X), not a balanced classifier. The dataset is roughly
balanced (T ~ 48%) anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_propensity(X: pd.DataFrame, T: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, T)
    return model


def cross_fitted_propensity(
    X: pd.DataFrame, T: np.ndarray, cv_folds: int = 5, seed: int = 42
) -> np.ndarray:
    from sklearn.model_selection import cross_val_predict

    model = LogisticRegression(max_iter=1000, random_state=seed)
    return np.clip(
        cross_val_predict(model, X, T, cv=cv_folds, method="predict_proba")[:, 1],
        0.02,
        0.98,
    )


def propensity_overlap_stats(e: np.ndarray) -> dict:
    return {
        "min": float(e.min()),
        "max": float(e.max()),
        "p01": float(np.percentile(e, 1)),
        "p99": float(np.percentile(e, 99)),
        "below_005": float((e < 0.05).mean()),
        "above_095": float((e > 0.95).mean()),
        "common_support_pct": round(float(((e >= 0.05) & (e <= 0.95)).mean()) * 100, 2),
    }
