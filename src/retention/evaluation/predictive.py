"""Factual-arm predictive metrics — p0 evaluated on controls, p1 on treated.

This is the ONLY correct way to score potential-outcome models on factual
data: p0 is never scored against treated outcomes (counterfactual there).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _safe(fn: Callable, y: np.ndarray, p: np.ndarray, default: float = -1.0) -> float:
    try:
        return round(float(fn(y, p)), 4)
    except ValueError:
        return default


def compute_metrics(
    Y: np.ndarray, p0: np.ndarray, p1: np.ndarray, T: np.ndarray
) -> dict[str, float]:
    control, treated = T == 0, T == 1
    factual = np.where(treated, p1, p0)
    return {
        "factual_auc": _safe(roc_auc_score, Y, factual),
        "factual_pr_auc": _safe(average_precision_score, Y, factual),
        "factual_logloss": _safe(log_loss, Y, np.clip(factual, 1e-7, 1 - 1e-7)),
        "factual_brier": _safe(brier_score_loss, Y, factual),
        "arm0_auc": _safe(roc_auc_score, Y[control], p0[control]),
        "arm1_auc": _safe(roc_auc_score, Y[treated], p1[treated]),
        "arm0_pr_auc": _safe(average_precision_score, Y[control], p0[control]),
        "arm1_pr_auc": _safe(average_precision_score, Y[treated], p1[treated]),
        "arm0_brier": _safe(brier_score_loss, Y[control], p0[control]),
        "arm1_brier": _safe(brier_score_loss, Y[treated], p1[treated]),
    }
