"""Calibration evaluation — arm-wise reliability tables and metrics."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.mean(np.abs(prob_true - prob_pred)))


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) == 0:
        return {"bins": [], "prob_true": [], "prob_pred": [], "ece": 0.0}
    return {
        "bins": list(range(len(prob_true))),
        "prob_true": [round(float(v), 4) for v in prob_true],
        "prob_pred": [round(float(v), 4) for v in prob_pred],
        "ece": round(ece(y_true, y_prob, n_bins), 4),
    }


def arm_calibration_report(Y: np.ndarray, p0: np.ndarray, p1: np.ndarray, T: np.ndarray) -> dict:
    control, treated = T == 0, T == 1
    return {
        "arm0_calibration": reliability_table(Y[control], p0[control]),
        "arm1_calibration": reliability_table(Y[treated], p1[treated]),
    }
