"""Probability calibration — Platt scaling per treatment arm."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, clone


def fit_calibrator(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(clone(model), method=method, cv=cv)
    calibrator.fit(X, y)
    return calibrator


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))
    brier = brier_score_loss(y_true, y_prob)
    return {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "n_bins": n_bins,
    }
