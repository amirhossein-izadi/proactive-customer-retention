"""Model ensembling — blends of outcome-model probabilities.

The simplest blend is a convex combination of two candidate p0 and p1
estimates. This often outperforms either component because:
- Logistic contributes smoothness and calibration.
- Boosting contributes interaction awareness.

The blend weight is tuned by grid search over factual log loss on the
development OOF predictions, separately per treatment arm.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def blend(p_a: np.ndarray, p_b: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend: alpha * p_a + (1 - alpha) * p_b."""
    return alpha * p_a + (1 - alpha) * p_b


def tune_arm_weights(
    p0_a: np.ndarray,
    p0_b: np.ndarray,
    p1_a: np.ndarray,
    p1_b: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_grid: int = 21,
) -> dict:
    """Grid search for per-arm blend weights minimising factual log loss."""
    alphas = np.linspace(0, 1, n_grid)
    best_alpha0, best_loss0 = 0.5, np.inf
    best_alpha1, best_loss1 = 0.5, np.inf
    control = T == 0
    treated = T == 1
    for a in alphas:
        blended_p0 = blend(p0_a, p0_b, a)
        blended_p1 = blend(p1_a, p1_b, a)
        if control.sum() > 5:
            loss0 = log_loss(Y[control], np.clip(blended_p0[control], 1e-7, 1 - 1e-7))
            if loss0 < best_loss0:
                best_alpha0, best_loss0 = a, loss0
        if treated.sum() > 5:
            loss1 = log_loss(Y[treated], np.clip(blended_p1[treated], 1e-7, 1 - 1e-7))
            if loss1 < best_loss1:
                best_alpha1, best_loss1 = a, loss1
    return {"alpha0": best_alpha0, "alpha1": best_alpha1, "loss0": best_loss0, "loss1": best_loss1}
