"""Doubly-robust policy value estimation.

Uplift convention: ``uplift = p0 - p1`` where p0 = P(churn | no contact) and
p1 = P(churn | contact). Positive uplift means contacting reduces churn.

The DR pseudo-outcome for tau(x) = mu0(x) - mu1(x) is:

    psi = (p0 - p1) + (1 - T)(Y - p0)/(1 - e) - T(Y - p1)/e

which is unbiased for the individual benefit of *not* being treated vs treated
(i.e., the churn reduction achieved by contact) under unconfoundedness.
"""

from __future__ import annotations

import numpy as np

E_CLIP = (0.02, 0.98)


def dr_pseudo_outcome(
    p0: np.ndarray, p1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    ec = np.clip(e, *E_CLIP)
    return (p0 - p1) + (1 - T) * (Y - p0) / (1 - ec) - T * (Y - p1) / ec


def dr_policy_value(
    uplift: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    e: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    k: int = 50,
) -> float:
    """Mean DR pseudo-outcome among the top-k customers ranked by ``uplift``.

    Interpretation: estimated average churn-probability reduction achieved by
    contacting the selected k customers instead of none.
    """
    psi = dr_pseudo_outcome(p0, p1, e, T, Y)
    idx = np.argsort(uplift)[::-1][:k]
    return float(psi[idx].mean())


def random_policy_value(
    p0: np.ndarray, p1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> float:
    """Value of contacting random customers — the ATE baseline (DR estimate)."""
    return float(dr_pseudo_outcome(p0, p1, e, T, Y).mean())
