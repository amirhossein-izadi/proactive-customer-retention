"""Uplift curve metrics — AUUC and Qini with the correct beneficial sign.

Uplift convention: positive uplift = contact reduces churn. A good ranking
places customers for whom contact helps most at the top, so the cumulative
incremental gain curve should rise above the random diagonal.

For the top-i selected customers, the estimated incremental benefit is::

    gain(i) = (churn_rate_control(i) - churn_rate_treated(i)) * i

i.e., the number of churns avoided by contacting these i customers, estimated
from the observed rates in each arm within the selected set.
"""

from __future__ import annotations

import numpy as np

from retention.evaluation.policy_value import dr_policy_value  # noqa: F401  (re-export)


def cumulative_gain_curve(uplift: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cumulative incremental gain (avoided churns) over the ranking."""
    order = np.argsort(uplift)[::-1]
    T_o, Y_o = T[order], Y[order]
    n = len(order)
    gain = np.zeros(n)
    n_t = np.cumsum(T_o == 1)
    n_c = np.cumsum(T_o == 0)
    y_t = np.cumsum(Y_o * (T_o == 1))
    y_c = np.cumsum(Y_o * (T_o == 0))
    idx = np.arange(1, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(n_t > 0, y_t / np.maximum(n_t, 1), 0.0)
        rate_c = np.where(n_c > 0, y_c / np.maximum(n_c, 1), 0.0)
    gain = (rate_c - rate_t) * idx
    return gain


def auuc(uplift: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """Area under the uplift curve (trapezoid, normalized by n). Higher = better."""
    curve = cumulative_gain_curve(uplift, T, Y)
    return float(np.sum((curve[1:] + curve[:-1]) / 2) / len(curve))


def qini(uplift: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """Qini coefficient: area between model curve and random diagonal."""
    curve = cumulative_gain_curve(uplift, T, Y)
    n = len(curve)
    diagonal = curve[-1] * np.arange(1, n + 1) / n
    qini_curve = curve - diagonal
    return float(np.sum((qini_curve[1:] + qini_curve[:-1]) / 2) / n)
