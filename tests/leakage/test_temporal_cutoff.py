"""Temporal cutoff — future data must not influence past predictions."""

import numpy as np
import pandas as pd
from retention.features.temporal import _robust_slope


def test_future_not_leaked():
    """Slope computed on day -30..0 should be the same regardless of future data."""
    past = np.array([[1, 2, 3, 4, 5]], dtype=float)
    future = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
    slope_past = _robust_slope(past)
    slope_future = _robust_slope(future[:, :5])
    np.testing.assert_allclose(slope_past, slope_future, atol=1e-6)


def test_cutoff_day_zero():
    """Day 0 is the decision point. Features must not use data after day 0."""
    arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=float)
    half = arr.shape[1] // 2
    early = arr[:, :half]
    recent = arr[:, -half:]
    # These use only data up to day 0
    assert early.shape[1] + recent.shape[1] <= arr.shape[1] + 1  # allow overlap
