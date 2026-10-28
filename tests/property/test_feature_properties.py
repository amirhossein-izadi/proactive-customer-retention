"""Feature invariants — building features must never produce NaN or out-of-range values."""

import numpy as np
from retention.features.temporal import _robust_slope, _max_zero_streak


def test_robust_slope_no_nan():
    arr = np.array([[1, 2, 3, np.nan, 5], [5, 4, 3, 2, 1]], dtype=float)
    s = _robust_slope(arr)
    assert not np.any(np.isnan(s)), "Slope should not produce NaN"


def test_robust_slope_short_sequence():
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    s = _robust_slope(arr)
    assert s.shape == (2,)


def test_max_zero_streak_non_negative():
    arr = np.array([[0, 0, 1, 0, 0, 0], [1, 2, 3, 4, 5, 6]], dtype=float)
    streaks = _max_zero_streak(arr)
    assert np.all(streaks >= 0)
    assert streaks[0] >= 2


def test_max_zero_streak_no_zeros():
    arr = np.array([[1, 2, 3]], dtype=float)
    assert _max_zero_streak(arr)[0] == 0
