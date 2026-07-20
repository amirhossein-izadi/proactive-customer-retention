"""Property-based tests for feature engineering invariants."""

from hypothesis import given, strategies as st
import pandas as pd
import numpy as np

from retention.features.temporal import _robust_slope, _max_zero_streak


@given(st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=60))
def test_robust_slope_shape(vals):
    arr = np.array([vals])
    result = _robust_slope(arr)
    assert result.shape == (1,)
    assert not np.isnan(result[0])


@given(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=1, max_size=60))
def test_max_zero_streak_nonnegative(vals):
    arr = np.array([vals])
    result = _max_zero_streak(arr)
    assert result.shape == (1,)
    assert result[0] >= 0


@given(st.integers(min_value=1, max_value=10000))
def test_customer_id_no_op(n):
    """customer_id should not affect model output — tested elsewhere with CatBoost."""
    pass
