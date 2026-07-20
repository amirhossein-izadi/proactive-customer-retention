"""Missingness feature tests — missing days vs zero days are never conflated."""

import numpy as np
from retention.features.missingness import missingness_features


def test_zeros_only():
    arr = np.array([[0.0, 0.0, 5.0, 0.0]])
    m = missingness_features(arr, expected_days=arr.shape[1])
    assert m["missing_days"][0] == 0
    assert m["zero_days"][0] == 3
    assert m["coverage_ratio"][0] == 1.0


def test_missing_only():
    arr = np.array([[np.nan, np.nan, 5.0, np.nan]])
    m = missingness_features(arr, expected_days=arr.shape[1])
    assert m["missing_days"][0] == 3
    assert m["zero_days"][0] == 0


def test_days_since_activity():
    arr = np.array([[0.0, np.nan, 3.0, 0.0, 0.0]])
    m = missingness_features(arr, expected_days=arr.shape[1])
    assert m["days_since_activity"][0] == 2.0  # last activity at idx 2, length 5 -> 5-1-2=2
