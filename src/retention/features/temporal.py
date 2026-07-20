"""Temporal feature engineering from 60-day usage history."""

from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial


def build_temporal_features(usage_df: pd.DataFrame) -> pd.DataFrame:
    grouped = usage_df.sort_values(["customer_id", "day"]).groupby("customer_id")
    values = grouped["daily_usage"].agg(list).to_dict()
    features = {"customer_id": list(values.keys())}
    seqs = [np.array(s, dtype=float) for s in values.values()]
    arr = np.array([np.pad(s, (0, max(0, 60 - len(s))), constant_values=np.nan)[:60] for s in seqs])

    early = arr[:, :30]
    recent = arr[:, 30:]
    last_7 = arr[:, -7:] if arr.shape[1] >= 7 else arr
    last_14 = arr[:, -14:] if arr.shape[1] >= 14 else arr
    last_30 = arr[:, -30:] if arr.shape[1] >= 30 else arr

    features["usage_mean"] = np.nanmean(arr, axis=1)
    features["usage_median"] = np.nanmedian(arr, axis=1)
    features["usage_std"] = np.nanstd(arr, axis=1)
    features["cv"] = features["usage_std"] / features["usage_mean"].clip(1e-6)
    features["recent_7_mean"] = np.nanmean(last_7, axis=1)
    features["recent_14_mean"] = np.nanmean(last_14, axis=1)
    features["recent_30_mean"] = np.nanmean(recent, axis=1)
    features["early_30_mean"] = np.nanmean(early, axis=1)
    features["recent_minus_early"] = features["recent_30_mean"] - features["early_30_mean"]
    features["recent_early_ratio"] = features["recent_30_mean"] / features["early_30_mean"].clip(
        1e-6
    )
    features["slope_60d"] = _robust_slope(arr)
    features["slope_30d"] = _robust_slope(recent)
    features["last_day_usage"] = arr[:, -1]
    features["zero_days"] = np.sum(np.isnan(arr) | (arr == 0), axis=1)
    features["active_days_pct"] = 1.0 - features["zero_days"] / 60.0
    features["longest_zero_streak"] = _max_zero_streak(arr)
    features["down_day_proportion"] = np.mean(
        np.diff(np.nan_to_num(arr, nan=0), axis=1) < 0, axis=1
    )
    features["max_drawdown"] = _max_drawdown(arr)
    features["days_since_activity"] = _days_since_last(arr)
    return pd.DataFrame(features)


def _robust_slope(arr: np.ndarray) -> np.ndarray:
    slopes = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        y = arr[i]
        mask = ~np.isnan(y)
        slopes[i] = (
            polynomial.polyfit(np.arange(arr.shape[1])[mask], y[mask], 1)[1]
            if mask.sum() >= 3
            else 0.0
        )
    return slopes


def _max_zero_streak(arr: np.ndarray) -> np.ndarray:
    """Longest run of consecutive observed zero-usage days (missing breaks the run)."""
    streaks = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        cur = max_s = 0
        for v in ~np.isnan(arr[i]) & (arr[i] == 0):
            cur = cur + 1 if v else 0
            max_s = max(max_s, cur)
        streaks[i] = max_s
    return streaks


def _max_drawdown(arr: np.ndarray) -> np.ndarray:
    dd = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        row = np.nan_to_num(arr[i], nan=0)
        peak = np.maximum.accumulate(row)
        dd[i] = np.max((peak - row) / peak.clip(1e-6))
    return dd


def _days_since_last(arr: np.ndarray) -> np.ndarray:
    """Days since the last observed positive usage (window length if never active)."""
    dsl = np.full(arr.shape[0], 60)
    for i in range(arr.shape[0]):
        nonzero = np.where(~np.isnan(arr[i]) & (arr[i] > 0))[0]
        if len(nonzero) > 0:
            dsl[i] = arr.shape[1] - 1 - nonzero[-1]
    return dsl


def _detrend(arr: np.ndarray) -> np.ndarray:
    """Residuals after removing a linear trend (NaN filled with 0 for stability)."""
    detrended = np.zeros_like(arr)
    x = np.arange(arr.shape[1])
    for i in range(arr.shape[0]):
        mask = ~np.isnan(arr[i])
        if mask.sum() >= 3:
            coef = polynomial.polyfit(x[mask], arr[i][mask], 1)
            detrended[i] = np.nan_to_num(arr[i], nan=0.0) - polynomial.polyval(x, coef)
    return detrended
