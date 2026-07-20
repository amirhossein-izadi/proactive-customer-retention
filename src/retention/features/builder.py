"""Feature engineering — static + temporal + PCA representations.

Guarantees:
- Usage columns are explicitly ordered chronologically (day -59 .. 0),
  never by lexicographic column order.
- Missing usage (NaN) is never conflated with observed zero usage.
- The preprocessor is a single, versioned object used by training,
  evaluation, bundling and serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention.contracts.raw_data import usage_columns
from retention.data.joining import pivot_usage as _pivot_usage  # noqa: F401  (re-export)
from retention.features.missingness import missingness_features
from retention.features.temporal import (
    _days_since_last,
    _detrend,
    _max_drawdown,
    _max_zero_streak,
    _robust_slope,
    build_temporal_features,
)

USAGE_COLS = usage_columns()

STATIC_INPUT = [
    "company_size",
    "monthly_revenue",
    "tenure_months",
    "subscription_type",
    "email_engagement",
    "support_tickets_30d",
]


def _chronological_usage_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return the usage matrix ordered by day ascending, regardless of input order."""
    if not isinstance(X, pd.DataFrame):
        return np.asarray(X, dtype=float)
    day_cols = [c for c in X.columns if str(c).startswith("usage_day_")]
    day_cols = sorted(day_cols, key=lambda c: int(str(c).replace("usage_day_", "")))
    if not day_cols:
        return np.zeros((len(X), 0))
    return X[day_cols].values.astype(float)


class TemporalEngineer(BaseEstimator, TransformerMixin):
    """Engineered temporal features: level, trend, inactivity, volatility, shape."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(index=getattr(X, "index", range(len(X))))
        arr = _chronological_usage_array(X)
        if arr.shape[1] < 10:
            for f in self.feature_names():
                out[f] = 0.0
            return out.values

        half = arr.shape[1] // 2
        early, recent = arr[:, :half], arr[:, -half:]
        last_7 = arr[:, -7:] if arr.shape[1] >= 7 else arr
        last_14 = arr[:, -14:] if arr.shape[1] >= 14 else arr

        miss = missingness_features(arr, expected_days=arr.shape[1])

        # Level
        out["usage_mean"] = np.nanmean(arr, axis=1)
        out["usage_median"] = np.nanmedian(arr, axis=1)
        out["usage_max"] = np.nanmax(arr, axis=1)
        out["usage_min"] = np.nanmin(arr, axis=1)
        out["usage_total"] = np.nansum(arr, axis=1)
        out["recent_7_mean"] = np.nanmean(last_7, axis=1)
        out["recent_14_mean"] = np.nanmean(last_14, axis=1)
        out["recent_30_mean"] = np.nanmean(recent, axis=1)
        out["early_30_mean"] = np.nanmean(early, axis=1)
        out["early_7_mean"] = np.nanmean(arr[:, :7], axis=1)
        out["early_14_mean"] = np.nanmean(arr[:, :14], axis=1)
        out["last_day_usage"] = arr[:, -1]

        # Trend
        out["slope_60d"] = _robust_slope(arr)
        out["slope_30d"] = _robust_slope(recent)
        out["slope_14d"] = _robust_slope(last_14)
        out["recent_minus_early"] = out["recent_30_mean"].values - out["early_30_mean"].values
        out["recent_early_ratio"] = out["recent_30_mean"].values / out["early_30_mean"].values.clip(
            1e-6
        )
        out["pct_decline"] = (out["early_30_mean"].values - out["recent_30_mean"].values) / out[
            "early_30_mean"
        ].values.clip(1e-6)

        # Inactivity — observed zeros and missing days are distinct signals
        out["zero_days"] = miss["zero_days"]
        out["missing_days"] = miss["missing_days"]
        out["active_days_pct"] = 1.0 - (out["zero_days"] + out["missing_days"]) / arr.shape[1]
        out["longest_zero_streak"] = _max_zero_streak(arr)
        out["days_since_activity"] = miss["days_since_activity"]
        out["recent_zero_days"] = np.sum(~np.isnan(recent) & (recent == 0), axis=1)

        # Volatility
        out["usage_std"] = np.nanstd(arr, axis=1)
        out["cv"] = out["usage_std"] / out["usage_mean"].clip(1e-6)
        out["mean_abs_daily_change"] = np.nanmean(np.abs(np.diff(arr, axis=1)), axis=1)
        out["down_day_proportion"] = _down_day_proportion(arr)
        out["max_drawdown"] = _max_drawdown(arr)
        out["trend_residual_var"] = np.nanvar(_detrend(arr), axis=1)

        # Shape
        out["first_half_total"] = np.nansum(arr[:, :half], axis=1)
        out["second_half_total"] = np.nansum(arr[:, half:], axis=1)
        out["halves_ratio"] = out["second_half_total"] / out["first_half_total"].clip(1e-6)

        return out.fillna(0.0).values

    def feature_names(self):
        return [
            "usage_mean",
            "usage_median",
            "usage_max",
            "usage_min",
            "usage_total",
            "recent_7_mean",
            "recent_14_mean",
            "recent_30_mean",
            "early_30_mean",
            "early_7_mean",
            "early_14_mean",
            "last_day_usage",
            "slope_60d",
            "slope_30d",
            "slope_14d",
            "recent_minus_early",
            "recent_early_ratio",
            "pct_decline",
            "zero_days",
            "missing_days",
            "active_days_pct",
            "longest_zero_streak",
            "days_since_activity",
            "recent_zero_days",
            "usage_std",
            "cv",
            "mean_abs_daily_change",
            "down_day_proportion",
            "max_drawdown",
            "trend_residual_var",
            "first_half_total",
            "second_half_total",
            "halves_ratio",
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names())


def _down_day_proportion(arr: np.ndarray) -> np.ndarray:
    """Proportion of observed day-over-day changes that are negative (NaN-aware)."""
    diffs = np.diff(arr, axis=1)
    valid = ~np.isnan(diffs)
    n_valid = valid.sum(axis=1).clip(min=1)
    return (valid & (diffs < 0)).sum(axis=1) / n_valid


class StaticEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df["log_company_size"] = np.log1p(df["company_size"])
        df["log_revenue"] = np.log1p(df["monthly_revenue"])
        df["revenue_per_employee"] = df["monthly_revenue"] / df["company_size"].clip(lower=1)
        df["tenure_years"] = df["tenure_months"] / 12.0
        df["is_annual"] = (df["subscription_type"] == "Annual").astype(int)
        df["is_monthly"] = (df["subscription_type"] == "Monthly").astype(int)
        df["tickets_per_revenue"] = df["support_tickets_30d"] / df["monthly_revenue"].clip(lower=1)
        df["tickets_per_employee"] = df["support_tickets_30d"] / df["company_size"].clip(lower=1)
        return df[self.feature_names()].values

    def feature_names(self):
        return [
            "log_company_size",
            "log_revenue",
            "revenue_per_employee",
            "tenure_years",
            "is_annual",
            "is_monthly",
            "tickets_per_revenue",
            "tickets_per_employee",
            "email_engagement",
            "support_tickets_30d",
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names())


class PCAEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(np.nan_to_num(_chronological_usage_array(X), nan=0.0))
        return self

    def transform(self, X):
        return self.pca.transform(np.nan_to_num(_chronological_usage_array(X), nan=0.0))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"pca_usage_{i}" for i in range(self.n_components)])


def build_preprocessor(
    include_pca: bool = True,
    scale: bool = True,
    raw_passthrough: bool = False,
) -> SkPipeline:
    """Canonical preprocessing pipeline.

    - One-hot encodes ``package`` and ``subscription_type``.
    - Optional raw 60-day usage passthrough for models that ingest raw sequences.
    - Optional StandardScaler: required for logistic/propensity convergence;
      monotone-invariant, so harmless for tree boosters.
    """
    transformers = [
        ("static", StaticEngineer(), STATIC_INPUT),
        ("temporal", TemporalEngineer(), USAGE_COLS),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["package", "subscription_type"],
        ),
    ]
    if include_pca:
        transformers.insert(2, ("pca", PCAEmbedding(n_components=5), USAGE_COLS))
    if raw_passthrough:
        transformers.append(("raw_usage", "passthrough", USAGE_COLS))
    steps: list = [("columns", ColumnTransformer(transformers))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return SkPipeline(steps)


def get_categorical_indices(preprocessor: SkPipeline) -> list[int]:
    """Return 0-based indices of OHE categorical features in the preprocessor output.

    CatBoost benefits from native categorical handling (ordered target encoding)
    rather than one-hot encoding. Pass these indices via ``cat_features``.
    """
    ct = preprocessor.named_steps["columns"]
    names = ct.get_feature_names_out()
    cat_starts = ("package_", "subscription_type_")
    return [i for i, n in enumerate(names) if any(n.startswith(p) for p in cat_starts)]



