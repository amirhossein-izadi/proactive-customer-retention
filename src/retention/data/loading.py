"""Data loading and DVC-tracked dataset access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retention.data.joining import join_customers_usage, pivot_usage
from retention.data.normalization import normalize_column_names, normalize_customer_frame
from retention.utils.paths import RAW as RAW_DATA


def load_customer_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load customers with normalized column names and categoricals."""
    return normalize_customer_frame(
        pd.read_csv(Path(path) if path else RAW_DATA / "customer_data.csv")
    )


def load_usage_data(path: str | Path | None = None) -> pd.DataFrame:
    df = normalize_column_names(
        pd.read_csv(Path(path) if path else RAW_DATA / "usage_timeseries.csv")
    )
    df["daily_usage"] = df["daily_usage"].astype(float)
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_customer_data(), load_usage_data()


def load_with_wide_usage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (merged, usage_wide) where usage_wide has 60 columns: usage_day_-59..0."""
    customers, usage = load_all()
    usage_wide = pivot_usage(usage)
    return join_customers_usage(customers, usage), usage_wide
