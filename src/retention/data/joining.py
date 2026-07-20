"""Join helpers — pivot long usage to wide and merge with customers.

Used identically by training, evaluation, bundling and serving so the
feature matrix is built exactly one way everywhere.
"""

from __future__ import annotations

import pandas as pd

from retention.contracts.raw_data import PROTECTED_COLUMNS


def pivot_usage(usage: pd.DataFrame) -> pd.DataFrame:
    """Pivot long (customer_id, day, daily_usage) to wide usage_day_* columns."""
    df = usage.copy()
    df["day_col"] = "usage_day_" + df["day"].astype(int).astype(str)
    wide = df.pivot(index="customer_id", columns="day_col", values="daily_usage")
    wide.columns.name = None
    return wide.reset_index()


def join_customers_usage(
    customers: pd.DataFrame,
    usage: pd.DataFrame,
    customer_ids: pd.Series | list | None = None,
) -> pd.DataFrame:
    """Merge customer attributes with wide usage for a given id set."""
    c = customers
    u = usage
    if customer_ids is not None:
        ids = pd.Series(customer_ids).unique()
        c = customers[customers["customer_id"].isin(ids)]
        u = usage[usage["customer_id"].isin(ids)]
    return c.merge(pivot_usage(u), on="customer_id", how="left")


def feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """Drop protected columns (id/treatment/outcome) from a merged frame."""
    feats = [c for c in merged.columns if c not in PROTECTED_COLUMNS]
    return merged[feats].reset_index(drop=True)
