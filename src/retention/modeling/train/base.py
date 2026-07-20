"""Shared data loading — dev/test sets with eligibility extras."""

from __future__ import annotations

import pandas as pd

from retention.data.joining import feature_frame, join_customers_usage
from retention.data.loading import load_all
from retention.utils.paths import SPLITS


def load_dev_test() -> tuple:
    """Load development and test sets from DVC-tracked splits.

    Returns (Xd, Td, Yd, cids_d, extras_d, Xt, Tt, Yt, cids_t, extras_t)
    where extras is a dict with tenure_months and monthly_revenue arrays.
    """
    dev_ids = pd.read_csv(SPLITS / "development_set.csv")["customer_id"]
    test_ids = pd.read_csv(SPLITS / "test_set.csv")["customer_id"]
    customers, usage = load_all()

    def extract(df_ids):
        m = join_customers_usage(customers, usage, df_ids)
        X = feature_frame(m)
        T = m["contacted"].values.astype(int)
        Y = m["churned"].values.astype(int)
        cids = m["customer_id"].values
        extras = {
            "tenure_months": m["tenure_months"].values.astype(float),
            "monthly_revenue": m["monthly_revenue"].values.astype(float),
        }
        return X, T, Y, cids, extras

    Xd, Td, Yd, cd, ed = extract(dev_ids)
    Xt, Tt, Yt, ct, et = extract(test_ids)
    return Xd, Td, Yd, cd, ed, Xt, Tt, Yt, ct, et
