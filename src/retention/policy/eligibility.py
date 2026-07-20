"""Eligibility rules — who can be scored and contacted."""

from __future__ import annotations

import pandas as pd


def eligible_customers(
    features: pd.DataFrame,
    customers: pd.DataFrame,
    min_usage_days: int = 45,
    min_tenure_months: int = 1,
    exclude_do_not_contact: bool = True,
) -> pd.Series:
    eligible = pd.Series(True, index=features.index)
    if "tenure_months" in features.columns:
        eligible &= features["tenure_months"] >= min_tenure_months
    if "active_days_pct" in features.columns:
        eligible &= features["active_days_pct"] >= (min_usage_days / 60)
    return eligible
