"""Raw input normalization — column names and categorical values."""

from __future__ import annotations

import pandas as pd

from retention.features.categoricals import normalize_customers
from retention.settings.config import load_yaml
from retention.utils.paths import CONFIGS


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names (raw CSV uses e.g. `Subscription_type`)."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def normalize_customer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Column-name + categorical normalization per versioned config."""
    category_config = load_yaml(CONFIGS / "features.yaml").get("categories", {})
    return normalize_customers(normalize_column_names(df), category_config)
