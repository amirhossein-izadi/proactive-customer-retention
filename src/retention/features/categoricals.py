"""Deterministic categorical normalization — whitespace/case/alias mapping.

Rules come from ``configs/features.yaml`` so normalization is versioned
configuration, not hidden code.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_category_value(
    value: Any, known: list[str], aliases: dict[str, str] | None = None
) -> tuple[str, bool]:
    """Return (normalized_value, was_normalized). Unknown values pass through."""
    if not isinstance(value, str):
        return value, False
    v = value.strip()
    aliases = aliases or {}
    lowered = v.lower()
    for k in known:
        if k.lower() == lowered:
            return k, k != v
    if lowered in {a.lower(): a for a in aliases}:
        alias_map = {a.lower(): target for a, target in aliases.items()}
        return alias_map[lowered], True
    return v, False


def normalize_customers(df: pd.DataFrame, category_config: dict | None = None) -> pd.DataFrame:
    """Normalize categorical columns of the customer table in a copy."""
    out = df.copy()
    category_config = category_config or {}
    for col in ("package", "subscription_type"):
        if col not in out.columns:
            continue
        cfg = category_config.get(col, {})
        known = cfg.get("known", sorted(out[col].dropna().unique().tolist()))
        aliases = cfg.get("aliases", {})
        out[col] = out[col].map(lambda v: normalize_category_value(v, known, aliases)[0])
    return out
