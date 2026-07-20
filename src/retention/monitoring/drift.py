"""Distribution drift detection — PSI and KS, used for dev/test shift reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention.monitoring.metrics import FEATURE_PSI


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index. >0.1 moderate shift, >0.25 significant."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.unique(np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-4, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    expected = np.sort(np.asarray(expected, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    all_vals = np.concatenate([expected, actual])
    cdf_e = np.searchsorted(expected, all_vals, side="right") / max(len(expected), 1)
    cdf_a = np.searchsorted(actual, all_vals, side="right") / max(len(actual), 1)
    return float(np.max(np.abs(cdf_e - cdf_a)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Per-column PSI/KS between two snapshots (e.g., development vs test)."""
    rows = []
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().values
        cur = pd.to_numeric(current[col], errors="coerce").dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue
        psi_val = psi(ref, cur)
        try:
            FEATURE_PSI.labels(feature=col).set(psi_val)
        except Exception:
            pass
        rows.append(
            {
                "feature": col,
                "psi": round(psi_val, 4),
                "ks": round(ks_statistic(ref, cur), 4),
                "shift_flag": psi_val > 0.25,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
