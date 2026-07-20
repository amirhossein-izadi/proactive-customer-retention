"""Subgroup analysis — model performance sliced by customer attribute."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def subgroup_table(
    group_labels: pd.Series,
    Y: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    T: np.ndarray,
) -> pd.DataFrame:
    """Per-group summary: size, churn rate (by arm), mean uplift, factual AUC."""
    rows = []
    for grp in group_labels.unique():
        mask = group_labels.values == grp
        if mask.sum() < 10:
            continue
        control = mask & (T == 0)
        treated = mask & (T == 1)
        factual = np.where(T[mask] == 1, p1[mask], p0[mask])
        auc = roc_auc_score(Y[mask], factual) if len(np.unique(Y[mask])) > 1 else -1
        rows.append(
            {
                "group": grp,
                "n": int(mask.sum()),
                "churn_control": round(float(Y[control].mean()), 4) if control.sum() > 5 else None,
                "churn_treated": round(float(Y[treated].mean()), 4) if treated.sum() > 5 else None,
                "mean_uplift": round(float((p0[mask] - p1[mask]).mean()), 4),
                "factual_auc": round(auc, 4) if auc >= 0 else None,
            }
        )
    return pd.DataFrame(rows)
