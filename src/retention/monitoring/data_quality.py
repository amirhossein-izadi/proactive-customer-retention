"""Data quality report — wired into `retention validate` and DVC validate stage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from retention.data.validation import check_join_integrity, check_usage_completeness
from retention.monitoring.metrics import DATA_QUALITY_MISSING_RATE


def run_data_quality(customers: pd.DataFrame, usage: pd.DataFrame) -> dict[str, Any]:
    """Structured data-quality summary persisted as a pipeline artifact."""
    join = check_join_integrity(customers, usage)
    completeness = check_usage_completeness(usage)
    missing = {c: int(customers[c].isna().sum()) for c in customers.columns}
    n = max(len(customers), 1)
    for col, cnt in missing.items():
        try:
            DATA_QUALITY_MISSING_RATE.labels(column=col).set(cnt / n)
        except Exception:
            pass
    return {
        "n_customers": int(len(customers)),
        "n_usage_rows": int(len(usage)),
        "treatment_rate": round(float(customers["contacted"].mean()), 4),
        "churn_rate": round(float(customers["churned"].mean()), 4),
        "churn_by_arm": {
            "control": round(
                float(customers.loc[customers["contacted"] == 0, "churned"].mean()), 4
            ),
            "treated": round(
                float(customers.loc[customers["contacted"] == 1, "churned"].mean()), 4
            ),
        },
        "missing_values": missing,
        "join_integrity": join,
        "usage_completeness": completeness,
    }
