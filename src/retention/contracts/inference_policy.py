"""Inference policy — input classification, OOD detection, abstention, fallback."""

from __future__ import annotations

import numpy as np

REJECT = "REJECT"
NORMALIZE = "NORMALIZE"
IMPUTE_AND_FLAG = "IMPUTE_AND_FLAG"
ACCEPT_AND_MONITOR = "ACCEPT_AND_MONITOR"

HARD_BOUNDS = {
    "company_size": (1, np.inf),
    "tenure_months": (0, np.inf),
    "monthly_revenue": (0, np.inf),
    "support_tickets_30d": (0, np.inf),
    "daily_usage": (0, np.inf),
    "email_engagement": (0, 1),
    "day": (-59, 0),
}

STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_DEGRADED = "DEGRADED"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_REJECT = "REJECT"


def classify_input(customer: dict) -> dict:
    """Classify inference input into REJECT/NORMALIZE/IMPUTE_AND_FLAG/ACCEPT_AND_MONITOR."""
    warnings = []
    status = STATUS_NORMAL

    # Hard domain bounds
    for field, (lo, hi) in HARD_BOUNDS.items():
        if field in customer:
            val = customer[field]
            if val < lo or val > hi:
                return {
                    "action": REJECT,
                    "status": STATUS_REJECT,
                    "reason": f"{field}_out_of_bounds",
                }

    # NaN/inf check
    for k, v in customer.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return {"action": REJECT, "status": STATUS_REJECT, "reason": f"{k}_invalid"}

    # Package normalization
    if "package" in customer:
        pkg = customer["package"].strip().lower()
        known = {"basic", "pro", "enterprise"}
        aliases = {"professional": "pro"}
        if pkg in known:
            pass  # valid
        elif pkg in aliases:
            customer["package"] = aliases[pkg]
            status = STATUS_WARNING
            warnings.append("PACKAGE_NORMALIZED")
        else:
            status = STATUS_WARNING
            warnings.append("UNKNOWN_PACKAGE")

    # Usage completeness
    if "usage_days" in customer:
        days = customer["usage_days"]
        if days < 45:
            status = STATUS_ABSTAIN
            warnings.append("INSUFFICIENT_USAGE")
        elif days < 55:
            status = STATUS_DEGRADED
            warnings.append("PARTIAL_USAGE")

    return {"action": ACCEPT_AND_MONITOR, "status": status, "warnings": warnings}


def should_abstain(
    uncertainty: float,
    uplift: float,
    max_uncertainty_ratio: float = 0.5,
    propensity: float | None = None,
) -> bool:
    if abs(uplift) < 1e-6:
        return True
    if uncertainty / abs(uplift) > max_uncertainty_ratio:
        return True
    if propensity is not None and (propensity < 0.02 or propensity > 0.98):
        return True
    return False
