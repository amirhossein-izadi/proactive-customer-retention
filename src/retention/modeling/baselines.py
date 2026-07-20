"""Policy baselines — simple decision rules for comparison.

The baselines are:
- ``random_policy_value``: contact a random subset (ATE estimate).
- ``risk_ranking_policy_value``: contact the top-k by churn risk p(Y=1) only.
- ``historical_policy_value``: the actual policy used during the experiment.
"""

from __future__ import annotations

import numpy as np

from retention.evaluation.policy_value import dr_policy_value


def random_policy_value(
    p0: np.ndarray, p1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> float:
    return float(dr_policy_value(np.ones(len(Y)), p0, p1, e, T, Y))


def risk_ranking_policy_value(
    p0: np.ndarray, p1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray, k: int = 50
) -> float:
    """Value of contacting the k customers with the highest factual-risk score.

    For demonstration this uses ``p0`` as the risk score — the churn probability
    under no contact.
    """
    return float(dr_policy_value(p0, p0, p1, e, T, Y, k=k))


def historical_policy_value(
    p0: np.ndarray, p1: np.ndarray, e: np.ndarray, T: np.ndarray, Y: np.ndarray, k: int = 50
) -> float:
    """Value of the actual historical treatment assignment.

    For the historical policy: select all who were actually contacted, then
    within that set the first k by customer_id (deterministic tie-break).
    """
    n = len(Y)
    uplift = np.zeros(n)
    # customers actually contacted, sorted by customer_id
    treated_idx = np.where(T == 1)[0]
    treated_sorted = treated_idx[np.argsort(treated_idx)]
    selected = treated_sorted[:k]
    uplift[selected] = 1.0
    return float(dr_policy_value(uplift, p0, p1, e, T, Y, k))
