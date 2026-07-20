"""Bootstrap confidence intervals for policy-value estimates."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a statistic over the sample."""
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        stats[i] = statistic(values[rng.integers(0, n, size=n)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(stats, [alpha * 100, (1 - alpha) * 100])
    return {
        "mean": float(stats.mean()),
        "std": float(stats.std(ddof=1)),
        f"CI_{int(ci * 100)}_lower": float(lo),
        f"CI_{int(ci * 100)}_upper": float(hi),
        "n_bootstrap": n_bootstrap,
    }
