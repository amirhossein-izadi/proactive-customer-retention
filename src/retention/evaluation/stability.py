"""Stability metrics — fold-agreement, Jaccard overlap, per-customer variance."""

from __future__ import annotations

import numpy as np


def jaccard_stability(top_sets: list[set], k: int = 50) -> float:
    """Mean pairwise Jaccard overlap across repeats/folds."""
    n = len(top_sets)
    if n < 2:
        return 1.0
    jacs = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jacs.append(inter / union if union else 0.0)
    return float(np.mean(jacs))


def repeat_uplift_stats(uplift_repeats: np.ndarray) -> np.ndarray:
    """Per-customer mean and std over repeats (n_samples × n_repeats)."""
    return np.array([uplift_repeats.mean(axis=1), uplift_repeats.std(axis=1, ddof=1)]).T
