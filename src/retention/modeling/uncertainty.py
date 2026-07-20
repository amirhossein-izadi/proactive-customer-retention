"""Uncertainty estimation — bootstrap for top-k stability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_uplift(
    uplift_scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(uplift_scores)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = uplift_scores[idx].mean()
    return means.mean(), means.std()


def top_k_stability(
    scores_per_fold: list[np.ndarray],
    customer_ids: np.ndarray,
    k: int = 50,
) -> dict:
    n_folds = len(scores_per_fold)
    top_sets = []
    for scores in scores_per_fold:
        top_idx = np.argsort(scores)[::-1][:k]
        top_sets.append(set(customer_ids[top_idx]))
    jaccards = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            intersection = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccards.append(intersection / union if union > 0 else 0)
    mean_jaccard = np.mean(jaccards) if jaccards else 0.0
    selection_freq = pd.Series({cid: sum(cid in s for s in top_sets) for cid in customer_ids})
    return {
        "mean_jaccard@50": round(mean_jaccard, 3),
        "std_jaccard@50": round(np.std(jaccards), 3) if jaccards else 0.0,
        "n_folds": n_folds,
        "selection_frequencies": selection_freq.sort_values(ascending=False).head(k).to_dict(),
    }
