"""Deterministic tie-breaking for ranking selection.

Uses customer_id as the final tie-breaker so that repeated runs with
the same model produce exactly the same ranking — no randomness from
any source (e.g., equal scores).
"""

from __future__ import annotations

import numpy as np


def deterministic_top_k(
    scores: np.ndarray,
    ids: np.ndarray,
    k: int,
    higher_better: bool = True,
) -> np.ndarray:
    """Select top-k indices with deterministic tie-breaking.

    Parameters
    ----------
    scores : ndarray of shape (n,)
    ids : ndarray of shape (n,) — unique customer identifiers.
    k : int — number of selections.
    higher_better : bool — if True, sort descending.

    Returns
    -------
    indices : ndarray of shape (k,) — indices of selected customers.
    """
    if len(scores) == 0:
        return np.array([], dtype=int)
    if np.any(np.isnan(scores)):
        raise ValueError("scores contain NaN — cannot rank")
    direction = -1 if higher_better else 1
    combined = np.lexsort((ids, direction * scores))
    k = min(k, len(scores))
    return combined[:k]
