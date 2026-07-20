"""Policy ranking tests — deterministic, unique, exactly k, DR sign."""

import numpy as np
from retention.policy.tie_breaking import deterministic_top_k
from retention.evaluation.policy_value import dr_pseudo_outcome


def test_exactly_k():
    scores = np.random.default_rng(42).random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 50)
    assert len(idx) == 50


def test_unique():
    scores = np.random.default_rng(42).random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 50)
    assert len(set(ids[idx])) == 50


def test_top_scores():
    rng = np.random.default_rng(42)
    scores = rng.random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 10)
    assert scores[idx[0]] >= scores[idx[-1]]


def test_dr_positive_for_beneficial_treatment():
    rng = np.random.default_rng(42)
    n = 200
    T = rng.integers(0, 2, size=n)
    p0 = 0.4 + 0.0 * T
    p1 = 0.2 * np.ones(n)
    e = np.full(n, 0.5)
    Y = ((1 - T) * (rng.random(n) < p0) + T * (rng.random(n) < p1)).astype(int)
    psi = dr_pseudo_outcome(p0, p1, e, T, Y)
    assert psi.mean() > 0, "DR pseudo-outcome should be > 0 when contact reduces churn"
