"""Ranking invariants — deterministic, unique, exactly k customers, correct ordering."""

import numpy as np
from retention.policy.tie_breaking import deterministic_top_k


def test_exactly_k():
    scores = np.random.default_rng(42).random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 50)
    assert len(idx) == 50


def test_unique():
    scores = np.random.default_rng(42).random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 50)
    assert len(set(idx)) == 50


def test_descending_scores():
    rng = np.random.default_rng(42)
    scores = rng.random(100)
    ids = np.arange(100)
    idx = deterministic_top_k(scores, ids, 10)
    assert scores[idx[0]] >= scores[idx[9]]


def test_deterministic():
    scores = np.array([0.5, 0.5, 0.3, 0.3])
    ids = np.array([42, 7, 15, 99])
    a = deterministic_top_k(scores, ids, 2)
    b = deterministic_top_k(scores, ids, 2)
    np.testing.assert_array_equal(a, b)


def test_empty():
    idx = deterministic_top_k(np.array([]), np.array([]), 5)
    assert len(idx) == 0
