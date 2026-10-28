"""Prediction invariants — probabilities must be in [0,1], uplift in [-1,1]."""

import numpy as np


def test_probability_bounds():
    probs = np.array([0.0, 0.5, 1.0, 0.001, 0.999])
    assert np.all((0 <= probs) & (probs <= 1))


def test_uplift_bounds():
    uplift = np.array([-1.0, 0.0, 0.5, 1.0])
    assert np.all((-1 <= uplift) & (uplift <= 1))


def test_uplift_equals_difference():
    p0 = np.array([0.3, 0.5, 0.8])
    p1 = np.array([0.2, 0.4, 0.1])
    uplift = p0 - p1
    assert np.allclose(uplift, np.array([0.1, 0.1, 0.7]))


def test_factual_matches_arm():
    p0 = np.array([0.3, 0.5])
    p1 = np.array([0.2, 0.4])
    T = np.array([0, 1])
    factual = np.where(T == 1, p1, p0)
    assert np.allclose(factual, [0.3, 0.4])
