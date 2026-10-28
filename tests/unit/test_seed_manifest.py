"""Seed manifest — deterministic derivation, reproducibility, distinct sub-seeds."""

from retention.data.splitting import generate_seed_manifest


def test_generates_six_seeds():
    sm = generate_seed_manifest(42)
    keys = {"master_seed", "holdout_seed", "outer_cv_seed", "inner_cv_seed", "model_seed", "bootstrap_seed"}
    assert keys.issubset(sm.keys()), f"Missing keys: {keys - sm.keys()}"


def test_reproducible_same_master():
    a = generate_seed_manifest(42)
    b = generate_seed_manifest(42)
    assert a == b


def test_different_master_different():
    a = generate_seed_manifest(42)
    b = generate_seed_manifest(99)
    assert a != b


def test_all_seeds_in_uint31_range():
    sm = generate_seed_manifest(42)
    for k, v in sm.items():
        if k == "master_seed":
            continue
        assert 0 <= v < 2**31, f"{k}={v} out of range"
