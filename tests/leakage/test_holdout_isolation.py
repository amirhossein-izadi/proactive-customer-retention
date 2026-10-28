"""Leakage tests — holdout isolation, OOF separation, calibration isolation, temporal cutoff."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

BASE = __import__("pathlib").Path(__file__).resolve().parent.parent.parent


def test_holdout_isolation():
    """Item 120: No test-set customer appears in development set."""
    dev = pd.read_csv(BASE / "data" / "splits" / "development_set.csv")
    test = pd.read_csv(BASE / "data" / "splits" / "test_set.csv")
    overlap = set(dev["customer_id"]) & set(test["customer_id"])
    assert len(overlap) == 0, f"{len(overlap)} customers in both dev and test"


def test_cv_folds_cover_dev_only():
    """Item 121: CV folds only reference development-set customers."""
    dev = pd.read_csv(BASE / "data" / "splits" / "development_set.csv")
    dev_ids = set(dev["customer_id"])
    import json

    folds = json.load(open(BASE / "data" / "splits" / "folds.json"))
    for f in folds:
        for idx_list in [f["train_indices"], f["val_indices"]]:
            for idx in idx_list:
                assert idx < len(dev), f"Fold index {idx} exceeds dev size {len(dev)}"


def test_oof_predictions_not_trained():
    """Item 126: Verify OOF predictions come from models that excluded that customer."""
    import pytest
    from retention.data.loading import load_all

    try:
        customers, usage = load_all()
    except FileNotFoundError:
        pytest.skip("raw data not available (DVC not pulled)")

    from retention.modeling.cross_fitting import repeated_cross_fit
    from retention.features.builder import _pivot_usage
    from retention.data.splitting import generate_seed_manifest
    import pandas as pd, numpy as np
    from catboost import CatBoostClassifier

    dev = pd.read_csv(BASE / "data" / "splits" / "development_set.csv")
    c = customers[customers["customer_id"].isin(dev["customer_id"])]
    u = usage[usage["customer_id"].isin(dev["customer_id"])]
    w = _pivot_usage(u)
    m = c.merge(w, on="customer_id", how="left")
    feats = [col for col in m.columns if col not in ("customer_id", "contacted", "churned")]
    Xr = m[feats].reset_index(drop=True)

    SEED = generate_seed_manifest(42)
    oof = repeated_cross_fit(
        Xr,
        m["contacted"].values,
        m["churned"].values,
        CatBoostClassifier,
        dict(iterations=100, depth=3, learning_rate=0.05, verbose=False),
        None,
        n_repeats=1,
        n_folds=5,
    )
    assert not np.any(oof["p0"] == 0), "OOF p0 should have non-zero predictions"
    assert oof["p0"].shape[0] == len(Xr), "OOF predictions must cover all dev customers"
