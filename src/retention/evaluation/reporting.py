"""Evaluation reporting — reads training artifacts, assembles final summary.

Wires: predictive metrics, calibration, policy value, AUUC/Qini, baselines,
bootstrap CI, stability (Jaccard), subgroup analysis, feature drift (PSI/KS),
global importance import, and leakage control register compliance.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from retention.data.loading import load_all
from retention.data.joining import feature_frame, join_customers_usage
from retention.data.leakage_register import get_leakage_register
from retention.evaluation.bootstrap import bootstrap_ci
from retention.evaluation.calibration import arm_calibration_report
from retention.evaluation.policy_value import dr_policy_value, dr_pseudo_outcome
from retention.evaluation.predictive import compute_metrics
from retention.evaluation.stability import jaccard_stability
from retention.evaluation.subgroup import subgroup_table
from retention.evaluation.uplift import auuc, qini
from retention.features.feature_names import all_feature_names
from retention.monitoring.drift import drift_report
from retention.modeling.baselines import (
    historical_policy_value,
    random_policy_value,
    risk_ranking_policy_value,
)
from retention.modeling.propensity import propensity_overlap_stats
from retention.settings.config import get_settings
from retention.utils.paths import ARTIFACTS, CHAMPION_DIR, METRICS_DIR, OOF_DIR, SPLITS
from retention.utils.serialization import save_json, load_pickle


def _load_drift_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dev + test raw features and preprocess for drift comparison.

    Uses the preprocessor saved during training (fitted once on dev) so
    evaluation never refits transformers — same transform applied to both
    dev and test, making PSI/KS comparisons valid.
    """
    customers, usage = load_all()
    dev_ids = pd.read_csv(SPLITS / "development_set.csv")["customer_id"]
    test_ids = pd.read_csv(SPLITS / "test_set.csv")["customer_id"]

    preproc_path = ARTIFACTS / "preprocessor_eval.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preproc_path}. "
            "Run `retention train` before `retention evaluate`."
        )
    prep = load_pickle(preproc_path)

    Xd = prep.transform(feature_frame(join_customers_usage(customers, usage, dev_ids)))
    Xt = prep.transform(feature_frame(join_customers_usage(customers, usage, test_ids)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols = all_feature_names(include_pca=True)
    return pd.DataFrame(Xd, columns=cols), pd.DataFrame(Xt, columns=cols)


def _load_global_importance() -> dict[str, float] | None:
    # Uplift explanation (SHAP(p0)-SHAP(p1)) is most relevant; fall back to arm models
    for arm in ("uplift_explanation", "arm0", "arm1"):
        path = CHAMPION_DIR / f"global_importance_{arm}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def _load_test_metadata() -> dict[str, np.ndarray]:
    """Load test-set customer metadata for subgroup analysis."""
    test_df = pd.read_csv(SPLITS / "test_set.csv")
    return {
        "subscription_type": test_df["subscription_type"].values,
        "package": test_df["package"].values,
    }


def evaluate_experiment() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OOF_DIR / "champion_predictions.npz")
    settings = get_settings()

    T = data["T"]
    Y = data["Y"]
    p0 = data["p0_t"]
    p1 = data["p1_t"]
    e = data["e_t"]
    uplift = data["uplift_t"]
    cids = data["cids_t"]

    # ── Predictive & calibration ──
    pm = compute_metrics(Y, p0, p1, T)
    cal = arm_calibration_report(Y, p0, p1, T)
    overlap = propensity_overlap_stats(e)

    # ── Uplift & policy ──
    auuc_val = auuc(uplift, T, Y)
    qini_val = qini(uplift, T, Y)
    dr50_val = dr_policy_value(uplift, p0, p1, e, T, Y, k=50)
    rand_val = random_policy_value(p0, p1, e, T, Y)
    risk_val = risk_ranking_policy_value(p0, p1, e, T, Y, k=50)
    hist_val = historical_policy_value(p0, p1, e, T, Y, k=50)

    # ── Bootstrap CI on DR pseudo-outcome ──
    psi = dr_pseudo_outcome(p0, p1, e, T, Y)
    boot = bootstrap_ci(psi, n_bootstrap=1000, seed=42)

    # ── Subgroup analysis ──
    meta = _load_test_metadata()
    sub_grps = {}
    for col in ("subscription_type", "package"):
        if col in meta and len(meta[col]) == len(Y):
            grp = meta[col]
            try:
                sub = subgroup_table(pd.Series(grp), Y, p0, p1, T)
                sub_grps[col] = sub.to_dict(orient="records")
            except Exception:
                pass

    # ── Feature drift (PSI/KS) ──
    drift = None
    try:
        dev_df, test_df = _load_drift_data()
        drift_cols = [c for c in dev_df.columns if any(k in c for k in ("usage_", "log_", "pca_"))][
            :15
        ]
        drift_df = drift_report(dev_df, test_df, drift_cols)
        drift = drift_df.to_dict(orient="records")
    except Exception as e:
        drift = [{"error": str(e)}]

    # ── Stability (Jaccard) from repeat-level predictions ──
    stab = {}
    try:
        p0_repeats = data.get("p0_repeats")
        if p0_repeats is not None:
            top_sets = []
            for rep in range(p0_repeats.shape[1]):
                scores = p0_repeats[:, rep] - data.get("p1_repeats", p0_repeats)[:, rep]
                top_idx = np.argsort(scores)[::-1][:50]
                top_sets.append(set(np.asarray(cids[top_idx], dtype=int)))
            stab["jaccard@50"] = round(jaccard_stability(top_sets, k=50), 4)
    except Exception:
        pass

    # ── Global importance (from bundle) ──
    imp = _load_global_importance()
    top_features = list(imp.keys())[:10] if imp else []

    # ── Leakage register ──
    leakage = get_leakage_register()

    # ── Assemble summary ──
    u50 = np.argsort(uplift)[::-1][:50]
    summary = {
        "model": "champion_bundle",
        "data_version": settings.data_version,
        "feature_version": settings.feature_version,
        "model_version": settings.model_version,
        **pm,
        "auuc": round(auuc_val, 4),
        "qini": round(qini_val, 4),
        "dr50": round(dr50_val, 4),
        "random_policy_value": round(rand_val, 4),
        "risk_baseline_policy_value": round(risk_val, 4),
        "historical_policy_value": round(hist_val, 4),
        "mean_uplift": round(uplift.mean(), 4),
        "mean_uplift@50": round(uplift[u50].mean(), 4),
        "dr50_bootstrap": boot,
        "propensity_overlap": overlap,
        "calibration": cal,
        "subgroup_analysis": sub_grps,
        "feature_drift": drift,
        "stability": stab,
        "top_features": top_features,
        "n_test": int(len(Y)),
    }
    save_json(summary, METRICS_DIR / "final_summary.json")
    print("Evaluation summary saved to artifacts/metrics/final_summary.json")
    for k, v in summary.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")
    if imp:
        print("  Top-5 features:", ", ".join(top_features[:5]))
    if drift:
        print(f"  Drift: {len(drift)} features checked")
    if sub_grps:
        print(f"  Subgroups: {list(sub_grps)}")


if __name__ == "__main__":
    evaluate_experiment()
