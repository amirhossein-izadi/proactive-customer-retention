"""Bundle builder — trains champion outcome models + calibrated serving models.

Architecture (estimator family follows the trained champion via
``champion_spec.json`` — legacy CatBoost fallback when no spec exists):
  - m0, m1:  outcome models (one per treatment arm) — used for SHAP explanations
  - cal0, cal1: CalibratedClassifierCV wrapping the champion with internal
                3-fold CV for Platt scaling — used for serving predictions
  - Serving pipeline: preprocessor → CalibratedClassifierCV → calibrated probabilities
  - The CalibratedClassifierCV trains its own models internally with train/cal split,
    ensuring proper calibration on held-out predictions within each arm.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from retention.bundle.metadata import write_champion_manifests

try:
    import shap

    from retention.explainability.uplift_explanations import uplift_explanations
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
from retention.data.joining import feature_frame, join_customers_usage
from retention.data.loading import load_all
from retention.features.builder import build_preprocessor
from retention.features.feature_names import all_feature_names
from retention.features.registry import FEATURE_VERSION
from retention.modeling.train.shared import model_kw
from retention.settings.config import get_settings
from retention.utils.paths import CHAMPION_DIR, OOF_DIR, SPLITS
from retention.utils.serialization import save_json, save_pickle

CHAMPION_PARAMS = {
    "iterations": 500, "depth": 4, "learning_rate": 0.022,
    "l2_leaf_reg": 25, "random_strength": 4.2, "subsample": 0.64,
    "verbose": False, "allow_writing_files": False,
}

_TREE_FAMILIES = ("catboost", "lightgbm", "xgboost")
_FAMILY_PRETTY = {
    "logistic": "Logistic", "catboost": "CatBoost",
    "lightgbm": "LightGBM", "xgboost": "XGBoost",
}
_ARCH_PRETTY = {
    "t_learner": "T-learner", "s_learner": "S-learner",
    "si_learner": "S-learner+interactions",
    "dr_learner": "DR-learner", "x_learner": "X-learner",
}


def _load_champion_spec() -> dict | None:
    """Champion spec written by the training runner; None → legacy fallback."""
    spec_path = OOF_DIR / "champion_spec.json"
    if not spec_path.exists():
        return None
    return json.loads(spec_path.read_text())


def _champion_estimator(spec: dict | None, seed_value: int):
    """Instantiate the champion estimator family with training-time params."""
    if spec is None:
        return _catboost(random_seed=seed_value)
    module = importlib.import_module(spec["cls_module"])
    cls = getattr(module, spec["cls_name"])
    return cls(**model_kw(spec["params"], spec.get("seed_key"), seed_value))


def _champion_family(spec: dict | None) -> tuple[str, str]:
    """(pretty family label, SHAP explainer kind) for the champion."""
    if spec is None:
        return "CatBoost T-learner", "tree"
    name = spec.get("name", "")
    family = name.split("_")[0] if "_" in name else name
    pretty = _FAMILY_PRETTY.get(family, family.replace("_", " ").title())
    arch = _ARCH_PRETTY.get(spec.get("ltype", ""), spec.get("ltype", ""))
    kind = "tree" if family in _TREE_FAMILIES else "linear"
    return f"{pretty} {arch}", kind


def _catboost(**kw) -> CatBoostClassifier:
    return CatBoostClassifier(**{**CHAMPION_PARAMS, **kw})


def build_champion_bundle() -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    spec = _load_champion_spec()
    if spec is None:
        print("  No champion_spec.json — legacy CatBoost fallback")
    family_label, explainer_kind = _champion_family(spec)
    print(f"  Champion family: {family_label}")

    dev_ids = pd.read_csv(SPLITS / "development_set.csv")["customer_id"]
    customers, usage = load_all()
    merged = join_customers_usage(customers, usage, dev_ids)
    X_raw = feature_frame(merged)
    T = merged["contacted"].values.astype(int)
    Y = merged["churned"].values.astype(int)

    prep = build_preprocessor(include_pca=True, scale=True)
    X_all = prep.fit_transform(X_raw)
    save_pickle(prep, CHAMPION_DIR / "preprocessor.pkl")

    # ── Outcome models (for SHAP / analysis) ──
    m0 = _champion_estimator(spec, 42).fit(X_all[T == 0], Y[T == 0])
    m1 = _champion_estimator(spec, 43).fit(X_all[T == 1], Y[T == 1])
    save_pickle(m0, CHAMPION_DIR / "m0.pkl")
    save_pickle(m1, CHAMPION_DIR / "m1.pkl")

    # ── Calibrated serving models ──
    # CalibratedClassifierCV does internal 3-fold CV: trains the champion on 2/3,
    # fits Platt scaler on held-out 1/3 per fold, averages 3 calibrated models.
    cal0 = CalibratedClassifierCV(_champion_estimator(spec, 44), method="sigmoid", cv=3).fit(
        X_all[T == 0], Y[T == 0]
    )
    cal1 = CalibratedClassifierCV(_champion_estimator(spec, 45), method="sigmoid", cv=3).fit(
        X_all[T == 1], Y[T == 1]
    )
    save_pickle(cal0, CHAMPION_DIR / "calibrator_0.pkl")
    save_pickle(cal1, CHAMPION_DIR / "calibrator_1.pkl")

    # ── Propensity ──
    prop = LogisticRegression(max_iter=1000, random_state=42).fit(X_all, T)
    save_pickle(prop, CHAMPION_DIR / "propensity.pkl")

    # ── SHAP explanations ──
    if _HAS_SHAP:
        try:
            fnames = all_feature_names(include_pca=True)
            sample = X_all[: min(len(X_all), 1000)]
            background = sample[: min(100, len(sample))]
            for arm_label, arm_model in [("arm0", m0), ("arm1", m1)]:
                if explainer_kind == "tree":
                    explainer = shap.TreeExplainer(arm_model)
                    sv = explainer.shap_values(sample)
                else:
                    explainer = shap.LinearExplainer(arm_model, background)
                    sv = explainer.shap_values(sample)
                if isinstance(sv, list):
                    sv = sv[1]
                imp = np.abs(sv).mean(0)
                imp_dict = {fnames[i]: round(float(imp[i]), 6) for i in range(len(fnames))}
                save_json(imp_dict, CHAMPION_DIR / f"global_importance_{arm_label}.json")
                print(f"  SHAP importance ({arm_label}): {len(fnames)} features")
            exp = uplift_explanations(m0, m1, sample)
            imp_diff = np.abs(exp).mean(0)
            diff_dict = {fnames[i]: round(float(imp_diff[i]), 6) for i in range(len(fnames))}
            save_json(diff_dict, CHAMPION_DIR / "uplift_explanation.json")
            print(f"  SHAP uplift explanation: {len(fnames)} features")
        except Exception as e:
            print(f"  SHAP explanations skipped ({e})")
    else:
        print("  SHAP not available — skipping explanations")

    n_features = X_all.shape[1]
    settings = get_settings()
    write_champion_manifests(
        CHAMPION_DIR,
        model_family=family_label,
        model_version=settings.model_version,
        feature_version=FEATURE_VERSION,
        n_features=n_features,
        n_dev_samples=len(X_all),
        extra_metadata={
            "champion_name": spec["name"] if spec else "catboost_t (legacy)",
            "champion_params": spec["params"] if spec else CHAMPION_PARAMS,
            "calibration": "CalibratedClassifierCV, Platt sigmoid, internal 3-fold CV",
        },
    )

    print(f"Champion bundle built at {CHAMPION_DIR}")
    for p in sorted(CHAMPION_DIR.iterdir()):
        sz = p.stat().st_size
        print(f"  {p.name}  ({sz:,} bytes)")


if __name__ == "__main__":
    build_champion_bundle()
