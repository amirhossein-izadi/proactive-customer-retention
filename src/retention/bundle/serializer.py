"""Bundle builder — trains champion on all dev data, calibrates, saves manifests.

Also computes global feature importance (SHAP TreeExplainer) and saves it
for downstream use in rankings and the evaluation report.
"""

from __future__ import annotations

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
from retention.data.joining import join_customers_usage, feature_frame
from retention.data.loading import load_all
from retention.features.builder import build_preprocessor
from retention.features.feature_names import all_feature_names
from retention.features.registry import FEATURE_VERSION
from retention.settings.config import get_settings
from retention.utils.paths import CHAMPION_DIR, SPLITS
from retention.utils.serialization import save_pickle, save_json

CHAMPION_PARAMS = dict(
    iterations=500,
    depth=4,
    learning_rate=0.022,
    l2_leaf_reg=25,
    random_strength=4.2,
    subsample=0.64,
    verbose=False,
    allow_writing_files=False,
)


def build_champion_bundle() -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    dev_ids = pd.read_csv(SPLITS / "development_set.csv")["customer_id"]
    customers, usage = load_all()
    merged = join_customers_usage(customers, usage, dev_ids)
    X_raw = feature_frame(merged)
    T = merged["contacted"].values.astype(int)
    Y = merged["churned"].values.astype(int)

    prep = build_preprocessor(include_pca=True, scale=True)
    X_all = prep.fit_transform(X_raw)
    save_pickle(prep, CHAMPION_DIR / "preprocessor.pkl")

    def _cb(**kw):
        return CatBoostClassifier(**{**CHAMPION_PARAMS, **kw})

    m0 = _cb(random_seed=42).fit(X_all[T == 0], Y[T == 0])
    m1 = _cb(random_seed=43).fit(X_all[T == 1], Y[T == 1])
    save_pickle(m0, CHAMPION_DIR / "m0.pkl")
    save_pickle(m1, CHAMPION_DIR / "m1.pkl")

    cal0 = CalibratedClassifierCV(_cb(random_seed=44), method="sigmoid", cv=3).fit(
        X_all[T == 0], Y[T == 0]
    )
    cal1 = CalibratedClassifierCV(_cb(random_seed=45), method="sigmoid", cv=3).fit(
        X_all[T == 1], Y[T == 1]
    )
    save_pickle(cal0, CHAMPION_DIR / "calibrator_0.pkl")
    save_pickle(cal1, CHAMPION_DIR / "calibrator_1.pkl")

    prop = LogisticRegression(max_iter=1000, random_state=42).fit(X_all, T)
    save_pickle(prop, CHAMPION_DIR / "propensity.pkl")

    # ── SHAP global + uplift explanations ──
    if _HAS_SHAP:
        try:
            fnames = all_feature_names(include_pca=True)
            sample = X_all[: min(len(X_all), 1000)]
            for arm_label, arm_model in [("arm0", m0), ("arm1", m1)]:
                explainer = shap.TreeExplainer(arm_model)
                sv = explainer.shap_values(sample)
                if isinstance(sv, list):
                    sv = sv[1]
                imp = np.abs(sv).mean(0)
                imp_dict = {fnames[i]: round(float(imp[i]), 6) for i in range(len(fnames))}
                save_json(imp_dict, CHAMPION_DIR / f"global_importance_{arm_label}.json")
                print(f"  SHAP importance ({arm_label}): {len(fnames)} features")

            exp = uplift_explanations(m0, m1, sample, fnames)
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
        model_family="CatBoost T-learner",
        model_version=settings.model_version,
        feature_version=FEATURE_VERSION,
        n_features=n_features,
        n_dev_samples=len(X_all),
        extra_metadata={
            "champion_params": CHAMPION_PARAMS,
            "propensity": "logistic, no class_weight",
        },
    )

    print(f"Champion bundle built at {CHAMPION_DIR}")
    for p in sorted(CHAMPION_DIR.iterdir()):
        sz = p.stat().st_size
        print(f"  {p.name}  ({sz:,} bytes)")


if __name__ == "__main__":
    build_champion_bundle()
