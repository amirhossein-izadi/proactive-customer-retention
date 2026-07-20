"""Feature ablation: static-only, temporal-only, no-PCA, full.

Run via ``retention ablate`` or ``python -m retention.evaluation.ablation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from retention.data.loading import load_all
from retention.features.builder import (
    PCAEmbedding,
    STATIC_INPUT,
    StaticEngineer,
    TemporalEngineer,
    _pivot_usage,
    usage_columns,
)
from retention.utils.paths import REPORTS, SPLITS
from retention.utils.seeds import generate_seeds

SEED = generate_seeds(42)
USAGE_COLS = usage_columns()
STATIC_FEATURE_COLS = STATIC_INPUT


def _prep_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dev = pd.read_csv(SPLITS / "development_set.csv")
    customers, usage = load_all()
    c = customers[customers["customer_id"].isin(dev["customer_id"])]
    u = usage[usage["customer_id"].isin(dev["customer_id"])]
    w = _pivot_usage(u)
    merged = c.merge(w, on="customer_id", how="left")
    feats = [col for col in merged.columns if col not in ("customer_id", "contacted", "churned")]
    return (
        merged[feats].reset_index(drop=True),
        merged["contacted"].values,
        merged["churned"].values,
    )


def _build_configs() -> dict[str, ColumnTransformer]:
    static = StaticEngineer()
    temporal = TemporalEngineer()
    pca = PCAEmbedding(n_components=5)
    return {
        "static_only": ColumnTransformer(
            [
                ("static", static, STATIC_FEATURE_COLS),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["package", "subscription_type"]),
            ]
        ),
        "temporal+cat": ColumnTransformer(
            [
                ("temporal", temporal, USAGE_COLS),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["package", "subscription_type"]),
            ]
        ),
        "static+temporal+cat": ColumnTransformer(
            [
                ("static", static, STATIC_FEATURE_COLS),
                ("temporal", temporal, USAGE_COLS),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["package", "subscription_type"]),
            ]
        ),
        "full": ColumnTransformer(
            [
                ("static", StaticEngineer(), STATIC_FEATURE_COLS),
                ("temporal", TemporalEngineer(), USAGE_COLS),
                ("pca", PCAEmbedding(n_components=5), USAGE_COLS),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["package", "subscription_type"]),
            ]
        ),
    }


def test_ablation() -> list[dict]:
    X_raw, T, Y = _prep_data()
    configs = _build_configs()
    stratum = np.array([f"{t}_{y}" for t, y in zip(T, Y)])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED["outer_cv_seed"])

    results = []
    for name, prep in configs.items():
        fold_scores = []
        for tr, va in skf.split(X_raw, stratum):
            X_tr = prep.fit_transform(X_raw.iloc[tr])
            X_va = prep.transform(X_raw.iloc[va])
            T_tr, Y_tr = T[tr], Y[tr]
            T_va, Y_va = T[va], Y[va]
            m0 = LogisticRegression(max_iter=1000).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            m1 = LogisticRegression(max_iter=1000).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            factual = np.where(
                T_va == 1, m1.predict_proba(X_va)[:, 1], m0.predict_proba(X_va)[:, 1]
            )
            fold_scores.append(
                {"auc": roc_auc_score(Y_va, factual), "brier": brier_score_loss(Y_va, factual)}
            )
        results.append(
            {
                "name": name,
                "auc": round(np.mean([s["auc"] for s in fold_scores]), 4),
                "brier": round(np.mean([s["brier"] for s in fold_scores]), 4),
            }
        )
        print(f"  {name:25s} AUC={results[-1]['auc']:.4f} Brier={results[-1]['brier']:.4f}")

    pd.DataFrame(results).to_csv(REPORTS / "ablation_results.csv", index=False)
    print(f"\nAblation saved to {REPORTS / 'ablation_results.csv'}")
    return results


if __name__ == "__main__":
    test_ablation()
