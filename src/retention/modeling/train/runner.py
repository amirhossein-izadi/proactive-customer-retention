"""Training orchestrator — load config, run challengers, select champion, save artifacts.

Supports use_optuna=True to auto-merge best params from artifacts/optuna/all_best.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from retention.utils.paths import ARTIFACTS, BASE, CHAMPION_DIR, OOF_DIR, REPORTS

mlflow.set_tracking_uri(f"sqlite:///{BASE / 'mlflow.db'}")

from retention.contracts.prediction import ChallengerPreds
from retention.utils.seeds import generate_seeds
from retention.evaluation.predictive import compute_metrics as factual_metrics
from retention.evaluation.policy_value import dr_policy_value as dr50_fn
from retention.features.builder import build_preprocessor
from retention.modeling.baselines import (
    historical_policy_value,
    random_policy_value,
    risk_ranking_policy_value,
)
from retention.modeling.ensembling import blend, tune_arm_weights
from retention.modeling.train.base import load_dev_test
from retention.modeling.train.config import load_challengers
from retention.modeling.train.dr_learner import train_dr_learner
from retention.modeling.train.s_int import train_s_learner_int
from retention.modeling.train.s_learner import train_s_learner
from retention.modeling.train.t_learner import train_t_learner
from retention.modeling.train.x_learner import train_x_learner
from retention.settings.config import get_settings
from retention.utils.serialization import save_json, save_pickle

SEED = generate_seeds(42)
OOF_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
_GIT_COMMIT = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
).stdout.strip() or "unknown"

_OPTUNA_KEY_MAP = {
    "logistic_t": "logistic_t_learner", "catboost_t": "catboost_t_learner",
    "lightgbm_t": "lightgbm_t_learner", "xgboost_t": "xgboost_t_learner",
    "logistic_s": "logistic_s_learner", "logistic_s_int": "logistic_s_learner",
    "catboost_s": "catboost_s_learner",
    "logistic_dr": "logistic_t_learner", "catboost_dr": "catboost_t_learner",
    "logistic_x": "logistic_t_learner", "catboost_x": "catboost_t_learner",
}


def _mlflow_params(params: dict) -> dict[str, str]:
    return {k: str(v) for k, v in params.items() if isinstance(v, (int, float, str, bool))}


def _mlflow_metrics(row: dict) -> dict[str, float]:
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and v is not None}


def train_experiment(use_optuna: bool = False) -> list[dict]:
    print("Loading split data...")
    Xd, Td, Yd, cids_d, extras_d, Xt, Tt, Yt, cids_t, extras_t = load_dev_test()
    print(f"Dev: {len(Xd)} | Test: {len(Xt)}")

    prep_static = build_preprocessor(include_pca=True, scale=True)
    prep_static.fit(Xd)
    save_pickle(prep_static, ARTIFACTS / "preprocessor_eval.pkl")

    mlflow.set_experiment("didar-retention")
    with mlflow.start_run(run_name="full-pipeline", nested=False):
        mlflow.set_tag("git_commit", _GIT_COMMIT)
        mlflow.set_tag("feature_version", "2.2.0")
        mlflow.set_tag("data_version", "didar_retention_v1")
        mlflow.set_tag("n_dev", str(len(Xd)))
        mlflow.set_tag("n_test", str(len(Xt)))

        loaded = load_challengers()
        if use_optuna:
            loaded = _merge_optuna(loaded)
        results, all_preds = [], {}

        for name, cls, params, seed_key, ltype in tqdm(loaded, desc="Models"):
            tqdm.write(f"\n--- {name} ({ltype}) ---")
            sys.stdout.flush()
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(_mlflow_params({"model_name": name, "learner_type": ltype}))
                if params:
                    mlflow.log_params(_mlflow_params(params))

                if ltype == "s_learner":
                    row, preds = train_s_learner(name, cls, params, seed_key, Xd, Td, Yd, Xt, Tt, Yt, SEED, cids_t, extras_t["monthly_revenue"])
                elif ltype == "si_learner":
                    row, preds = train_s_learner_int(name, cls, params, seed_key, Xd, Td, Yd, Xt, Tt, Yt, SEED, cids_t, extras_t["monthly_revenue"])
                elif ltype == "dr_learner":
                    row, preds = train_dr_learner(name, cls, params, seed_key, Xd, Td, Yd, Xt, Tt, Yt, SEED, cids_t, extras_t["monthly_revenue"])
                elif ltype == "x_learner":
                    row, preds = train_x_learner(name, cls, params, seed_key, Xd, Td, Yd, Xt, Tt, Yt, SEED, cids_t, extras_t["monthly_revenue"])
                else:
                    row, preds = train_t_learner(name, cls, params, seed_key, Xd, Td, Yd, Xt, Tt, Yt, SEED, cids_t, extras_t["monthly_revenue"])

                mlflow.log_metrics(_mlflow_metrics(row))
                mlflow.set_tag("git_commit", _GIT_COMMIT)

            tqdm.write(f"  Dev (OOF CV)   AUC={row['dev_auc']:.4f}  Brier={row['dev_brier']:.4f}  DR@50={row['dev_dr50']:.4f}")
            tqdm.write(f"  Test (holdout)  AUC={row['test_auc']:.4f}  Brier={row['test_brier']:.4f}  DR@50={row['test_dr50']:.4f}  Arm0 AUC={row['test_arm0_auc']:.4f}  Arm1 AUC={row['test_arm1_auc']:.4f}  U50={row['test_uplift50']:.4f}")
            results.append(row)
            all_preds[name] = preds

        CAL_THRESH = 0.20
        calibrated = [r for r in results if r.get("dev_brier", 1.0) < CAL_THRESH]
        valid = calibrated if calibrated else results
        if not calibrated:
            print(f"\nWARN: no model passed Brier<{CAL_THRESH}")
        champion = max(valid, key=lambda r: r["dev_dr50"])
        print(f"\nChampion: {champion['model']} (Dev DR@50={champion['dev_dr50']:.4f})")
        print(f"  Test DR@50={champion['test_dr50']:.4f}  AUC={champion['test_auc']:.4f}")
        mlflow.set_tag("champion", champion["model"])
        mlflow.log_metric("champion_dev_dr50", champion["dev_dr50"])
        mlflow.log_metric("champion_test_dr50", champion["test_dr50"])

        # Baselines
        cp: ChallengerPreds = all_preds[champion["model"]]
        print("\n--- Policy Baselines (on test set) ---")
        cap = get_settings().top_k
        rand_val = random_policy_value(cp["p0_t"], cp["p1_t"], cp["e_t"], Tt, Yt)
        risk_val = risk_ranking_policy_value(cp["p0_t"], cp["p1_t"], cp["e_t"], Tt, Yt, k=cap)
        hist_val = historical_policy_value(cp["p0_t"], cp["p1_t"], cp["e_t"], Tt, Yt, k=cap)
        print(f"  Random: {rand_val:.4f}  Risk: {risk_val:.4f}  Historical: {hist_val:.4f}")
        for label, val in [("baseline_random", rand_val), ("baseline_risk", risk_val), ("baseline_historical", hist_val)]:
            mlflow.log_metric(label, val)
        baseline_rows = [
            {"model": "baseline_random", "test_dr50": round(rand_val, 4)},
            {"model": "baseline_risk", "test_dr50": round(risk_val, 4)},
            {"model": "baseline_historical", "test_dr50": round(hist_val, 4)},
        ]

        comparison_path = REPORTS / "model_comparison.csv"
        all_rows = results + baseline_rows
        pd.DataFrame(all_rows).to_csv(comparison_path, index=False)
        save_json(all_rows, ARTIFACTS / "metrics" / "predictive_metrics.json")
        mlflow.log_artifact(str(comparison_path))

        champion_path = OOF_DIR / "champion_predictions.npz"
        test_keys = ["p0_t", "p1_t", "e_t", "uplift_t", "se_t", "cids_t", "rev_t"]
        np.savez(champion_path, **{k: cp[k] for k in test_keys},
                 tenure_months=extras_t.get("tenure_months", np.ones(len(cids_t)) * 12),
                 Y=Yt, T=Tt)
        mlflow.log_artifact(str(champion_path))
        print(f"Champion test predictions saved to {champion_path}")

        save_json({"python": sys.version, "sklearn": __import__("sklearn").__version__,
                    "numpy": __import__("numpy").__version__, "catboost": __import__("catboost").__version__,
                    "lightgbm": __import__("lightgbm").__version__, "xgboost": __import__("xgboost").__version__,
                    "cpus": os.cpu_count(), "champion": champion["model"]},
                   ARTIFACTS / "metrics" / "run_environment.json")
        mlflow.log_artifact(str(ARTIFACTS / "metrics" / "run_environment.json"))

        # Blending
        blend_name = "blend_t"
        if blend_name not in all_preds and "logistic_t" in all_preds and "catboost_t" in all_preds:
            try:
                prep = build_preprocessor(include_pca=True, scale=True)
                Xd_all, Xt_all = prep.fit_transform(Xd), prep.transform(Xt)
                lp, cbp = all_preds["logistic_t"], all_preds["catboost_t"]
                bw = tune_arm_weights(lp["p0_d"], cbp["p0_d"], lp["p1_d"], cbp["p1_d"], Yd, Td)
                p0_b = blend(lp["p0_t"], cbp["p0_t"], bw["alpha0"])
                p1_b = blend(lp["p1_t"], cbp["p1_t"], bw["alpha1"])
                e_b = np.clip(LogisticRegression(max_iter=1000, random_state=42).fit(Xd_all, Td).predict_proba(Xt_all)[:, 1], 0.02, 0.98)
                fm_b = factual_metrics(Yt, p0_b, p1_b, Tt)
                dr50_b = dr50_fn(p0_b - p1_b, p0_b, p1_b, e_b, Tt, Yt, k=50)
                tqdm.write(f"\n--- {blend_name} (a0={bw['alpha0']:.2f}, a1={bw['alpha1']:.2f}) ---")
                tqdm.write(f"  Test AUC={fm_b['factual_auc']:.4f}  Brier={fm_b['factual_brier']:.4f}  DR50={dr50_b:.4f}")
                # Append blend to comparison CSV
                blend_row = {"model": blend_name, "test_auc": round(fm_b['factual_auc'], 4),
                             "test_brier": round(fm_b['factual_brier'], 4), "test_dr50": round(dr50_b, 4)}
                all_rows.append(blend_row)
                pd.DataFrame(all_rows).to_csv(comparison_path, index=False)
                save_json(all_rows, ARTIFACTS / "metrics" / "predictive_metrics.json")
            except Exception as exc:
                tqdm.write(f"  Blend skipped: {exc}")

    print("Training complete.")
    return results


def _merge_optuna(challengers: list[tuple]) -> list[tuple]:
    optuna_path = ARTIFACTS / "optuna" / "all_best.json"
    if not optuna_path.exists():
        print("[optuna] no all_best.json — using config defaults")
        return challengers
    with open(optuna_path) as f:
        optuna_results = json.load(f)
    merged = []
    for name, cls, params, seed_key, ltype in challengers:
        key = _OPTUNA_KEY_MAP.get(name, name)
        if key in optuna_results:
            best = optuna_results[key]
            override = {k: v for k, v in best.items() if k not in ("best_value", "architecture", "model", "n_trials")}
            params = {**params, **override}
            tqdm.write(f"  [optuna] {name}: merged {len(override)} params")
        merged.append((name, cls, params, seed_key, ltype))
    return merged
