"""Cross-fitted data splitting with seed manifest and stratification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from retention.utils.seeds import generate_seeds, save_seeds


def create_stratum(df: pd.DataFrame) -> pd.Series:
    return df["contacted"].astype(str) + "_" + df["churned"].astype(str)


generate_seed_manifest = generate_seeds


def create_holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed_manifest: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seed_manifest is None:
        seed_manifest = generate_seed_manifest()
    stratum = create_stratum(df)
    dev_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=stratum,
        random_state=seed_manifest["holdout_seed"],
    )
    return df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()


def create_outer_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed_manifest: dict | None = None,
) -> list[dict]:
    if seed_manifest is None:
        seed_manifest = generate_seed_manifest()
    folds = []
    for repeat in range(n_repeats):
        seed = seed_manifest["outer_cv_seed"] + repeat
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        stratum = create_stratum(df)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratum)):
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                    "seed": seed,
                }
            )
    return folds


def save_split_manifest(
    df: pd.DataFrame,
    split_label: str,
    output_path: str | Path,
    seed_manifest: dict | None = None,
) -> None:
    if seed_manifest is None:
        seed_manifest = generate_seed_manifest()
    manifest = {
        "dataset": "didar_retention",
        "split": split_label,
        "n_customers": len(df),
        "seed_manifest": seed_manifest,
        "columns": list(df.columns),
        "stratum_counts": create_stratum(df).value_counts().to_dict(),
        "sha256": hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest(),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def build_split_pipeline(
    df: pd.DataFrame,
    output_dir: str | Path = "data/splits",
    test_size: float = 0.2,
    n_folds: int = 5,
    n_repeats: int = 3,
    master_seed: int = 42,
) -> None:
    seed_manifest = generate_seed_manifest(master_seed)
    dev_df, test_df = create_holdout_split(df, test_size, seed_manifest)
    folds = create_outer_folds(dev_df, n_folds, n_repeats, seed_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_split_manifest(dev_df, "development", out / "development_manifest.json", seed_manifest)
    save_split_manifest(test_df, "final_test", out / "test_manifest.json", seed_manifest)
    dev_df.to_csv(out / "development_set.csv", index=False)
    test_df.to_csv(out / "test_set.csv", index=False)
    with open(out / "folds.json", "w") as f:
        json.dump(folds, f, indent=2)
    save_seeds(seed_manifest, out / "seed_manifest.json")
    print(
        f"Split pipeline complete. {len(dev_df)} dev / {len(test_df)} test, {len(folds)} outer folds."
    )
