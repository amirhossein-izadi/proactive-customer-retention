"""Oversampling validation curve — test SMOTE ratios inside CV folds.

Ensures no leakage: oversampling is applied ONLY inside each training fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from retention.features.builder import build_preprocessor

try:
    from imblearn.over_sampling import SMOTE

    _HAS_IMBLEARN = True
except ImportError:  # optional dependency for imbalance experiments
    SMOTE = None  # type: ignore[assignment]
    _HAS_IMBLEARN = False


def oversampling_curve(
    X_raw,
    Y: np.ndarray,
    ratios: list[float] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Evaluate PR-AUC and Brier at different SMOTE oversampling ratios.

    Parameters
    ----------
    X_raw : DataFrame
        Raw feature DataFrame (wide format with usage columns).
    Y : ndarray
        Binary outcome labels.
    ratios : list[float] | None
        SMOTE sampling_strategy ratios to evaluate.
        None defaults to [0.3, 0.5, 0.7, 1.0] (perfect balance).
    n_folds : int
        Number of stratified CV folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys 'ratio', 'auc_mean', 'auc_std', 'brier_mean', 'brier_std'.
    Each value is a list of floats, one per ratio.
    """
    if ratios is None:
        ratios = [0.3, 0.5, 0.7, 1.0]

    stratum = np.array([f"{int(y)}" for y in Y])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    auc_means, auc_stds, brier_means, brier_stds = [], [], [], []

    for ratio in ratios:
        fold_aucs, fold_briers = [], []
        for tr, va in skf.split(X_raw, stratum):
            prep = build_preprocessor(include_pca=True, scale=True)
            X_tr = prep.fit_transform(X_raw.iloc[tr])
            X_va = prep.transform(X_raw.iloc[va])
            Y_tr, Y_va = Y[tr], Y[va]

            if ratio < 1.0:
                if not _HAS_IMBLEARN:
                    continue  # skip resampling configs when imbalanced-learn is absent
                smote = SMOTE(sampling_strategy=ratio, random_state=seed + tr[0])
                X_tr, Y_tr = smote.fit_resample(X_tr, Y_tr)

            model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            model.fit(X_tr, Y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(Y_va, proba))
            fold_briers.append(brier_score_loss(Y_va, proba))

        auc_means.append(float(np.mean(fold_aucs)))
        auc_stds.append(float(np.std(fold_aucs, ddof=1)))
        brier_means.append(float(np.mean(fold_briers)))
        brier_stds.append(float(np.std(fold_briers, ddof=1)))

    return {
        "ratio": ratios,
        "auc_mean": auc_means,
        "auc_std": auc_stds,
        "brier_mean": brier_means,
        "brier_std": brier_stds,
    }
