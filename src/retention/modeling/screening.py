"""Treatment-interaction screening — identify features that modify treatment effect.

Use before building an S-learner: screen each candidate feature for a
significant interaction with treatment. Helps select interaction terms
for improved CATE estimation.

Usage::

    from retention.modeling.screening import screen_interactions
    stable = screen_interactions(X, T, Y, top_k=10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def screen_interactions(
    X: np.ndarray | pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    top_k: int = 15,
    C: float = 10.0,
) -> pd.DataFrame:
    """Screen each feature for treatment-effect modification.

    For each feature j, fits::

        logit(P(Y=1)) = beta0 + beta_j * X[:, j] + gamma * T + delta_j * X[:, j] * T

    Reports delta_j (interaction coefficient) and its Z-score for ranking.
    Features with large |delta| and low p are plausible effect modifiers.
    """
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X, dtype=float)
        names = [f"f{i}" for i in range(X_arr.shape[1])]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    n_features = X_arr.shape[1]
    rows = []

    for j in range(n_features):
        xj = X_scaled[:, j].reshape(-1, 1)
        interaction = xj.ravel() * T
        X_design = np.column_stack([xj, T.reshape(-1, 1), interaction.reshape(-1, 1)])

        # Skip near-constant features
        if np.std(xj) < 1e-8:
            continue

        try:
            model = LogisticRegression(max_iter=2000, C=C, random_state=42).fit(
                X_design, Y
            )
            coefs = model.coef_[0]
            # delta is the interaction coefficient (last column)
            delta = coefs[2]

            # Simple Z-score approximation (Wald test)
            proba = model.predict_proba(X_design)[:, 1]
            W = np.diag(proba * (1 - proba))
            try:
                XtWX = X_design.T @ W @ X_design
                se = np.sqrt(np.diag(np.linalg.inv(XtWX + np.eye(3) * 1e-6)))
                z_score = delta / max(se[2], 1e-10)
            except np.linalg.LinAlgError:
                z_score = 0.0
                se = np.zeros(3)

            rows.append(
                {
                    "feature": names[j],
                    "delta": round(float(delta), 6),
                    "z_score": round(float(z_score), 4),
                    "abs_z": round(abs(float(z_score)), 4),
                }
            )
        except Exception:
            rows.append(
                {
                    "feature": names[j],
                    "delta": 0.0,
                    "z_score": 0.0,
                    "abs_z": 0.0,
                }
            )

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return result
    return result.sort_values("abs_z", ascending=False).head(top_k).reset_index(drop=True)
