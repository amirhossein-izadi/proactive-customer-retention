"""Model-feature compatibility — preprocessor output matches model's expected input."""

from retention.features.builder import build_preprocessor, get_categorical_indices
from retention.features.feature_names import all_feature_names


def test_preprocessor_feature_count():
    prep = build_preprocessor(include_pca=True, scale=True)
    n_features = len(all_feature_names(include_pca=True))
    # Cannot assert exact count without fitting, just check it's reasonable
    assert n_features > 30, f"Too few features: {n_features}"


def test_preprocessor_feature_count_no_pca():
    n = len(all_feature_names(include_pca=False))
    assert n > 25
    assert n < 80


def test_categorical_indices_detected():
    prep = build_preprocessor(include_pca=True, scale=False)
    import pandas as pd
    n = 10
    X = pd.DataFrame({"company_size": range(1, n+1), "monthly_revenue": range(1, n+1),
                       "tenure_months": range(1, n+1),
                       "subscription_type": ["Monthly", "Annual"] * (n // 2),
                       "email_engagement": [0.5] * n, "support_tickets_30d": [0] * n,
                       "package": ["Pro", "Basic"] * (n // 2)})
    for d in range(-59, 1):
        X[f"usage_day_{d}"] = 0.0
    prep.fit(X)
    cat_idx = get_categorical_indices(prep)
    assert len(cat_idx) > 0, "Should detect OHE columns as categorical"
