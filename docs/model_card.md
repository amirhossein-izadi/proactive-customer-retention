# Model Card — Didar Retention Uplift Model

## Model Overview

| Field | Value |
|---|---|
| **Model name** | `didar-retention-uplift-v1` |
| **Champion architecture** | Logistic T-learner (two separate L2-regularized logistic regression models per treatment arm) |
| **Task** | CATE estimation (individual treatment effect) |
| **Target** | τ(x) = P(churn | no contact, x) − P(churn | contact, x) |
| **Framework** | scikit-learn 1.7.2, CatBoost 1.2.10, Optuna 4.9.0 |
| **License** | MIT (code), Apache 2.0 (CatBoost) |

## Champion Performance

| Metric | Dev (OOF CV) | Test (Holdout) |
|--------|-------------|----------------|
| ROC-AUC | 0.8165 | **0.8036** |
| PR-AUC | — | 0.6394 |
| Brier | 0.1576 | **0.1626** |
| Log loss | — | 0.4885 |
| DR@50 | 0.3826 | **0.5692** |
| Uplift@50 | — | 0.3578 |
| Arm0 AUC (control) | — | 0.7800 |
| Arm1 AUC (treated) | — | 0.8309 |

**Baselines**: Random DR@50 = 0.007, Risk-based = -0.110, Historical = -0.001

## Training Data

| Source | Rows | Columns |
|---|---|---|
| `customer_data.csv` | 4,000 | 10 (incl. treatment & outcome) |
| `usage_timeseries.csv` | 240,000 | 3 (60 days per customer) |

- **Churn rate**: 31.4% | **Treatment rate**: 48.5%
- **Split**: 80/20 stratified by (contacted, churned), 5-fold × 3-repeat CV
- **No missing values**, 100% join coverage, 60-day history for all customers

## Features (53 total)

| Family | Count | Examples |
|--------|-------|----------|
| Static | 10 | log_company_size, log_revenue, tenure_years, is_annual, tickets_per_revenue |
| Temporal | 33 | usage_mean, recent_7_mean, slope_60d, zero_days, active_days_pct, max_drawdown, halves_ratio |
| PCA | 5 | pca_usage_0 through pca_usage_4 |
| OHE | 5 | package_Basic/Enterprise/Pro, subscription_type_Annual/Monthly |

## Calibration

| Arm | Brier | ECE |
|-----|-------|-----|
| Control (p0) | 0.1709 | 0.083 |
| Treated (p1) | 0.1539 | 0.052 |

Calibrated via CalibratedClassifierCV (Platt scaling, internal 3-fold CV). Champion selection enforces Brier < 0.20 threshold.

## Models Compared

9 models across 4 architectures: S-learner (3 variants), T-learner (2), DR-learner (2), X-learner (2). Plus logistic-catboost blend.

## Causal Assumptions

1. **Consistency**: contacted=1 matches the intervention being modeled
2. **Positivity**: Propensity clipping [0.02, 0.98]; 100% of dev customers in common support
3. **Conditional ignorability**: Not guaranteed — account-manager judgment, relationship quality, and unresolved complaints are not in the dataset

**Limitation**: The uplift estimate is observational, not experimental. Recommend randomized exploration (10-20% of capacity) in production.

## Monitoring

Three Grafana dashboards:
1. **Model & Business**: Uplift distribution, mean uplift, model loaded status, common support ratio, top SHAP features
2. **Operations**: Request rate, p50/p95 latency, error rate, active endpoints
3. **Data & Drift**: PSI per feature, active days coverage, schema validation failures

Five Prometheus alert rules: ModelNotLoaded (critical), HighRequestErrorRate, HighInferenceLatency, SchemaValidationFailures, LowActiveDaysCoverage.

## Version History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-07-21 | Final training pipeline, Optuna-tuned params, full Docker stack verified |
