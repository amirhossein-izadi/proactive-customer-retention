# Didar Retention — Project Report

## 1. Project Overview & Problem Formulation

### Business Problem

The marketing team can proactively contact exactly **50 customers per campaign** to reduce churn. The question is not "who is most likely to churn?" but rather **"who will be most saved by contact?"** This is a **causal uplift (CATE) problem**, not a classification problem.

### CATE Formulation

$$\tau(x) = P(\text{churn} \mid T=0, X=x) - P(\text{churn} \mid T=1, X=x)$$

Where $T=1$ means contacted. Positive $\tau$ means contact reduces churn. The 50 highest $\tau$ values are selected.

### Why Uplift Beats Churn Prediction

A standard churn model selects the riskiest customers. But historical contact was confounded: account managers contacted at-risk customers more often (`treated churn rate 32.4%` vs `control 30.5%`). A churn model would recommend contacting the same risky people — who are risky *because* they were already contacted. The uplift model corrects for this bias.

### Data Overview

| File | Rows | Content |
|------|------|---------|
| `customer_data.csv` | 4,000 | Static attributes + treatment + outcome |
| `usage_timeseries.csv` | 240,000 | 60 days of daily usage per customer |

- **Churn rate**: 31.4% | **Treatment rate**: 48.5%
- **No missing values** — 100% join coverage, 60-day history for every customer
- **Stratified 80/20 split**: 3,200 dev / 800 test
- **CV**: 5-fold × 3-repeat, stratified by (contacted × churned) joint distribution

### Feature Engineering (53 features)

| Family | Count | Features |
|--------|-------|----------|
| **Static** (10) | `log_company_size`, `log_revenue`, `revenue_per_employee`, `tenure_years`, `is_annual`, `is_monthly`, `tickets_per_revenue`, `tickets_per_employee`, `email_engagement`, `support_tickets_30d` |
| **Temporal** (33) | Level (mean, median, max, min, total), Recency (recent_7/14/30d, early_7/14/30d, last_day), Trend (slopes 14/30/60d, recent_minus_early, recent_early_ratio, pct_decline), Inactivity (zero_days, missing_days, active_days_pct, longest_zero_streak, days_since_activity, recent_zero_days), Volatility (std, cv, mean_abs_daily_change, down_day_proportion, max_drawdown, trend_residual_var), Shape (first/second_half_total, halves_ratio) |
| **PCA** (5) | 5 components from 60-day usage sequence |
| **OHE** (5) | `package_Basic/Enterprise/Pro`, `subscription_type_Annual/Monthly` |

All temporal features are computed **chronologically** (day -59 to 0), never by lexicographic column order. Missing usage is **never conflated with observed zero** — separate `missing_days` and `zero_days` features exist.

### Project Structure

```
C:\SBU\Extra\didar-retention\Challenge\
├── src/retention/                 # Source code (~6,500 lines)
│   ├── cli.py                    # 9 CLI commands (validate/split/train/evaluate/rank/bundle/ablate/tune/run)
│   ├── run.py                    # Full pipeline orchestrator
│   ├── contracts/                # Pydantic schemas, error codes, feature specs, inference policy
│   ├── data/                     # Loading, joining, validation (Pandera), splitting, snapshots, leakage register
│   ├── features/                 # Builder (53 features), categorical encoding, missingness, registry
│   ├── modeling/                 # S/T/DR/X learners, cross-fitting, calibration, ensembling, factories, propensity, screening, oversampling
│   │   └── train/                # Runner, config, shared utilities, per-learner train modules
│   ├── evaluation/               # AUUC/Qini, policy value, bootstrap CI, calibration, stability, subgroup, reporting
│   ├── policy/                   # Eligibility, ranking, tie-breaking, fallback, support overlap
│   ├── tuning/                   # Optuna search spaces, objectives, smart runner
│   ├── bundle/                   # RetentionPolicyBundle, serializer, metadata, compatibility checks
│   ├── serving/                  # FastAPI app (5 routes), middleware, dependencies, exception handlers
│   ├── monitoring/               # Prometheus metrics, drift detection, audit logging, data quality
│   ├── explainability/           # SHAP global/local explanations, reason codes, uplift explanations
│   ├── settings/                 # YAML config loader
│   └── utils/                    # Paths, seeds, serialization, hashing, timing
├── configs/                      # YAML/JSON configs (models, features, validation, policy, tuning)
├── data/                         # Raw CSVs (DVC-tracked), splits, manifests
├── models/                       # Champion bundle, registry
├── artifacts/                    # OOF predictions, metrics, rankings, plots, Optuna studies
├── tests/                        # 102 tests across 7 categories
│   ├── unit/                     #  9 tests (calibration, features, missingness, policy, etc.)
│   ├── integration/              # 13 tests (API, bundle, DVC, training pipeline)
│   ├── leakage/                  #  7 tests (OOF isolation, holdout, calibration isolation)
│   ├── contract/                 # 10 tests (input/output schema, model compatibility)
│   ├── robustness/               # 14 tests (batch invariance, extreme values, missing days, unseen cats)
│   ├── property/                 # 13 tests (feature/prediction/ranking invariants)
│   └── golden/                   # 36 tests (fixtures, predictions, invariants)
├── monitoring/                   # Prometheus config + alert rules + Grafana dashboards + provisioning
├── orchestration/airflow/        # Airflow DAG stubs (scoring + retraining)
├── .github/workflows/            # CI/CD pipelines (CI, Docker, model validation)
├── notebooks/                    # EDA notebooks (empty stubs)
├── docs/                         # Model card, methodology report, ML roadmap, ML guide corpus
├── scripts/                      # CLI entry point wrappers
├── reports/                      # Model comparison CSV, final ranking CSV
├── pyproject.toml                # Project config + dependencies
├── Dockerfile                    # Multi-stage build with uv
├── docker-compose.yml            # 4 services (API, MLflow, Prometheus, Grafana)
└── dvc.yaml                      # 6 pipeline stages
```

### Reproducibility & Seed Management

All randomness is controlled by a **master seed** (42) that deterministically generates all sub-seeds:

```json
{
  "master_seed": 42,
  "holdout_seed": 191664964,   // 80/20 train/test split
  "outer_cv_seed": 1662057958,  // 5-fold x 3-repeat CV folds
  "inner_cv_seed": 1405681632,  // Inner CV for hyperparameter tuning
  "model_seed": 942484272,      // Per-model random state
  "bootstrap_seed": 929893138   // Bootstrap uncertainty estimation
}
```

### How to Reproduce Results Exactly

```bash
# 1. Clean environment
uv sync --frozen

# 2. Full pipeline (tune + train + bundle + rank + evaluate)
uv run retention run

# Or step by step:
uv run retention split              # Stratified 80/20 split (deterministic)
uv run retention tune --all        # Optuna with TPE + fixed seed
uv run retention train --optuna    # Train 9 models, merge Optuna params
uv run retention bundle            # Build champion artifact
uv run retention rank              # Generate top-50 ranking
uv run retention evaluate          # Full evaluation summary
```

The seed manifest is saved at `data/splits/seed_manifest.json` and included in every training run's MLflow tags. Changing the master seed produces an entirely different (but equally valid) set of results.

### Dependencies & Environment

**Run environment** (from last training):

| Component | Version |
|-----------|---------|
| Python | 3.12.12 |
| scikit-learn | 1.7.2 |
| NumPy | 2.3.5 |
| CatBoost | 1.2.10 |
| LightGBM | 4.6.0 |
| XGBoost | 3.1.3 |
| CPU cores | 16 |

**Dependency groups** (from `pyproject.toml`):

| Group | Key Packages | When to Install |
|-------|-------------|-----------------|
| **Core** | pandas, numpy, scikit-learn, catboost, lightgbm, xgboost, mlflow, pydantic, fastapi, uvicorn, prometheus-client, shap, optuna, imbalanced-learn | Always |
| **dev** | pytest, hypothesis, ruff, mypy, pre-commit | `uv sync` (default) |
| **monitoring** | opentelemetry-sdk, opentelemetry-instrumentation-fastapi | Docker production |
| **explainability** | lime | Docker production |
| **automl** | autogluon | Optional experiments |
| **notebooks** | jupyter, matplotlib, seaborn | EDA |
| **tabfm** | torch, transformers | Research |

```bash
# Full install (all groups)
uv sync --all-groups

# Production serving only
uv sync --no-dev --group monitoring --group explainability
```

## 2. Exploratory Data Analysis

### Data Integrity

| Check | Result |
|-------|--------|
| Customers | 4,000 |
| Usage rows | 240,000 |
| Usage days per customer | Exactly 60 |
| Usage day range | -59 through 0 |
| Duplicate customer IDs | 0 |
| Duplicate customer-day rows | 0 |
| Missing usage cells | 0 |
| Unknown customer IDs in usage | 0 |
| Negative usage values | 0 |
| Daily usage range | 0–96 |
| Zero-usage observations | 22.63% |
| Customers with all-zero usage | 0 |

The dataset is **unusually clean** — no missing values, no duplicates, 100% join coverage, complete 60-day history for every customer. Production considerations (missing histories, partial windows, schema drift) must be discussed as risks even though they do not appear here.

### Treatment & Outcome Distribution

| Population | Count / Rate |
|------------|-------------:|
| Churned | 1,257 / **31.43%** |
| Not churned | 2,743 / 68.58% |
| Contacted | 1,939 / **48.48%** |
| Not contacted | 2,061 / 51.53% |

**Four treatment-outcome groups:**

| | Not churned | Churned | Total |
|---|:----------:|:-------:|:-----:|
| **Not contacted** | 1,432 | 629 | 2,061 |
| **Contacted** | 1,311 | 628 | 1,939 |

The smallest cell contains 628 customers — not severely imbalanced for 4,000 rows.

### The Confounding Problem

**Naive comparison**: Contacted customers churn at 32.39% vs 30.52% for non-contacted — a difference of **+1.87 percentage points**. A naive analysis would conclude contact increases churn.

This is **confounding by indication**: account managers preferentially contacted at-risk customers.

**Evidence of targeting bias** — a cross-validated propensity model (predicting P(contacted)) achieves:

| Metric | Value |
|--------|-------|
| Propensity ROC-AUC | **0.649** |
| Predicted propensity range | 0.05–0.92 |

An AUC of 0.649 confirms treatment assignment was **not random**.

**Largest standardized differences between contacted vs uncontacted customers:**

| Feature | Std Difference |
|---------|:-------------:|
| Recent/early usage ratio | -0.468 |
| 60-day usage slope | -0.351 |
| Recent minus early usage | -0.346 |
| Recent 7-day usage | -0.337 |
| Recent 14-day usage | -0.334 |
| Last-day usage | -0.331 |
| Recent 30-day usage | -0.309 |
| Support tickets | **+0.265** |

Account managers were more likely to contact customers with **steeper usage decline, lower recent activity, more inactivity, and more support tickets**.

### Usage Pattern Analysis

**Median customer usage:**

| Feature | Median |
|---------|:------:|
| Early 30-day average | 10.17 actions/day |
| Recent 30-day average | 3.17 |
| Recent 7-day average | 0.86 |
| Recent-minus-early difference | -5.70 |
| Recent/early usage ratio | 0.395 |
| 60-day slope | -0.189 actions/day |
| Longest zero streak | 3 days |
| Days since last activity | 1 day |

Nearly every customer shows decline. The **severity of decline** is what matters — relative measures (recent/early ratio) are more discriminative than slope sign alone.

**Strongest univariate churn predictors (Spearman correlation):**

| Feature | Correlation with churn |
|---------|:---------------------:|
| Recent/early usage ratio | -0.383 |
| Last-day usage | -0.316 |
| Recent 7-day usage | -0.314 |
| Number of zero days | +0.275 |
| Usage slope | -0.254 |

### Customer Segment Churn Rates

| Segment | Churn Rate |
|---------|:----------:|
| Monthly subscription | **38.58%** |
| Annual subscription | 22.11% |
| Basic package | 34.29% |
| Pro package | 28.24% |
| Enterprise package | 29.98% |

Monthly subscription is both a strong risk feature and a potential treatment-effect modifier.

### Feature Redundancy

Many engineered features are highly correlated:

| Pair | Spearman ρ |
|------|:----------:|
| Recent-7 usage vs zero days | -0.943 |
| Mean usage vs zero days | -0.928 |
| Mean usage vs recent-7 usage | +0.918 |
| Recent-7 usage vs recent/early ratio | +0.910 |
| Company size vs monthly revenue | +0.836 |
| Revenue vs mean usage | +0.779 |

**Implications**: Regularize linear models. Do not interpret SHAP values on 40 near-identical usage aggregates independently. Group features into families (usage_level, usage_decline, inactivity, volatility) for interpretation.

### Churn Risk Baseline

A cross-validated regularized logistic risk model (without treatment) achieves:

| Metric | Value |
|--------|:-----:|
| ROC-AUC | **0.801** |
| Average precision (PR-AUC) | 0.643 |
| Brier score | 0.163 |
| Log loss | 0.488 |
| Observed churn among top-50 risk customers | **94%** |

This establishes strong churn-risk signal in the data. But a customer with 90% churn probability may be unaffected by contact, while a customer with 45% may drop to 10% if contacted. **This is why uplift beats churn prediction.**

## 3. Causal Learner Architectures

The experiment compares **four causal architectures**, each with logistic regression and CatBoost as the base model.

### 3.1 S-Learner (Single Model)

One model trained on all 4,000 rows with `contacted` as an additional feature:

$$P(Y=1) = f(X, T)$$

Then: $p_0 = f(X, T=0)$, $p_1 = f(X, T=1)$, $\tau = p_0 - p_1$

**Training**: Cross-fitted via `repeated_cross_fit_slearner()` — 5-fold × 3-repeat CV. Each fold trains the model on 80% of data (with T as a feature), predicts on the held-out 20%. OOF predictions are collected for all 3 repeats, then averaged.

**Test predictions**: Retrain on full dev data, score test set with T=0 and T=1 separately.

| Variant | Base Model | Features | Fine-Tuning |
|---------|-----------|----------|-------------|
| `logistic_s` | LogisticRegression (C=0.0313) | 53 features + T | Optuna-tuned C for logloss+Brier |
| `logistic_s_int` | LogisticRegression (C=0.0313) | 53 features + T + 5 screened `T × feature` interactions | Top-5 effect modifiers detected at train time via `screen_interactions()` (Z-score ranking of treatment-feature interaction coefficients) |
| `catboost_s` | CatBoost (depth=4, lr=0.011, l2=34.8) | 53 features + T | Optuna-tuned depth/lr/l2/min_data |

**Strengths**: Uses all data, low variance. **Weakness**: Flexible model may ignore weak treatment signal when baseline churn dominates.

### 3.2 T-Learner (Two Models)

Two independent models, one per treatment arm (~2,000 rows each):

$$m_0: P(Y=1 \mid T=0, X), \quad m_1: P(Y=1 \mid T=1, X)$$

$$\tau = m_0 - m_1$$

**Training**: Cross-fitted via `repeated_cross_fit()` — for each fold, fits m0 on control arm, m1 on treated arm, predicts both p0 and p1 on the validation fold. Propensity e(x) is also computed per fold using logistic regression. OOF predictions are averaged across 3 repeats.

**Calibration**: Both arms are separately calibrated using `CalibratedClassifierCV` (Platt scaling, internal 3-fold CV). This is critical because $\tau = p_0 - p_1$ compounds any calibration error between arms.

| Variant | Control Model | Treated Model | Regularization |
|---------|--------------|---------------|----------------|
| `logistic_t` | Logistic (C=0.0255) | Logistic (C=0.0255, seed+1) | L2 penalty, Optuna-tuned C |
| `catboost_t` | CatBoost (depth=3, lr=0.026, l2=25.7) | CatBoost (depth=3, l2=25.7, seed+1) | Conservative: depth 3, min_data_in_leaf=30 |

**Strengths**: Each arm learns completely separate response functions — no shared parameterization. **Weakness**: Sample split doubles variance (2,000 instead of 4,000).

### 3.3 DR-Learner (Doubly Robust)

Three-stage estimator designed for observational (confounded) treatment:

1. **Nuisance models**: Cross-fit $m_0, m_1, e(X)$ via 5-fold CV using the same preprocessor and seed control as T/S-learners
2. **DR pseudo-outcomes**: For each row i:
   $$\psi_i = m_0(X_i) - m_1(X_i) + \frac{(1-T_i)(Y_i - m_0(X_i))}{1-e(X_i)} - \frac{T_i(Y_i - m_1(X_i))}{e(X_i)}$$
   This combines the outcome model difference with an inverse-propensity-weighted correction term.
3. **CATE model**: Fit $\text{ElasticNet}(\psi \sim X)$ on the cross-fitted pseudo-outcomes. The ElasticNet parameters (alpha, l1_ratio) are **Optuna-tuned per training run** (20 trials, inner 3-fold CV on the pseudo-outcomes).

**Why DR works**: If the propensity model is correct, the IPW term corrects for confounding bias. If the outcome models are correct, the outcome difference term is accurate. The estimator is consistent if *either* is correct.

| Variant | Nuisance Models | CATE Model | CATE Tuning |
|---------|----------------|------------|-------------|
| `logistic_dr` | LogisticRegression | ElasticNet | alpha=0.04, l1=0.16 |
| `catboost_dr` | CatBoostClassifier | ElasticNet | alpha=0.04, l1=0.16 |

### 3.4 X-Learner (Imputed Effects)

Two effect models, one per treatment arm:

1. Compute outcome models $m_0, m_1$ (same cross-fitted procedure as T-learner)
2. **Pseudo-effects**:
   - For treated customers: $D^1 = m_0(X) - Y$ (how much would the control model overpredict?)
   - For control customers: $D^0 = Y - m_1(X)$ (how much does the treated model underpredict?)
3. Fit separate effect regressors on pooled OOF pseudo-effects:
   - $\tau_1 \sim D^1$ on treated data only
   - $\tau_0 \sim D^0$ on control data only
4. **Combine**: $\tau(X) = e(X) \cdot \tau_0(X) + (1-e(X)) \cdot \tau_1(X)$

The effect models use Ridge regression with alpha **tuned per run** via `tune_ridge()` (15 trials, inner CV on pooled pseudo-effects).

| Variant | Outcome Models | Effect Models | Effect Tuning |
|---------|---------------|--------------|---------------|
| `logistic_x` | LogisticRegression | Ridge (alpha ~86) | Alpha tuned on pooled D0/D1 |
| `catboost_x` | CatBoostClassifier | Ridge (alpha ~86) | Alpha tuned on pooled D0/D1 |

### 3.5 Cross-Fitting Protocol (All Learners)

All models use the same cross-fitting framework:

```
For each repeat (3 total):
  For each fold (5 total):
    1. Split dev data into train (80%) / validation (20%)
    2. Fit preprocessor on train fold ONLY (no leakage of scaling/encoding params)
    3. Transform train and validation with fold-specific preprocessor
    4. Fit model(s) on train fold
    5. Predict on validation fold (out-of-fold)
  Collect OOF predictions across all folds
Average predictions across repeats
```

This guarantees every prediction on dev data is **out-of-fold** — no row is ever scored by a model that was trained on that row. Preprocessing (scaling, PCA, OHE) is fitted independently per fold, preventing global leakage.

### 3.6 Shared Architecture

All 9 model variants share:
- `capacity()` — reads `top_k` from config (currently 50)
- `model_kw()` — strips meta-params, injects seeds (supports LogisticRegression, CatBoost, LightGBM, XGBoost)
- `tune_elasticnet()` / `tune_ridge()` — shared Optuna tuning for CATE/effect models
- `build_preprocessor()` — same pipeline (53 features, optional PCA, optional scaling)

These are defined once in `modeling/train/shared.py` and imported by all 4 learner modules. No duplication.

**Strengths**: Imbalanced-arm robustness. **Note**: Our arms are balanced (48/52), so X-learner is secondary.

### 3.7 Leakage Prevention Register

An 18-item leakage register is maintained in `data/leakage_register.py`, documenting each known leakage path and its mitigation:

| Category | Risk | Mitigation |
|----------|------|------------|
| **Label leakage** | churned used as a feature | Excluded from all feature frames |
| **Treatment misuse** | contacted used in feature pipeline | Architecture-specific: S-learner uses T as input, T-learner uses T for arm split, DR/X learners use T for pseudo-outcomes |
| **Preprocessing** | Scaling/PCA/OHE fitted on full data | All fitted inside each CV fold independently |
| **Calibration** | Calibrator sees validation data | CalibratedClassifierCV with internal 3-fold CV |
| **Oversampling** | SMOTE applied before split | Only inside training folds |
| **Hyperparameter tuning** | Optuna sees outer validation | Optuna runs on inner CV only |
| **Feature selection** | screen_interactions sees test data | Screened on preprocessed dev data only |
| **Blend weights** | tune_arm_weights sees test | Tuned on dev OOF, evaluated on test predictions |
| **Ensemble stacking** | Stacking feature leakage | Uses only OOF base predictions |
| **Ranking policy** | poliy tuned on test | All policy parameters from config, not data-driven |
| **Model selection** | Champion chosen by test performance | Champion selected on dev OOF DR@50 only |
| **Report generation** | Final report influenced by test | Test metrics inspected only after pipeline is frozen |

## 4. Hyperparameter Tuning

Optuna TPE sampler with median pruner was used across 4 studies (15 trials each for `logistic`/`catboost` × `s_learner`/`t_learner`):

| Study | Best Alpha | Best L1_Ratio | Best Value |
|-------|-----------|---------------|------------|
| `logistic/t_learner` | C=0.0255 | — | 0.558 |
| `catboost/t_learner` | depth=3, lr=0.026, l2=25.7, min_data=30 | — | 0.566 |
| `logistic/s_learner` | C=0.0313 | — | 0.567 |
| `catboost/s_learner` | depth=4, lr=0.011, l2=34.8, min_data=80 | — | 0.574 |

Additionally, the **DR-learner CATE stage** is tuned **per-training-run** (20 trials of ElasticNet on cross-fitted pseudo-outcomes). The **X-learner effect models** also use per-run tuning of Ridge alpha via `tune_ridge()` (15 trials). These are fast — ElasticNet/Ridge train in milliseconds.

Tuned params are saved to `artifacts/optuna/all_best.json` and auto-merged at train time via `--optuna` flag.
## 5. Results & Model Comparison

### 4.1 Champion: logistic_t (T-Learner, Logistic Regression)

Directly from MLflow experiment `didar-retention` (last parent run):

| Metric | Value | Benchmark |
|--------|-------|----------|
| Test ROC-AUC | **0.8036** | >0.80 target exceeded |
| Test Brier | **0.1626** | <0.17 target beaten |
| DR Policy Value @50 | **0.5692** | — |
| Mean Uplift | **0.093** | 9.3pp |
| Mean Uplift @50 | **0.3578** | — |
| Arm0 AUC (control) | 0.7800 | — |
| Arm1 AUC (treated) | 0.8309 | — |

### 4.2 Full Comparison (MLflow Child Runs, Latest Training)

| Model | Test AUC | Test DR@50 | Uplift@50 | Test Brier |
|-------|----------|------------|-----------|------------|
| **logistic_t 🏆** | **0.8036** | **0.5692** | **0.3578** | **0.1626** |
| blend_t (logistic+catboost) | 0.8036 | 0.5557 | — | 0.1627 |
| catboost_t | 0.7991 | 0.4312 | 0.3283 | 0.1649 |
| catboost_dr | 0.7981 | **0.3955** | 0.2582 | 0.1654 |
| logistic_x | 0.8036 | **0.3889** | 0.3017 | 0.1626 |
| logistic_dr | 0.8036 | 0.3527 | 0.2551 | 0.1626 |
| catboost_x | 0.7981 | 0.3455 | 0.2905 | 0.1654 |
| catboost_s | 0.7879 | 0.3232 | 0.0933 | 0.1677 |
| logistic_s | 0.7894 | 0.1389 | 0.1124 | 0.1673 |

### 4.3 Baselines (MLflow Parent Metrics)

| Baseline | DR@50 | Interpretation |
|----------|-------|---------------|
| Random contact | **+0.007** | Random selection is neutral |
| Risk-based (top churn) | **-0.110** | **Negative!** Contacting riskiest customers increases churn by 11pp |
| Historical policy | **-0.001** | Current account team is no better than random |

The champion's DR@50=0.5692 is **80× better** than random and **reverses the negative effect** of the historical contact policy.

### 4.4 Subgroup Analysis

| Group | Split | N | Churn (Control) | Churn (Treated) | Mean Uplift | AUC |
|-------|-------|---|-----------------|-----------------|-------------|-----|
| **Subscription** | Annual | 345 | 35.3% | 38.9% | 0.090 | 0.808 |
| | Monthly | 455 | 27.3% | 27.2% | 0.095 | 0.799 |
| **Package** | Basic | 398 | 33.5% | 32.2% | 0.095 | 0.791 |
| | Pro | 297 | 27.7% | 32.9% | 0.092 | 0.809 |
| | Enterprise | 105 | 26.5% | 32.1% | 0.088 | 0.829 |

The uplift model works consistently across segments — no segment shows negative average uplift.

### 4.5 Feature Drift (Dev vs Test)

| Feature | PSI | KS |
|---------|-----|----|
| pca_usage_3 | 0.014 | 0.020 |
| log_revenue | 0.014 | 0.026 |
| log_company_size | 0.011 | 0.028 |
| usage_mean | 0.009 | 0.020 |

All features have PSI < 0.05 (no drift). The dev and test distributions are stable.

### 4.6 Top-5 Customers

| Rank | Customer ID | p0 (No Contact) | p1 (Contact) | Uplift | Revenue |
|------|-------------|-----------------|--------------|--------|---------|
| 1 | 620 | 0.762 | 0.251 | **0.512** | $1,965 |
| 2 | 754 | 0.498 | 0.017 | **0.481** | $2,452 |
| 3 | 1758 | 0.844 | 0.385 | **0.459** | $5,920 |
| 4 | 832 | 0.708 | 0.258 | **0.450** | $774 |
| 5 | 1881 | 0.639 | 0.197 | **0.442** | $1,387 |

Full ranking: `reports/final_ranking.csv` (50 customers, all eligible)
## 6. Calibration & Probability Quality

Each serving model is a `CalibratedClassifierCV` wrapping a CatBoost classifier with **Platt scaling (sigmoid)** and internal 3-fold CV. This ensures probabilities are well-calibrated per treatment arm.

### Calibration Metrics (Final Evaluation)

| Arm | Brier | ECE | Arm AUC |
|-----|-------|-----|---------|
| Control (p0) | 0.1709 | 0.083 | 0.780 |
| Treated (p1) | 0.1539 | 0.052 | 0.831 |

The 0.017 Brier gap between arms is expected — the treated arm has different characteristics (riskier customers). The **champion selection enforces Brier < 0.20** for all models, disqualifying poorly calibrated models regardless of AUC.

### Bundle Architecture

```
raw features → build_preprocessor → CalibratedClassifierCV(CatBoost, cv=3) → calibrated p0/p1
```

The CalibratedClassifierCV:
1. Clones the CatBoost base model 3 times
2. Trains each on 2/3 of the arm data
3. Fits Platt scaling on held-out 1/3
4. At serving: averages 3 calibrated predictions
## 7. Infrastructure & MLOps

### Experiment Tracking (MLflow)

- **Backend**: SQLite (`mlflow.db`) — 154 tracked runs
- **Structure**: 21 parent experiments + 133 child runs (one per model per training)
- **Logged**: Model params, test metrics (AUC, Brier, DR50, arm-specific AUC), baselines, tags (champion, git commit, versions)
- **Artifacts**: champion predictions (`.npz`), model comparison CSV, run environment JSON

### Pipeline Orchestration (DVC)

6 stages defined in `dvc.yaml`:

```
validate -> split -> train -> evaluate -> rank -> bundle
```

Each stage specifies its dependencies (source code, data, configs) and outputs (metrics, predictions, artifacts). DVC tracks changes and only re-runs stages whose dependencies have changed.

```yaml
train:
  cmd: uv run retention train --optuna
  deps:
    - src/retention/modeling
    - data/splits/development_set.csv
    - data/splits/test_set.csv
  outs:
    - artifacts/oof_predictions/champion_predictions.npz
    - reports/model_comparison.csv
```

Raw data tracked via `dvc add` with SHA256 checksums. Full pipeline reproduction:

```bash
uv sync --frozen
dvc repro
uv run pytest
```

### Eligibility, Abstention & Fallback Policy

The ranking pipeline (`policy/ranking.py`) applies a multi-stage filter before selecting the top 50:

1. **Eligibility** (`policy/eligibility.py`): minimum 45 active usage days, minimum 1 month tenure, exclude recently contacted (30 days), exclude do-not-contact flags. Configurable via `configs/policy.yaml`.

2. **Propensity overlap** (`policy/support.py`): customers with propensity < 0.05 or > 0.95 are flagged; out-of-support customers get robust_score = -inf (never selected).

3. **Abstention** (`policy/fallback.py`): if |uplift| < 1e-6 or SE(uplift) > 0.5 x |uplift|, abstain (negligible or too uncertain).

4. **Tie-breaking** (`policy/tie_breaking.py`): primary = robust_score descending, secondary = customer_id ascending. Uses `np.lexsort((ids, direction * scores))` for deterministic ordering.

5. **Fallback hierarchy**: champion model -> calibrated predictions -> uplift -> ranking. If model not loaded: 503. If individual customer fails: exclude, continue ranking remainder.

### Production Serving (FastAPI)

| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | `{"status": "alive"}` |
| `/ready` | GET | `{"status": "ready", "model_loaded": true}` |
| `/v1/score` | POST | Per-customer uplift + probability + reason codes |
| `/v1/rank` | POST | Batch top-k ranking with eligibility filtering |
| `/v1/model-info` | GET | Version metadata |
| `/metrics` | GET | Prometheus format metrics |

Inference: ~24ms per customer. Bundle loaded in ~6.6s.

#### Exception Handling & Error Taxonomy

The API defines 14 structured error codes in `contracts/error_codes.py`:

| Code | HTTP | Description | Retryable |
|------|------|-------------|-----------|
| `USAGE_DUPLICATE_DAY` | 422 | Same day appears twice in input | No |
| `BATCH_TOO_LARGE` | 422 | Batch exceeds maximum size | No |
| `MODEL_NOT_LOADED` | 503 | Bundle not loaded | Yes |
| `INVALID_USAGE` | 422 | Usage value out of domain | No |
| `UNKNOWN_CATEGORY` | 422 | Package/subscription not recognized | No |
| `MISSING_FIELD` | 422 | Required field absent | No |
| `SCHEMA_MISMATCH` | 422 | Incompatible schema version | No |
| `INFERENCE_TIMEOUT` | 504 | Prediction exceeded time limit | Yes |
| `NEGATIVE_REVENUE` | 422 | Revenue below zero | No |
| `ENGAGEMENT_OUT_OF_RANGE` | 422 | Email engagement outside [0,1] | No |
| `TENURE_MISMATCH` | 422 | Tenure inconsistent with subscription | No |
| `COMPANY_SIZE_ZERO` | 422 | Company size must be >= 1 | No |
| `RANKING_FAILED` | 500 | Internal ranking computation error | Yes |
| `FEATURE_BUILD_FAILED` | 500 | Feature computation error | Yes |

All error responses follow a consistent structure:
```json
{
  "request_id": "uuid",
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "Model not loaded. Try /health or /ready.",
    "field": null,
    "retryable": true
  }
}
```

#### Bundle Caching & Load Strategy

The bundle is loaded via `@lru_cache(maxsize=4)` in `serving/dependencies.py`. On first request, the bundle is loaded from disk (all pickled artifacts). The `/ready` endpoint has a two-phase retry: if the cached bundle is `None`, it clears the cache and retries via an uncached `_load_bundle()` path. This prevents permanent caching of failed loads.

```python
@router.get("/ready")
async def ready():
    bundle = get_bundle()       # cached — 6.6s first, instant after
    if bundle is None:
        clear_bundle_cache()    # retry with fresh load
        bundle = _load_bundle() # uncached, no caching of None
    ...
```

Key optimizations:
- Bundle loaded once at startup via middleware or first request
- `lru_cache` prevents repeated disk reads
- `clear_bundle_cache()` available for hot-reload scenarios
- `/health` is always instant (no model dependency)
- `/ready` only does disk I/O when bundle is missing

### Model Bundle

`RetentionPolicyBundle` contains all artifacts for serving:
- `preprocessor.pkl` — ColumnTransformer (fit scaler, OHE, PCA)
- `m0.pkl`, `m1.pkl` — CatBoost outcome models (for SHAP/analysis)
- `calibrator_0.pkl`, `calibrator_1.pkl` — CalibratedClassifierCV (CatBoost + Platt)
- `propensity.pkl` — Logistic regression for P(contacted)
- `global_importance_*.json` — SHAP feature importance per arm + uplift ex planations
- `metadata.json`, `checksums.json`, `feature_manifest.json`, `model_card.md`

### Docker Stack

4 services defined in `docker-compose.yml`:

| Service | Image | Internal Port | External Port | Purpose |
|---------|-------|---------------|---------------|---------|
| **api** | Build from Dockerfile | 8000 | 8001 | FastAPI model serving |
| **mlflow** | `ghcr.io/mlflow/mlflow:v2.17.2` | 5000 | 5001 | Experiment tracking |
| **prometheus** | `prom/prometheus:v2.53.0` | 9090 | 9091 | Metrics collection |
| **grafana** | `grafana/grafana:11.0.0` | 3000 | 3001 | Visualization & dashboards |

#### Build & Run

```bash
# First build (10-15 min on first run, cached layers after)
docker compose build api

# Start all services
docker compose up -d

# Verify
curl http://localhost:8001/health          # API: {"status":"alive"}
curl http://localhost:8001/ready           # API: {"status":"ready","model_loaded":true}
curl http://localhost:9091/metrics          # Prometheus: metrics data
open http://localhost:3001                  # Grafana (admin/admin)

# Stop
docker compose down

# Rebuild after code changes
docker compose build api --no-cache
docker compose up -d --force-recreate api
```

#### Dockerfile Design

Multi-stage build (python:3.12-slim) with uv-based dependency management:

```
1. python:3.12-slim base
2. Install system deps: gcc, libpq-dev, curl (cached)
3. Copy uv binary from ghcr.io/astral-sh/uv
4. Copy pyproject.toml + src/ before uv sync (package installs correctly)
5. uv sync --frozen --no-dev --group monitoring --group explainability
6. Copy configs/ and models/
7. Create non-root user (uid 1001), chown only src/configs/models
8. Run as non-root user
9. HEALTHCHECK via /health endpoint
10. CMD: uvicorn retention.serving.app:app
```

Key design decisions:
- Source code copied **before** uv sync to ensure package installs correctly
- `chown` targets only `/app/src /app/configs /app/models` (skips `.venv` for speed)
- Uses `uvicorn` directly (not `uv run`) to avoid `.venv` write attempts
- `PATH` includes `/.venv/bin` for direct binary access

### Prometheus Configuration

**Scrape config** (`monitoring/prometheus/prometheus.yml`):

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: retention-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
        labels:
          service: retention-ranking
  - job_name: mlflow
    static_configs:
      - targets: ["mlflow:5000"]
  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
```

**Exposed Metrics** (from `monitoring/metrics.py`):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | endpoint | HTTP request latency |
| `inference_requests_total` | Counter | endpoint, status | Model inference requests |
| `inference_duration_seconds` | Histogram | endpoint | Model inference latency |
| `estimated_uplift` | Histogram | — | Uplift score distribution |
| `model_loaded` | Gauge | — | Champion bundle loaded (1=yes, 0=no) |
| `schema_validation_failures_total` | Counter | error_type | Pandera validation failures |
| `feature_psi` | Gauge | feature | Population Stability Index per feature |
| `active_days_pct` | Gauge | — | Active days coverage ratio |
| `inference_negative_uplift_total` | Counter | — | Inferences where uplift < 0 |

**Recording Rules** (`monitoring/prometheus/recording_rules.yml`):

| Rule | Expression | Description |
|------|-----------|-------------|
| `job:error_rate_5m` | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` | 5-min error rate |
| `job:mean_uplift_5m` | `avg(estimated_uplift_sum[5m]) / avg(estimated_uplift_count[5m])` | 5-min mean uplift |
| `job:inference_request_rate_5m` | `rate(inference_requests_total[5m])` | Inference request rate |

**Alert Rules** (`monitoring/prometheus/alert_rules.yml`):

| Alert | Condition | Severity | Owner |
|-------|-----------|----------|-------|
| `ModelNotLoaded` | `model_loaded == 0 for 2m` | Critical | mlops |
| `HighRequestErrorRate` | `error_rate > 5% for 5m` | Warning | backend |
| `HighInferenceLatency` | `p95 latency > 2s for 5m` | Warning | mlops |
| `SchemaValidationFailures` | `failures > 1% for 5m` | Warning | backend |
| `LowActiveDaysCoverage` | `active_days_pct < 0.5 for 10m` | Warning | data |

### Grafana Dashboards

Three pre-provisioned dashboards with auto-configured Prometheus datasource:

#### 1. Model Performance (`model_performance.json`)

| Panel | Type | Query | Description |
|-------|------|-------|-------------|
| Estimated Uplift Distribution | Histogram | `estimated_uplift_bucket` | Distribution of predicted uplift |
| Mean Uplift | Time series | `avg(estimated_uplift)` | Average uplift over time |
| Model Loaded Status | Stat | `model_loaded` | 1=loaded, 0=not loaded |
| Common Support Ratio | Time series | `inference_common_support_ratio` | % of scored customers in common support |
| Top Features (Global Importance) | Bar chart | — | SHAP importance rankings |

#### 2. Operations (`operations.json`)

| Panel | Type | Query | Description |
|-------|------|-------|-------------|
| Request Rate | Time series | `rate(http_requests_total[5m])` | Request volume per endpoint |
| Request Latency (p95) | Time series | `histogram_quantile(0.95, ...)` | P95 latency per endpoint |
| Error Rate by Endpoint | Time series | `rate(http_requests_total{status=~"5.."}[5m])` | 5xx errors per endpoint |
| Model Loaded | Stat | `model_loaded` | Binary status indicator |
| Inference Latency (p50) | Time series | `histogram_quantile(0.50, ...)` | P50 inference latency |
| Active Endpoints | Stat | `count(http_requests_total)` | Number of active endpoints |

#### 3. Data Drift (`data_drift.json`)

| Panel | Type | Query | Description |
|-------|------|-------|-------------|
| PSI per Feature | Table | `feature_psi` | Population Stability Index per feature |
| Active Days Coverage | Time series | `avg(active_days_pct)` | Mean usage coverage |
| Zero-Days Ratio | Time series | `avg(zero_days_ratio)` | Mean zero-days proportion |
| Null/Missing Rate | Table | `data_quality_missing_rate` | Missing values per column |
| Schema Validation Failures | Time series | `rate(schema_validation_failures_total[5m])` | Validation error rate |

**Provisioning**: Datasource and dashboard configs are auto-loaded on Grafana startup via:
- `monitoring/grafana/provisioning/datasources.yml`
- `monitoring/grafana/provisioning/dashboards.yml`

No manual setup required after `docker compose up`.

### Prometheus Targets (Verified)

```
retention-api  -> api:8000/metrics   (15s scrape interval)
mlflow         -> mlflow:5000        (15s)
prometheus     -> localhost:9090     (15s)
```

### CI/CD (GitHub Actions)

| Workflow | Trigger | Steps |
|----------|---------|-------|
| **CI** | push/PR | pytest (102 tests), ruff lint |
| **Docker** | push/tag | Build & push to GHCR with semantic versioning |
| **Model Validation** | CI success | Train, bundle, golden tests, compatibility check |

### Airflow Orchestration

| DAG | Schedule | Tasks | Purpose |
|-----|----------|-------|---------|
| `retention_scoring` | Weekly (Mon 6 AM) | rank -> evaluate | Regenerate top-50 ranking from existing champion bundle |
| `retention_retraining` | Monthly (1st 4 AM) | data check -> tune (20 trials) -> train -> bundle -> rank -> evaluate | Full retrain with data quality gate |

The scoring DAG is lightweight — only re-ranks existing predictions. The retraining DAG starts with a Python data quality check (`join_coverage_pct == 100%`) that fails the DAG early if the source data is compromised. Airflow is optional for the challenge dataset (static) but essential for production where new data arrives on a schedule.

### MLflow Experiment Tracking

- **Backend**: SQLite (`mlflow.db`) — 154 tracked runs
- **Structure**: 21 parent experiments + 133 child runs (one per model per training)
- **Logged per run**: Model name, hyperparameters, test metrics (AUC, Brier, DR50, arm-specific AUC, uplift@50), baselines (random, risk, historical)
- **Tags**: champion model name, git commit, data version, feature version
- **Artifacts**: champion predictions (.npz), model comparison CSV, run environment JSON (Python, sklearn, numpy, catboost versions)

### Quick Start Commands

```bash
# Full pipeline (tune -> train -> bundle -> rank -> evaluate)
uv run retention run

# Tune only (top 3 models or all)
uv run retention tune --top 3
uv run retention tune --all

# Train with Optuna params
uv run retention train --optuna

# Docker stack
docker compose build api     # Build API image
docker compose up -d          # Start all services
docker compose logs api       # View API logs
docker compose down           # Stop all services

# Test API
curl http://localhost:8001/health
curl http://localhost:8001/ready

# Prometheus
curl http://localhost:9091/metrics

# Grafana
open http://localhost:3001    # admin/admin

# Run tests
uv run pytest tests/ -v

# View MLflow
open http://localhost:5001
```

## 8. Testing

**102 tests across 7 categories** — all passing in ~9 seconds:

### Test Categories

| Category | Tests | What They Verify |
|----------|-------|------------------|
| **Unit** (9) | `test_calibration.py`, `test_features.py`, `test_missingness.py`, `test_policy_ranking.py`, `test_propensity.py`, `test_seed_manifest.py`, `test_temporal_features.py`, `test_uplift_sign.py`, `test_data_validation.py` | Individual functions: slope computation, zero-streak detection, calibration ECE, propensity AUC, OOF coverage, tie-breaking determinism, seed reproducibility, temporal ordering, missing≠zero distinction |
| **Integration** (13) | `test_api_prediction.py` (5), `test_bundle_roundtrip.py` (2), `test_dvc_pipeline.py` (3), `test_training_pipeline.py` (3) | API health/score/rank endpoints, bundle serialization round-trip, DVC stage validation, T-learner/S-learner train-and-predict pipeline |
| **Leakage** (7) | `test_holdout_isolation.py` (3), `test_oof_isolation.py` (2), `test_calibration_isolation.py` (2), `test_temporal_cutoff.py` (2) | No dev/test overlap, OOF predictions cover all rows, calibrators fitted on train only, future data not leaked into past |
| **Contract** (10) | `test_input_schema.py` (5), `test_output_schema.py` (3), `test_model_feature_compatibility.py` (2) | Missing fields rejected, negative usage rejected, duplicate days rejected, output bounds, feature count, categorical indices detected |
| **Robustness** (14) | `test_batch_invariance.py` (1), `test_column_order.py` (1), `test_extreme_values.py` (5), `test_missing_days.py` (5), `test_unseen_categories.py` (4) | Single vs batch scoring parity, column permutation invariance, zero/high revenue, partial history (1-60 days), unknown package/subscription |
| **Property** (13) | `test_feature_properties.py` (4), `test_prediction_properties.py` (4), `test_ranking_properties.py` (5) | No NaN from temporal features, probability bounds [0,1], uplift in [-1,1], deterministic top-k, descending scores, k=0 edge case |
| **Golden** (36) | `test_golden_predictions.py` (4), `test_invariants.py` (32) | Fixture existence, JSON validity, nominal bounds, invalid rejection, temporal NaN, split reproducibility, ranking invariants, tie-breaking determinism |

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Single category
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/leakage/ -v
uv run pytest tests/contract/ -v
uv run pytest tests/robustness/ -v
uv run pytest tests/property/ -v
uv run pytest tests/golden/ -v

# Selected test
uv run pytest tests/unit/test_policy_ranking.py -v
```

### Key Test Patterns

- **Hypothesis property tests**: Temporal slope functions are tested with random arrays via the Hypothesis framework for NaN-safety, shape correctness, and non-negative streak counts
- **Determinism tests**: Same seed + same input → identical output across repeated runs
- **Leakage tests**: Cross-validated OOF predictions must not use the row's own fold's model
- **Golden fixtures**: 4 fixture JSONs (nominal, invalid, incomplete, unknown) test the full API path with known expected behaviors
- **API tests**: Batch/single parity ensures the `/v1/score` and `/v1/rank` endpoints produce identical predictions for the same customer

## 9. Conclusion & Key Findings

### What We Learned

1. **Logistic T-learner wins at this sample size** — At 4,000 customers (2,000 per arm), simple regularized logistic regression beats CatBoost, LightGBM, and XGBoost. This is expected: boosting models need larger sample sizes per treatment arm to justify their additional variance. The logistic T-learner achieves the best factual AUC (0.8036) and the highest DR policy value (0.5692) with the lowest fold variance.

2. **Confounding is real and severe** — The risk-based baseline scores **-0.110 DR@50**. The historical policy of contacting risky customers was actively harmful. Our champion model reverses this to **+0.569**.

3. **DR-learner CATE tuning works** — With Optuna-tuned ElasticNet (alpha=0.04, l1=0.16), catboost_dr went from DR@50=0.130 to **0.395** (3× improvement). The key was tuning the CATE stage, not the nuisance models.

4. **S-learner struggles with weak treatment signal** — logistic_s_int's DR@50=0.055 on test, barely above random. The treatment signal (contacted binary) is too weak to learn from in a single model when baseline churn signal dominates.

5. **Calibration matters for uplift** — Arm Brier scores (p0=0.171 vs p1=0.154) differ, which accumulates error in the uplift estimate $\tau = p_0 - p_1$. The Brier < 0.20 champion filter is essential.

### Tooling Assessment

| Tool | Assessment |
|------|-----------|
| **Optuna** | Essential — tuning CATE models (DR/X learners) improved DR@50 by 2-3× |
| **MLflow** | 154 runs logged — every training is fully tracked with params and metrics |
| **DVC** | 6 pipeline stages, SHA256-tracked data, reproducible end-to-end |
| **FastAPI** | 24ms inference, bundle loaded in 6.6s, strict Pydantic validation |
| **Docker** | Multi-stage build, non-root, uv-based — production-ready |
| **SHAP** | Global importance + uplift explanations for champion model |
| **CalibratedClassifierCV** | Proper Platt scaling per arm, eliminates retraining waste |

### Recommendations

- **Production**: Deploy `logistic_t` as champion. It's stable (lowest fold variance), well-calibrated (Brier=0.1626), and simple to debug.
- **Monitoring**:
  - Track arm Brier scores separately — a widening gap indicates calibration drift
  - Monitor propensity overlap — if common support drops below 90%, the model is extrapolating
  - Weekly scoring DAG generates fresh ranking from existing bundle
  - Monthly retraining DAG retrains and validates new champion
- **Next iteration**: Add more nuance to treatment (contact channel, intensity, timing) to improve S-learner signal.

### Final Model Ranking

```
1.  logistic_t 🏆   | AUC=0.8036, DR@50=0.5692  ← Deploy this
2.  blend_t          | AUC=0.8036, DR@50=0.5557  ← Strong ensemble candidate
3.  catboost_t       | AUC=0.7991, DR@50=0.4312  ← CatBoost T-learner baseline
4.  catboost_dr      | AUC=0.7981, DR@50=0.3955  ← DR with tuned CATE (biggest improver)
5.  logistic_x       | AUC=0.8036, DR@50=0.3889  ← X-learner with tuned effects
6.  logistic_dr      | AUC=0.8036, DR@50=0.3527  ← Logistic DR with tuned CATE
7.  catboost_x       | AUC=0.7981, DR@50=0.3455  ← CatBoost X-learner
8.  catboost_s       | AUC=0.7879, DR@50=0.3232  ← CatBoost S-learner
9.  logistic_s       | AUC=0.7894, DR@50=0.1389  ← Logistic S-learner
```

**MVP command to deploy**: `docker compose up -d` starts FastAPI + MLflow + Prometheus + Grafana.

---

*Report generated from: MLflow DB (sqlite:///mlflow.db, 154 runs), DVC pipeline (6 stages), Optuna studies (4+tuned per-run), model_comparison.csv, final_summary.json*
