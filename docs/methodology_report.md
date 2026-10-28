# Retention Uplift — Final Methodology Report

## 1. Problem Formulation

**Task**: Prioritize 50 customers for proactive contact to minimize churn.

**Reframing**: This is NOT a churn classification problem. It is a **causal treatment-effect estimation (CATE)** problem:

$$\tau(x) = P(\text{churn} \mid T=0, x) - P(\text{churn} \mid T=1, x)$$

where $T=1$ means contacted. The 50 customers with the highest positive $\hat{\tau}(x)$ are selected.

**Key insight**: Historical account managers contacted riskier customers (confounding by indication). Naive comparison shows +1.87pp churn among contacted — a biased estimate. The champion model estimates ~57pp churn reduction for the top 50.

## 2. Data

- 4,000 customers x 60-day usage history (240,000 rows)
- 31.4% churn rate, 48.5% treatment rate
- No missing values, 100% join coverage
- Propensity model AUC = 0.649, confirming non-random treatment assignment

## 3. Feature Engineering (53 features)

| Family | Count | Description |
|--------|-------|-------------|
| Static | 10 | log company_size, log revenue, revenue_per_employee, tenure_years, is_annual/monthly, tickets_per_revenue/employee, email_engagement, support_tickets_30d |
| Temporal | 33 | Level (mean/median/max/min/total, recent_7/14/30d, early_7/14/30d), Trend (slopes 14/30/60d, recent_minus_early, recent_early_ratio, pct_decline), Inactivity (zero_days, missing_days, active_days_pct, longest_zero_streak, days_since_activity), Volatility (std, cv, mean_abs_daily_change, down_day_proportion, max_drawdown, trend_residual_var), Shape (halves total/ratio) |
| PCA | 5 | 5 components from 60-day usage sequence |
| OHE | 5 | package_Basic/Enterprise/Pro, subscription_type_Annual/Monthly |

## 4. Validation Protocol

- **Split**: Stratified 80/20 by (contacted x churned) joint distribution
- **Cross-validation**: 5-fold x 3-repeat, stratified
- **Preprocessing**: All scaling, PCA, OHE fitted inside each training fold
- **Calibration**: CalibratedClassifierCV (Platt scaling, internal 3-fold CV) per treatment arm
- **Champion selection**: Dev OOF DR@50 with Brier < 0.20 calibration threshold

## 5. Models Compared

9 models across 4 architectures:

| Architecture | Logistic | CatBoost |
|-------------|----------|----------|
| S-learner (single model + T as feature) | logistic_s, logistic_s_int (with interactions) | catboost_s |
| T-learner (separate per arm) | logistic_t | catboost_t |
| DR-learner (doubly robust pseudo-outcomes) | logistic_dr | catboost_dr |
| X-learner (imputed effect models) | logistic_x | catboost_x |

Plus logistic-catboost blend and 3 baselines (random, risk-based, historical).

## 6. Results

### Champion: logistic_t

| Metric | Value |
|--------|-------|
| Test AUC | 0.8036 |
| Test Brier | 0.1626 |
| DR@50 | 0.5692 |
| Mean uplift@50 | 0.3578 |
| Arm0 AUC | 0.7800 |
| Arm1 AUC | 0.8309 |

### Model Ranking (by Test DR@50)

| Rank | Model | AUC | DR@50 | Uplift@50 |
|------|-------|-----|-------|-----------|
| 1 | logistic_t | 0.8036 | **0.5692** | 0.3578 |
| 2 | blend_t | 0.8036 | 0.5557 | — |
| 3 | catboost_t | 0.7991 | 0.4312 | 0.3283 |
| 4 | catboost_dr | 0.7981 | 0.3955 | 0.2582 |
| 5 | logistic_x | 0.8036 | 0.3889 | 0.3017 |
| 6 | logistic_dr | 0.8036 | 0.3527 | 0.2551 |
| 7 | catboost_x | 0.7981 | 0.3455 | 0.2905 |
| 8 | catboost_s | 0.7879 | 0.3232 | 0.0933 |
| 9 | logistic_s | 0.7894 | 0.1389 | 0.1124 |

### Baselines

- Random: +0.007 (near zero)
- Risk-based (top churn): **-0.110** (confounding confirmed)
- Historical: -0.001 (no better than random)

## 7. Key Findings

1. **Logistic T-learner wins** — At 4,000 customers (2,000 per arm), simple regularized logistic regression beats all tree-based models. Boosting needs more data to justify its variance.

2. **Confounding is real** — The risk-based baseline = -0.110. Contacting the riskiest customers was actively harmful.

3. **DR-learner CATE tuning works** — catboost_dr went from DR@50=0.130 to 0.395 after Optuna-tuned ElasticNet.

4. **Calibration threshold matters** — Brier < 0.20 ensures the champion produces reliable probabilities.

5. **The champion is deployable** — 24ms inference, 6.6s bundle load, FastAPI with 14 error codes, Prometheus metrics, Grafana dashboards.

## 8. Infrastructure

- **CLI**: 9 commands (validate/split/train/evaluate/rank/bundle/ablate/tune/run)
- **MLflow**: 154 runs, SQLite backend, full metrics + artifacts
- **DVC**: 6 pipeline stages (validate -> split -> train -> evaluate -> rank -> bundle)
- **FastAPI**: 5 endpoints (health/ready/score/rank/model-info) + /metrics
- **Docker**: 4 services (API + MLflow + Prometheus + Grafana), non-root user
- **Testing**: 102 tests across 7 categories
- **Airflow**: Scoring DAG (weekly) + Retraining DAG (monthly)
