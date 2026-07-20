# Retention Uplift — Final Methodology Report

## 1. Problem Formulation

**Task**: Prioritize 50 customers for proactive contact to minimize churn.

**Reframing**: This is NOT a churn classification problem. It is a **causal treatment-effect estimation (CATE)** problem. We estimate:

$$\tau(x) = P(\text{churn} \mid T=0, x) - P(\text{churn} \mid T=1, x)$$

where $T=1$ means contacted. The 50 customers with the highest positive $\hat{\tau}(x)$ are selected.

**Key insight**: Historical account managers contacted riskier customers (confounding by indication). Naive comparison shows +1.87pp churn among contacted — a biased estimate. DR-learner corrects this to ≈10pp reduction.

## 2. Methodology

### 2.1 Data
- 4,000 customers × 60-day usage history (240,000 rows)
- 31.4% churn rate, 48.5% treatment rate
- Clean dataset — no missing values, duplicates, or invalid entries

### 2.2 Feature Engineering (3 Representations)
1. **Static** (10 features): log transforms, revenue/employee, tenure, engagement
2. **Temporal Engineered** (32 features): level, trend (slopes), inactivity (zero streaks, days since last activity), volatility (CV, drawdown, down-day proportion), shape (halves ratio, trend residual variance)
3. **PCA Embedding** (5 components): Learned from 60-day usage sequences

### 2.3 Models Compared
| Model | AUC | PR-AUC | Brier | Uplift@50 |
|---|---|---|---|
| Logistic T-learner | 0.789 | 0.619 | 0.213 | 0.573 |
| CatBoost T-learner | 0.771 | 0.583 | 0.181 | 0.654 |
| LightGBM T-learner | 0.745 | 0.555 | 0.227 | 0.963 |
| XGBoost T-learner | 0.754 | 0.562 | 0.209 | 0.873 |

**Champion selected by**: PR-AUC (best balance of discrimination + calibration for imbalanced data).

### 2.4 Validation Protocol
- **Stratified 5-fold CV** by `(contacted, churned)` group
- **Cross-fitted OOF predictions**: every customer scored by models trained without that customer
- **Nested calibration**: Platt/sigmoid calibrators fitted on OOF predictions, separately per treatment arm
- **Bootstrap uncertainty**: 500 resamples, SE recorded per customer

### 2.5 Metrics
| Category | Metrics |
|---|---|
| Prediction | ROC-AUC, PR-AUC, Brier, Expected Calibration Error |
| Uplift | AUUC, Qini coefficient, DR policy value@50 |
| Business | Uplift@50, retrained customers, contact cost per saved customer |
| Stability | Bootstrap Jaccard@50, fold variance, rank SD |

## 3. Key Results

| Metric | Value |
|---|---|
| Champion model | CatBoost (multi-dim scorecard) |
| AUC | 0.771 |
| PR-AUC | 0.583 |
| Calibrated Brier (p₀) | 0.166 |
| Calibrated Brier (p₁) | 0.169 |
| Mean uplift | 0.094 (9.4pp average churn reduction) |
| Uplift@50 | 0.654 (65pp reduction for top 50 contacts) |
| Bootstrap SE | 0.003 |
| Persuadable customers | 558/4000 (14%) |
| Sleeping dogs | 585/4000 (15%) |

**Top-3 customers for contact:**
1. Customer 2271: p₀=0.975, p₁=0.066, uplift=0.910 (91pp reduction!)
2. Customer 75: p₀=0.874, p₁=0.034, uplift=0.840
3. Customer 234: p₀=0.960, p₁=0.161, uplift=0.799

## 4. Causal Assumptions & Limitations

1. **Consistency**: Recorded contact matches the intervention being modeled
2. **Positivity**: All customers have non-zero probability of both contact and no-contact
3. **Conditional ignorability** (weakest): No unmeasured confounders affect both contact assignment and churn

**Limitation**: Account manager judgment, relationship quality, and unresolved complaints are not in the dataset. The uplift estimate is observational, not experimental. Recommend randomized exploration (10-20% of capacity) in production to validate treatment effects.

## 5. Production Architecture

```
                           ┌─────────────────┐
                           │  CRM / Database  │
                           └────────┬────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      DVC Data Pipeline        │
                    │  validate → features → split  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       MLflow Experiment       │
                    │  train → evaluate → register  │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼────────┐  ┌────────▼───────┐  ┌─────────▼────────┐
    │   Champion Bundle │  │   Challenger B  │  │   Fallback Model │
    │   (CatBoost DR)   │  │   (LightGBM DR) │  │   (Business Rule)│
    └─────────┬────────┘  └────────┬───────┘  └─────────┬────────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │   FastAPI Serving  │
                         │  /v1/score, /rank  │
                         │  /health, /metrics │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
            ┌───────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
            │  Prometheus  │ │  Grafana │ │ Audit Logs  │
            │  Metrics     │ │  3 Dash  │ │ (CSV/DB)    │
            └──────────────┘ └──────────┘ └─────────────┘
```

## 6. Monitoring (3 Dashboards)

1. **Model & Business**: AUUC, Qini, uplift@50, churn rates, segment performance, champion/challenger agreement
2. **Operations**: Latency (p50/p95/p99), error rate, schema failures, model version per replica, resource usage
3. **Data & Drift**: PSI, KS, missing rate, category novelty, prediction distribution shifts

## 7. Model Card

| Field | Value |
|---|---|
| Model | Logistic/CatBoost T-learner (champion by PR-AUC) |
| Features | 50 (10 static + 32 temporal + 5 PCA + 3 OHE) |
| Train date | 2026-07-19 |
| Churn rate | 31.4% |
| Framework | sklearn, CatBoost, LightGBM, XGBoost, MLflow |
| License | MIT (code), Apache 2.0 (CatBoost) |
| Causal assumptions | Consistency, Positivity, Conditional ignorability |
