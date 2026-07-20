# Model Card — Didar Retention Uplift Model

## Model Overview

| Field | Value |
|---|---|
| **Model name** | `didar-retention-uplift-v1` |
| **Model type** | CatBoost T-learner + Doubly Robust Learner |
| **Task** | CATE estimation (individual treatment effect) |
| **Target** | `τ(x) = P(churn | no contact, x) − P(churn | contact, x)` |
| **Framework** | scikit-learn, CatBoost, MLflow |
| **License** | MIT (code), CatBoost Apache 2.0 |

## Intended Use

- Rank customers by predicted churn reduction from proactive contact
- Select top 50 customers for retention campaigns
- **Not** for autonomous decision-making without human review
- **Not** for applications requiring guaranteed causal inference

## Training Data

| Source | Rows | Columns |
|---|---|---|
| `customer_data.csv` | 4,000 | 10 (incl. treatment & outcome) |
| `usage_timeseries.csv` | 240,000 | 3 (60 days per customer) |

- **Churn rate**: 31.4% | **Treatment rate**: 48.5%
- **Split**: 80/20 stratified by `(contacted, churned)`, 5-fold CV

## Features

34 features across families:
- **Static**: log(company_size), log(revenue), revenue_per_employee, tenure_years, is_annual, tickets_per_revenue
- **Usage level**: mean, median, std, cv, recent_7/14/30d means, early_30d mean
- **Trend**: 60d/30d slope, recent_minus_early, recent_early_ratio, last_day_usage
- **Inactivity**: zero_days, active_days_pct, longest_zero_streak, days_since_activity
- **Volatility**: down_day_proportion, max_drawdown

## Performance

| Metric | Value |
|---|---|
| Churn model ROC-AUC | ~0.80 |
| Mean uplift@50 | ~0.011 |
| Mean p₀@50 (no contact risk) | ~0.98 |
| Mean p₁@50 (contact risk) | ~0.96 |
| Brier score (p₀) | ~0.00 (near-perfect separation) |

## Causal Assumptions (Required for Interpretation)

1. **Consistency**: The observed contact matches the intervention being modeled
2. **Positivity**: Every customer has non-zero probability of both contact and no-contact
3. **Conditional ignorability (unconfoundedness)**: No unmeasured confounders affect both contact assignment and churn

**Limitation**: Assumption 3 is the weakest — account manager judgment, relationship quality, and unresolved complaints are not in the dataset. The uplift estimate is observational, not experimental.

## Ethical Considerations

- The model may systematically deprioritize certain segments (e.g., low-revenue customers)
- Contact effectiveness may vary by channel (call, email, etc.) — binary treatment masks this
- Monitor for feedback loops: retraining on model-driven contact decisions can create self-fulfilling predictions

## Monitoring

Three dashboards:
1. **Model & Business**: AUUC, Qini, uplift@50, segment performance
2. **Operations**: latency, error rate, throughput
3. **Data & Drift**: PSI, KS, missing rate, OOD rate

## Version History

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-07-19 | Initial training pipeline + top-50 ranking |
