# Regression Metrics and Metric Decision Tables

Classification gets most of the attention, but a large fraction of business ML is regression: predicting price, demand, revenue, duration, lifetime value, click probability (in some setups), and so on. You need to know these metrics as well as the classification ones.

## Core Regression Metrics

### MAE — Mean Absolute Error

```
MAE = (1/N) * Σ |y_i - ŷ_i|
```

- **What it measures**: average absolute deviation from true value.
- **Unit**: same as target. Easy to interpret ("off by $5 on average").
- **Robustness**: less sensitive to outliers than MSE/RMSE.
- **Gradient**: not differentiable at 0, so requires special handling in some optimization contexts (subgradient).
- **Use when**: business cares about typical error magnitude and you want interpretability.

### MSE — Mean Squared Error

```
MSE = (1/N) * Σ (y_i - ŷ_i)²
```

- **What it measures**: average squared deviation.
- **Unit**: squared. Hard to interpret directly. Use RMSE instead.
- **Robustness**: very sensitive to outliers because errors are squared.
- **Use when**: large errors are disproportionately bad, or when optimization theory is the priority (MSE gives nice gradients).

### RMSE — Root Mean Squared Error

```
RMSE = sqrt(MSE)
```

- **What it measures**: square root of MSE, in the same unit as target.
- **Property**: still penalizes large errors more than small ones.
- **Use when**: you want MAE's interpretability but still want to penalize large errors.
- **Trap**: if the data has one huge outlier, RMSE will be dominated by it.

### MAPE — Mean Absolute Percentage Error

```
MAPE = (1/N) * Σ | (y_i - ŷ_i) / y_i |  * 100%
```

- **What it measures**: average percentage error.
- **Unit**: percentage. Friendly to non-technical stakeholders.
- **Trap 1**: explodes when y is near zero. Dividing by tiny numbers creates huge errors.
- **Trap 2**: asymmetric. Predicting 10 when truth is 20 gives 50% error. Predicting 30 when truth is 20 also gives 50% error. Business may care that these are not equivalent.
- **Trap 3**: if some y are zero or negative, MAPE is undefined or misleading.
- **Use when**: targets are strictly positive, business thinks in percentages, and you can handle the asymmetry.

### RMSLE — Root Mean Squared Logarithmic Error

```
RMSLE = sqrt( (1/N) * Σ (log(1 + y_i) - log(1 + ŷ_i))² )
```

- **What it measures**: RMSE in log space.
- **Property**: penalizes under-prediction more than over-prediction.
- **Use when**: target spans several orders of magnitude and under-predicting large values is worse than over-predicting. Common in demand forecasting, ride-share ETA, insurance claims.

### R² — Coefficient of Determination

```
R² = 1 - SS_res / SS_tot = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
```

- **What it measures**: fraction of variance explained by the model, compared to predicting the mean.
- **Range**: can be negative (model worse than mean), 0 (model same as mean), 1 (perfect).
- **Trap**: R² is not a business metric. A high R² does not mean the model is making money.
- **Trap**: R² always increases when you add features, even useless ones. Use Adjusted R² for variable selection:
```
Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
```
where p is number of features.

### Quantile Loss (Pinball Loss)

```
QuantileLoss(τ) = (1/N) * Σ [τ * max(y_i - ŷ_i, 0) + (1 - τ) * max(ŷ_i - y_i, 0)]
```

- **What it measures**: asymmetric loss for quantile estimation.
- **Use when**: you want a prediction interval, not just a point estimate. τ=0.5 is the median.
- **Why it matters in business**: inventory, capacity planning, risk management all need quantile forecasts ("90th percentile demand"), not just point forecasts.

## The Metric Selection Table

| Situation | Primary metric | Secondary metrics | Why |
|---|---|---|---|
| Balanced classification | accuracy, macro F1 | ROC-AUC, log loss | accuracy is acceptable if costs are equal |
| Rare fraud detection | PR-AUC, recall@precision, precision@k | ROC-AUC, calibration | positives are rare; false alerts are costly |
| Medical screening | recall/sensitivity, Fβ with β>1 | specificity, PR-AUC | missing positives is dangerous |
| Spam filtering | precision, false positive rate | recall, PR-AUC | false positives may hide real emails |
| Credit default | ROC-AUC, KS, calibration | PR-AUC, expected profit | ranking and calibrated risk both matter |
| Search/recommendation | NDCG, MRR, MAP | CTR/conversion online | order matters more than class label |
| Model used for probability | log loss, Brier, calibration | ROC-AUC | probability quality matters |
| Model used for manual review queue | precision@k, recall@k | PR-AUC | humans inspect top-k cases |
| Severe class imbalance | PR-AUC, Fβ, recall@precision | balanced accuracy | accuracy is misleading |
| Business-cost-sensitive case | expected cost/profit | precision, recall | metric should encode cost matrix |
| Price/demand prediction | MAE, MAPE | RMSE, R² | business cares about typical error or percentage |
| Outlier-heavy target | MAE, Huber | RMSE for comparison | MSE/RMSE explode with outliers |
| Wide-range positive target (e.g., demand) | RMSLE | MAE, RMSE | log-space error is more meaningful |
| Forecasting with intervals | quantile loss / pinball | coverage, CRPS | business needs ranges, not just points |
| Capacity planning | quantile loss (90th percentile) | service level achieved | need to plan for high demand days |

## Ranking Metrics: A Quick Note

For search, recommendation, and ranking problems, the labels are ordinal or relative, not categorical. The relevant metrics are:

### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality. Higher-ranked correct items count more. Normalized to [0, 1]. Use NDCG@k for top-k evaluation.

### MAP (Mean Average Precision)

Average of precision values at each rank where a relevant item appears. Useful when you care about the position of relevant items.

### MRR (Mean Reciprocal Rank)

1 / (rank of first relevant item), averaged over queries. Use when only the first correct answer matters.

### Hit Rate@k / Recall@k

Fraction of users/queries for which at least one relevant item appears in top-k. Simple and interpretable.

These metrics are essential for search/recommendation systems. In an interview for a role that touches ranking, you must know them.

## Why You Should Not Just Optimize One Metric

A model optimized for one metric often fails on another:

- Optimize accuracy: gets stuck predicting majority class.
- Optimize ROC-AUC: may produce high alert volume, hurting precision.
- Optimize RMSE: overweights outliers, may hurt MAE.
- Optimize R²: increases with more features, even useless ones.
- Optimize log loss: requires calibration, may sacrifice ranking quality.
- Optimize MAPE: may produce unstable predictions for low-volume cases.

Production teams track a **primary metric** (the decision metric) and **guardrail metrics** (the things that must not degrade). Example: optimize revenue per user (primary), guardrail is page load time and customer support tickets.

## The Big Idea

**A metric is not a property of the model. A metric is a property of the model, the data, the threshold, and the business cost structure.** If you change any of those, the right metric may change too.

That is the level of nuance interviewers are looking for.
