# Validation Strategies and Data Leakage

Validation is the most underappreciated part of ML. A model that "looks great" in offline evaluation but fails in production almost always has a validation problem. This section covers how to validate correctly.

## The Goal of Validation

Validation answers the question: **how will this model perform on data it has not seen?**

If your validation strategy is wrong, the answer is meaningless, and your model selection is a lottery.

## The Three Splits

Most ML projects use three splits:

1. **Training set**: used to fit model parameters.
2. **Validation set**: used to tune hyperparameters and select between models.
3. **Test set**: used to estimate final generalization performance.

Critical rules:
- The test set is touched **once**, at the very end.
- The validation set can be used repeatedly during model development.
- The training set is used for fitting only.

If you tune hyperparameters on the test set, your test estimate is no longer unbiased.

## Cross-Validation

When data is limited, hold out a single validation set wastes data. K-fold cross-validation uses all data for both training and validation, in turns.

### Standard k-fold

1. Split data into k equal folds.
2. For each fold i:
   - Train on the other k-1 folds.
   - Evaluate on fold i.
3. Average the k evaluation scores.

Typical k = 5 or 10. Higher k = lower bias, higher variance, more compute.

### Stratified k-fold

Same as k-fold, but each fold preserves the class distribution. Essential for imbalanced classification.

### Time series CV (rolling/expanding window)

For time series data, you cannot use random k-fold. The future must not leak into the past. Use:

- **Expanding window**: train on [t1, t_k], validate on [t_{k+1}, t_{k+h}]. Then expand the training window.
- **Rolling window**: train on [t_{k-w+1}, t_k], validate on [t_{k+1}, t_{k+h}]. Fixed window.

scikit-learn's `TimeSeriesSplit` implements this.

### Group k-fold

When multiple rows belong to the same entity (user, patient, transaction), random splitting can put the same entity in train and test. This leaks information.

Group k-fold keeps all rows of a group in the same fold. Use for:
- Multiple sessions per user
- Multiple records per patient
- Multiple transactions per account

### Leave-one-out (LOO) CV

Special case of k-fold with k = n. Maximum data usage, but very expensive and high variance.

## Validation Strategy Comparison

| Split strategy | Use when | Pros | Cons | Leakage risk |
|---|---|---|---|---|
| Random split | IID data, no time/user leakage | simple, efficient | unrealistic for temporal problems | medium |
| Stratified split | classification with class imbalance | preserves class ratio | still may leak users/time | medium |
| Time-based split | forecasting, churn, fraud, production prediction | realistic future simulation | less data for training | low if done correctly |
| Group split | multiple rows per user/customer/device | avoids same entity in train/test | fewer independent groups | low for entity leakage |
| K-fold CV | small/medium IID data | efficient estimate | can leak if groups/time ignored | medium-high |
| Nested CV | model selection + unbiased evaluation | cleaner performance estimate | expensive | lower |
| Backtesting | time series | simulates repeated forecasting | more complex | low |
| Online A/B test | final business validation | causal impact | costly, needs infra | low if randomized correctly |

## Data Leakage: The Silent Killer

Data leakage happens when information from outside the training set influences the model. The result is a model that looks great in validation but fails in production.

### Types of leakage

**Train-test contamination**: information from test set used during training (e.g., fit preprocessing on full data before splitting).

**Temporal leakage**: future information used as a feature (e.g., using tomorrow's price to predict today's price).

**Target leakage**: a feature that is a proxy for the target, or that wouldn't be available at prediction time.

**Group leakage**: multiple rows of the same entity in train and test (e.g., same patient in both).

**Preprocessing leakage**: scaler, encoder, or feature selector fit on full data, then applied to train and test separately.

**Cross-validation leakage**: time-based or grouped data shuffled randomly, breaking the data-generating process.

### Common examples

1. **Scaling before splitting**: the test set's statistics influence the scaler. Always split first, then fit the scaler on train only.
2. **Filling missing values with the mean of the full dataset**: same problem. Use train mean only.
3. **Target encoding using the full dataset's target statistics**: same problem.
4. **Features that include the label or its proxy**: e.g., "is_premium" used to predict churn when "is_premium" was assigned based on whether the user churned.
5. **Features that are forward-looking**: e.g., "next_purchase_date" to predict "will_purchase_again."
6. **Same user in train and test**: the model memorizes user-level patterns that don't generalize.
7. **Time-shifted features**: using a rolling average that includes the current time step.

### How to detect leakage

1. **Sanity check feature availability**: at prediction time, will this feature be available?
2. **Check feature importance**: suspiciously high importance on a single feature is a red flag.
3. **Temporal ordering**: ensure all features at time t only use information from times ≤ t.
4. **Group integrity**: check that no group appears in both train and test.
5. **Train-test performance gap**: huge gap often signals leakage.
6. **Permutation test**: shuffle the target, refit; the model should not learn anything. If it does, leakage.

### How to prevent leakage

1. **Split first, then preprocess**: fit the scaler, encoder, imputer on train only. Apply to train and test.
2. **Time-aware features**: only use information from before the prediction time.
3. **Group-aware splits**: ensure entities don't cross train/test.
4. **Use scikit-learn Pipelines**: they chain fit/transform correctly within CV folds.
5. **Be skeptical of very high performance**: if a model is too good, suspect leakage.
6. **Maintain a strict prediction-time feature list**: only allow features that can be computed in production at the prediction moment.

## Validation for Time-Based Problems

Time-based problems need special care because the past is not like the future.

### The wrong way

Random train/test split on time-stamped data. The model sees future patterns during training. It overfits and fails on truly new data.

### The right way

Time-based split: train on older data, validate on newer data. Use a "gap" between train and validation to avoid leakage from target window.

Example: predicting churn in the next 7 days.
- Features: from days [t-30, t-1]
- Label: did user churn in [t, t+7]?
- Train: features from month 1, labels from month 1+7days.
- Test: features from month 2 (later), labels from month 2+7days.

The "gap" matters: if the label window is [t, t+7], you cannot have a test sample at time t+3 that uses a feature at time t+5. There is no leakage if train end < test start.

### Backtesting

For time series, simulate the past:
- Use data up to time T1 to predict T1+1 to T1+h.
- Use data up to time T2 to predict T2+1 to T2+h.
- ...
- Roll forward through time.

This gives a realistic estimate of how the model would have performed in production.

## Validation for Imbalanced Data

Standard accuracy and F1 are unstable on rare classes. Use:

- **Stratified k-fold**: preserves class ratio in each fold.
- **PR-AUC and Fβ**: more informative than accuracy.
- **Repeated CV**: average over multiple shuffles to reduce variance.
- **Class-weighted training**: optional, but consistent with the metric you care about.

## Validation for Production Models

For models that will run in production, validation should simulate production conditions:

1. **Latency budget**: can the model respond within the time limit?
2. **Feature freshness**: are features available when needed, or are they delayed?
3. **Distribution shift**: how does the model perform on data unlike training?
4. **Edge cases**: empty inputs, missing features, rare categories.
5. **Failure modes**: what happens when a downstream service is down?

Shadow mode is the gold standard: run the new model in parallel with the current model, compare predictions, but do not act on them.

## The Validation Checklist

Before considering a model "validated", check:

- [ ] No data leakage (time, group, target, preprocessing).
- [ ] Split is appropriate for the data structure.
- [ ] Test set is held out and untouched.
- [ ] Validation metric matches business objective.
- [ ] Confidence interval on the performance estimate.
- [ ] Performance checked on relevant slices (segments).
- [ ] Performance checked on edge cases.
- [ ] Comparison to baseline (mean, simple model, current production model).
- [ ] Resource use: latency, memory, model size.
- [ ] Reproducibility: same split and seeds produce same result.
- [ ] Test on data from a different time period (generalization check).

If any of these are missing, the model is not validated. It is just trained.
