# Robustness Comparison by Scenario

Knowing the algorithms is not enough. You also need to know **which algorithm survives which data situation**. This is the table interviewers use to filter senior candidates from junior ones.

## The Scenario Robustness Table

| Scenario | Linear / logistic | SVM | Decision tree | Random forest | Gradient boosting | Neural net |
|---|---|---|---|---|---|---|
| Small data | strong baseline | good if tuned | overfits | good | can overfit | risky |
| Large tabular data | good baseline | kernel SVM bad; linear ok | weak alone | good | excellent | possible but not always best |
| High-dimensional sparse text | very good | very good | weak | often weak | medium | transformer stronger with data |
| Nonlinear interactions | weak unless features engineered | good with kernel | good | good | excellent | excellent |
| Noisy labels | medium | medium-good | weak | good | can chase noise | can overfit |
| Outliers in regression | weak under MSE | medium | medium | good | medium-good | depends on loss |
| Class imbalance | threshold/weights needed | weights needed | biased | weights/sampling needed | weights/threshold needed | weighted loss/sampling needed |
| Need interpretability | excellent | medium | excellent | medium | lower | low |
| Need low latency | excellent | linear SVM good | excellent | medium | medium | depends on size |
| Need calibrated probability | good | needs calibration | weak | often needs calibration | often needs calibration | often needs calibration |
| Need top performance on tabular | medium | medium | low | good | often strongest | mixed |

## Detailed Scenario Analysis

### Small data (n < 10,000)

Best choices: linear/logistic regression, regularized models, random forest with shallow trees, simple SVM.

Why: deep neural networks overfit. Deep trees overfit. Boosting overfits without enough data.

Pitfall: trusting cross-validation scores on small data. The variance of those estimates is huge. Use repeated CV or bootstrap confidence intervals.

### Large tabular data (n > 100,000)

Best choices: gradient boosting (XGBoost/LightGBM/CatBoost), random forest, regularized linear models for very high dimensions.

Why: tree-based models handle heterogeneous features, missing values (in modern implementations), and nonlinear interactions with less manual feature engineering.

Pitfall: not using early stopping, not using proper time/group splits, trusting a single validation score.

### High-dimensional sparse text (e.g., TF-IDF)

Best choices: linear models (logistic regression, linear SVM), Naive Bayes for very small data, fine-tuned transformers if data is large.

Why: linear models on sparse features are very strong, fast, and well-calibrated. Trees struggle because they make axis-aligned splits on sparse data.

Pitfall: trying to use deep learning without enough data. A logistic regression on TF-IDF is a very hard baseline to beat with traditional methods.

### Nonlinear interactions

Best choices: tree-based models (capture interactions natively), kernel SVM, neural networks.

Pitfall: linear models without explicit interaction features will underfit.

### Noisy labels

Best choices: random forest (averaging reduces noise impact), regularized linear models, robust loss functions (Huber, MAE).

Why: gradient boosting will chase noisy labels aggressively, because the residuals include the noise.

Pitfall: training gradient boosting without label-cleaning or robust validation. Boosting models can hit 100% training accuracy and 50% test accuracy on noisy data.

### Outliers in regression

Best choices: MAE, Huber loss, quantile loss, tree-based models (less sensitive).

Pitfall: MSE/RMSE will be dominated by outliers. Even one extreme value can make the model fit the outlier instead of the bulk of the data.

### Class imbalance

Best choices: any model with **proper handling**:
- Class weights in the loss (most libraries support this).
- Resampling (oversample minority, undersample majority, SMOTE).
- Threshold tuning to optimize business cost.
- PR-AUC and Fβ instead of accuracy.

Pitfall: trusting accuracy. With 1% positive rate, predicting all-negative gives 99% accuracy and zero recall.

### Need interpretability

Best choices (in order of interpretability):
1. Linear / logistic regression.
2. Shallow decision tree.
3. Random forest + SHAP or permutation importance.
4. Gradient boosting + SHAP.
5. Neural nets (use SHAP, LIME, attention, integrated gradients).

Pitfall: arguing that a model is "interpretable" just because it is simple. Stakeholders often need feature-level explanations, not just coefficients.

### Need low latency

Best choices:
1. Linear / logistic regression (microseconds).
2. Shallow trees (microseconds).
3. Linear SVM (microseconds).
4. Small neural nets (milliseconds).
5. Distilled models (small model trained to mimic large model).

Pitfall: shipping a 1000-tree gradient boosting model that takes 200ms per prediction. The latency budget may be 10ms.

### Need calibrated probability

Best choices: logistic regression (calibrated by construction), then calibration (Platt or isotonic) on top of other models.

Pitfall: using raw random forest probabilities for cost-sensitive decisions. They are usually uncalibrated.

### Need top performance on tabular

Best choices: gradient boosting (XGBoost/LightGBM/CatBoost). With proper tuning, they almost always beat other methods on structured/tabular data.

Pitfall: not establishing strong baselines first. A well-tuned logistic regression may be 90% as good as gradient boosting with 1% of the complexity.

## The Three Tradeoffs You Must Name

Every modeling decision is a tradeoff between these three:

### 1. Interpretability vs Performance

Linear models are interpretable but limited. Deep models are powerful but opaque. Random forest is in between.

**Senior answer**: "I will use linear models as a baseline. If performance gain from a more complex model is X%, and X% matters for the business, I will use the complex model with SHAP explanations. Otherwise, I will keep it simple."

### 2. Latency vs Accuracy

A 1000-tree boosting model is more accurate than a 50-tree model, but 20x slower.

**Senior answer**: "I will measure the marginal accuracy gain per additional tree, and pick the smallest model that meets the business accuracy target."

### 3. Variance vs Bias

Complex models have low bias, high variance. Simple models have high bias, low variance.

**Senior answer**: "I will use cross-validation to estimate the variance of my model and pick the simplest model whose performance is within one standard deviation of the best model."

## Special Case: The Adversarial Setting

Fraud, abuse, content moderation, and security are adversarial. The distribution shifts because bad actors adapt to the model.

What changes:
- Labels are delayed and incomplete (you only know about caught fraud, not all fraud).
- Models decay faster than in non-adversarial settings.
- Retraining cadence must be higher.
- Drift detection is harder because the "concept" itself is changing.
- Online learning or frequent retraining is often required.
- Shadow mode and A/B tests must run longer to capture enough positive examples.

If you are interviewing for a role that touches fraud or abuse, mention these factors. They signal real production experience.

## Special Case: The Rare-Event Setting

Medical diagnosis, equipment failure, conversion prediction, and rare disease detection are imbalanced by nature.

What changes:
- Accuracy is useless. PR-AUC, recall@precision, Fβ.
- Threshold tuning is critical.
- Class weighting or sampling is needed.
- Validation split must preserve class ratio.
- Often need calibration to set business thresholds.

## What This Section Teaches You

Every algorithm has strengths and weaknesses. The job is to match the algorithm to the situation. The interview is testing whether you can reason about the match, not whether you memorized hyperparameters.

A good answer pattern: "Given [data situation] and [business priority], I would use [algorithm] because [strength], watching out for [weakness], and validating with [split strategy] using [primary metric]."
