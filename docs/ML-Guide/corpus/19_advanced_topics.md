# Appendix: Advanced Topics and Common Failure Modes

This appendix covers topics that come up in senior interviews and production work, where the depth of the discussion separates competent candidates from excellent ones.

## Calibration in Depth

### Why calibration matters

A model is calibrated when its predicted probabilities match observed frequencies. Two important failure modes:

1. **The model says "0.8 confidence" for fraud, but only 50% of those flagged cases are actually fraud.** The model is overconfident. The threshold-based business cost calculation is wrong.
2. **The model says "0.3 confidence" for churn, but 60% of those users churn.** The model is underconfident. The retention offer cost is misjudged.

### Platt scaling

Fits a logistic regression on top of the model's scores:

```
P(y=1 | s) = 1 / (1 + exp(A * s + B))
```

where `s` is the model's score and `A`, `B` are learned from a held-out set. Works well for SVMs and similar.

### Isotonic regression

A non-parametric monotonic mapping from score to calibrated probability. More flexible than Platt scaling, but needs more data to fit reliably.

### Temperature scaling

A common technique for neural networks. Divide logits by a learned scalar `T`:

```
P(y | x) = softmax(z / T)
```

`T > 1` softens the probabilities (reduces overconfidence). `T < 1` sharpens. `T` is fit by minimizing NLL on a held-out set.

### When to recalibrate

- After deploying: monitor calibration over time. Recalibrate if drift is detected.
- After retraining: the new model may have different calibration properties.
- When changing decision thresholds: thresholds only make sense for calibrated probabilities.

## Cost-Sensitive Learning

Standard classification assumes equal cost for FP and FN. Real business problems have asymmetric costs.

### Approaches

1. **Class weighting**: weight the loss by inverse class frequency or by cost ratio.
   - scikit-learn: `class_weight='balanced'` or a custom dict.
   - XGBoost: `scale_pos_weight`.
   - PyTorch: pass `weight` to the loss function.

2. **Resampling**: oversample minority, undersample majority, or use SMOTE.
   - Risk: oversampling can cause overfitting; undersampling loses data.

3. **Threshold tuning**: change the decision threshold to optimize business cost.
   - Cheapest to implement, often sufficient.

4. **Custom loss**: write a loss function that encodes the cost structure directly.
   - Most flexible, hardest to maintain.

5. **Focal loss**: down-weights easy examples, focuses on hard ones. Common in object detection.

### Expected cost framework

For threshold `t`:
```
Expected cost = C_FP * FP(t) + C_FN * FN(t)
```

For a calibrated probability `p`, the optimal threshold is:
```
t* = C_FN / (C_FN + C_FP)
```

This is the Bayes-optimal decision rule under cost minimization. In practice, you tune on validation data because the costs are often approximate.

## Multi-Class Classification Strategies

### One-vs-Rest (OvR)

Train K binary classifiers, one per class. Pick the class with the highest probability.

Pros: simple, parallelizable.
Cons: ignores correlations between classes; can be uncalibrated.

### One-vs-One (OvO)

Train K*(K-1)/2 binary classifiers, one per pair of classes. Vote on the final class.

Pros: each classifier sees balanced data.
Cons: many classifiers; slower training and inference.

### Multinomial (softmax)

Single model with K outputs, softmax activation, multi-class cross-entropy.

```
P(y=k | x) = exp(z_k) / Σ exp(z_j)
Loss = -Σ y_k * log(P(y=k))
```

Pros: jointly calibrated, captures class correlations.
Cons: harder to train, needs all classes represented.

### Macro vs Micro Averaging

For multi-class metrics:

**Macro averaging**: compute metric per class, then average.
- Treats all classes equally.
- Sensitive to performance on rare classes.

**Micro averaging**: aggregate TP, FP, FN across classes, then compute metric.
- Treats all instances equally.
- Dominated by majority class.

Choose macro if all classes matter equally; micro if overall accuracy matters most.

## Imbalanced Classification Strategies

### Class weighting

```python
# scikit-learn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# PyTorch
weights = torch.tensor([1.0, 5.0])  # negative, positive
criterion = nn.CrossEntropyLoss(weight=weights)
```

### Resampling

- **Random oversampling**: duplicate minority samples.
- **Random undersampling**: drop majority samples.
- **SMOTE**: synthesize new minority samples by interpolation.
- **ADASYN**: SMOTE variant focused on hard examples.

### Threshold tuning

Pick a threshold that gives the desired precision/recall tradeoff.

### Cost-sensitive learning

Use a custom loss that encodes the asymmetric cost of FP vs FN.

### Anomaly detection framing

If positives are extremely rare, treat as anomaly detection: one-class SVM, isolation forest, autoencoder. The model learns what "normal" looks like and flags deviations.

## Multi-Label Classification

Each sample can have multiple labels (e.g., a movie can be both "action" and "comedy").

### Approaches

- **Binary relevance**: train one binary classifier per label. Ignores label correlations.
- **Classifier chains**: train classifiers sequentially, passing previous predictions as features.
- **Label powerset**: treat each unique label set as a class. Only works for small label spaces.
- **Neural approaches**: multi-label sigmoid output, binary cross-entropy per label.

### Metrics

- **Hamming loss**: fraction of labels predicted wrong.
- **F1 (micro/macro)**: averaged across labels.
- **Subset accuracy**: fraction of samples with all labels correct.
- **Coverage error**: average number of labels needed to cover all true labels.

## Ensemble Methods

### Bagging

Train many independent models on bootstrap samples, average predictions.

Reduces variance, not bias. Best when base model has high variance (e.g., decision trees).

### Boosting

Train models sequentially, each focusing on the previous model's errors.

Reduces bias, can reduce variance with regularization. Best when base model has high bias (e.g., shallow trees).

### Stacking

Train multiple base models, then a meta-model on their predictions.

Most flexible, can capture model complementarity. Risk of overfitting if not careful.

### Blending

Like stacking, but the meta-model is trained on a holdout set.

Simpler than stacking, less prone to overfitting.

### Voting

Hard or soft voting across models. Simple ensemble.

## Feature Engineering Patterns

### Interaction features

```python
df['price_per_sqft'] = df['price'] / df['sqft']
df['income_to_debt_ratio'] = df['income'] / df['debt']
```

### Polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### Target encoding

Replace categorical values with the mean of the target for that category. Needs out-of-fold computation to avoid leakage.

```python
# For each fold, compute target mean using only training data
# Apply to validation
```

### Frequency encoding

Replace categorical values with their frequency in the data.

### Embedding features

Use a pretrained model's embedding as a feature. Common in NLP (sentence embeddings), images (CNN features), or graphs (node embeddings).

### Time-based features

- Hour, day of week, month, season.
- Days since last event.
- Rolling counts, sums, averages over windows.
- Trend, acceleration.

## Handling Missing Data

### Strategies

- **Drop rows**: only if missing is rare and MCAR.
- **Impute with mean/median/mode**: simple, works for low missingness.
- **Impute with model**: more accurate, but more complex.
- **Add "is_missing" indicator**: signal to the model that the value was missing.
- **Use algorithms that handle missing natively**: XGBoost, LightGBM, CatBoost.

### Why imputing with mean is dangerous

- Reduces variance.
- Distorts correlations.
- Treats all "missing" the same way, even if reasons differ.
- May bias the model if missingness is not random.

### When to use what

- Missing < 5%: mean/median imputation is usually fine.
- Missing 5-30%: model-based imputation or add "is_missing" indicator.
- Missing > 30%: investigate why; consider dropping the feature or collecting it differently.

## Outlier Detection and Treatment

### Detection methods

- **Z-score**: |x - mean| / std > 3.
- **IQR method**: outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
- **Isolation Forest**: tree-based anomaly score.
- **Local Outlier Factor (LOF)**: density-based.
- **DBSCAN**: clustering-based.

### Treatment

- **Remove**: if outlier is a data error.
- **Cap (winsorize)**: cap at the 1st/99th percentile.
- **Transform**: log, sqrt, Box-Cox.
- **Robust model**: use MAE instead of MSE, Huber loss, tree-based models.
- **Keep**: if outlier is real and informative (e.g., fraud).

## The Reproducibility Checklist

- [ ] Random seeds set for NumPy, PyTorch, Python `random`.
- [ ] Library versions pinned in `requirements.txt`.
- [ ] Data version recorded (snapshot ID or hash).
- [ ] Code version recorded (git commit hash).
- [ ] Hyperparameters logged.
- [ ] Environment documented (OS, Python, CUDA).
- [ ] End-to-end pipeline runs from raw data to predictions.
- [ ] Experiment tracking (MLflow, W&B) captures everything.

## The Bias-Variance Tradeoff in Production

The bias-variance tradeoff is not just a textbook concept. It is a daily production concern.

**High bias, low variance (underfitting)**:
- Model is too simple for the data.
- Train and val performance are both bad.
- Fix: more features, more complex model, less regularization.

**Low bias, high variance (overfitting)**:
- Model memorizes training data.
- Train performance is great, val performance is bad.
- Fix: more data, regularization, simpler model, dropout, augmentation, early stopping.

**Bias-variance diagnosis**:
- Train score >> val score: overfitting.
- Train score ≈ val score ≈ low: underfitting.
- Train score ≈ val score ≈ high: sweet spot.

The goal is to find the simplest model within one standard deviation of the best model's performance. Senior candidates say this explicitly in interviews.

## The Calibration Curve in Practice

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('Calibration curve')
plt.legend()
plt.show()
```

Read it: a curve below the diagonal is overconfident; a curve above is underconfident.

## The Calibration-Importance Tradeoff

- Logistic regression: well-calibrated by default, but limited capacity.
- Gradient boosting: high capacity, often poorly calibrated.
- Neural networks: high capacity, can be calibrated with temperature scaling.

The pattern: simpler models are better calibrated, complex models are more accurate. Pick based on what matters for the use case.
