# Evaluation Metrics Foundations: The Confusion Matrix and Everything Built On It

This is the most important starting point because every model-development interview eventually asks: "How do you know the model is good?" If you cannot answer that crisply, nothing else you say about algorithms will land.

## The Confusion Matrix

Every classification metric comes from this 2x2 table.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

Read it carefully:

- **TP**: model said positive, reality was positive. Correct and useful.
- **TN**: model said negative, reality was negative. Correct and useful, but often ignored.
- **FP** (Type I error): model said positive, reality was negative. False alarm.
- **FN** (Type II error): model said negative, reality was positive. Missed case.

The cost of FP and FN is rarely equal. A spam filter that misses a spam email is cheap; a cancer screener that misses a tumor is dangerous. Metrics exist to make these tradeoffs explicit.

## The Core Classification Metrics

### Accuracy

```
Accuracy = (TP + TN) / (TP + FP + TN + FN)
```

- **What it asks**: "What fraction of predictions were correct overall?"
- **When it works**: balanced classes, equal error costs.
- **When it is dangerous**: imbalanced data. If only 1% of users churn, a model that predicts "no churn" for everyone has 99% accuracy and zero business value.
- **Interview sentence**: "Accuracy is only reliable when class distribution and error costs are not pathological."

### Precision

```
Precision = TP / (TP + FP)
```

- **What it asks**: "When the model predicts positive, how often is it correct?"
- **Use when**: false positives are expensive. Example: spam filter marking legitimate email as spam.
- **What it ignores**: false negatives. A model with high precision may still miss most real positives.

### Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

- **What it asks**: "Of all real positives, how many did we catch?"
- **Use when**: false negatives are expensive. Example: cancer screening, fraud catching.
- **What it ignores**: false positives. A model with high recall may flood you with false alarms.
- **Note**: Recall and Sensitivity are the same thing. TPR = Recall.

### Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

- **What it asks**: "Of all real negatives, how many did we reject?"
- **Use when**: false positives matter. Example: a system that should only flag true fraud.
- **Interview insight**: Specificity is recall for the negative class.

### F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

- **What it is**: the harmonic mean of precision and recall.
- **Why harmonic mean instead of arithmetic mean**: the harmonic mean punishes imbalance between precision and recall. If either is 0, F1 is 0. Arithmetic mean would let a perfect 1.0 in one dimension mask a 0.0 in the other.
- **What it ignores**: true negatives. F1 says nothing about how well you handle the negative class.
- **Limitation**: requires a chosen threshold. F1 at threshold 0.5 is not the same as F1 at threshold 0.3.
- **scikit-learn reference**: `sklearn.metrics.f1_score` computes this with the harmonic mean definition, treating precision and recall with equal weight.

### Fβ-Score

```
Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```

- **What it is**: a generalized F-score that weights recall and precision differently.
- **β > 1**: recall matters more (e.g., β=2 weighs recall twice as much as precision).
- **β < 1**: precision matters more (e.g., β=0.5 weighs precision twice as much as recall).
- **β = 1**: equal weighting, reduces to F1.
- **Interview sentence**: "Use β>1 when missing a positive is more expensive than a false alarm; use β<1 when the reverse is true."

### Balanced Accuracy

```
Balanced Accuracy = (Recall + Specificity) / 2
                  = (TP / (TP+FN) + TN / (TN+FP)) / 2
```

- **What it is**: average of recall on each class.
- **When to use**: imbalanced classification where raw accuracy is misleading.
- **Insight**: a model that always predicts the majority class has 0% recall on the minority class, so balanced accuracy is at most 0.5 for such a model.

### Logarithmic Loss (Log Loss / Cross-Entropy)

```
LogLoss = - (1/N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

- **What it asks**: how well do predicted probabilities match actual outcomes?
- **Why use it**: when you care about the *confidence* of predictions, not just the class label.
- **Property**: heavily penalizes confident wrong predictions. Predicting 0.99 for a true 0 is much worse than predicting 0.51.
- **Use when**: probabilities feed into business decisions (e.g., expected loss, expected revenue per prediction).

### Brier Score

```
Brier = (1/N) * Σ (p_i - y_i)²
```

- **What it is**: mean squared error between predicted probability and actual outcome (0 or 1).
- **Use when**: probability calibration matters more than ranking.
- **Difference from log loss**: Brier is less aggressive about penalizing confident wrong predictions, but they often rank models similarly.

## The Single Most Important Rule

**For imbalanced classification, never say "accuracy is high, so the model is good."**

For fraud, churn, disease, abuse detection, search relevance, or rare-event prediction, you must discuss:
- precision
- recall
- PR-AUC
- thresholding
- business cost

If you only mention accuracy on an imbalanced problem, the interviewer marks you as junior immediately. Always pair the metric with the business cost structure.

## Why F1 Ignores True Negatives

F1 is built from precision and recall, both of which only involve positives (TP, FP, FN). True negatives never enter the formula.

Implication: a model can score very high F1 on a class-imbalanced problem while being terrible at confirming "no fraud" or "no churn." This is fine if you only care about the positive class, but dangerous if you ever argue "the model is good" without specifying the cost structure.

A more complete metric for class-imbalanced settings is **Matthews Correlation Coefficient (MCC)**, which uses all four cells of the confusion matrix. It returns a value between -1 and +1, where +1 is perfect prediction, 0 is no better than random, and -1 is total disagreement.

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

When you want one number that respects class imbalance and uses the full confusion matrix, MCC is often a better choice than F1.
