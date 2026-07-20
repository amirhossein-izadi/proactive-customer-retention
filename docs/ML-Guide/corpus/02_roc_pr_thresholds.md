# ROC Curves, PR Curves, and the Art of Choosing Thresholds

Once you understand precision, recall, and F1, the next layer is what happens when you vary the decision threshold. That is where ROC-AUC and PR-AUC live, and where most interview confusion comes from.

## The Threshold Problem

A classifier that outputs a probability (e.g., logistic regression) does not give you a class label directly. You must pick a threshold, often 0.5, to convert probabilities into class predictions.

But 0.5 is almost never correct in production. The right threshold depends on:
- The cost of false positives
- The cost of false negatives
- The base rate of the positive class
- The business action that follows a "positive" prediction

This is why we need **threshold-independent metrics** like ROC-AUC and PR-AUC, and why we need a principled way to pick a final threshold.

## ROC Curve and ROC-AUC

### What it is

The ROC (Receiver Operating Characteristic) curve plots:
- **Y-axis**: True Positive Rate (TPR) = Recall = TP / (TP + FN)
- **X-axis**: False Positive Rate (FPR) = FP / (TN + FP)

As you sweep the decision threshold from 1.0 (predict everything negative) to 0.0 (predict everything positive), you trace a curve. The area under that curve is **ROC-AUC**.

### What ROC-AUC actually measures

ROC-AUC measures the probability that a randomly chosen positive example is ranked above a randomly chosen negative example. It is a **ranking quality** metric, not a classification metric.

- ROC-AUC = 1.0: perfect ranking. All positives above all negatives.
- ROC-AUC = 0.5: random. No better than a coin flip.
- ROC-AUC < 0.5: anti-ranker (invert predictions to get a good model).

### Why ROC-AUC is popular

- Threshold-independent.
- Single number summary of separability.
- Comparable across models on the same dataset.
- Symmetric in class labels (flipping labels gives the same score in reverse).

### The hidden trap: ROC-AUC under imbalance

When positives are very rare (1% base rate), a model that ranks a few positives well can still get high ROC-AUC even if its absolute precision is bad. The TN dominates the FPR calculation, so FPR stays small even when the model produces many false positives.

Example: 100 positives, 9900 negatives, model with 100 false positives.
- TPR = 80 / 100 = 0.80
- FPR = 100 / 9900 = 0.0101
- ROC-AUC will look great.
- Precision = 80 / 180 = 0.44 (more than half of positive predictions are wrong).

ROC-AUC told you the model is good. But alert volume would be enormous and unsustainable.

## PR Curve and PR-AUC

### What it is

The PR (Precision-Recall) curve plots:
- **Y-axis**: Precision = TP / (TP + FP)
- **X-axis**: Recall = TP / (TP + FN)

As you sweep the threshold, you trace another curve. The area under that curve is **PR-AUC** (also called Average Precision in some implementations).

### What PR-AUC actually measures

It measures how well the model retrieves positives while keeping precision high across different recall levels. It is **sensitive to the positive class quality**.

### Why PR-AUC is better for rare positives

PR-AUC does not involve TN anywhere. This is a feature, not a bug, when positives are rare. A model cannot hide its poor precision behind a huge TN count.

For the same fraud example:
- PR-AUC will visibly drop as the model produces more false positives at low thresholds.
- It reflects what the business actually cares about: when you flag something as fraud, how often are you right?

## The Big Comparison

| Dimension | ROC-AUC | PR-AUC | F1-score |
|---|---|---|---|
| Uses threshold? | No, evaluates across thresholds | No, evaluates across thresholds | Yes, computed at one threshold |
| Main focus | ranking positives above negatives | positive-class retrieval quality | balance of precision and recall at one threshold |
| Axes | TPR vs FPR | Precision vs Recall | single number from precision/recall |
| Sensitive to class imbalance? | Less visibly sensitive; can look high under imbalance | Very sensitive to positive prevalence | Sensitive through precision/recall |
| Includes TN? | Yes, through FPR | No direct TN focus | No |
| Best for | general ranking/separability | rare positive class | deployment decision at a specific threshold |
| Bad for | deciding business threshold | comparing across datasets with different prevalence | comparing ranking quality independent of threshold |
| Example | "Can model rank fraud above non-fraud?" | "When we catch fraud, how many alerts are useful?" | "At this alert threshold, is the tradeoff acceptable?" |
| Interview trap | "High ROC-AUC means production-ready" | "PR-AUC alone gives threshold" | "F1 is always best for imbalance" |

**Rule of thumb:**
- Use **ROC-AUC** to evaluate ranking separability.
- Use **PR-AUC** when positives are rare and alert quality matters.
- Use **F1/Fβ/precision@recall** when you need a concrete operating threshold.

## Threshold Tuning

Picking a threshold is a business decision, not a default 0.5 decision. Here is how you do it properly.

### Method 1: Maximize F1 (or Fβ) on validation

Simple. Pick the threshold that gives the best F1 on a held-out validation set.

Pros: easy, single number.
Cons: ignores true negatives and business cost asymmetry.

### Method 2: Minimize expected business cost

Define:
- `C_FP` = cost of a false positive
- `C_FN` = cost of a false negative

Expected cost at threshold t:
```
Cost(t) = FP(t) * C_FP + FN(t) * C_FN
```

Pick the threshold that minimizes Cost(t) on the validation set.

This is the production-grade method. If you do not know the costs, estimate them from operations. A typical fraud team might say "missing fraud costs us $500 per case; investigating a false alert costs $20." Plug those in.

### Method 3: Fix recall or precision, find the other

If business says "we must catch at least 80% of fraud," set recall = 0.80 and pick the threshold that achieves it. Then look at precision as a secondary metric.

If business says "we can only review 50 alerts per day, given the top of the score distribution," pick the threshold so that only 50 positives are predicted per day.

### Method 4: Cost-sensitive decision rule

If the model outputs calibrated probabilities `p`, predict positive when:
```
p > C_FN / (C_FN + C_FP)
```

This is the Bayes-optimal threshold for cost minimization when probabilities are calibrated.

## Calibration

A model is **calibrated** when its predicted probabilities match observed frequencies.

- Predict 0.80 on 100 cases. About 80 should actually be positive.
- If only 50 are positive, the model is overconfident. If 90 are, the model is underconfident.

**A model can rank well (high ROC-AUC) and be badly calibrated at the same time.** Random forests and gradient boosting are often uncalibrated by default. Logistic regression is usually reasonably calibrated.

### Calibration plots

A reliability diagram plots predicted probability (x-axis) against observed frequency (y-axis). The diagonal is perfect calibration. Curves below the diagonal are overconfident; curves above are underconfident.

### Calibration methods

- **Platt scaling**: fit a logistic regression on top of the model's scores. Works well for SVMs and similar.
- **Isotonic regression**: non-parametric monotonic mapping. More flexible, needs more data.
- **Temperature scaling**: divide logits by a single learned scalar T. Common in deep learning.

### Why calibration matters

- Threshold selection is honest only when probabilities are calibrated.
- Expected value calculations (revenue per prediction, expected loss per default) require calibrated probabilities.
- Combining models (ensembling, blending) needs calibrated outputs.

scikit-learn documents `CalibratedClassifierCV` and probability calibration as a separate concern from ranking, exactly because a model can rank well and be badly calibrated at the same time.

## The Production Workflow for Choosing a Threshold

1. **Train** the model.
2. **Compare ranking metrics**: ROC-AUC, PR-AUC.
3. **Check calibration**: calibration curve, Brier, log loss.
4. **Choose threshold** based on business cost, recall target, or alert budget.
5. **Validate threshold** on untouched validation/test data.
6. **Monitor precision, recall, drift, and delayed labels** after deployment.

If you skip any of these steps, you will eventually ship a model that is "great offline" but terrible in production.
