# Decision Trees, Bagging, and Boosting: The Tree Family

Tree-based models are the workhorses of modern tabular ML. Most production Kaggle wins and most deployed business models are some form of gradient boosting. You need to know this family inside and out.

## Decision Trees

### How a tree is built

A decision tree recursively partitions the feature space using axis-aligned splits. At each node, the algorithm:

1. Considers every possible split on every feature.
2. Picks the split that most reduces impurity.
3. Recurses on each child node.
4. Stops when a stopping criterion is met (max depth, min samples, pure node).

The result is a piecewise-constant predictor: each leaf predicts a single value (mean for regression, majority class for classification).

### Splitting criteria

#### For classification

**Gini impurity** (used by default in scikit-learn):
```
Gini = 1 - Σ p_k²
```
where `p_k` is the proportion of class k in the node. Gini = 0 means pure node. Lower is better.

**Entropy / Information gain** (used by `criterion='entropy'`):
```
Entropy = -Σ p_k log(p_k)
```
Information gain is the reduction in entropy from a split. Both Gini and entropy usually pick similar splits; Gini is faster to compute.

#### For regression

**MSE reduction** (used by `DecisionTreeRegressor`):
```
MSE_node = (1/n) * Σ (y_i - ȳ_node)²
```
Pick the split that minimizes the weighted average MSE of the children.

Other options: **MAE reduction** (more robust to outliers), **Friedman MSE** (approximation that often works as well).

### Overfitting and how to control it

Decision trees have **high variance**: small changes in data can produce very different trees. A fully grown tree memorizes the training set.

Controls:

| Hyperparameter | What it does | Effect of increasing |
|---|---|---|
| `max_depth` | maximum tree depth | deeper = more overfitting, more interactions |
| `min_samples_split` | minimum samples to split a node | higher = less overfitting, smoother boundary |
| `min_samples_leaf` | minimum samples in a leaf | higher = less overfitting, less granularity |
| `max_features` | features considered per split | lower = more randomness, more variance reduction |
| `max_leaf_nodes` | maximum number of leaves | fewer = simpler tree, more bias |
| `min_impurity_decrease` | minimum impurity reduction to split | higher = fewer splits, simpler tree |
| `ccp_alpha` | cost-complexity pruning parameter | higher = more aggressive pruning |

Post-pruning (cost-complexity pruning) is often more effective than pre-pruning: grow the full tree, then collapse weak branches.

### Pros and cons

**Pros:**
- Easy to understand and visualize.
- Handles nonlinear relationships and feature interactions automatically.
- Handles both numerical and categorical features (with proper encoding).
- No feature scaling required.
- Makes few assumptions about the data distribution.
- Fast inference.

**Cons:**
- High variance: unstable to small data changes.
- Easily overfits without regularization.
- Greedy: each split is locally optimal, not globally optimal.
- Biased toward features with more levels (more possible splits).
- Can be biased to majority class without adjustment.

### Missing values

scikit-learn's `DecisionTreeClassifier` does not natively handle NaN (until recent versions). In practice, you either:
- Impute missing values before training.
- Use a library that handles them (XGBoost, LightGBM, CatBoost).
- Treat "missing" as its own category.

## Random Forest (Bagging Trees)

### Core idea

Train many decision trees on **bootstrap samples** of the data, with **random feature subsets** at each split, and **average** their predictions. This reduces variance dramatically.

### Algorithm

For B trees:
```
For b = 1 to B:
    1. Draw a bootstrap sample of size n from the training data (sample with replacement).
    2. Grow a tree on this sample, but at each split:
       - Consider only a random subset of m features (m = sqrt(p) for classification, m = p/3 for regression).
       - Pick the best split among these m features.
    3. Do not prune. Let each tree overfit.

Predict:
    Classification: majority vote across B trees.
    Regression: average prediction across B trees.
```

### Why it works

Each individual tree has high variance and low bias. Averaging many uncorrelated trees:
- Reduces variance by factor of ~1/B (if trees are independent).
- Keeps bias the same (each tree is a flexible model).
- Result: low variance, low bias, low error.

The randomness (bootstrap + feature subsampling) **decorrelates** the trees. Without that, all trees would make the same splits on the same dominant features.

### Hyperparameters

| Hyperparameter | Effect |
|---|---|
| `n_estimators` (B) | number of trees. More = better (diminishing returns) and slower |
| `max_features` | randomness at splits. Lower = more decorrelated, more variance reduction |
| `max_depth` | depth of individual trees. Usually left unconstrained; bagging handles overfitting |
| `min_samples_leaf` | smoothing. Higher = simpler trees, faster training |
| `bootstrap` | whether to sample with replacement. True is standard |
| `oob_score` | use out-of-bag samples to estimate generalization error (free cross-validation) |
| `n_jobs` | parallelization. Set to -1 for all cores |

### Out-of-bag (OOB) error

Each tree is trained on ~63% of the data (the bootstrap sample). The remaining ~37% (the OOB samples) can be used to estimate test error without a separate validation set. This is a free, built-in cross-validation when `oob_score=True`.

### Feature importance

Random forests can report two kinds of importance:

1. **Gini importance (impurity-based)**: total reduction in impurity from splits on each feature, averaged across trees. Fast but biased toward high-cardinality features.
2. **Permutation importance**: shuffle one feature's values, measure how much error increases. Slow but more reliable.

Use permutation importance when you need trustworthy explanations.

### scikit-learn framing

scikit-learn describes random forests and ExtraTrees as randomized "perturb-and-combine" tree ensembles whose predictions average individual trees. The `RandomForestClassifier` documentation explicitly describes fitting many decision trees on sub-samples and averaging to improve predictive accuracy and control overfitting.

That framing is the production-grade answer. Random forest is not magic; it is a variance-reduction wrapper around decision trees.

### When to use random forest

- Tabular data with mixed feature types.
- Need a strong baseline with low tuning effort.
- Want built-in feature importance.
- Need OOB error estimates without cross-validation.
- Data is small to medium (under ~100K samples). For larger data, gradient boosting usually wins.

## Gradient Boosting

### Core idea

Sequentially add weak learners (usually shallow trees) to an ensemble, where each new learner fits the **residuals** (or gradients of the loss) of the current ensemble. This reduces both bias and variance.

### Algorithm (regression with MSE)

```
Initialize F_0(x) = mean(y)

For m = 1 to M:
    1. Compute residuals: r_i = y_i - F_{m-1}(x_i)
    2. Fit a shallow tree h_m to predict r_i from x_i
    3. Update: F_m(x) = F_{m-1}(x) + ν * h_m(x)
       where ν is the learning rate (shrinkage)

Final: F_M(x) = F_0(x) + ν * Σ h_m(x)
```

### Why it works

Each new tree is fit to the errors of the current model. The ensemble iteratively reduces the loss, so the final model is much stronger than any individual tree.

### Generalization to arbitrary losses

The "residuals" in the algorithm above are actually the **negative gradients** of the loss with respect to the predictions. For MSE, gradient = residual. For log loss, gradient = (p - y). For MAE, gradient = sign(y - p).

This makes gradient boosting a generic framework: pick a loss, compute its gradient, fit a tree to the gradient.

### Learning rate and number of trees

There is a tradeoff between `ν` (learning rate) and `M` (number of trees):
- Small `ν` + many trees = better generalization, slower training.
- Large `ν` + few trees = faster training, worse generalization.

A common starting point: `ν = 0.1`, then use early stopping to pick `M`.

### Early stopping

Split training data into train + validation. After each new tree, evaluate on validation. Stop when validation loss stops improving for some number of rounds (e.g., 50). This is the single most important regularization technique for gradient boosting.

### Major implementations

| Library | Strengths | Notes |
|---|---|---|
| **XGBoost** | Mature, fast, well-documented | Industry default; L1+L2 regularization built in |
| **LightGBM** | Very fast on large data, histogram-based | Leaf-wise growth; great with categorical features |
| **CatBoost** | Best native categorical handling, ordered boosting | Good default for mixed-type tabular data |
| **scikit-learn GradientBoosting** | Simple, no extra dependencies | Slow on large data; good for learning |

### Hyperparameters

| Hyperparameter | Effect | Tuning guidance |
|---|---|---|
| `n_estimators` (M) | number of trees | use early stopping, start at 1000 |
| `learning_rate` (ν) | shrinkage per tree | 0.01 to 0.1; smaller needs more trees |
| `max_depth` | depth of each tree | 3 to 8 for shallow trees; lower = more bias, more robust |
| `min_samples_leaf` | minimum samples per leaf | 1 to 50; higher = more regularization |
| `subsample` | fraction of samples per tree | 0.5 to 1.0; < 1.0 adds stochasticity |
| `colsample_bytree` | fraction of features per tree | 0.5 to 1.0; reduces overfitting |
| `reg_alpha` (L1) | sparsity regularization | 0 to 1; rarely needs tuning |
| `reg_lambda` (L2) | shrinkage regularization | 0 to 10; default usually fine |

### When to use gradient boosting

- Tabular data with high signal-to-noise.
- Need top performance on classification/regression.
- Willing to spend time tuning and validating.
- Can afford sequential training (boosting does not parallelize across trees).

### When NOT to use gradient boosting

- Noisy labels: boosting will chase the noise.
- Time-based data with leakage risk: boosting exploits leakage ruthlessly.
- Real-time strict latency: a 1000-tree model is slow.
- Tiny datasets: a single tree or linear model may be enough.

## Bagging vs Boosting: The Comparison You Must Master

| Dimension | Bagging | Boosting |
|---|---|---|
| Full name | bootstrap aggregating | sequential additive modeling |
| Training | parallel/independent models | sequential models |
| Each model sees | bootstrap sample / randomized features | reweighted/residual-focused data |
| Main goal | reduce variance | reduce bias, improve difficult examples |
| Typical example | random forest | gradient boosting, AdaBoost, XGBoost/LightGBM |
| Noise robustness | usually better | can overfit noisy labels |
| Imbalance | not solved automatically | not solved automatically, but weighting helps |
| Interpretability | medium | lower |
| Hyperparameter sensitivity | lower | higher |
| Best when | base model has high variance | baseline underfits or needs stronger nonlinear power |
| Production risk | model size/latency | overfit/leakage/sensitivity |
| Interview sentence | "Bagging stabilizes unstable learners." | "Boosting sequentially corrects errors/residuals." |

### The fundamental difference

Bagging says: "I have an unstable model with high variance. Let me train many of them on different data and average. Variance drops, bias stays."

Boosting says: "I have a weak model with high bias. Let me train it, see where it fails, and train another model focused on those failures. Bias drops, variance may rise but I regularize."

In production, you usually try both, then pick the one that performs better on held-out validation.

## Decision Tree vs Random Forest vs Gradient Boosting

| Dimension | Decision tree | Random forest / bagging | Gradient boosting |
|---|---|---|---|
| Training style | one tree | many independent trees | trees added sequentially |
| Main error reduction | none alone | reduces variance | reduces bias, can also reduce variance |
| Overfitting risk | high | lower than one tree | high if too many trees/depth/learning rate bad |
| Noise sensitivity | high | medium-low | medium-high, can chase noisy labels |
| Interpretability | high | medium | lower |
| Tuning complexity | low-medium | medium | high |
| Handles nonlinear interactions | yes | yes | yes, very well |
| Handles scaling | no scaling needed | no scaling needed | no scaling needed |
| Works well on tabular data | sometimes | yes | very often excellent |
| Training speed | fast | parallelizable | sequential, can be slower |
| Production latency | very fast | can be heavier | can be heavy depending on number/depth |
| Best first use | explainable baseline | robust tabular baseline | performance-oriented tabular model |
| Main failure | unstable/memorizes | less interpretable, big model | leakage/noise overfit, tuning burden |

## Practical Advice for Interviews

When asked "which tree model would you use?", give a structured answer:

1. **"It depends on the data and the priority."**
2. Start with a **single decision tree** as a sanity-check baseline.
3. Try **random forest** as a robust nonlinear baseline.
4. Try **gradient boosting** if performance is the priority and you have time to tune.
5. Use **early stopping** to avoid overfitting.
6. Validate with the right split (time-based for temporal problems, group-based for entity leakage).
7. Monitor performance and drift after deployment.

That sequence tells the interviewer you understand the progression, the tradeoffs, and the production reality.
