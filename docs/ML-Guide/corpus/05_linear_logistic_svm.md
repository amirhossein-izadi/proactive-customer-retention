# Linear Regression, Logistic Regression, and SVM: A Deep Dive

These three models are deceptively simple. The interview signal is not whether you know the formulas; it is whether you understand their assumptions, failure modes, and when they are the right choice.

## Linear Regression

### The model

```
y = Xβ + ε
```

Predicted value: `ŷ = Xβ̂`. The model assumes the true target is a linear combination of features plus noise.

### The objective

Minimize the sum of squared errors:

```
L(β) = Σ (y_i - X_i β)² = ||y - Xβ||²
```

### Closed-form solution (the Normal Equation)

```
β̂ = (X^T X)^(-1) X^T y
```

This is the exact minimum. It exists when `X^T X` is invertible (no perfect multicollinearity, enough samples).

### When closed-form is impractical

- Large `p` (many features) or large `n` (many samples): inverting `X^T X` is `O(p³)`.
- Ill-conditioned `X^T X`: numerical instability.
- Streaming data: you cannot recompute the inverse every batch.

In these cases, use **gradient descent**:

```
β_{t+1} = β_t - α * ∇L(β_t)
       = β_t - α * (2 X^T (X β_t - y))
```

### Gradient descent variants

- **Batch GD**: uses all data per step. Slow per step, stable.
- **Stochastic GD (SGD)**: uses one sample per step. Fast per step, noisy.
- **Mini-batch GD**: uses a batch (e.g., 32-512 samples) per step. Industry default.

### Assumptions (Gauss-Markov theorem)

For OLS to give the **Best Linear Unbiased Estimator (BLUE)**, the data must satisfy:

1. **Linearity**: the relationship between features and target is linear.
2. **Independence**: errors are independent across samples.
3. **Homoscedasticity**: errors have constant variance.
4. **No multicollinearity**: features are not perfectly correlated.
5. **No autocorrelation** (for time series).

If these hold, OLS is unbiased and has minimum variance among linear unbiased estimators.

For statistical inference (p-values, confidence intervals on coefficients), you also need:
- **Normality of errors** (for exact p-values; large samples relax this by CLT).

### Regularization

#### Ridge (L2)

```
L(β) = ||y - Xβ||² + λ ||β||²
```

- Shrinks coefficients toward zero, but rarely to exactly zero.
- Good when all features are useful but correlated.
- Solution: `β̂ = (X^T X + λI)^(-1) X^T y` (always invertible).

#### Lasso (L1)

```
L(β) = ||y - Xβ||² + λ ||β||₁
```

- Shrinks some coefficients to exactly zero (feature selection).
- Good for sparse models.
- Unstable when features are highly correlated (picks one arbitrarily).

#### ElasticNet (L1 + L2)

```
L(β) = ||y - Xβ||² + λ₁ ||β||₁ + λ₂ ||β||²
```

- Combines L1 sparsity with L2 stability.
- Good for correlated features and high-dimensional data.

### Diagnostics

| Diagnostic | What it tells you | What to do |
|---|---|---|
| Residual plot | Linearity, homoscedasticity | If curved, add features; if funnel-shaped, transform target |
| Q-Q plot of residuals | Normality of errors | If heavy tails, robust regression or transform |
| Leverage / Cook's distance | Influential points | Investigate outliers, possibly remove |
| VIF (Variance Inflation Factor) | Multicollinearity | Drop or combine features, use Ridge |
| R² / Adjusted R² | Variance explained | Track over iterations, but do not over-interpret |

### Prediction vs Inference

Linear regression is used for two very different purposes:

- **Prediction**: you only care about ŷ. Use cross-validation, regularization, ignore p-values.
- **Inference**: you care about whether feature X_j has a real effect. Use p-values, confidence intervals, residual diagnostics, worry about confounders.

Interviewers love asking about this distinction. Memorize the framing.

## Logistic Regression

### The model

Despite the name, logistic regression is a **classification** algorithm, not a regression algorithm.

```
P(y = 1 | x) = σ(z) = 1 / (1 + exp(-z))
where z = β^T x
```

The sigmoid function σ maps any real number to (0, 1), so the output is a probability.

### The loss: binary cross-entropy (log loss)

For a single example:
```
L = -[y log(p) + (1 - y) log(1 - p)]
```

For N examples:
```
L = -(1/N) * Σ [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

This is the negative log-likelihood under a Bernoulli model.

### Why this loss, not MSE

- The output is a probability, not a continuous value. MSE assumes Gaussian residuals, which is wrong here.
- Log loss is the proper maximum-likelihood objective for classification.
- Log loss heavily penalizes confident wrong predictions, which is what you want.

### Decision boundary

The decision boundary is where P(y=1|x) = 0.5, which is where `z = β^T x = 0`. This is a **linear** boundary in feature space. Logistic regression is a **linear classifier**, no matter how nonlinear the features are.

### Coefficient interpretation

The coefficient `β_j` represents the change in the **log-odds** of the positive class for a one-unit increase in `x_j`:

```
log(p / (1 - p)) = β^T x
```

So `β_j > 0` means increasing `x_j` increases the odds of y=1. To get the multiplicative effect on odds: `exp(β_j)`.

### Multiclass extensions

- **One-vs-Rest (OvR)**: train K binary classifiers, one per class. Pick the class with the highest probability.
- **Multinomial (softmax)**: train one model with K outputs using softmax activation and multi-class cross-entropy. Generally better calibrated.

scikit-learn's `LogisticRegression` defaults to OvR; you can set `multi_class='multinomial'` for softmax.

### Regularization

Same L1/L2 options as linear regression. L1 gives sparse models, L2 is the default and most stable.

### Why logistic regression is a strong production baseline

- Fast to train, fast to predict.
- Outputs calibrated-ish probabilities.
- Easy to debug, easy to interpret.
- Robust to many feature engineering choices.
- Works well as a baseline before trying fancier models.

Interview tip: when asked "what would you do first?", lead with logistic regression. Then justify upgrading to a more complex model only if there is a real reason.

## Support Vector Machines (SVM)

### Core idea

Find the hyperplane that separates classes with the **maximum margin** between them. The margin is the distance from the hyperplane to the nearest points on each side. The points on the margin are the **support vectors**.

### Hard margin SVM

For linearly separable data, solve:
```
min (1/2) ||w||²
subject to: y_i (w^T x_i + b) ≥ 1 for all i
```

The objective minimizes `||w||²`, which is equivalent to maximizing the margin. The constraints ensure all points are correctly classified with margin at least 1.

### Soft margin SVM

Real data is rarely linearly separable. Allow some violations using slack variables `ξ_i ≥ 0`:
```
min (1/2) ||w||² + C Σ ξ_i
subject to: y_i (w^T x_i + b) ≥ 1 - ξ_i for all i
```

The parameter `C` controls the penalty for margin violations:
- **Large C**: heavy penalty, narrow margin, fewer misclassifications (risk: overfit).
- **Small C**: light penalty, wide margin, more misclassifications allowed (risk: underfit).

### The kernel trick

For nonlinear boundaries, replace `x_i · x_j` (dot product) with `K(x_i, x_j)` (kernel function). This implicitly maps data to a higher-dimensional space without computing the mapping.

Common kernels:

| Kernel | Formula | Use when |
|---|---|---|
| Linear | K(x, x') = x · x' | high-dimensional sparse data, text |
| Polynomial | K(x, x') = (γ x · x' + r)^d | moderate nonlinearity, known degree |
| RBF (Gaussian) | K(x, x') = exp(-γ ||x - x'||²) | general nonlinear, local patterns |
| Sigmoid | K(x, x') = tanh(γ x · x' + r) | historically used for NN-like boundaries |

RBF is the default. The `γ` parameter controls how far the influence of a single training example reaches:
- **Large γ**: narrow influence, complex boundary (risk: overfit).
- **Small γ**: wide influence, smoother boundary (risk: underfit).

### Scaling

SVM depends on distances and dot products. If features are on different scales, the kernel measures meaningless distances. **Standardize features before fitting SVM.** This is a common interview trap.

### Strengths

- Effective in high-dimensional spaces (text, genomics).
- Memory efficient: only support vectors matter.
- Works well with clear margin of separation.
- Kernel trick handles nonlinearity flexibly.

### Weaknesses

- Slow on large datasets (`O(n²)` to `O(n³)` for kernel SVM).
- No natural probability output. Need Platt scaling for probabilities.
- Sensitive to hyperparameter `C` and kernel parameters (`γ`, degree).
- Hard to debug and explain.
- Not the first choice for tabular data at scale (gradient boosting usually wins).

### When to use SVM

- Medium-sized dataset (< 100K samples).
- High-dimensional features (text classification, small images).
- Clear margin of separation in the data.
- Need a strong classical baseline with minimal feature engineering.

### When NOT to use SVM

- Very large datasets (use linear models or boosting).
- Need fast inference on millions of predictions (use simpler models).
- Need calibrated probabilities (logistic regression is better by default).
- Tabular data with mixed feature types (gradient boosting is more practical).

## Three-Way Comparison

| Dimension | Linear regression | Logistic regression | SVM |
|---|---|---|---|
| Problem type | regression | classification | classification/regression/outlier detection |
| Output | continuous value | probability-like score via sigmoid/softmax | class score / margin |
| Loss | MSE | log loss / cross-entropy | hinge loss or variants |
| Decision boundary | N/A for regression | linear unless features transformed | linear or nonlinear with kernel |
| Probability quality | N/A | often reasonably calibrated | not naturally probabilistic |
| Outlier sensitivity | high under MSE | medium | support vectors influence boundary |
| Main regularization | Ridge/Lasso/ElasticNet | L1/L2 | C, kernel params |
| Scaling needed | yes for regularization/GD | yes | yes, very important |
| Best use | numeric prediction baseline | interpretable classification baseline | high-dimensional margin-based classification |
| Interview trap | using it for classification probabilities directly | thinking threshold must be 0.5 | thinking kernel SVM scales well to huge data |
