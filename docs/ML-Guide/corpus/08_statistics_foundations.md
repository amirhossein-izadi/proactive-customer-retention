# Statistics Foundations for ML Interviews

The job description says "Solid foundation in statistics, including hypothesis testing, experimental design, and data interpretation." This is not optional. Most candidates fail statistics questions not because they are hard, but because they are taught badly.

This section builds statistics from the ground up.

## Core Vocabulary

### Population

The full group you care about. Examples:
- All users of an app
- All transactions in a year
- All customers of a company

The population is usually too large or too expensive to measure directly.

### Sample

The observed subset of the population. The training data is a sample. The A/B test data is a sample.

Key insight: every statistical conclusion is a claim about the population, derived from the sample, with uncertainty quantified by statistics like standard error.

### Parameter

A true unknown quantity describing the population. Examples:
- True mean revenue per user
- True conversion rate
- True treatment effect of a new model

Parameters are fixed but unknown. We never observe them directly.

### Statistic

A quantity computed from the sample. Examples:
- Sample mean
- Sample conversion rate
- Sample coefficient in regression

Statistics are random (they change with the sample). Parameters are fixed.

### Distribution

A description of the possible values a random variable can take, and their probabilities. Examples:
- Normal (Gaussian) distribution
- Bernoulli distribution (binary outcomes)
- Poisson distribution (counts)

## Descriptive Statistics: The Basics

### Measures of center

- **Mean**: `μ = (1/n) Σ x_i`. Sensitive to outliers.
- **Median**: middle value when sorted. Robust to outliers.
- **Mode**: most frequent value. Useful for categorical data.

When the distribution is symmetric, mean ≈ median. When skewed, they differ; median is usually more representative of "typical."

### Measures of spread

- **Variance**: `σ² = (1/n) Σ (x_i - μ)²`. Average squared deviation.
- **Standard deviation**: `σ = sqrt(σ²)`. Same unit as the data.
- **Range**: max - min. Sensitive to extremes.
- **IQR (interquartile range)**: 75th percentile - 25th percentile. Robust spread measure.
- **MAD (median absolute deviation)**: median(|x_i - median|). Very robust.

### Skewness and kurtosis

- **Skewness**: measures asymmetry. Positive = right tail, negative = left tail.
- **Kurtosis**: measures tail heaviness. High kurtosis = more extreme outliers.

For business data, distributions are often right-skewed (revenue, order value, session duration). Reporting the mean is misleading; report the median or geometric mean.

## The Normal Distribution

The normal (Gaussian) distribution is the most important distribution in classical statistics. It has:

- Mean μ and standard deviation σ
- Symmetric bell shape
- 68% of mass within 1σ, 95% within 2σ, 99.7% within 3σ (the "68-95-99.7 rule")

### Why it matters

- The Central Limit Theorem (CLT) says the sampling distribution of the mean approaches normal as sample size grows, regardless of the underlying distribution.
- Many statistical tests assume normality (or rely on CLT to make it approximately true).
- Many ML models assume Gaussian residuals (linear regression) or work best with roughly Gaussian features.

### When the normal assumption fails

- Heavy-tailed data (financial returns, response times): use t-distribution or robust methods.
- Count data (number of events): use Poisson or negative binomial.
- Proportions near 0 or 1: use beta distribution.
- Strictly positive skewed data: use log-normal, gamma.

For ML, you can usually get away with non-normal features as long as the model is flexible enough. The normality assumption matters most for **statistical inference** (p-values, confidence intervals).

## Sampling Distribution and Standard Error

### Sampling distribution

The distribution of a statistic over hypothetical repeated samples from the same population.

Example: if you sample 1000 users 1000 times and compute the mean revenue each time, you get 1000 sample means. Their distribution is the sampling distribution of the mean.

### Standard error

The standard deviation of the sampling distribution. For the sample mean:
```
SE = σ / sqrt(n)
```

Key insight: the standard error shrinks as `sqrt(n)` grows. To halve the standard error, you need 4x the data.

The standard error is **not the same as the standard deviation**:
- Standard deviation describes the data.
- Standard error describes the uncertainty in an estimate.

## Confidence Intervals

A confidence interval (CI) is a range of plausible values for a parameter, derived from the sample.

For the mean with known or large-sample standard deviation:
```
CI = x̄ ± z * (σ / sqrt(n))
```

For 95% confidence, `z = 1.96`. So a 95% CI for the mean is `x̄ ± 1.96 * SE`.

### What "95% confidence" actually means

If you repeated the sampling procedure infinitely many times, 95% of the constructed intervals would contain the true parameter. It does **not** mean "there is a 95% probability the parameter is in this specific interval" (the parameter is fixed; the interval is random).

### Common CI patterns

- 95% CI: `x̄ ± 1.96 * SE`
- 99% CI: `x̄ ± 2.58 * SE`
- 90% CI: `x̄ ± 1.65 * SE`

Wider CIs = more confidence, less precision.

### CI for proportions

For a sample proportion `p̂` with sample size `n`:
```
SE = sqrt(p̂(1-p̂) / n)
CI = p̂ ± z * SE
```

For rare events (p̂ near 0 or 1), use Wilson or Clopper-Pearson intervals instead of the normal approximation.

### CI for difference of means (A/B testing)

For treatment and control means `x̄_T` and `x̄_C`:
```
SE_diff = sqrt(SE_T² + SE_C²)
CI_diff = (x̄_T - x̄_C) ± z * SE_diff
```

If the CI does not include 0, the difference is statistically significant at the corresponding level.

## The Central Limit Theorem (CLT)

Statement: when you take many independent samples from any distribution with finite mean and variance, the distribution of the sample mean approaches a normal distribution as n grows, with mean equal to the population mean and standard error equal to `σ / sqrt(n)`.

Why it matters:
- It justifies using normal-based CIs and tests even when the underlying data is not normal.
- It explains why larger samples give more precise estimates.
- It is the foundation of A/B testing.

Practical rule: `n ≥ 30` is usually enough for CLT to kick in for unimodal, roughly symmetric distributions. For skewed distributions, you may need `n ≥ 100` or more.

## Covariance and Correlation

### Covariance

```
Cov(X, Y) = (1/n) Σ (x_i - x̄)(y_i - ȳ)
```

Measures how X and Y move together. Positive = same direction, negative = opposite, zero = no linear relationship.

Units depend on the units of X and Y. Hard to interpret.

### Pearson correlation

```
ρ(X, Y) = Cov(X, Y) / (σ_X * σ_Y)
```

Unitless, between -1 and +1. Measures linear relationship only.

- ρ = +1: perfect positive linear.
- ρ = -1: perfect negative linear.
- ρ = 0: no linear relationship (but there may be nonlinear).
- ρ = 0.7: strong positive linear relationship.

### Spearman correlation

Same idea, but uses ranks. Captures monotonic (not just linear) relationships. More robust to outliers.

### Critical warning: correlation ≠ causation

Two variables can be correlated because:
1. X causes Y.
2. Y causes X.
3. A third variable Z causes both (confounding).
4. Coincidence (especially with many variables).

You cannot determine causation from correlation alone. That is what experimental design and causal inference are for.

## Probability Refresher

### Basic rules

- P(A and B) = P(A) * P(B) (if independent)
- P(A or B) = P(A) + P(B) - P(A and B)
- P(not A) = 1 - P(A)
- P(A | B) = P(A and B) / P(B)

### Bayes' theorem

```
P(A | B) = P(B | A) * P(A) / P(B)
```

- P(A) = prior probability of A.
- P(B | A) = likelihood of B given A.
- P(A | B) = posterior probability of A given B.

This is the foundation of probabilistic reasoning, Naive Bayes classifiers, and Bayesian A/B testing.

### Common distributions

| Distribution | Use case | Parameters |
|---|---|---|
| Bernoulli | binary outcome (click/no-click) | p |
| Binomial | count of successes in n trials | n, p |
| Normal | continuous, symmetric, unbounded | μ, σ |
| Log-normal | continuous, positive, right-skewed | μ, σ (of log) |
| Poisson | count of events in interval | λ |
| Exponential | time between events | λ |
| Beta | probability of probability | α, β |
| t-distribution | like normal with heavier tails | degrees of freedom |
| Chi-squared | sum of squared normals | degrees of freedom |
| F-distribution | ratio of two chi-squared | d1, d2 |

## The Statistical Inference Workflow

1. **Define the question**: what parameter do you want to estimate or test?
2. **State hypotheses**: null (H₀) and alternative (H₁).
3. **Choose a test**: based on data type and assumptions.
4. **Check assumptions**: normality, independence, equal variance.
5. **Compute the test statistic**: t, z, F, chi-square, etc.
6. **Compute p-value**: probability of seeing this extreme a result under H₀.
7. **Decide**: reject H₀ if p-value < α (typically 0.05).
8. **Report effect size and CI**: not just p-value.
9. **Consider practical significance**: is the effect big enough to matter?

This is the skeleton of every hypothesis test. Memorize it.
