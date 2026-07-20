# Hypothesis Testing, p-values, and Confidence Intervals in Depth

The job description explicitly says "hypothesis testing, experimental design, and data interpretation." This is one of the most common places candidates fail interviews. We are going to fix that.

## Hypothesis Testing: The Big Picture

A hypothesis test is a formal framework for deciding between two competing claims about a population, using sample data.

### The hypotheses

- **Null hypothesis (H₀)**: the default claim. Usually "no effect," "no difference," "no relationship."
- **Alternative hypothesis (H₁ or Ha)**: what you are trying to show. Usually "there is an effect," "there is a difference."

The test assumes H₀ is true and asks: "How unlikely is the data we observed under this assumption?"

### The decision

If the data is very unlikely under H₀, we reject H₀ in favor of H₁. The threshold for "very unlikely" is the significance level, α, usually 0.05.

### Critical point

**Hypothesis testing does not prove H₁ is true.** It says "H₀ is unlikely to have produced this data." That is different from "H₁ is true."

## The p-value: The Most Misunderstood Concept in Statistics

### Correct definition

A **p-value is the probability, assuming the null hypothesis is true, of observing data at least as extreme as the one we observed**.

In symbols:
```
p-value = P(test statistic ≥ observed | H₀ is true)
```

### What p-value is NOT

| Wrong statement | Why wrong |
|---|---|
| "The probability that the null hypothesis is true" | p-value assumes the null; it does not assign probability to it |
| "The probability that the result happened by chance" | Too vague; "by chance" is undefined |
| "Small p-value means big effect" | Small p-value can happen with tiny effect and huge sample |
| "p > 0.05 means no effect" | Could be underpowered experiment |
| "p < 0.05 means the result is real" | Could be one of many false positives |
| "p-value proves causality" | Causality depends on design, not p-value |

### A clean example

You run an A/B test on a new model.

- H₀: new model has no effect on conversion.
- Observed: treatment conversion is 5.1%, control is 4.8%.
- p-value = 0.03.

Interpretation: "If the new model truly had no effect, and all test assumptions were valid, we would see a difference of 5.1% vs 4.8% or more extreme in about 3% of repeated experiments."

It does **not** mean:
- "There is a 3% chance the new model has no effect." (frequentist vs Bayesian confusion)
- "There is a 97% chance the new model works." (no, p-values do not give that)
- "The new model increases conversion by 0.3%." (point estimate is the data; uncertainty is the CI)

### How p-values behave

- **Larger effect** → smaller p-value (easier to detect).
- **Larger sample** → smaller p-value (more power).
- **More noise** → larger p-value (harder to detect).
- **More tests** → more small p-values by chance (multiple testing problem).

### Common misinterpretations to avoid

1. **"p = 0.049, just barely significant!"** — A 0.049 and 0.051 are not categorically different. The threshold is arbitrary.
2. **"p = 0.00001, extremely significant!"** — p-values are not effect sizes. A tiny effect with a huge sample gives tiny p-values.
3. **"We failed to reject H₀, so H₀ is true."** — Absence of evidence is not evidence of absence. The test may be underpowered.

## Type I and Type II Errors

| | H₀ is true | H₀ is false |
|---|---|---|
| Reject H₀ | **Type I error** (false positive) | Correct (true positive) |
| Fail to reject H₀ | Correct (true negative) | **Type II error** (false negative) |

- **Type I error rate** = α (significance level).
- **Type II error rate** = β.
- **Power** = 1 - β = probability of correctly rejecting H₀ when H₁ is true.

### Setting up a hypothesis test

1. Choose α (typically 0.05). This caps the false positive rate.
2. Compute the test statistic from the data.
3. Compare to the critical value from the null distribution, or compute the p-value.
4. If p < α, reject H₀.

### Power analysis

Power analysis tells you:
- Given an effect size, sample size, and α, what is the power?
- Given a desired power, α, and effect size, how many samples do you need?

Required sample size for two-proportion test:
```
n = (z_α/2 * sqrt(2 * p̄(1-p̄)) + z_β * sqrt(p_T(1-p_T) + p_C(1-p_C)))² / (p_T - p_C)²
```

For simplicity, online calculators handle this. The point is: you can compute the required sample size **before** running the experiment.

## Effect Size

Statistical significance ≠ practical significance.

A 0.001% lift in conversion with p = 0.001 is "highly significant" but worth almost nothing to the business.

A 5% lift with p = 0.20 is not significant but might be the most important business opportunity.

Always report:
- **Effect size** (the magnitude of the difference)
- **Confidence interval** (the uncertainty)
- **p-value** (the strength of evidence)

Common effect size measures:
- **Cohen's d** (difference in means / pooled standard deviation). 0.2 = small, 0.5 = medium, 0.8 = large.
- **Cohen's h** (difference in proportions, transformed). 0.2 = small, 0.5 = medium, 0.8 = large.
- **Odds ratio, risk ratio** (for binary outcomes).
- **R²** (for regression, but not standardized).

## Common Hypothesis Tests

| Test | Use case | Test statistic |
|---|---|---|
| One-sample t-test | Is sample mean different from known value? | t |
| Two-sample t-test | Are two group means different? | t |
| Paired t-test | Are paired measurements different? | t |
| One-proportion z-test | Is sample proportion different from known? | z |
| Two-proportion z-test | Are two group proportions different? | z |
| Chi-squared test | Are categorical variables independent? | χ² |
| ANOVA | Are 3+ group means different? | F |
| Wilcoxon / Mann-Whitney | Non-parametric version of t-test | rank-based |
| Kruskal-Wallis | Non-parametric version of ANOVA | rank-based |
| Pearson correlation test | Is correlation significantly different from 0? | t |
| Shapiro-Wilk | Is data normally distributed? | W |
| Levene's / Bartlett's | Do groups have equal variance? | F / χ² |

## Confidence Intervals: The Cleaner Way to Report Results

A confidence interval contains all parameter values that would not be rejected by a hypothesis test at the corresponding significance level.

For example, a 95% CI for the difference in conversion rates contains all values where the null "no difference" would not be rejected at α = 0.05.

**Senior practice: report CIs, not just p-values.**

"Conversion increased from 4.8% to 5.1% (95% CI: 0.05% to 0.55%, p = 0.03)."

This tells the reader:
- The point estimate (5.1% - 4.8% = 0.3%).
- The uncertainty (CI: 0.05% to 0.55%).
- The statistical evidence (p = 0.03).

## Multiple Testing Problem

If you run 20 hypothesis tests at α = 0.05, you expect **1 false positive by chance** (5% of 20 = 1). The more tests you run, the more false positives appear.

### Corrections

- **Bonferroni**: divide α by the number of tests. α' = α / m. Conservative.
- **Holm-Bonferroni**: step-down procedure, slightly less conservative.
- **Benjamini-Hochberg (FDR)**: controls false discovery rate. Use when you can tolerate some false positives.
- **Bonferroni-Holm**: step-down version, more powerful than Bonferroni.

In ML context, multiple testing appears when:
- You test many metric variations.
- You slice metrics by user segment.
- You run many A/B tests across the year.

The fix: pre-register the primary metric, use guardrail metrics (limited set), and apply multiple-testing correction when exploring.

## The Statistical Decision Workflow

1. **State the question** in terms of a parameter.
2. **Formulate H₀ and H₁**.
3. **Choose the test** based on data type and assumptions.
4. **Check assumptions** (normality, independence, variance).
5. **Choose α and required power**.
6. **Compute sample size** if planning the experiment.
7. **Collect data** (or use existing data).
8. **Compute test statistic and p-value**.
9. **Compute effect size and CI**.
10. **Decide**: reject H₀ or not, with effect size context.
11. **Report**: point estimate, CI, p-value, effect size, sample size.

This is the workflow you should walk through in interviews when asked about experiment analysis.
