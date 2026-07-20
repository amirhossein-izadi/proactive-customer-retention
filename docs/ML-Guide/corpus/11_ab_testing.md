# A/B Testing: The Complete Guide

A/B testing is the most direct way to prove model impact, which is exactly what the job description demands. This section is a complete interview-ready treatment.

## What A/B Testing Is

A controlled experiment where:
- Users (or sessions, or items) are **randomly assigned** to a control group (current experience) or treatment group (new experience).
- Outcomes are measured on both groups.
- The difference in outcomes is the **estimated treatment effect**.

Randomization is the magic: in expectation, it makes the two groups equivalent on every variable except the treatment. So any difference in outcomes can be attributed to the treatment.

## Key Concepts

### Unit of randomization

The entity that is assigned to control or treatment. Must be:
- **Independent**: one unit's outcome does not affect another's.
- **Stable**: stays in the same group throughout the test.
- **Trackable**: outcomes can be measured per unit.

Common units:
- **User ID**: standard for consumer products.
- **Session ID**: short-term tests, but users may re-enter.
- **Device ID**: when users have multiple devices.
- **Account ID**: for B2B products.
- **Item ID**: for content experiments.
- **Geography**: for marketplace or policy tests.

Wrong unit choices break independence. Examples:
- Randomizing by session in a returning-user product: same user in both groups.
- Randomizing by user in a social network: treated users influence control users.

### Control and treatment

- **Control**: receives the current experience. Baseline for comparison.
- **Treatment**: receives the new model, feature, design, or policy.

Multiple treatments (A/B/n) are possible but require more samples and multiple-testing adjustment.

### Primary metric

The single metric used to make the decision. Pre-registered before the test.

If you have multiple metrics, pick one as primary. The others are secondary or guardrail.

### Guardrail metrics

Metrics that must not degrade. They protect the business from unintended side effects.

Examples for a new recommendation model:
- Primary: clicks per session
- Guardrails: page load time, unsubscribe rate, content diversity, return rate

### Sample size

The number of units needed to detect the minimum effect of interest with sufficient power.

Required sample size depends on:
- **Baseline rate** (or mean)
- **Minimum detectable effect (MDE)**: the smallest effect that matters to the business
- **Significance level (α)**: usually 0.05
- **Power (1 - β)**: usually 0.80

Smaller MDE → more samples needed. Higher power → more samples needed.

### Minimum detectable effect (MDE)

The smallest effect size that the test should be able to detect. It links statistics to business value.

- "We care about a 1% lift in conversion." → MDE = 1%.
- "Anything below 5% is not worth the engineering cost." → MDE = 5%.

If you set MDE too small, the test will be enormous. If you set it too large, you will miss real effects.

### Statistical power

The probability of detecting a real effect of size MDE, given the sample size and significance level. Power = 1 - β. Standard target: 80%.

### p-value and confidence interval

After the test, compute the effect size, p-value, and confidence interval.

- p-value: strength of evidence against the null (no effect).
- Confidence interval: plausible range for the true effect.
- Effect size: the magnitude of the observed effect.

## The A/B Testing Workflow

1. **Define the hypothesis**: "Treatment X will increase metric Y by at least Z%."
2. **Choose metrics**: primary (decision) + guardrails + secondary (exploratory).
3. **Estimate baseline and MDE** from historical data.
4. **Compute required sample size** (use calculator or formula).
5. **Choose unit of randomization** (user, session, etc.).
6. **Run a pilot / ramp-up** to catch bugs before full launch.
7. **Run the test** until the planned sample size is reached.
8. **Check SRM** (sample ratio mismatch): is the actual split correct?
9. **Analyze**: effect size, CI, p-value, segment breakdown.
10. **Make a decision**: ship, iterate, or kill.
11. **Document** and share learnings.

## Critical Issues in A/B Testing

### Sample Ratio Mismatch (SRM)

If the actual number of users in each group does not match the planned ratio, the test is broken.

Example: planned 50/50, observed 48/52. That is a 2% deviation. With 100K users, the probability of that happening by chance is very small. There is a bug.

**SRM causes**: bug in randomization, bot traffic, redirects, eligibility changes mid-test.

**Always check SRM first** before analyzing the test. If SRM exists, the test result is meaningless.

### Peeking

Repeatedly checking the p-value and stopping when it is significant inflates the false positive rate.

If you peek every day for 14 days, your chance of a false positive is much higher than 5%, even if you stop at the planned sample size.

**Solutions**:
- Use a fixed horizon and do not check p-values until then.
- Use sequential testing methods (mSPRT, always-valid CIs).
- Use Bonferroni or similar to penalize multiple looks.

### Novelty effect

Users react to change initially (positive or negative), then revert. The treatment effect is not stable over time.

**Mitigation**: run the test for at least one full business cycle (typically 1-2 weeks for consumer products). Some teams run for 4+ weeks for retention tests.

### Network effects (spillovers)

If treated users can influence control users (social network, marketplace, two-sided platform), the independence assumption breaks.

Example: in a social feed test, treated users see better content and engage more, which affects their posts, which affects control users' feeds.

**Mitigations**: cluster randomization (e.g., by geography), ghost treatments (render the new feed for control users but do not actually show it), longer time windows.

### Multiple testing

Running many variants or many metrics inflates false positive rates.

**Mitigations**: pre-register metrics, use Bonferroni or FDR corrections, use a single primary metric.

### Long-term effects

Some effects (e.g., retention) take months to manifest. A 2-week test may show no effect when there is a real one.

**Mitigations**: run long enough, or use proxy metrics that respond quickly (and validate them against long-term outcomes periodically).

### Seasonality

Day-of-week, month, holidays, and external events can affect outcomes. A test run during a holiday season may not generalize.

**Mitigations**: run during representative periods, run multiple tests, use stratification by time.

## When A/B Testing Is Valid

| Situation | A/B test? | Why |
|---|---|---|
| Independent users, measurable short-term KPI | strong | clean randomization possible |
| UI/model ranking change for users | strong | treatment can be isolated |
| Fraud model affecting bad actors | medium | adversarial adaptation and delayed labels |
| Marketplace pricing | hard | treatment can affect supply/demand equilibrium |
| Social network feed | hard | treated users influence control users |
| Rare event metric | hard | huge sample needed |
| Long-term retention | hard | long test duration needed |
| Ethical/safety-critical change | limited | cannot expose harmful treatment |
| Offline historical policy comparison | not A/B | needs causal/off-policy methods |
| Personalized treatment assignment | advanced | randomization and eligibility become complex |

## When A/B Testing Is Hard or Invalid

| Situation | Problem | Possible solutions |
|---|---|---|
| Marketplace/network effects | Treatment leaks between groups | Cluster randomization, ghost treatments |
| Very rare outcome | Needs huge sample | Longer test, choose more common proxy |
| Long-term outcome | Test takes too long | Use proxy, then validate with longer test |
| Ethical/safety issue | Cannot randomize harmful treatment | Stagger rollout, restricted population, observational methods |
| Offline historical data only | No live users to randomize | Causal inference (DiD, synthetic control, IV) |
| Strong seasonality | Effects confounded with time | Run across multiple cycles, use blocking |
| Limited user base | Cannot reach required sample size | Switch to within-subject design, use lower power |
| High-cost treatment | Budget for full test is infeasible | Run smaller pilot, use lower power |
| Novelty effect | Short-term gain, long-term loss | Run longer, plan for reversion |
| Selection bias in assignment | Treated ≠ control populations | Check covariate balance, stratify |

## Special A/B Test Designs

### Multi-armed bandits (MAB)

Adaptive allocation: shift more users to the better-performing arm during the test. Faster decisions, but more complex analysis.

Use when: opportunity cost of bad treatment is high (e.g., pricing), and you want to learn fast.

Don't use when: you need a clean estimate of the average effect, or guardrail analysis requires fixed groups.

### Factorial designs (A/B/n, A/B/C/D)

Test multiple factors at once. A/B/C/D can be analyzed as multiple A/B tests with multiple-testing correction, or with ANOVA-style models for interactions.

Use when: factors may interact (e.g., new model + new UI).

### Switchback / time-based designs

When the unit of randomization is time (e.g., hour or day), with all users seeing the same treatment during each interval.

Use when: network effects are severe and cluster randomization is impractical (e.g., ride-share).

### Within-subject designs

Same user sees both treatments (e.g., interleaved ranking). Requires the order to be randomized and carryover effects to be controlled.

Use when: between-subject variance is high relative to within-subject variance, and carryover is small.

## A/B Testing Pitfalls in Practice

1. **Stopping early because p < 0.05** at first peek. → Use fixed horizon or sequential test.
2. **Running multiple tests on overlapping users.** → Interaction effects, dilution.
3. **Not checking SRM.** → Randomization bug goes undetected.
4. **Optimizing the wrong metric.** → Local optima that hurt business KPIs.
5. **Trusting short-term lifts.** → Novelty effects, seasonality, regression to the mean.
6. **Ignoring guardrail degradation.** → Solving one metric by breaking another.
7. **Drawing conclusions from underpowered tests.** → Missing real effects.
8. **Forcing pre-registered decisions.** → Cherry-picking to fit the data.
9. **Not documenting the test.** → Same debates every quarter.
10. **A/A tests skipped.** → Use A/A tests to verify the platform works.

## What the Modern Literature Says

Modern reviews of online controlled experiments emphasize that large-scale A/B testing creates statistical challenges beyond simple textbook tests:

- Multiple metrics and segments require correction.
- Long-term effects are hard to capture.
- Network effects and interference break independence.
- Peeking and early stopping inflate false positives.
- The platform itself (logging, randomization, assignment) needs to be tested.

A famous reference is Kohavi's "Practical Guide to Controlled Experiments on the Web," which frames web A/B testing as random assignment of live users to variants and emphasizes the engineering discipline required.

## The Senior Interview Answer

When asked "How would you design an A/B test for a new recommendation model?", give this structure:

1. **Unit of randomization**: user ID (assuming independence).
2. **Primary metric**: clicks per session (or business metric if measurable quickly).
3. **Guardrails**: page load time, return rate, content diversity.
4. **Baseline and MDE**: from historical data, pick the smallest effect worth detecting (e.g., 2% relative lift in CTR).
5. **Sample size**: compute using power analysis (α=0.05, power=0.80).
6. **Duration**: cover at least 1-2 business cycles (typically 2-4 weeks).
7. **Ramp-up**: 1% → 5% → 50% → 100% to catch bugs.
8. **SRM check**: every day, verify the actual split matches the planned split.
9. **Analysis**: effect size, CI, p-value, segment breakdown.
10. **Decision**: ship if positive and significant with no guardrail degradations; iterate or kill otherwise.
11. **Long-term validation**: monitor for novelty effect decay; consider a longer test for retention.

That structure is what senior candidates give. Memorize it.
