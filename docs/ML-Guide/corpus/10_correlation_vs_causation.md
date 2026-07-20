# Correlation vs Causation: The Deep Dive

This is the most important concept in business data science, and the one most often misunderstood. The job description explicitly calls out "data interpretation" and "A/B testing." This section ties them together.

## The Fundamental Distinction

| | Correlation | Causation |
|---|---|---|
| Meaning | X and Y move together | Changing X changes Y |
| Can be measured from observational data? | Yes | Not safely without assumptions/design |
| Main question | "Are they associated?" | "What happens if we intervene?" |
| Example | Users who use feature X pay more | Enabling feature X causes users to pay more |
| Main enemy | Spurious relationship | Confounding, selection bias, reverse causality |
| Tool | Correlation, regression, EDA | Randomized experiments, causal inference |
| Business danger | Wrong decision from fake pattern | Shipping changes that do not actually help |

## Why "Correlation Does Not Imply Causation" Is Not Enough

The phrase is true but unhelpful. The real question is: **when does correlation suggest causation, and how do I tell the difference?**

Correlation can be causal, non-causal, or coincidental. You need additional information to distinguish them.

## The Four Reasons Two Variables Can Be Correlated

### 1. X causes Y (the real deal)

Example: drug lowers blood pressure. The drug truly causes the effect. This is what we want to confirm.

### 2. Y causes X (reverse causality)

Example: high-value customers get access to a premium feature. The feature is correlated with revenue, but the relationship is that revenue causes feature access, not feature access causing revenue.

Common in:
- Loyalty programs: premium members spend more, but maybe they were already high spenders.
- Product usage: heavy users convert more, but they might be high-intent users to begin with.

### 3. Confounding (a third variable Z causes both X and Y)

Example: power users use the feature more AND have higher revenue. The feature is not the cause; being a power user is.

Common in observational data:
- "Customers who buy product A also buy product B" — maybe they are just customers in general.
- "Users who get a discount return more" — maybe they are price-sensitive users who buy more of everything.

### 4. Coincidence (spurious correlation)

With many variables, some will be correlated by chance. With 100 variables tested pairwise, you will find many "significant" correlations.

## Selection Bias

Selection bias happens when the sample you observe is not representative of the population you want to make claims about.

### Types of selection bias

**Sampling bias**: your sample over-represents some subgroup.
- Survey responses from engaged users (engaged users answer surveys).
- App analytics from opted-in users (different from non-opt-in).

**Survivorship bias**: your sample only includes cases that "survived" some process.
- Studying active users (excludes churned users).
- Looking at successful startups (excludes failures).

**Self-selection**: people choose whether to be in the treatment.
- Customers who ask for refunds are different from those who don't.
- Patients who take medication are different from those who don't.

**Berkson's bias**: conditioning on a collider creates spurious correlation in subgroups.
- Studying hospital patients (sicker than general population).

## Confounding in Detail

A confounder is a variable that affects both the supposed cause and the supposed effect.

### Classic example: ice cream and drowning

Ice cream sales and drowning deaths are correlated. The confounder is summer heat (causes both ice cream sales and swimming). Eating ice cream does not cause drowning.

### How to detect confounders

1. Domain knowledge: what variables could plausibly affect both?
2. Causal diagrams (DAGs): draw the assumed causal structure.
3. Stratification: check if the correlation holds within each level of the confounder.
4. Adjustment in regression: include the confounder as a covariate.

### How to deal with confounders

1. **Randomization** (best): randomly assign treatment, which breaks the link between confounders and treatment.
2. **Matching**: pair treated and untreated units with similar confounder values.
3. **Stratification**: analyze within strata of the confounder.
4. **Regression adjustment**: include confounders as covariates.
5. **Propensity score methods**: model the probability of being treated, then match/weight/stratify on it.
6. **Instrumental variables**: use a variable that affects treatment but not the outcome directly.

## The Causal Inference Toolkit

### Randomized Controlled Trial (RCT / A/B test)

The gold standard. Random assignment breaks the link between confounders and treatment, in expectation. This is why A/B testing is so important.

Strengths:
- Unbiased estimate of average treatment effect.
- No need to model confounders.
- Strong external validity (if done on a representative sample).

Weaknesses:
- Expensive and slow.
- Sometimes unethical or impractical.
- May have novelty effects or network spillovers.

### Observational causal methods

Used when randomization is not possible. Stronger assumptions required.

**Difference-in-differences**: compare changes over time between treated and control groups. Assumes parallel trends.

**Regression discontinuity**: when treatment is assigned based on a threshold (e.g., test score), compare units just above and just below the threshold.

**Instrumental variables**: find a variable that affects treatment but not outcome (except through treatment). Estimate causal effect using the IV.

**Propensity score methods**: model P(treatment | covariates), then match/weight/stratify treated and untreated units with similar propensities.

**Synthetic control**: create a synthetic comparison group from a weighted combination of untreated units. Useful for policy analysis.

**Causal graphs / DAGs**: explicit representation of assumed causal structure. Tools: do-calculus, instrumental variables, front-door adjustment.

## Counterfactual Reasoning

Causal inference is fundamentally about **counterfactuals**: what would have happened to this unit under a different treatment?

- "Would this user have churned anyway if we had not sent the retention email?"
- "Would this customer have converted without the discount?"

The counterfactual is never observed for the same unit. That is the fundamental problem of causal inference. Methods estimate it by finding comparable units, modeling assumptions, or experimental design.

## Common Business ML Scenarios and Causal Traps

### "This model has high ROC-AUC, so it drives business value"

Wrong. ROC-AUC measures ranking, not causal effect. The model could be picking up on confounding (e.g., "users about to convert are more likely to be shown this offer") without actually causing the conversion.

To prove business value: run an A/B test where the model's predictions actually drive the action.

### "We A/B tested and got a 2% lift, ship it!"

Good. But check:
- Is the lift consistent across segments, or driven by a small group?
- Is there a novelty effect (lift decays over time)?
- Are there guardrail degradations you missed?
- Could the treatment be leaking into the control (e.g., shared channels)?

### "Power users love this feature, so let's roll it out to everyone"

Survivorship bias. Power users are not representative. Roll out to a randomized subset first.

### "Customers who bought product A are most likely to buy product B"

This is correlation, often due to general buying tendency. Recommending B to A buyers may not increase total revenue, just shift basket composition.

## Practical Causal Thinking for ML Interviews

When asked "how would you measure the impact of this model?", the senior answer structure is:

1. **Randomize** the model's predictions to drive the action in some cases and not in others (or use a control model).
2. **Pre-register** the primary metric (revenue, conversion, retention) and guardrails.
3. **Run the test** for long enough to capture delayed effects.
4. **Analyze**: effect size, CI, p-value, segment breakdown.
5. **Validate the assumption**: was randomization respected (SRM check)?
6. **Decide** based on the result, not on offline metrics.

When asked "how would you handle confounding?", the answer is: "I'd randomize if possible, otherwise use a causal method (matching, propensity scores, IV, DiD) and explicitly state the assumptions."

The phrase "correlation does not imply causation" is true but junior. The senior version is: "Correlation is a hypothesis generator. To claim causation, you need either a randomized experiment or strong causal assumptions with appropriate methods."

## The Key Insight

**Every claim about "model X improved metric Y" is a causal claim.** It says the model caused the improvement. The only way to verify that is randomized experimentation. Everything else (observational analysis, regression adjustment, even sophisticated causal methods) requires assumptions.

That is why A/B testing and experiment design are central to this role, and why "data interpretation" in the job description is shorthand for "causal reasoning under uncertainty."
