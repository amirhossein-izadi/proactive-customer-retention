# Model Monitoring, Drift, and Performance Decay

Models are not "deploy once, run forever." They decay. The job description explicitly says "Continuously monitor the performance of deployed models, diagnose issues, and implement necessary retraining and improvements." This section is the depth on monitoring.

## Why Models Decay

Three reasons:

1. **Data drift**: the input distribution changes. The model sees data unlike training.
2. **Concept drift**: the relationship between input and target changes. The same input now means something different.
3. **Label drift**: the target distribution changes. The base rate shifts.

In production, all three often happen simultaneously. Rarely is a deployed model stable for long.

## The Four Kinds of Drift

### Data drift (covariate shift, feature drift)

**Definition**: P(X) changes. The distribution of input features changes.

**Example**: a model trained on pre-pandemic data sees pandemic-era inputs. Demographics, behavior, transaction patterns shift.

**Detection**: compare training feature distributions to recent production feature distributions.

**Methods**:
- Kolmogorov-Smirnov test (continuous features).
- Chi-squared test (categorical features).
- Population Stability Index (PSI).
- Jensen-Shannon divergence.
- Wasserstein distance.
- Adversarial validation (train a classifier to distinguish train from production; if it can, distributions differ).

**What to do**: investigate the cause, retrain on recent data, possibly add new features.

### Concept drift (model drift, real drift)

**Definition**: P(Y | X) changes. The relationship between input and target changes.

**Example**: a fraud model trained on last year's patterns misses new fraud techniques. The features look the same, but the labels are different.

**Detection**: hard, because you need labels to detect it. Use proxy metrics, slice analysis, delayed evaluation.

**What to do**: retrain on recent labeled data, change features, redesign the model.

### Label drift (prior probability shift)

**Definition**: P(Y) changes. The base rate of the positive class shifts.

**Example**: fraud rate spikes during a holiday season. A model calibrated to last quarter's base rate is now miscalibrated.

**Detection**: monitor the predicted probability distribution; if it shifts, label distribution likely shifted too. Confirm with delayed labels when available.

**What to do**: recalibrate the model, retrain with recent data, retune threshold.

### Prediction drift (output drift)

**Definition**: the distribution of model outputs changes.

**Example**: a fraud model's score distribution suddenly drops. Either the input is changing, the model is broken, or the upstream data pipeline is corrupted.

**Detection**: monitor mean, variance, and quantiles of predictions over time.

**What to do**: investigate; could be a data pipeline issue, a model bug, or a real distribution shift.

## Monitoring Taxonomy

| Monitoring type | Question | Detection | Action |
|---|---|---|---|
| Data quality | Is the input broken? | Schema validation, null checks, range checks, categorical value sets | Block/alert/fallback |
| Data drift | Has input distribution changed? | KS test, chi-squared, PSI, JS divergence, adversarial validation | Investigate/retrain |
| Prediction drift | Has output distribution changed? | Mean, std, quantiles of predictions over time | Compare to feature drift |
| Concept drift | Has X→Y relationship changed? | Delayed labels, proxy metrics, slice metrics | Retrain/change features |
| Label drift | Has target distribution changed? | Delayed labels, predicted probability shifts | Recalibrate/retrain |
| Performance drift | Are metrics worse? | Precision, recall, F1, etc. on delayed labels | Rollback/retrain |
| Calibration drift | Are probabilities wrong? | Calibration plot, Brier score over time | Recalibrate |
| Latency/error | Is service healthy? | p50/p95/p99 latency, error rates, throughput | Scale/optimize/fallback |
| Business KPI | Is model helping business? | Conversion, revenue, retention, satisfaction | A/B decision |

## The Performance Monitoring Problem

The hardest part of monitoring: **true labels are delayed**.

For many business problems:
- Churn: known 30-60 days after inactivity.
- Default: known 90+ days after missed payment.
- Lifetime value: known only after the customer leaves.
- Conversion: known only after the customer has had time to act.

So you cannot compute precision/recall in real time. You need proxies.

### Proxy metrics

- **Engagement drop** as proxy for churn.
- **Page views, session length, return rate** as proxies for retention.
- **Click-through rate** as proxy for relevance.
- **Investigation rate, alert volume** as proxies for fraud model health.

Choose proxies that are:
- **Leading indicators**: they change before the true label.
- **Sensitive**: small model degradations show up as proxy changes.
- **Actionable**: changes in proxy should trigger investigation.

Validate the proxy periodically against the true label. If the proxy and truth diverge, the proxy is broken.

### Delayed evaluation pipeline

Build a pipeline that:
1. Logs every prediction with a unique ID and timestamp.
2. When the label arrives (days/weeks later), joins it back to the prediction.
3. Computes performance metrics over rolling windows.
4. Alerts when metrics degrade.

This is essential for any production model with non-trivial label delay.

## Alerting and Thresholds

Monitoring without alerting is a dashboard, not a system. Set thresholds for automatic alerts.

**Common alert patterns**:
- Latency p95 > 200ms for 5 minutes.
- Error rate > 1% for 5 minutes.
- Data drift PSI > 0.2 for any feature.
- Predicted positive rate drops by 50% vs baseline.
- Calibration error > 5% over the last week.
- Business KPI drop > X% in A/B test.

**Avoid alert fatigue**: too many alerts = ignored alerts. Set thresholds based on business impact, not statistical noise.

**Alert on rate of change, not absolute**: a slow drift may not trip an absolute threshold but the rate of change is alarming.

## Slice-Based Monitoring

Aggregate metrics hide slice-specific failures.

**What is a slice**:
- A user segment (new vs returning, region, device).
- A feature value range (low income, high spenders).
- A time period (weekend vs weekday).
- A prediction confidence range.

**Why it matters**:
- A model can have stable overall accuracy but degrade for one segment.
- Bias and fairness issues show up in slice analysis.
- Business KPIs may differ by segment in ways the model misses.

**Implementation**:
- Compute metrics per slice.
- Set slice-specific alerts.
- Investigate any slice where performance diverges from the overall.

## The Monitoring Stack

A typical production monitoring stack includes:

1. **Data validation**: Great Expectations, TFX Data Validation, Deequ, pandas-profiling.
2. **Drift detection**: Evidently AI, whylogs, Alibi Detect.
3. **Performance tracking**: MLflow, Weights & Biases, custom dashboards.
4. **Business KPI monitoring**: A/B test dashboards, BI tools (Looker, Tableau, Mode).
5. **Infrastructure monitoring**: Prometheus, Grafana, Datadog.
6. **Alerting**: PagerDuty, Opsgenie, Slack integrations.

For a single model, you can build this in-house. For many models, invest in a platform.

## The Drift Response Playbook

When an alert fires, follow this sequence:

1. **Confirm the alert**: is it real or a fluke? Check multiple windows.
2. **Check data quality first**: a sudden drift is often a pipeline issue, not a real shift.
3. **Check the model itself**: was there a recent deployment, retraining, or config change?
4. **Check external events**: is there a holiday, news event, marketing campaign that explains the shift?
5. **Investigate the slice breakdown**: is one segment driving the shift?
6. **Compute recent performance** if labels are available: is the model actually worse?
7. **Decide**: investigate further, retrain, roll back, or wait and watch.

Always have a way to roll back. The cost of an investigation is small; the cost of a broken model in production is large.

## Retraining Strategy

Once you decide to retrain:

1. **Use the most recent data**: include data from after the drift started.
2. **Validate on a holdout** that reflects the new distribution: not just random holdout, but recent holdout.
3. **Compare to current production model** on:
   - Overall metrics.
   - Slice metrics (especially slices that drifted).
   - Latency.
   - Calibration.
4. **Deploy carefully**: canary, not full replacement.
5. **Monitor the new model** more aggressively for the first few days.

## What Senior Candidates Say

When asked "how do you monitor a deployed model?", give this structure:

1. **Pre-deployment baseline**: log everything (inputs, outputs, latencies) so you have a comparison.
2. **Data quality**: schema validation, null checks, value range checks, alerts on breakage.
3. **Drift detection**: PSI/KS tests on key features, distribution shifts on outputs.
4. **Performance**: delayed evaluation pipeline that joins labels back to predictions; proxy metrics for fast feedback.
5. **Slice analysis**: monitor key segments separately; alerts when slices diverge.
6. **Calibration**: calibration plots, Brier score over time; recalibrate when needed.
7. **Latency and errors**: standard SRE practices; alerts tied to business impact.
8. **Business KPI**: integrated with A/B test dashboards; the final arbiter of model value.
9. **Retraining triggers**: scheduled + drift-triggered + performance-triggered.
10. **Rollback plan**: tested, fast, with a runbook.

That is the depth expected.
