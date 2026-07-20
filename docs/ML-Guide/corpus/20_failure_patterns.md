# Common ML Failure Patterns and Anti-Patterns

This section catalogs the most common ways ML projects fail. Knowing these is what separates engineers who build working systems from those who build impressive-looking failures.

## The Top ML Anti-Patterns

### 1. Solving the wrong problem

The team builds a model with 95% accuracy on a proxy metric. The business KPI does not move.

Cause: the ML target is not aligned with the business decision. Example: optimizing click-through rate when the business cares about revenue.

Fix: trace the chain from business objective to ML metric. Validate via A/B test that the proxy correlates with the real KPI.

### 2. Trusting offline metrics

The team celebrates a 3% AUC improvement and ships. In production, the model is no better than the old one.

Cause: offline validation is flawed (leakage, wrong split, wrong metric, no concept drift handling).

Fix: validate with the right split, the right metric, and the right comparison. A/B test before declaring victory.

### 3. Ignoring data quality

The team assumes the data is correct. Production predictions are terrible.

Cause: missing values, schema changes, encoding issues, duplicates, time zone bugs, bad joins.

Fix: data validation as a first-class concern. Schema checks, range checks, null rate monitoring, freshness monitoring.

### 4. Sampling bias in training

The team trains on a sample that does not represent production traffic. The model fails on real data.

Cause: historical data, biased cohorts, easy-to-label users, opt-in users.

Fix: profile training data vs production data. Use adversarial validation to detect distribution shift. Re-sample to match production.

### 5. Leakage everywhere

The team has a model with 0.99 AUC. They cannot reproduce the performance in production.

Cause: target leakage, time leakage, group leakage, preprocessing leakage.

Fix: strict time-based cutoffs, group-aware splits, fit preprocessing only on training data, sanity-check feature availability at inference time.

### 6. No monitoring

The team deploys and forgets. Six months later, the model is silently broken.

Cause: no drift detection, no performance tracking, no alerting.

Fix: build monitoring from day one. Track data quality, drift, performance, calibration, latency. Alert on rate of change, not just absolute thresholds.

### 7. No retraining strategy

The model decays. Performance degrades. The team manually retrains when someone notices.

Cause: no scheduled retraining, no drift triggers, no documented procedure.

Fix: scheduled retraining, drift-triggered retraining, performance-triggered retraining. Document the procedure. Automate it.

### 8. Latency budget violation

The team picks the most accurate model. Inference takes 5 seconds. The product is unusable.

Cause: ignoring latency constraints, no early performance testing.

Fix: latency budget is a hard constraint. Measure it early. Use simpler models, distillation, batch inference, or model compression.

### 9. The model is a black box

The model makes bad predictions. Nobody knows why. The business loses trust.

Cause: no interpretability tools, no error analysis, no slice analysis.

Fix: SHAP values for feature-level explanations, slice analysis for failure modes, error analysis for systematic issues. Build a feedback loop with stakeholders.

### 10. No rollback plan

The model is deployed. It misbehaves. The team scrambles for hours.

Cause: no versioned artifacts, no tested rollback procedure, no monitoring alerts.

Fix: keep previous model artifacts in the registry. Test rollback. Set up monitoring alerts. Document the runbook.

### 11. Optimizing for the wrong user

The model optimizes for the average user but fails for important subgroups.

Cause: no slice analysis, no fairness checks, ignoring tail behavior.

Fix: slice analysis is a first-class evaluation step. Check performance for important subgroups. Set up alerts for slice-specific issues.

### 12. Feedback loops

The model influences user behavior, which influences future labels, which influence the model. Over time, the model becomes self-reinforcing.

Cause: deploying a model that affects the data-generating process (recommendations, content, ads, fraud flags).

Fix: monitor for feedback loops. Use delayed labels when possible. Occasionally retrain on a fresh sample. Test counterfactual policies offline.

## The Production ML Failure Cascade

Most production ML failures are cascades of small issues:

1. A pipeline breaks silently.
2. Training data has missing or wrong values.
3. The model trains on bad data.
4. Validation looks good because the same bad data is in the validation set.
5. The model is deployed.
6. Production traffic has different (correct) data.
7. The model makes bad predictions.
8. No monitoring catches it.
9. The business loses money for weeks.
10. Someone notices.

The fix: defense in depth. Data validation, model validation, deployment checks, monitoring, alerting, retraining, rollback. Each layer catches what the previous one missed.

## The "I Built a Model" Anti-Pattern

A common failure: someone trains a model, gets good offline metrics, and declares success. The system is not productionized.

The full delivery includes:
- Data validation.
- Reproducible training pipeline.
- Experiment tracking.
- Model registry.
- Deployment with safety checks.
- Monitoring.
- Alerting.
- Retraining pipeline.
- Rollback plan.
- Documentation.
- Stakeholder communication.

Each of these is a separate project. The model is maybe 20% of the work.

## The "I Built a Complex System" Anti-Pattern

The opposite failure: the team over-engineers. They build a complex MLOps platform for a model that is not even proven.

Fix: start simple. A script that trains a model and saves predictions is a good start. Add complexity only when the simple version is in production and generating value.

## When ML Is the Wrong Solution

Sometimes ML is not the answer. Recognizing this is senior.

ML is wrong when:
- The problem is deterministic and rules work fine (e.g., simple validation logic).
- There is no signal in the data (humans cannot predict it either).
- The cost of wrong predictions is too high and uncertainty is irreducible.
- The data is too small to learn from.
- The problem changes too fast for any model to stay current.

A senior candidate will say "I would start with a rules-based baseline and only move to ML if the rules are not capturing the patterns we care about."

## The Model Is Not the Product

The model is a component. The product is the system that uses the model to make decisions.

- The product includes the data pipeline, the model, the serving infrastructure, the monitoring, the alerting, the retraining, the documentation, the on-call rotation.
- The product serves a business decision.
- The product has SLAs, error budgets, and stakeholders.

A model that is never deployed is a research project, not a product. A model that is deployed but not monitored is a liability. A model that is monitored but not retrained decays. The full lifecycle is the product.

## The Documentation Anti-Pattern

"The code is self-documenting" is the calling card of teams that cannot maintain their own systems six months later.

For ML systems, document:
- Data sources, schemas, and freshness SLAs.
- Feature definitions and computation logic.
- Model architecture and hyperparameters.
- Training data version, code version, environment.
- Validation strategy and metric definitions.
- Deployment procedure and rollback plan.
- Monitoring setup and alert thresholds.
- Known failure modes and limitations.
- On-call runbook.

The doc that matters most is the runbook: what to do when X happens.

## The "It Works on My Machine" Anti-Pattern

The model trains on the engineer's laptop. It fails in production because the environment is different.

Causes:
- Different library versions.
- Different data versions.
- Different hardware (CPU vs GPU).
- Different preprocessing.
- Different feature definitions.

Fix: containerized environments (Docker), pinned versions, automated pipelines that run the same code in dev and prod.

## The Communication Anti-Pattern

The team builds a great model. Stakeholders do not know what it does, do not trust it, or do not use it.

Causes:
- No stakeholder communication during development.
- No interpretability or explanation.
- No A/B test results shared.
- No documentation.

Fix: involve stakeholders early. Share progress. Explain the model with SHAP or similar. Run a proper A/B test and share results. Document everything.

## The Final Word

Most ML project failures are not model failures. They are:
- Problem framing failures.
- Data quality failures.
- Validation failures.
- Communication failures.
- Operational failures.

The model is the easy part. Everything around the model is the hard part.

This is why the job description emphasizes "end-to-end," "productionize," "validation and optimization," and "performance monitoring." These are the things that matter.
