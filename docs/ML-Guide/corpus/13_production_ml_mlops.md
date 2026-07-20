# Production ML and MLOps: The Full Lifecycle

The job description says "Work with engineering teams to successfully productionize models, ensuring they are scalable, reliable, and perform efficiently in real-time environments." This is not a "nice to have." It is the second most common reason candidates fail senior ML interviews, after statistics.

## The ML Lifecycle

Every production ML system goes through these stages:

| Stage | What you do | What can go wrong | What to monitor |
|---|---|---|---|
| Problem framing | define business objective, ML target, decision action | optimizing proxy metric | business KPI |
| Data extraction | SQL, joins, cutoff dates | leakage, duplicates, wrong joins | row counts, nulls, freshness |
| Feature engineering | create predictive variables | train/serve skew, future features | feature distributions |
| Labeling | define target window | delayed labels, noisy labels | label rate, delay |
| Offline validation | test model before deploy | wrong split, metric mismatch | slice metrics |
| Experiment tracking | log params/metrics/artifacts | unreproducible experiments | run lineage |
| Model registry | version/promote models | wrong model deployed | version, stage, approval |
| Deployment | batch/online/shadow/canary | latency, dependency failure | latency, errors |
| Online validation | A/B test or canary | randomization bugs | KPI, guardrails, SRM |
| Monitoring | observe live behavior | drift, silent failure | data drift, prediction drift |
| Retraining | update model | feedback loops, stale labels | retrain triggers |
| Rollback | revert safely | no fallback | incident metrics |

We will cover each of these in detail.

## Problem Framing

The most important step, often skipped.

**The trap**: "We need to predict churn." But what does the business do with the prediction? Send a discount? Trigger a customer success call? Show a different message?

Each action has different cost-benefit. The metric you optimize should reflect that.

**Senior approach**:
1. Start with the business question: "Reduce churn by 5% in Q1."
2. Define the decision: "Which users get a retention offer?"
3. Define the ML target: "P(churn in next 30 days)."
4. Define the metric: "Conversion rate on retention offers × customer LTV."
5. Translate to ML metric: optimize for expected profit per user, not just AUC.

The ML metric is downstream of the business metric. Always trace the chain.

## Data Extraction

The first engineering step. Get the data, with the right cutoffs, with the right joins.

**Common pitfalls**:
- Joining on the wrong key.
- Using data from after the prediction cutoff (leakage).
- Duplicates (a user appearing multiple times due to bad join).
- Time zone issues (same UTC date, different local date).
- Inactive users (need to filter out users with no recent activity).

**Senior practice**:
- Use SQL with explicit cutoff dates.
- Document the data dictionary.
- Verify row counts at each step.
- Use versioned data snapshots for reproducibility.

## Feature Engineering

The biggest leverage point in most projects.

**Categories of features**:
- **Aggregate features**: counts, sums, averages, percentiles over windows.
- **Time-based features**: recency, frequency, trends, seasonality.
- **Categorical encodings**: target encoding, frequency encoding, one-hot, embeddings.
- **Interaction features**: ratios, products, differences of existing features.
- **Text features**: TF-IDF, embeddings, n-grams, topic models.
- **Image features**: CNN embeddings, color histograms.
- **Embedding features**: learned from related models (e.g., user embeddings from a recommender).

**The train/serve skew problem**:
Features must be computed the same way in training and inference. If training uses "purchases in last 30 days" computed daily, but serving uses the same feature computed on demand, you can get subtle differences.

**Solution**: use a feature store. Compute features once, store them, serve them the same way to both training and inference.

## Labeling

The hardest part of supervised ML.

**The target window problem**:
"Predict churn" needs a definition. When is a user "churned"?
- No purchase in 30 days? 60 days? 90 days?
- Cancellation of subscription? Account closure?
- Specific event? (e.g., "deleted app")

Each definition gives different labels and different models. Pick based on the business action.

**Delayed labels**:
For some problems, you only know the true label after a delay:
- Churn: known 30+ days after inactivity.
- Default: known months after missed payment.
- Conversion: known after the customer has had time to act.

**Solutions**:
- Use proxy metrics that respond quickly (e.g., engagement drop as proxy for churn).
- Build a delayed evaluation pipeline.
- Train on the most recent fully-labeled data, accept that the freshest samples have noisier labels.

**Noisy labels**:
- Human-labeled: humans disagree, get tired, follow inconsistent guidelines.
- Inferred: based on a definition, but the definition may be wrong.
- Self-reported: subject to social desirability bias.

**Solutions**:
- Multiple annotators, measure agreement.
- Clean labels using heuristics or models.
- Use robust loss functions (Huber, MAE).

## Offline Validation

We covered this in the previous section. Recap:
- Right split for the data structure.
- Right metric for the business objective.
- Right baseline (mean, simple model, current production).
- Slice analysis to find failure modes.
- Confidence interval on the estimate.

## Experiment Tracking

Without it, you cannot reproduce a model or explain a result.

**What to log**:
- Code version (git commit hash).
- Data version (snapshot ID, date range).
- Hyperparameters.
- Training metrics per step/epoch.
- Validation metrics.
- Final test metrics.
- Model artifact (file path or hash).
- Environment (Python version, library versions).
- Notes and rationale.

**Tools**:
- **MLflow Tracking**: open-source, language-agnostic, integrates with most ML libraries.
- **Weights & Biases**: hosted, with rich visualization.
- **DVC**: data and pipeline versioning.
- **Neptune, Comet, SageMaker Experiments**: hosted alternatives.

MLflow Tracking is documented as a tool for logging parameters, code versions, metrics, and output files in MLflow. It is the most common open-source choice.

## Model Registry

A model registry is a centralized store for trained models, with versioning, lineage, and lifecycle management.

**What a registry provides**:
- Model versioning (v1, v2, v3).
- Stage transitions (staging → production → archived).
- Aliases (champion, challenger).
- Lineage (which data, code, hyperparameters produced this model).
- Metadata (owner, description, approval status).
- Annotations (links to A/B test results, business reviews).

**Tools**: MLflow Model Registry, Vertex AI Model Registry, SageMaker Model Registry, custom solutions.

MLflow's Model Registry is documented as a centralized model store with lineage, versioning, aliases, metadata, and lifecycle management support. That maps directly to the "preferred competency" line in the job description.

## Deployment

Three main modes, with very different tradeoffs.

### Batch inference

- Run the model on a schedule (e.g., hourly) over a batch of inputs.
- Write predictions to a database or feature store.
- Downstream systems read predictions.

**Use when**: predictions are not needed in real-time, can be precomputed, or are needed for many users at once.

**Pros**: simple, scalable, no latency pressure.
**Cons**: predictions may be stale; not appropriate for time-sensitive use cases.

### Online inference (real-time)

- Model is deployed as a service (REST, gRPC, or in-process).
- Each request triggers inference.
- Response time matters (e.g., < 100ms).

**Use when**: real-time decisions per user (recommendations, fraud, search ranking).

**Pros**: fresh predictions, immediate response.
**Cons**: latency budget, scaling challenges, infrastructure cost.

### Streaming inference

- Model processes events as they arrive (e.g., Kafka stream).
- Stateful: maintains context over time.

**Use when**: fraud detection, monitoring, real-time personalization at scale.

**Pros**: fresh, low-latency, can handle high throughput.
**Cons**: complex infrastructure, harder to debug.

### Deployment strategies

**Direct deployment (replacement)**: stop the old model, start the new one. Risky; if the new model is broken, all traffic is broken.

**Shadow deployment**: new model receives traffic and makes predictions, but they are not used. Old model is still the decision-maker. Compare predictions, latency, errors.

**Canary deployment**: new model gets a small percentage of traffic (e.g., 1%, then 5%, then 25%, then 100%). Monitor at each step.

**A/B test deployment**: new model and old model serve different user groups, business metrics are compared. Statistical test decides the winner.

**Champion-challenger**: production model is the "champion." New models are "challengers" running in parallel, evaluated offline and online. Promote the winner.

### Latency considerations

- Linear models: microseconds.
- Shallow trees: microseconds.
- Random forest: milliseconds.
- Gradient boosting: milliseconds to tens of milliseconds.
- Neural nets: milliseconds to hundreds of milliseconds, depending on size.

If the budget is 10ms, you cannot run a 1000-tree gradient boosting model. Use a smaller model, a distilled model, or batch inference.

## Online Validation

The bridge between offline metrics and production reality.

**Pre-launch checks**:
- SRM check.
- Latency under load.
- Memory and CPU usage.
- Error handling for malformed inputs.
- Logging and tracing.
- Rollback procedure tested.

**A/B testing** (covered in detail in another section):
- Primary metric, guardrails, secondary metrics.
- Power analysis and MDE.
- Run for at least one full business cycle.
- Pre-register the decision rule.

**Canary analysis**:
- Start with a small fraction of traffic.
- Watch for errors, latency spikes, business metric drops.
- Increase gradually.
- Roll back if anything looks bad.

## Monitoring

Once in production, the work is not done. Models decay.

**What to monitor**:

| Monitoring type | Question answered | Example signal | What to do |
|---|---|---|---|
| Data quality | is input broken? | null spike, schema change, invalid category | block/alert/fallback |
| Data drift | did input distribution change? | age/location/device distribution shift | investigate/retrain |
| Prediction drift | did output distribution change? | fraud score suddenly lower | compare with feature drift |
| Concept drift | did X→Y relationship change? | same features no longer predict churn | retrain/change features |
| Label drift | did target distribution change? | churn base rate changed | recalibrate/retrain |
| Performance drift | are metrics worse? | precision/recall drop after labels arrive | rollback/retrain |
| Calibration drift | are probabilities wrong? | predicted 0.8 means observed 0.6 | recalibrate |
| Latency/error | is service healthy? | p95 latency spike | scale/optimize/fallback |
| Business KPI | did model help business? | conversion/revenue/retention | A/B decision |

### Drift detection

**Statistical tests for drift**:
- Kolmogorov-Smirnov test (continuous features).
- Chi-squared test (categorical features).
- Population Stability Index (PSI).
- Jensen-Shannon divergence.

**Tools**: Evidently AI, whylogs, Arize, Fiddler, internal tools.

Evidently's documentation describes evaluation, testing, and monitoring of data and AI systems, including tabular/text data and monitoring use cases from experiments to production.

### The delayed label problem

For most models, true labels arrive days, weeks, or months after prediction. Performance monitoring is delayed.

**Mitigations**:
- Use proxy metrics (engagement drop as proxy for churn).
- Backtest on historical data when full labels arrive.
- Use a delayed evaluation dashboard that updates as labels come in.

## Retraining

Models decay. Retraining keeps them fresh.

**Triggers**:
- **Scheduled**: retrain every N days/weeks.
- **Drift-triggered**: retrain when drift exceeds a threshold.
- **Performance-triggered**: retrain when proxy metrics degrade.
- **Event-triggered**: retrain after major business changes (e.g., new product launch).

**Patterns**:
- **Full retrain**: train new model on all available data, replace old model.
- **Incremental / online learning**: continuously update model with new data.
- **Warm start**: initialize new model from old model weights (for deep learning).

**Pitfalls**:
- Feedback loops: predictions influence future labels (e.g., a fraud model that flags transactions, then transactions are investigated, then the investigation results become labels).
- Catastrophic forgetting: model loses performance on older patterns.
- Stale features: training data has changed but features are still computed the old way.

## Rollback and Incident Handling

When a model fails in production, you need a way back.

**Prerequisites**:
- Previous model artifact preserved (in registry).
- Old feature pipeline available.
- Ability to switch model version quickly (config flag, routing rule).
- Monitoring alerts that catch issues fast.
- Runbook with clear incident steps.

**Incident steps**:
1. Alert fires.
2. On-call investigates: data quality, drift, errors, latency.
3. Decide: bug, drift, or external event.
4. If model-related: roll back to previous version.
5. Communicate to stakeholders.
6. Post-mortem: root cause, prevention.

## MLOps Maturity Levels

A useful framework to think about production ML maturity:

**Level 0: Manual**
- Scripts, no tracking, manual deployment.
- No monitoring.
- Acceptable for research, not for production.

**Level 1: DevOps for ML**
- Automated training pipelines.
- Basic experiment tracking.
- Manual deployment.

**Level 2: Automated training**
- CI/CD for model training.
- Model registry.
- Basic monitoring.

**Level 3: Automated deployment**
- Automated deployment with safety checks.
- Drift detection.
- Auto-retraining on triggers.

**Level 4: Full MLOps**
- Feature store.
- Real-time monitoring with alerts.
- Continuous training and deployment.
- A/B testing integrated.
- Model governance and lineage.

The job description hints at Level 2-3. The "preferred competency" of MLflow points to Level 2-3 maturity.

## The Senior Interview Answer

When asked "how would you productionize this model?", give this structure:

1. **Validation**: time-based split, business-aligned metric, comparison to baseline.
2. **Tracking**: log code, data, hyperparameters, metrics in MLflow.
3. **Registry**: version model in MLflow Model Registry, attach lineage.
4. **Deployment**: canary (1% → 5% → 25% → 100%) with rollback.
5. **Monitoring**: data quality, drift, prediction drift, latency, business KPI.
6. **A/B test**: pre-registered primary metric, guardrails, sample size computed.
7. **Retraining**: scheduled + drift-triggered, with rollback to previous version.
8. **Incident handling**: runbook, alerts, on-call rotation.

That is the level of detail expected for a senior ML role.
