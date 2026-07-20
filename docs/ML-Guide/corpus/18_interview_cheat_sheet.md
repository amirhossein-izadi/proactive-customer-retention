# The ML Interview Cheat Sheet and Study Order

This section is the synthesis. Use it to review, to organize your study, and to prepare for the actual interview.

## The Top-Down Reasoning Pattern

For every interview question, structure your answer top-down:

1. **Context**: what kind of problem is this?
2. **Data**: what data do we have, and what are the constraints?
3. **Metric**: what metric matters and why?
4. **Validation**: how do we validate the approach?
5. **Model**: what model fits the context?
6. **Tradeoffs**: what are we giving up?
7. **Production**: how does it ship and stay healthy?

Memorize this skeleton. Every answer can use it.

## Interview Question Bank

### "How do you know the model is good?"

Strong answer: "It depends on the business cost structure. For balanced data with equal error costs, accuracy or ROC-AUC. For imbalanced data, PR-AUC, recall@precision, or business cost-based metric. For probability-based decisions, log loss or Brier score. I always also check calibration, because a model that ranks well can still be poorly calibrated, and that breaks cost-sensitive thresholds."

### "Why not just use accuracy?"

Strong answer: "On imbalanced data, accuracy is misleading. A model that always predicts the majority class gets 99% accuracy with zero business value. I use precision, recall, F1, or PR-AUC instead, and tune the threshold based on the cost of false positives vs false negatives."

### "ROC-AUC or PR-AUC?"

Strong answer: "ROC-AUC measures ranking quality across thresholds. PR-AUC focuses on positive-class performance and is more informative when positives are rare. For fraud detection (rare positives, alert cost matters), I prefer PR-AUC. For a marketing model with balanced classes, ROC-AUC is fine."

### "How do you choose the threshold?"

Strong answer: "Not 0.5 by default. I look at the business cost: false positive cost vs false negative cost. On the validation set, I plot the expected cost across thresholds and pick the minimum. Or I fix a recall target based on business constraints and read off the threshold. I also check calibration, because the threshold is meaningful only if probabilities are calibrated."

### "Random forest or gradient boosting?"

Strong answer: "Random forest is a robust, low-tuning baseline that reduces variance through bagging. Gradient boosting is stronger on tabular data but more sensitive to overfitting, especially with noisy labels. I'd start with random forest as a baseline, then try gradient boosting if the performance gain justifies the tuning cost and monitoring burden."

### "Why logistic regression as a baseline?"

Strong answer: "It's fast, interpretable, well-calibrated, and easy to debug. It sets a performance floor that any complex model needs to beat by enough to justify its complexity. Many 'deep learning' projects would do just as well with logistic regression on engineered features."

### "How do you avoid leakage?"

Strong answer: "Three rules: split before preprocessing, define a strict cutoff date for time-aware features, and use group-aware splits when entities have multiple rows. I always use a scikit-learn Pipeline to ensure preprocessing is fit only on training data, and I check for suspiciously high validation scores as a leakage red flag."

### "How would you validate a model's business impact?"

Strong answer: "A/B test. Pre-register a primary metric (the business KPI, not a proxy), set guardrail metrics to catch unintended harm, compute sample size for the minimum detectable effect, run for at least one full business cycle, check SRM, and report effect size with confidence interval, not just p-value. Then monitor long-term for novelty effects."

### "How do you monitor a deployed model?"

Strong answer: "Several layers. Data quality: schema validation, null checks, value ranges. Drift: PSI or KS test on input features, distribution shifts on outputs. Performance: delayed evaluation pipeline that joins labels back to predictions; proxy metrics for fast feedback. Calibration: reliability diagrams and Brier score over time. Latency and errors: standard SRE monitoring. Business KPIs: integrated with A/B dashboards. Slice analysis: monitor key segments separately."

### "How do you debug a deep learning model?"

Strong answer: "First, check the data. Visualize samples, verify labels, check normalization. Then, try to overfit a small batch. If that fails, the model has a bug. If it succeeds, train on more data and watch the loss curve. If loss is NaN, lower the learning rate and add gradient clipping. If loss plateaus, try a different architecture or features. Always compare to a simple baseline."

### "How do you retrain?"

Strong answer: "Three triggers: scheduled (every N days), drift-triggered (PSI or KS exceeds threshold), and performance-triggered (proxy metrics degrade). I retrain on the most recent data, validate on a recent holdout, compare to the current production model on overall and slice metrics, and deploy via canary. I always keep the previous model ready for rollback."

### "How do you productionize?"

Strong answer: "Track experiments in MLflow with code, data, hyperparameters, and metrics. Register the model in MLflow Model Registry with lineage. Validate offline with the right split and metric. Deploy via canary (1% → 5% → 25% → 100%) with rollback ready. Monitor data quality, drift, performance, calibration, latency, and business KPI. Set retraining triggers. Document the runbook."

### "Why is correlation not causation?"

Strong answer: "Two variables can be correlated because of a confounder (a third variable causing both), reverse causality (Y causes X, not X causing Y), selection bias, or coincidence. To establish causation, you need either a randomized experiment or strong assumptions with appropriate methods (matching, instrumental variables, difference-in-differences, etc.)."

### "What is a p-value?"

Strong answer: "A p-value is the probability, assuming the null hypothesis is true, of observing data at least as extreme as what we got. It is not the probability the null is true, not the probability the result is due to chance, and not the effect size. I always report it together with the effect size, confidence interval, and the assumptions of the test."

### "How do you choose sample size for an A/B test?"

Strong answer: "From the baseline metric, the minimum detectable effect that matters to the business, the significance level (usually 0.05), and the desired power (usually 0.80). I use a power analysis formula or calculator. Smaller MDE requires more samples. Higher power requires more samples. The required sample size is often the binding constraint on test duration."

### "When can't you do an A/B test?"

Strong answer: "When there's strong network effects (treated users influence control), when the outcome is too rare to detect (need huge samples), when the outcome is too long-term (test would take years), when randomization is unethical or impractical, or when you only have historical data and need causal inference methods instead."

## The Priority Order for Study

### Phase 1: Evaluation and Metrics (highest priority)

Master this before algorithms.
- Confusion matrix and all derived metrics.
- ROC-AUC vs PR-AUC vs F1.
- Threshold tuning and business cost.
- Regression metrics: MAE, RMSE, MAPE, R², RMSLE, quantile loss.
- Calibration.
- Slice-based evaluation.

### Phase 2: Validation and Leakage

- Train/val/test split, cross-validation.
- Time-based splits, group splits.
- Data leakage: types, detection, prevention.
- Backtesting for time series.

### Phase 3: Classical ML

- Linear regression: closed form, GD, assumptions, regularization.
- Logistic regression: loss, regularization, multiclass, calibration.
- Decision trees: splits, overfitting, pruning.
- Random forest: bagging, feature subsampling, OOB.
- Gradient boosting: residuals, learning rate, early stopping.
- SVM: margin, kernels, C, gamma, scaling.
- Naive Bayes and KNN as secondary topics.

### Phase 4: Statistics and Experiments

- Distributions: normal, Bernoulli, Poisson, etc.
- Sampling distribution, standard error, CLT.
- Confidence intervals.
- p-values, hypothesis testing, power, effect size.
- Multiple testing corrections.
- A/B testing: design, sample size, SRM, peeking.
- Correlation vs causation.

### Phase 5: Production ML and MLOps

- ML lifecycle.
- Data leakage in production.
- MLflow: tracking, registry, deployment.
- Drift detection.
- Monitoring: data quality, drift, performance, calibration.
- Retraining strategies.
- Rollback and incident handling.
- Batch vs online inference.

### Phase 6: PyTorch and Deep Learning

- Tensor operations, autograd.
- `nn.Module`, training loop.
- Loss functions, optimizers, LR schedules.
- DataLoader, batch size, num_workers.
- Checkpointing, device management.
- Overfitting control, debugging.
- CNN, RNN, transformer basics.
- Transfer learning, fine-tuning.

### Phase 7: NLP and SQL

- NLP: tokenization, TF-IDF, embeddings, BERT, fine-tuning, evaluation.
- SQL: joins, window functions, CTEs, time-based features, A/B analysis, performance.

## The "Must-Master" List Before Interview

You should be able to explain these fluently, without notes:

1. Precision vs recall vs F1.
2. ROC-AUC vs PR-AUC under imbalance.
3. Why F1 ignores true negatives.
4. Why threshold 0.5 is often wrong.
5. How to choose a threshold using business cost.
6. MAE vs RMSE vs MAPE.
7. Linear regression closed form and assumptions.
8. Logistic regression loss and coefficient interpretation.
9. SVM margin, kernel, C, gamma.
10. Decision tree split criteria and overfitting control.
11. Bagging vs boosting.
12. Random forest vs gradient boosting.
13. Why boosting can overfit noisy labels.
14. Class imbalance handling: class weights, sampling, thresholding, PR metrics.
15. Train/test leakage, especially time leakage.
16. Random split vs time split vs group split.
17. p-value, confidence interval, power, effect size.
18. Correlation vs causation.
19. A/B test design: primary metric, guardrails, sample size, SRM, peeking.
20. ML lifecycle: data → features → train → validate → registry → deploy → monitor → retrain.
21. PyTorch training loop: zero_grad, forward, loss, backward, step.
22. Calibration and why probability quality matters.
23. Model monitoring: drift, delayed labels, performance decay.
24. SQL feature extraction with cutoff dates.

## The Common Interview Traps to Avoid

1. **Saying "use accuracy" on imbalanced data** → always address class imbalance first.
2. **Picking 0.5 as threshold by default** → always justify threshold choice.
3. **Confusing correlation with causation** → always mention experimental design.
4. **Not checking data leakage** → always ask about time and entity structure.
5. **Optimizing the wrong metric** → tie metric to business objective.
6. **Trusting a single validation score** → use CI, cross-validation, slice analysis.
7. **Ignoring production constraints** → latency, freshness, monitoring all matter.
8. **Treating the model as the deliverable** → the system is the deliverable.
9. **Forgetting the baseline** → "is it better than a simple model?"
10. **Skipping the business context** → the model serves a decision, not the other way around.

## The Day-Before-Interview Checklist

- [ ] Can you write a PyTorch training loop from memory?
- [ ] Can you write a leakage-safe SQL feature query?
- [ ] Can you explain PR-AUC vs ROC-AUC in your sleep?
- [ ] Can you describe a recent A/B test you would design?
- [ ] Can you list the seven stages of the ML lifecycle?
- [ ] Can you explain what a p-value is and is not?
- [ ] Can you describe the bias-variance tradeoff?
- [ ] Can you compare bagging and boosting?
- [ ] Can you explain why logistic regression is a strong baseline?
- [ ] Can you describe how to monitor a deployed model?

If yes to all, you are ready.

## Resources to Use (No URLs, Just Categories)

- **scikit-learn user guide**: best for classical ML, metrics, validation, cross-validation.
- **MLflow documentation**: best for tracking, registry, deployment.
- **PyTorch official tutorials**: best for autograd, training loop, transfer learning.
- **NIST/SEMATECH handbook of statistical methods**: best for classical statistics.
- **Kohavi et al. papers on controlled experiments**: best for A/B testing at scale.
- **"Practical Statistics for Data Scientists"** (Bruce, Bruce, Gedeck): best applied stats book.
- **"An Introduction to Statistical Learning"** (James, Witten, Hastie, Tibshirani): best for classical ML with statistical depth.
- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** (Géron): best practical ML book.
- **Evidently AI documentation**: best for monitoring and drift detection.

Read these once, use them as references, and return to them when studying specific topics.
