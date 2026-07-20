# The Master Context Map for ML Interviews

Before memorizing any single topic, internalize this: every ML problem is a chain of decisions, and the interviewer is testing whether you reason across the whole chain.

> **Business problem → data situation → model family → metric choice → validation design → production risk → monitoring/retraining strategy.**

If you can walk that chain fluently, you sound senior. If you only talk about algorithms, you sound junior no matter how clever your formulas are.

## The Master Context Table

This is the top-down map. Every topic in the job description fits somewhere in it.

| Context | Typical business problem | Data shape | Best first models | Stronger models | Main metrics | Validation style | Main interview trap |
|---|---|---|---|---|---|---|---|
| Binary classification | churn, fraud, conversion, lead scoring, default prediction | tabular, labels 0/1 | Logistic regression, decision tree | Random forest, gradient boosting, neural net | Precision, recall, F1, ROC-AUC, PR-AUC, log loss | stratified split, time split if temporal | using accuracy on imbalanced data |
| Imbalanced classification | fraud, rare disease, abuse detection, churn minority | positive class rare | Logistic regression with class weights | Gradient boosting with class weights, calibrated model | PR-AUC, recall@precision, precision@k, Fβ | stratified/time split; keep base rate realistic | saying ROC-AUC alone is enough |
| Multiclass classification | category prediction, intent detection, product classification | one label among many | Multinomial logistic regression | boosted trees, neural nets, transformers | macro/micro F1, top-k accuracy, log loss | stratified split | confusing macro vs micro averaging |
| Ranking / recommendation | search ranking, feed ranking, product ranking | user-item interactions | logistic ranker, matrix factorization | LambdaMART, neural ranker, two-tower model | NDCG, MAP, MRR, CTR, conversion | time split, user split, online A/B | offline metric improves but business metric drops |
| Regression | price, demand, revenue, duration prediction | numeric target | Linear/ridge regression | Random forest regressor, gradient boosting regressor, neural net | MAE, RMSE, MAPE, RMSLE, R² | random/time split depending on problem | optimizing RMSE when business cares about absolute error |
| Forecasting | demand, traffic, sales, capacity | time series | naive baseline, ARIMA-style, regression with lags | GBM with lag features, deep sequence models | MAE, RMSE, MAPE, pinball loss | strictly time-based backtesting | leakage from future information |
| Anomaly detection | fraud prefilter, system monitoring, outlier detection | few/no labels | isolation forest, one-class SVM | autoencoder, density model | precision@k, recall@k, investigation yield | temporal validation, human review | treating unsupervised score as calibrated probability |
| NLP classification | sentiment, ticket routing, intent, toxicity | text | TF-IDF + logistic regression/SVM | BERT/transformer fine-tuning | F1, PR-AUC, calibration, latency | stratified or time split | using large transformer without baseline |
| NLP retrieval / RAG | semantic search, FAQ, document QA | text corpus + queries | BM25, TF-IDF | embeddings, cross-encoder reranker | recall@k, MRR, NDCG, human eval | query-level split | evaluating only generation, not retrieval |
| Computer vision | defect detection, object classification | images | pretrained CNN feature extractor | ViT/CNN fine-tuning | accuracy, F1, mAP, IoU | image-level or group split | data leakage from near-duplicate images |
| A/B testing | prove model/business impact | online user data | current system as control | new model as treatment | primary KPI + guardrails | randomized controlled experiment | peeking, multiple testing, bad randomization |
| Production ML | deploy model reliably | live data stream/batch | simple robust model | model selected by latency/cost/quality tradeoff | online KPI, latency, drift, error rate | shadow/canary/A-B | offline validation but no monitoring |
| MLOps | reproducible ML lifecycle | code + data + model artifacts | scripts + manual tracking | MLflow/model registry/CI/CD | reproducibility, lineage, model health | promotion gates | no experiment tracking/model versioning |
| Deep learning | unstructured data, representation learning | image/text/audio/large data | MLP/CNN/RNN baseline | transformer, fine-tuned foundation model | task-specific + loss curve + calibration | holdout + early stopping | not knowing training-loop/debugging basics |
| Statistical inference | "is this effect real?" | sample data | t-test/proportion test | regression/causal methods | p-value, CI, effect size, power | experiment design | confusing statistical significance with practical significance |
| Causal inference | "did X cause Y?" | observational or experimental | randomized A/B test | matching, diff-in-diff, IV, causal graphs | treatment effect, CI, p-value | design-dependent | treating correlation as causation |

## Why this table matters

When the interviewer asks "How would you approach this problem?", the right answer is not "use XGBoost." The right answer walks through several rows of this table and only then commits to a model.

For example, for a fraud detection problem, you would say:

1. **Context**: imbalanced binary classification with delayed labels and adversarial behavior.
2. **First model**: logistic regression with class weights (calibrated, interpretable baseline).
3. **Stronger model**: gradient boosting with proper time-based split, class weights, threshold tuning.
4. **Primary metric**: PR-AUC, recall@precision, since positives are rare and alert volume is a business cost.
5. **Validation**: time-based split; never random split for fraud because tomorrow's fraud is not like yesterday's.
6. **Production**: monitor precision, recall, drift, and delayed label performance; retrain on a schedule and on drift triggers.
7. **Business proof**: A/B test the new model against the current one with guardrail metrics (false positive cost, customer friction).

That answer demonstrates seniority. Memorize the structure, not just the rows.
