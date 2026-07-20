# Classical ML Algorithm Master Comparison

This is the table that ties together every classical algorithm. Use it as a decision framework in interviews, not as a list to memorize.

## The Master Algorithm Table

| Model | Core idea | Handles nonlinearity? | Scaling needed? | Good for | Bad for | Robust to noise? | Imbalance behavior | Interpretability | Production notes |
|---|---|---|---|---|---|---|---|---|---|
| Linear regression | linear relationship, squared error | no, unless feature engineering | yes for regularization/GD | simple regression, baseline, interpretation | nonlinear patterns, outliers | weak to outliers | not classification | high | fast, stable, easy to monitor |
| Ridge regression | linear + L2 penalty | no | yes | multicollinearity, many correlated features | sparse feature selection | better than OLS | N/A | high-medium | stable baseline |
| Lasso regression | linear + L1 penalty | no | yes | feature selection, sparse models | correlated features can be unstable | medium | N/A | high | useful for explainable sparse models |
| ElasticNet | linear + L1 + L2 | no | yes | mix of L1 sparsity and L2 stability | pure prediction tasks | medium | N/A | medium | good when features are correlated |
| Logistic regression | linear classifier, probabilistic | linear boundary | yes | baseline classification, calibrated-ish probabilities | complex interactions | medium | needs class weights/threshold tuning | high | strong production baseline |
| Naive Bayes | Bayes rule + conditional independence | limited | usually no | text baseline, small data | correlated features | surprisingly robust | can handle imbalance with priors | medium | very fast |
| KNN | prediction by nearby points | yes | yes | small data, local patterns | high-dimensional/large data | weak to noisy features | bad unless balanced/sampled | low | poor latency at scale |
| Linear SVM | max-margin linear classifier | linear | yes | high-dimensional sparse data, text | probability output, huge nonlinear problems | good margin robustness | needs class weights | medium | good text baseline |
| RBF SVM | max-margin with kernel | yes | yes | medium-sized nonlinear data | large datasets | can overfit with bad gamma/C | needs class weights | low | expensive; SVC fit scales poorly on large data |
| Decision tree | recursive feature splits | yes | no | explainable nonlinear rules | high variance | weak | biased to majority unless adjusted | high | unstable alone |
| Random forest | bagging many trees | yes | no | robust tabular baseline | very high-dimensional sparse data, latency-sensitive huge models | good | needs class weights/sampling/threshold | medium | strong baseline, less tuning |
| Gradient boosting | sequential trees correcting residuals | yes | no | tabular SOTA-style performance | noisy labels, leakage, poor validation | medium-low if overfit | supports weights but needs metric/threshold care | medium-low | powerful but must monitor carefully |
| MLP | learned nonlinear function | yes | yes | large structured/unstructured data | small tabular data without care | depends on regularization | needs loss weighting/sampling | low | requires training/debugging skill |
| CNN | convolutional feature learning | yes | normalization needed | images, spatial data | tabular data | good with augmentation | needs sampling/loss strategy | low | GPU + data pipeline important |
| Transformer | attention-based sequence modeling | yes | normalization/tokenization | NLP, multimodal, long-range dependencies | tiny data, strict latency | depends on data/fine-tuning | needs careful sampling/loss | low | expensive; often fine-tune/pretrain |

scikit-learn's supervised-learning guide organizes these into linear models, SVMs, nearest neighbors, Gaussian processes, decision trees, ensembles, Naive Bayes, and more. Its SVM docs note that SVMs are effective in high-dimensional spaces but the SVC implementation may become impractical for very large sample sizes.

## Why This Table Works

It answers the questions interviewers actually ask:

1. **"When would you use this model?"** (Good for / Bad for columns)
2. **"What are its failure modes?"** (Robust to noise / Imbalance columns)
3. **"Why would you pick it over alternatives?"** (Interpretability + Production notes)
4. **"How do you productionize it?"** (Production notes + Scaling)

Memorize the structure: any algorithm can be described in these eight dimensions. If you do not have an opinion on each dimension, you do not know the algorithm well enough.

## Quick Decision Rules

### "Which model should I try first on tabular data?"

1. **Logistic regression / linear regression**: as a baseline. Always.
2. **Random forest**: as a robust nonlinear baseline. Low tuning, high reliability.
3. **Gradient boosting (LightGBM/XGBoost/CatBoost)**: as a performance-oriented model.
4. **Neural network**: only if you have lots of data, latency budget, and the rest have plateaued.

### "Which model for text classification?"

1. **TF-IDF + logistic regression or linear SVM**: strong baselines, fast, interpretable.
2. **Fine-tuned transformer (BERT-style)**: when you have enough data and need top performance.

### "Which model for ranking?"

1. **Logistic ranker or gradient boosting with pairwise/listwise loss**: starting point.
2. **LambdaMART, neural rankers, two-tower models**: for high performance.

### "Which model for interpretability?"

1. **Linear / logistic regression**: highest.
2. **Shallow decision tree**: high.
3. **Random forest + SHAP values**: medium.
4. **Gradient boosting + SHAP values**: medium-low.
5. **Deep networks**: low (use attention or saliency maps if needed).

### "Which model for low-latency inference?"

1. **Linear / logistic regression**: smallest, fastest.
2. **Shallow trees**: small, fast.
3. **Random forest**: heavier (many trees).
4. **Gradient boosting with few trees**: medium.
5. **Neural network**: depends on size; can be small (distilled) or huge (transformer).

## The "Try in this Order" Heuristic for Tabular ML

Most Kaggle and production winners follow this ladder:

1. Baseline: predict the mean / majority class.
2. Linear/logistic regression with proper preprocessing.
3. Random forest with default hyperparameters.
4. Gradient boosting with cross-validated hyperparameters.
5. Stack/blend the strongest models.

Stop climbing the ladder when:
- The marginal improvement no longer justifies the complexity cost.
- The new model is harder to debug, monitor, or explain.
- The latency budget is exceeded.

## Common Algorithm Failures

| Failure mode | Models most affected | What to do |
|---|---|---|
| Overfit small data | Deep nets, deep trees, KNN | regularize, use linear/forest baselines |
| High variance | Single trees, KNN | bagging (random forest), more data |
| High bias | Linear models on nonlinear data | add features, use trees/boosting, neural nets |
| Chasing noise | Boosting (without regularization) | use early stopping, regularize, clean labels |
| Slow at inference | Random forest (many trees), large neural nets | distill, prune, use linear model |
| Bad on extreme imbalance | Tree-based models without class weights | class weights, sampling, threshold tuning |
| Cannot extrapolate | Tree-based models (predict in observed range only) | use linear models for trends outside training range |
| Misses interactions | Linear models | explicit interaction features, trees, neural nets |
| Sensitive to feature scale | SVM, KNN, neural nets, regularized linear | standardize/normalize inputs |
| Leaks at inference | Models trained with future features | strict time-based cutoffs in features |
| Bad calibration | Tree-based models | Platt scaling, isotonic regression |

## What scikit-learn Specifically Teaches

scikit-learn's supervised-learning documentation frames classical models as a family with consistent APIs. Random forests and ExtraTrees are described as randomized "perturb-and-combine" tree ensembles whose predictions average individual trees. The `RandomForestClassifier` documentation explicitly describes fitting many decision trees on sub-samples and averaging to improve predictive accuracy and control overfitting.

That is the production-grade framing: random forests are not a "black box algorithm." They are a variance-reduction wrapper around decision trees, and that framing is what you should give in interviews.
