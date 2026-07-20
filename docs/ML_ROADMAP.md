# ML Master Roadmap — Didar Retention Challenge

## The Core Insight

**This is an uplift/causal problem, not a churn classification problem.**  
The goal is to rank customers by `τ(x) = P(churn|no contact) − P(churn|contact)`, not by raw churn probability.

---

## 1. Problem Framing

| Dimension | Decision |
|---|---|
| Task | CATE estimation (individual treatment effect) |
| Target | `contacted=1` → `churned=0` (prevent churn) |
| Ranking metric | `robust_score = τ(x) − λ × SE(τ(x))` |
| Capacity | Top 50 customers |
| Key risk | **Confounding by indication** — historical contact was biased toward at-risk customers |

---

## 2. Data & Validation

- **Leakage-safe split**: Stratified by `(contacted, churned)` group; 80/20 development/test
- **Cross-fitting**: 5-fold outer × 3-fold inner, nested CV
- **Oversampling**: Only inside training folds if needed (class weights first; SMOTE as challenger)
- **Oversampling validation curve**: Compare ratios [native, 0.5, 0.75, 1.0] on PR-AUC, Brier, uplift@50

**Critical rule**: Preprocessing (scaling, imputation, encoding) fitted on train fold only.

---

## 3. Modeling Hierarchy

| Level | Model | Role |
|---|---|---|
| Baseline | Random selection, churn-risk-only, revenue-only | Show ML adds value |
| P0 | Logistic T-learner | Interpretable, calibrated baseline |
| P0 | CatBoost T-learner | Strong gradient-boosted baseline |
| P0 | **CatBoost DR-learner** (champion candidate) | Doubly robust protection |
| P1 | LightGBM / XGBoost DR-learner | Challenger ensemble diversity |
| P1 | Causal forest | Heterogeneity-focused challenger |
| P2 | AutoGluon DR pipeline | Predictive upper bound |
| P3 | TabFM T-learner / DR-assisted | Zero-shot research challenger |

---

## 4. Architecture

```mermaid
flowchart TD
    A[Raw CRM + Usage] --> B[Data Validation]
    B --> C[Feature Engineering]
    C --> D[Cross-fitted Split]
    D --> E[Propensity e(x)]
    D --> F[Outcome m₀(x)]
    D --> G[Outcome m₁(x)]
    E & F & G --> H[Doubly Robust Pseudo-outcomes]
    H --> I[CATE Model]
    I --> J[Calibration]
    J --> K[Top-50 Policy]
    K --> L[FastAPI / Batch]
```

---

## 5. Metrics Scorecard

| Dimension | Primary | Secondary |
|---|---|---|
| Predictive | PR-AUC, ROC-AUC | Lift@50 |
| Probability | Brier, Log loss | Calibration slope, ECE |
| Treatment ranking | AUUC, Qini coefficient | Uplift@50 |
| Business policy | DR policy value@50 | Incremental retained customers |
| Stability | Bootstrap Jaccard@50 | Rank SD across folds |
| Causal support | Propensity overlap % | Clipped-weight % |

---

## 6. Probability Reliability

- Calibrate `p₀` and `p₁` **separately** on out-of-fold data
- Use Platt scaling (sigmoid) as default; isotonic as challenger
- **Groupwise uplift validation**: decile bins → IPW/DR observed uplift
- Report: reliability diagrams, Brier, calibration slope/intercept per treatment arm

---

## 7. Explainability (SHAP/LIME)

- Explain `p₀(x)` and `p₁(x)` separately — NOT uplift directly
- Group correlated features into families: `usage_level`, `usage_decline`, `inactivity`, `volatility`, `value`, `contract`
- **SHAP ≠ causal** → never claim features "cause" contact effectiveness
- LIME: secondary local sanity check only
- Convert to CRM reason codes: "Usage fell 68%", "11 days inactive", "Monthly subscription"

---

## 8. Dataset & Pipeline Versioning

| Tool | Versions |
|---|---|
| DVC | Raw CSVs, processed features, split manifests, rankings |
| MLflow | Runs, params, metrics, model registry, artifacts |
| Git | Code, config, DVC pointer files, lockfiles |

**Seed manifest**: Split seeds, model seeds, bootstrap seeds all derived from master seed but persisted explicitly.

---

## 9. Model Bundle

One `pyfunc` bundle containing: preprocessor, propensity model, outcome models, calibrators, CATE model, ranking policy, feature schema, model card, golden tests.

```python
bundle = RetentionPolicyBundle.load("models/champion/")
output = bundle.predict(customer_batch)  # {"rank", "p0", "p1", "uplift", "reasons"}
```

---

## 10. Serving (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /ready` | Model-loaded readiness |
| `GET /v1/model-info` | Version metadata |
| `POST /v1/score` | Single customer (ad-hoc) |
| `POST /v1/rank` | Batch ranking |
| `GET /metrics` | Prometheus scrape |

**Key design**: `/v1/rank` is a batch endpoint because ranking depends on the full cohort.

Pydantic strict schemas with `extra="forbid"`, domain validation (usage ≥ 0, engagement ∈ [0,1], day ∈ [-59,0]).

---

## 11. Docker

Multi-stage build with `uv sync --frozen`, non-root user, healthcheck, pinned Python 3.12-slim.

`docker-compose.yml` with: API, MLflow, Prometheus, Grafana.

---

## 12. Monitoring

Three dashboards:

| Dashboard | Audience | Key metrics |
|---|---|---|
| Model & Business | ML/Product | AUUC, uplift@50, retained customers, segment performance |
| Operations | MLOps/SRE | Latency, error rate, throughput, model version consistency |
| Data & Drift | Data/ML | PSI, KS, missing rate, OOD rate, category novelty |

**Alert rules**: p95 latency breach, schema failures >1%, score distribution shift, model version mismatch.

---

## 13. Testing

| Layer | Tools |
|---|---|
| Unit | pytest, Hypothesis (property-based) |
| Data | Pandera schema validation |
| Model | Golden fixtures, serialization round-trip, batch equivalence |
| Integration | Container smoke test, FastAPI TestClient |
| Robustness | Artificial missingness, category corruption, extreme values |

---

## 14. AutoML & TabFM Usage

**AutoGluon**: Nuisance model ensembles (propensity, m₀, m₁) inside outer CV. Not for final CATE.

**TabFM**: Zero-shot challenger, non-commercial license, context-size sensitive.  
Experiments:
1. TabFM as risk model (baseline)
2. TabFM T-learner (m₀ and m₁ separately)
3. TabFM as DR nuisance estimator
4. TabFM as stacking feature (OOF predictions → meta-model)

**Expected champion**: CatBoost DR-learner (best calibration, stability, latency, license profile).

---

## 15. Complete uv Setup Commands

```bash
# Create environment
uv venv
source .venv/bin/activate

# Core dependencies
uv add pandas numpy scikit-learn scipy catboost lightgbm xgboost
uv add mlflow pydantic pyyaml
uv add fastapi uvicorn[standard] prometheus-client structlog shap dvc

# Dev dependencies
uv add --group dev pytest pytest-cov hypothesis ruff mypy pre-commit

# Optional groups
uv add --group automl autogluon
uv add --group tabfm torch transformers sentencepiece
uv add --group monitoring opentelemetry-sdk opentelemetry-instrumentation-fastapi
uv add --group notebooks jupyter matplotlib seaborn
uv add --group explainability lime

# Sync lockfile
uv sync --frozen
```

---

## 16. Priority Execution Plan

| Phase | What | Who |
|---|---|---|
| P0 | Leakage-safe data split, feature engineering, CatBoost DR-learner, T-learner baseline, policy value@50 | ML engineer |
| P0 | Calibration, SHAP explainability, bootstrap stability, top-50 CSV | ML engineer |
| P1 | DVC versioning, MLflow tracking, model bundle | ML engineer |
| P1 | FastAPI endpoints, Dockerfile, docker-compose | Backend/ML |
| P1 | Unit/property/golden tests | ML engineer |
| P2 | AutoGluon ensemble, TabFM challenger | ML engineer (research) |
| P2 | Prometheus/Grafana dashboards, alert rules | MLOps |
| P3 | Airflow DAGs, champion/challenger pipeline, randomized exploration | MLOps/Platform |

---

## 17. Failure Mode Checklist

- [ ] Leakage: time-based, preprocessing, target, group, stacking
- [ ] Calibration: fitted on training predictions (WRONG) → must be OOF
- [ ] Oversampling before split (WRONG) → must be inside fold
- [ ] SHAB explained as causal (WRONG) → explains model, not reality
- [ ] `contacted` used as a feature (WRONG) → it's the treatment
- [ ] Single seed 42 (WRONG) → multi-seed stability required
- [ ] AUC alone as success metric (WRONG) → policy value@50 is primary
- [ ] Missing day 0 temporal semantics (WRONG) → define explicitly
- [ ] Silent fallback (WRONG) → log fallback reason and level
- [ ] `customer_id` as Prometheus label (WRONG) → high cardinality
