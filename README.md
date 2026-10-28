# Didar Retention 
## Causal Uplift Model for Churn Reduction

A causal uplift modeling project on [Didar](https://didar.me) CRM data for proactive churn prevention. Given customer attributes and 60 days of product-usage history per customer, the system estimates each customer's churn reduction from being contacted and prioritizes exactly the top 50 customers for a retention campaign — ranking by treatment effect, not by raw churn risk.

> **Project links:** [Didar](https://didar.me) · [Technical Report](Didar-Customer-Retention-Technical-Report.pdf) · [Full Report](report.md) · [Model Card](docs/model_card.md) · [Methodology](docs/methodology_report.md)

**Champion**: logistic_t (T-learner with L2 logistic regression, Optuna-tuned)
**Test AUC**: 0.8036 | **Test DR@50**: 0.5692 | **Test Brier**: 0.1626

## Quick Start

```bash
# Install dependencies
uv sync

# Full pipeline (tune → train → bundle → rank → evaluate)
uv run retention run

# Or step by step:
uv run retention train --optuna       # 9 models + blend
uv run retention bundle                # champion bundle with SHAP
uv run retention rank                  # top-50 ranking
uv run retention evaluate              # full evaluation summary

# Docker stack
docker compose build api
docker compose up -d
curl http://localhost:8001/health
```

## Results

| Policy | DR@50 | Meaning |
|--------|-------|---------|
| **logistic_t (champion)** | **+0.569** | Contacting top 50 reduces churn by ~57pp |
| Random contact | +0.007 | Near zero |
| Risk-based (top churn) | -0.110 | **Negative** — historical contact was harmful |
| Historical policy | -0.001 | No better than random |

See `report.md` for full model comparison (9 models, 102 tests passing).

## CLI Commands

| Command | Description |
|---------|-------------|
| `retention validate` | Pandera schema validation |
| `retention split` | Stratified 80/20 dev/test split |
| `retention tune --top 3` | Optuna hyperparameter search |
| `retention train --optuna` | Train 9 models, auto-merge Optuna params |
| `retention evaluate` | Full evaluation summary |
| `retention rank` | Generate top-50 ranking |
| `retention bundle` | Build champion bundle |
| `retention ablate` | Feature ablation study |
| `retention run` | Full pipeline: tune → train → bundle → rank → evaluate |

## Architecture

9 models across 4 causal architectures (S/T/DR/X learners), each with logistic and CatBoost.
Champion selection: Dev OOF DR@50 with Brier < 0.20 calibration threshold.
All predictions cross-fitted (5-fold x 3-repeat). Preprocessing fitted inside folds.

## Key Features

- **53 features**: 10 static + 33 temporal + 5 PCA + 5 OHE
- **Optuna tuning**: S/T learner params + per-run CATE tuning (DR/X learners)
- **Calibration**: CalibratedClassifierCV with Platt scaling per treatment arm
- **Bundle**: Preprocessor + models + calibrators + propensity + SHAP explanations
- **API**: FastAPI with 5 endpoints, 14 error codes, Prometheus metrics
- **Docker**: Non-root user, uv-based, 4 services (API + MLflow + Prometheus + Grafana)
- **MLflow**: 154 runs logged (SQLite), full metrics + artifacts per model
- **DVC**: 6 pipeline stages, SHA256-tracked data
- **Tests**: 102 passing (unit, integration, leakage, contract, robustness, property, golden)

## Causal Assumptions

1. **Consistency**: `contacted=1` matches the intervention.
2. **Positivity**: Propensity clipping [0.02, 0.98] — 100% of dev customers in common support.
3. **Ignorability**: **Not guaranteed** — historical contact was confounded by indication.

## Project Structure

```
src/retention/
├── cli.py              # 9 CLI commands
├── run.py              # Full pipeline orchestrator
├── contracts/          # Pydantic schemas, error codes, inference policy
├── data/               # Loading, validation (Pandera), splitting, leakage register
├── features/           # Builder (53 features), temporal, missingness, registry
├── modeling/           # S/T/DR/X learners, cross-fitting, calibration, propensity
│   └── train/          # Runner, config, shared utilities, per-learner modules
├── tuning/             # Optuna search spaces, objectives, smart runner
├── evaluation/         # AUUC/Qini, policy value, bootstrap, calibration, subgroups
├── policy/             # Eligibility, ranking, tie-breaking, fallback
├── bundle/             # RetentionPolicyBundle, serializer, compatibility
├── serving/            # FastAPI (5 endpoints, error codes, metrics)
├── monitoring/         # Prometheus metrics, drift, audit logging
├── explainability/     # SHAP global/local, reason codes, uplift explanations
├── settings/           # YAML config loader
└── utils/              # Paths, seeds, serialization, hashing
```

## License

See `LICENSE`.
