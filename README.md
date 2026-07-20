# Didar Retention — Causal Uplift Model for Churn Reduction

**Status**: Structured ML experiment with production-oriented scaffolding.
**Champion architecture**: CatBoost T-learner (two separate outcome models).

A full pipeline trains multiple challenger models (logistic, CatBoost, LightGBM,
XGBoost, DR-learner — each as T-learner or S-learner), selects the champion by
dev OOF DR policy value, and generates a deterministic top-50 customer ranking.

## Quick Start

```bash
uv sync --frozen --all-groups          # no `uv sync` needed if already synced
uv pip install --no-deps -e .          # install the `retention` CLI
retention validate                     # pandera schema validation + quality report
retention split                        # stratified dev/test split (20%)
retention train                        # all challengers → champion (takes ~10 min)
retention evaluate                     # champion metric summary
retention rank                         # save reports/final_ranking.csv
retention bundle                       # model artifacts to models/champion/
retention ablate                       # feature ablation table
```

## Key Architectural Decisions

- **Scaled preprocessor**: `build_preprocessor` in `features/builder.py` wraps
  the ColumnTransformer with a `StandardScaler` — necessary for logistic convergence
  and harmless for tree models.
- **Package one-hot encoding**: `package` (Basic/Pro/Enterprise) is now passed
  to all models. Previously it was silently dropped.
- **Chronological temporals**: usage columns are always ordered by integer day,
  never by lexicographic string sort, so “recent” features are correct.
- **Missing ≠ zero**: `missing_days` and `zero_days` are separate features.
- **Champion selected on dev**: test set is evaluated only once for the final
  champion — no test-set leakage.
- **S-learner OOF**: both T-learner and S-learner produce real out-of-fold
  predictions; dev metrics are not placeholders.

## Causal Assumptions

1. **Consistency**: `contacted=1` matches the intervention. Satisfied by data.
2. **Positivity**: propensity clipping `[0.02, 0.98]` — ~1.5% of customers outside.
3. **Ignorability**: **not guaranteed** — account manager targeting confounds
   treatment assignment. The uplift estimate is observational.

## Structure

```
src/retention/
├── cli.py                       # 7 pipeline commands
├── contracts/                   # Pydantic schemas + error codes + inference policy
├── data/                        # loading, validation, splitting, joining, snapshots
├── features/                    # builder (canonical preprocessor), temporal, missingness, registry
├── modeling/                    # learners (S/T/DR), cross_fitting, calibration, factories
│   └── train/                   # modular training runner (base, config, runner, s/t learner)
├── evaluation/                  # predictive metrics, uplift curves, baselines, bootstrap
├── policy/                      # eligibility, support, tie-breaking, ranking
├── bundle/                      # serialization, metadata, compatibility check
├── serving/                     # FastAPI (health, score, rank, model-info)
├── monitoring/                  # prometheus metrics, drift, data quality, audit logging
└── utils/                       # paths, hashing, serialization, timing
```

## What Is Known To Work (Verified by Import / Smoke Tests)

- `retention validate` — pandera checks on 4,000 customers, 240,000 usage rows
- `retention split` — stratified joint (T,Y) 80/20 split
- `retention evaluate` (with synthetic npz) — metric summary including DR policy value
- `retention rank` (with synthetic npz) — deterministic top-50 with support filter
- **S/T/DR learners** — unit-tested against synthetic beneficial treatment
- **Temporal feature ordering** — permutation-invariance tested
- **Missingness** — NaN not conflated with zero, tested
- **FastAPI /health, /ready, /v1/score, /v1/rank** — import-smoke tested
- **Uplift sign** — positive for synthetic beneficial treatment (both S and T learner)
- **Holdout isolation** — zero overlap between dev and test customer ids
- **Bundle roundtrip** — pickle → unpickle preserves predictions

## Known Gaps

- **Full training run** not executed end-to-end after the refactor — the previous
  run crashed on the S-learner branch. A full `retention train` is needed to
  produce actual `champion_predictions.npz` and populate `reports/` + `artifacts/`.
- **SHAP, MLflow pyfunc, 500-sample bootstrap, CausalForest** — code structures exist
  but are not yet wired into the execution path (see `docs/EXPERT_SCORECARD.md` warning).
- **Docker serving** requires the bundle to exist first (`retention bundle`).
- **Monitoring dashboards** (Grafana JSON) are shipped as static configuration,
  not verified with live data.

## License

See `LICENSE` (delivered as part of the challenge submission).
