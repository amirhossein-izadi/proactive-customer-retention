# Model Card: Didar Retention Champion

- **Model family**: CatBoost T-learner
- **Model version**: 1.1.0
- **Feature version**: 2.2.0
- **Data version**: didar_retention_v1
- **Training samples**: 3200
- **Features**: 53

## Limitations
- Observational (not experimental) — unmeasured confounding remains a risk.
- Calibration on dev only; test-set calibration not reported.
- Top-k ranking is deterministic only within the same bundle version.
