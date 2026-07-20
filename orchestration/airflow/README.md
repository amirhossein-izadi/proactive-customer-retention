# Airflow Orchestration

Two DAGs are provided:

- **retention_scoring_dag**: runs `retention rank` periodically (e.g., weekly)
  to produce the current top-k customer ranking.
- **retention_retraining_dag**: runs the full pipeline (train → evaluate →
  bundle → rank) when triggered by new data or on a schedule.

Both expect the `retention` CLI installed and the `.venv` available. Set
`AIRFLOW__CORE__DAGS_FOLDER` or symlink these DAGs into your dags/ dir.
