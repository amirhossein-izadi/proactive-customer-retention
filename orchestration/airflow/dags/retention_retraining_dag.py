"""Retention retraining DAG — full pipeline retrain on schedule or trigger."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _check_data_quality():
    """Optional: validate new data before retraining."""
    q_path = Path("/app/artifacts/metrics/data_quality.json")
    if not q_path.exists():
        raise FileNotFoundError("No data quality report")
    q = json.loads(q_path.read_text())
    if q.get("join_coverage_pct", 0) < 100:
        raise ValueError(f"Join coverage {q['join_coverage_pct']}% < 100%")


with DAG(
    dag_id="retention_retraining",
    description="Full pipeline retrain (monthly)",
    schedule="0 4 1 * *",  # First day of each month at 4 AM
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args=default_args,
    tags=["retention", "retraining"],
) as dag:

    check_data = PythonOperator(
        task_id="check_data_quality",
        python_callable=_check_data_quality,
    )

    tune = BashOperator(
        task_id="hyperparameter_tuning",
        bash_command="cd /app && uv run retention tune --all --trials 20",
    )

    train = BashOperator(
        task_id="retrain_models",
        bash_command="cd /app && uv run retention train --optuna",
    )

    bundle = BashOperator(
        task_id="build_champion_bundle",
        bash_command="cd /app && uv run retention bundle",
    )

    rank = BashOperator(
        task_id="generate_ranking",
        bash_command="cd /app && uv run retention rank",
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /app && uv run retention evaluate",
    )

    check_data >> tune >> train >> bundle >> rank >> evaluate
