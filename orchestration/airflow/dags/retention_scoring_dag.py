"""Retention scoring DAG — runs weekly to generate top-k customer ranking."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retention_scoring",
    description="Generate top-50 retention ranking (weekly)",
    schedule="0 6 * * 1",  # Every Monday at 6 AM
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args=default_args,
    tags=["retention", "scoring"],
) as dag:

    rank = BashOperator(
        task_id="generate_ranking",
        bash_command="cd /app && uv run retention rank",
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /app && uv run retention evaluate",
    )

    rank >> evaluate
