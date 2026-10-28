"""Full pipeline orchestrator — one command from tune to evaluate."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from retention.bundle.serializer import build_champion_bundle
from retention.evaluation.reporting import evaluate_experiment
from retention.modeling.train.runner import train_experiment
from retention.policy.ranking import generate_final_ranking
from retention.tuning.optuna_runner import smart_tune
from retention.utils.paths import ARTIFACTS, CHAMPION_DIR, OOF_DIR, REPORTS


def run_pipeline(skip_tune: bool = False) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")

    optuna_results = ARTIFACTS / "optuna" / "all_best.json"
    if not skip_tune and not optuna_results.exists():
        print("\n=== Phase 1/5: Hyperparameter Tuning ===")
        smart_tune(all_models=True)
    elif skip_tune:
        print("\n=== Phase 1/5: Tuning skipped (--skip-tune) ===")
    else:
        print(f"\n=== Phase 1/5: Tuning cached ({optuna_results}) ===")

    print("\n=== Phase 2/5: Training ===")
    train_experiment(use_optuna=optuna_results.exists())

    print("\n=== Phase 3/5: Bundle ===")
    build_champion_bundle()

    print("\n=== Phase 4/5: Ranking ===")
    generate_final_ranking()

    print("\n=== Phase 5/5: Evaluation ===")
    evaluate_experiment()

    print(f"\n=== Syncing to {run_dir} ===")
    for src, name in [
        (CHAMPION_DIR, "champion"),
        (OOF_DIR, "oof_predictions"),
        (ARTIFACTS / "rankings", "rankings"),
        (ARTIFACTS / "metrics", "metrics"),
        (REPORTS, "reports"),
    ]:
        if src.exists():
            dst = run_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
            print(f"  {name}/")

    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "timestamp": datetime.now().isoformat()
    }, indent=2, default=str))
    (ARTIFACTS / "runs" / "latest.txt").write_text(run_id)

    print(f"\nPipeline complete. Run: {run_id}")
    return run_id
