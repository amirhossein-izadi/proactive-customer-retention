"""One CLI entry point for all retention pipeline commands."""

from __future__ import annotations

import argparse
import sys

from retention.bundle.serializer import build_champion_bundle
from retention.data.loading import load_all
from retention.data.splitting import build_split_pipeline
from retention.data.validation import validate_customers, validate_usage
from retention.evaluation.ablation import test_ablation
from retention.evaluation.reporting import evaluate_experiment
from retention.modeling.train.runner import train_experiment
from retention.policy.ranking import generate_final_ranking
from retention.run import run_pipeline
from retention.tuning.optuna_runner import smart_tune
from retention.utils.timing import timed


def _timed(name: str):
    return timed(name)


def cmd_validate() -> None:
    with _timed("validate"):
        customers, usage = load_all()
        validate_customers(customers)
        validate_usage(usage)
        print("Validation passed.")


def cmd_split() -> None:
    with _timed("split"):
        customers, _ = load_all()
        build_split_pipeline(customers)


def cmd_train() -> None:
    parser = argparse.ArgumentParser(prog="retention train")
    parser.add_argument("--optuna", action="store_true", help="Auto-merge Optuna best params")
    args, _ = parser.parse_known_args()
    with _timed("train"):
        train_experiment(use_optuna=args.optuna)


def cmd_evaluate() -> None:
    with _timed("evaluate"):
        evaluate_experiment()


def cmd_rank() -> None:
    with _timed("rank"):
        generate_final_ranking()


def cmd_bundle() -> None:
    with _timed("bundle"):
        build_champion_bundle()


def cmd_ablate() -> None:
    with _timed("ablate"):
        test_ablation()


def cmd_tune() -> None:
    parser = argparse.ArgumentParser(prog="retention tune")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top", type=int, default=0, help="Tune top N models by test DR@50")
    group.add_argument("--all", action="store_true", help="Tune all enabled models")
    group.add_argument("--arch", type=str, default="", help="Specific architecture")
    parser.add_argument("--model", type=str, default="", help="Specific model")
    parser.add_argument("--trials", type=int, default=0, help="Override trial count")
    args, _ = parser.parse_known_args()
    with _timed("tune"):
        smart_tune(top_k=args.top, all_models=args.all, arch=args.arch, model=args.model, n_trials=args.trials)


def cmd_run() -> None:
    parser = argparse.ArgumentParser(prog="retention run")
    parser.add_argument("--skip-tune", action="store_true", help="Skip hyperparameter tuning")
    args, _ = parser.parse_known_args()
    with _timed("run"):
        run_pipeline(skip_tune=args.skip_tune)


CMDS = {
    "validate": cmd_validate, "split": cmd_split,
    "train": cmd_train, "evaluate": cmd_evaluate,
    "rank": cmd_rank, "bundle": cmd_bundle,
    "ablate": cmd_ablate, "tune": cmd_tune, "run": cmd_run,
}

USAGE = f"Usage: retention {{{'|'.join(CMDS.keys())}}}"


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd not in CMDS:
        print(f"Unknown: {cmd}. Available: {list(CMDS)}")
        sys.exit(1)
    CMDS[cmd]()


if __name__ == "__main__":
    main()
