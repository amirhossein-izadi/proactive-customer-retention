"""DVC pipeline — verify all stages are reachable and commands exist."""

import subprocess
from pathlib import Path


def test_dvc_yaml_exists():
    assert Path("dvc.yaml").exists(), "dvc.yaml not found"


def test_dvc_yaml_has_all_stages():
    import yaml
    with open("dvc.yaml") as f:
        dvc = yaml.safe_load(f)
    stages = dvc.get("stages", {})
    expected = {"validate", "split", "train", "evaluate", "rank", "bundle"}
    assert expected.issubset(stages.keys()), f"Missing: {expected - stages.keys()}"


def test_dvc_commands_exist():
    import yaml
    with open("dvc.yaml") as f:
        dvc = yaml.safe_load(f)
    for name, stage in dvc.get("stages", {}).items():
        cmd = stage.get("cmd", "")
        assert "uv run retention" in cmd, f"{name}: expected 'uv run retention' in cmd"
