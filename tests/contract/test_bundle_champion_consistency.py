"""Bundle champion consistency — served arm models must match the reported champion.

Regression guard for train/serve skew: reporting and ranking describe the model
named in ``metadata.json``, so the pickled arm models must be that same family.
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path

import pytest

from retention.bundle.compatibility import check_bundle

CHAMPION_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "champion"


@pytest.fixture()
def bundle_copy():
    if not (CHAMPION_DIR / "metadata.json").exists() or not (CHAMPION_DIR / "m0.pkl").exists():
        pytest.skip("champion bundle not present")
    tmp = Path(tempfile.mkdtemp()) / "champion"
    shutil.copytree(CHAMPION_DIR, tmp)
    return tmp


def test_committed_bundle_matches_reported_champion(bundle_copy):
    result = check_bundle(bundle_copy)
    consistency = result["checks"].get("champion_consistency", "")
    assert result["passed"], f"bundle invalid: {result['errors']}"
    assert consistency.startswith("ok"), f"champion consistency not verified: {consistency}"


def test_skew_is_detected(bundle_copy):
    """Claiming a different family than the pickles must fail validation."""
    meta = json.loads((bundle_copy / "metadata.json").read_text())
    current = type(pickle.loads((bundle_copy / "m0.pkl").read_bytes())).__name__
    meta["champion_name"] = "catboost_t" if "Logistic" in current else "logistic_t"
    (bundle_copy / "metadata.json").write_text(json.dumps(meta))

    result = check_bundle(bundle_copy)

    assert not result["passed"], "train/serve skew must fail bundle validation"
    assert any("skew" in e.lower() for e in result["errors"]), result["errors"]
