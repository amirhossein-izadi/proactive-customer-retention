"""Validate champion bundle — used by CI/CD model_validation workflow."""
from retention.bundle.compatibility import check_bundle

result = check_bundle("models/champion")
if not result["passed"]:
    print(f"FAILED: {result['errors']}")
    raise SystemExit(1)
print(f"Bundle OK: {len(result['checks'])} checks passed")
