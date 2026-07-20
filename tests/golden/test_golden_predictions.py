"""Golden test: verify prediction pipeline on fixture inputs."""
import json
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

def test_fixtures_exist():
    assert (FIXTURES / "nominal_customer.json").exists()
    assert (FIXTURES / "incomplete_history.json").exists()
    assert (FIXTURES / "unknown_category.json").exists()
    assert (FIXTURES / "invalid_usage.json").exists()

def test_fixtures_valid_json():
    for f in FIXTURES.glob("*.json"):
        data = json.loads(f.read_text())
        assert "customer_id" in data
        assert "package" in data

def test_nominal_values_in_bounds():
    data = json.loads((FIXTURES / "nominal_customer.json").read_text())
    assert 0 <= data["email_engagement"] <= 1
    assert data["monthly_revenue"] >= 0
    assert data["company_size"] >= 1

def test_invalid_rejected():
    data = json.loads((FIXTURES / "invalid_usage.json").read_text())
    assert data["monthly_revenue"] < 0
    assert data["email_engagement"] > 1
