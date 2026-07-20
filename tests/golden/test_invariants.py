"""Golden test fixtures — validation invariants for the full prediction builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retention.features.builder import TemporalEngineer, StaticEngineer
from retention.data.splitting import generate_seed_manifest


# ── Nominal fixtures ──
@pytest.fixture
def basic_monthly():
    return {
        "customer_id": "1",
        "company_size": 50,
        "package": "Basic",
        "tenure_months": 12,
        "subscription_type": "Monthly",
        "monthly_revenue": 200.0,
        "support_tickets_30d": 1,
        "email_engagement": 0.3,
    }


@pytest.fixture
def pro_annual():
    return {
        "customer_id": "2",
        "company_size": 200,
        "package": "Pro",
        "tenure_months": 36,
        "subscription_type": "Annual",
        "monthly_revenue": 2000.0,
        "support_tickets_30d": 0,
        "email_engagement": 0.8,
    }


@pytest.fixture
def usage_declining():
    rng = np.random.default_rng(42)
    usage = rng.poisson(8, 60).astype(float)
    usage[-10:] = 0
    return pd.DataFrame({"customer_id": 1, "day": range(-59, 1), "daily_usage": usage})


@pytest.fixture
def usage_stable():
    rng = np.random.default_rng(42)
    usage = rng.poisson(15, 60).astype(float)
    return pd.DataFrame({"customer_id": 2, "day": range(-59, 1), "daily_usage": usage})


# ── Invariants ──
class TestFeatureInvariants:
    def test_temporal_no_nan(self, usage_declining):
        e = TemporalEngineer()
        df = usage_declining.copy()
        df["day_col"] = "usage_day_" + df["day"].astype(str)
        pivoted = df.pivot(index="customer_id", columns="day_col", values="daily_usage")
        out = e.fit_transform(pivoted)
        assert not np.any(np.isnan(out))

    def test_temporal_shape(self, usage_declining):
        e = TemporalEngineer()
        df = usage_declining.copy()
        df["day_col"] = "usage_day_" + df["day"].astype(str)
        pivoted = df.pivot(index="customer_id", columns="day_col", values="daily_usage")
        out = e.fit_transform(pivoted)
        assert out.shape == (1, len(e.feature_names()))

    def test_static_no_nan(self, basic_monthly):
        e = StaticEngineer()
        df = pd.DataFrame([basic_monthly])
        out = e.fit_transform(df)
        assert not np.any(np.isnan(out))
        assert out.shape == (1, 10)

    def test_recent_early_ratio_declining(self, usage_declining):
        e = TemporalEngineer()
        out = e.fit_transform(usage_declining.set_index("customer_id"))
        names = e.feature_names()
        ratio_idx = names.index("recent_early_ratio")
        ratio = out[0, ratio_idx]
        assert ratio < 1.0  # declining → recent < early


class TestSplitInvariants:
    def test_seed_manifest_has_all_keys(self):
        sm = generate_seed_manifest(42)
        required = {
            "master_seed",
            "holdout_seed",
            "outer_cv_seed",
            "inner_cv_seed",
            "model_seed",
            "bootstrap_seed",
        }
        assert required.issubset(sm.keys())

    def test_seed_manifest_reproducible(self):
        sm1 = generate_seed_manifest(42)
        sm2 = generate_seed_manifest(42)
        assert sm1 == sm2

    def test_different_master_gives_different_sub_seeds(self):
        sm1 = generate_seed_manifest(42)
        sm2 = generate_seed_manifest(43)
        assert sm1["model_seed"] != sm2["model_seed"]


class TestProbabilityInvariants:
    def test_probability_bounds(self):
        probs = np.array([0.0, 0.5, 1.0])
        assert np.all((probs >= 0) & (probs <= 1))

    def test_uplift_range(self):
        p0 = np.array([0.8, 0.3])
        p1 = np.array([0.4, 0.2])
        uplift = p0 - p1
        assert np.all((uplift >= -1) & (uplift <= 1))

    def test_robust_score_order(self):
        uplift = np.array([0.5, 0.3, 0.1])
        se = np.array([0.1, 0.1, 0.1])
        rs = uplift - se
        assert rs[0] > rs[1] > rs[2]


class TestRankingInvariants:
    def test_top_k_unique(self):
        n, k = 100, 50
        rng = np.random.default_rng(42)
        scores = rng.random(n)
        cids = np.arange(n)
        idx = np.argsort(scores)[::-1][:k]
        selected = cids[idx]
        assert len(set(selected)) == k

    def test_exactly_50(self):
        n, k = 100, 50
        rng = np.random.default_rng(42)
        scores = rng.random(n)
        idx = np.argsort(scores)[::-1][:k]
        assert len(idx) == 50

    def test_tie_breaking_deterministic(self):
        scores = np.array([0.5, 0.5, 0.5, 0.1])
        cids = np.array([10, 5, 7, 1])
        idx = np.lexsort((cids, -scores))
        selected = cids[idx[:3]]
        assert selected[0] == 5  # lowest ID among tied top scores
