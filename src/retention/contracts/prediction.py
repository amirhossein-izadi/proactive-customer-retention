"""Prediction output contracts — validated records for ranking and serving
and internal TypedDicts documenting the training pipeline data flow.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ── Ranking output schemas (API / CSV) ──


class RankedCustomer(BaseModel):
    """One row of the final campaign ranking."""

    model_config = ConfigDict(frozen=True)

    rank: Annotated[int, Field(ge=1)]
    customer_id: int
    p_churn_no_contact: Annotated[float, Field(ge=0.0, le=1.0)]
    p_churn_contact: Annotated[float, Field(ge=0.0, le=1.0)]
    estimated_uplift: Annotated[float, Field(ge=-1.0, le=1.0)]
    uplift_se: Annotated[float, Field(ge=0.0)]
    propensity: Annotated[float, Field(ge=0.0, le=1.0)]
    in_common_support: bool
    eligible: bool
    abstain: bool
    ranking_score: float
    monthly_revenue: Annotated[float, Field(ge=0.0)]


class RankingResult(BaseModel):
    """Validated campaign output — exactly top_k unique customers."""

    model_config = ConfigDict(frozen=True)

    customers: list[RankedCustomer]
    top_k: int
    model_version: str
    feature_version: str
    policy_version: str
    data_version: str

    def validate_campaign(self) -> RankingResult:
        ids = [c.customer_id for c in self.customers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate customer_id in ranking")
        if len(self.customers) > self.top_k:
            raise ValueError("ranking exceeds capacity")
        ranks = [c.rank for c in self.customers]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("ranks are not consecutive from 1")
        return self


# ── Internal pipeline data contracts (TypedDict) ──


class ChallengerPreds(TypedDict):
    """Standardised return type for ``train_t_learner`` / ``train_s_learner``.

    Every array is 1-D with length matching the respective (dev / test) set.
    """

    p0_t: np.ndarray  # P(churn | no contact) on test
    p1_t: np.ndarray  # P(churn | contact) on test
    e_t: np.ndarray  # Propensity on test
    uplift_t: np.ndarray  # p0_t - p1_t
    se_t: np.ndarray  # Std error of uplift across repeats
    cids_t: np.ndarray  # Test customer IDs
    rev_t: np.ndarray  # Test monthly revenue
    p0_d: np.ndarray  # Dev OOF P(churn | no contact)
    p1_d: np.ndarray  # Dev OOF P(churn | contact)


class ChallengerMetrics(TypedDict):
    """Per-challenger comparison row written to ``model_comparison.csv``."""

    model: str
    dev_auc: float
    dev_brier: float
    dev_dr50: float
    test_auc: float
    test_brier: float
    test_arm0_auc: float
    test_arm1_auc: float
    test_dr50: float
    test_uplift50: float
