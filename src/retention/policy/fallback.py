"""Fallback logic when primary model abstains."""

from __future__ import annotations


class FallbackPolicy:
    def __init__(self, rule_based: bool = True):
        self.rule_based = rule_based

    def should_abstain(
        self, uncertainty: float, uplift: float, max_uncertainty_ratio: float = 0.5
    ) -> bool:
        if abs(uplift) < 1e-6:
            return True
        return uncertainty / abs(uplift) > max_uncertainty_ratio
