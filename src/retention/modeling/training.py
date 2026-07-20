"""Thin re-export — delegates to ``train.runner`` for modular training.

Kept so ``from retention.modeling.training import train_experiment``
(used by cli.py) continues to work after the refactor.
"""

from __future__ import annotations

from retention.modeling.train.runner import train_experiment  # noqa: F401

__all__ = ["train_experiment"]
