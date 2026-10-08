"""Tier 3: an optional second opinion on an ambiguous candidate. v0.1 ships the null one."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Arbiter(Protocol):
    def decide(self, feats: np.ndarray, candidate: str) -> float | None:
        """Probability that `candidate` is right, or None when unavailable."""
        ...


class NullArbiter:
    def decide(self, feats: np.ndarray, candidate: str) -> float | None:
        return None
