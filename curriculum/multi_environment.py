"""
Multi-Environment Training curriculum strategy.

Every candidate is evaluated simultaneously across multiple gravity levels
each generation. Fitness is a weighted mean, with the highest weight on the
target gravity (Earth). This forces policies to generalise across gravities
rather than specialising for any single one.

Training cost: proportional to the number of eval gravities (3 by default).

Default evaluation gravities and weights
-----------------------------------------
  g = -1.6  (Moon)     weight 0.20
  g = -5.0  (mid)      weight 0.30
  g = -9.81 (Earth)    weight 0.50

`gravity()` always returns target_gravity (-9.81) so that Earth probes and
final evaluations use the correct reference point.

Parameters
----------
eval_gravities  : tuple[float, ...]  default (-1.6, -5.0, -9.81)
    Gravities at which each candidate is evaluated. All must be negative.
weights         : tuple[float, ...]  default (0.20, 0.30, 0.50)
    Per-gravity fitness weights. Automatically normalised to sum to 1.
target_gravity  : float  default -9.81
    Returned by gravity() for probe/final-eval purposes.
seed            : int  default 0   (unused; kept for API consistency)
"""

from __future__ import annotations

import numpy as np

from .base import CurriculumBase


class MultiEnvironment(CurriculumBase):
    """Simultaneous multi-gravity evaluation with weighted fitness aggregation."""

    name = "multi_environment"

    def __init__(
        self,
        eval_gravities: tuple[float, ...] = (-1.6, -5.0, -9.81),
        weights: tuple[float, ...] = (0.20, 0.30, 0.50),
        target_gravity: float = -9.81,
        seed: int = 0,
    ):
        assert len(eval_gravities) == len(weights), (
            "eval_gravities and weights must have the same length"
        )
        assert all(g < 0 for g in eval_gravities), "All gravities must be negative"
        assert target_gravity < 0, "target_gravity must be negative"
        self._eval_gravities = [float(g) for g in eval_gravities]
        raw = [float(w) for w in weights]
        total = sum(raw)
        self._weights = [w / total for w in raw]
        self.target_gravity = float(target_gravity)

    # ------------------------------------------------------------------ core
    def gravity(self, gen: int, max_gen: int) -> float:
        """Primary gravity (always target) — used for probes and final eval."""
        return self.target_gravity

    def phase_label(self, gen: int, max_gen: int) -> str:
        return f"multi({len(self._eval_gravities)})"

    def describe(self) -> str:
        g_str = ",".join(f"{g:.1f}" for g in self._eval_gravities)
        w_str = ",".join(f"{w:.2f}" for w in self._weights)
        return f"MultiEnvironment  g=[{g_str}]  w=[{w_str}]"

    # ---------------------------------------------------------- multi-gravity
    def eval_gravities(self, gen: int, max_gen: int) -> list[float]:
        return list(self._eval_gravities)

    def combine_fitness(self, scores: list[float]) -> float:
        return float(sum(s * w for s, w in zip(scores, self._weights)))
