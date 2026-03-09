"""
Staged Evolution curriculum strategy.

Gravity advances through discrete stages at fixed generation boundaries.
At each stage transition the training loop restarts CMA-ES centred on the
best solution found so far ("population transfer" analogue for CMA-ES).

Schedule (defaults, 500 generations, 3 equal stages)
-----------------------------------------------------
  gen   0 – 166 : g = -1.6  (Moon gravity)
  gen 167 – 333 : g = -5.0  (intermediate)
  gen 334 – 499 : g = -9.81 (Earth gravity)

At gen 167 and gen 334 `at_stage_boundary()` returns True, causing the
training loop to restart CMA-ES from `best_params` (same sigma as initial).

Parameters
----------
stage_gravities : tuple[float, ...]  default (-1.6, -5.0, -9.81)
    Gravity value for each stage. All must be negative.
stage_fracs     : tuple[float, ...]  default (1/3, 1/3, 1/3)
    Fraction of max_gen assigned to each stage. Automatically normalised.
seed            : int  default 0   (unused; kept for API consistency)
"""

from __future__ import annotations

from .base import CurriculumBase


class StagedEvolution(CurriculumBase):
    """Discrete-stage gravity curriculum with ES restart at stage boundaries."""

    name = "staged_evolution"

    def __init__(
        self,
        stage_gravities: tuple[float, ...] = (-1.6, -5.0, -9.81),
        stage_fracs: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3),
        seed: int = 0,
    ):
        assert len(stage_gravities) == len(stage_fracs), (
            "stage_gravities and stage_fracs must have the same length"
        )
        assert all(g < 0 for g in stage_gravities), "All gravities must be negative"
        self.stage_gravities = [float(g) for g in stage_gravities]
        total = sum(stage_fracs)
        self.stage_fracs = [float(f) / total for f in stage_fracs]
        # Internal state for boundary detection
        self._last_stage: int = -1

    # ------------------------------------------------------------------ core
    def _get_stage(self, gen: int, max_gen: int) -> int:
        cumulative = 0.0
        for i, frac in enumerate(self.stage_fracs[:-1]):
            cumulative += frac
            if gen < cumulative * max_gen:
                return i
        return len(self.stage_gravities) - 1

    def gravity(self, gen: int, max_gen: int) -> float:
        return self.stage_gravities[self._get_stage(gen, max_gen)]

    def phase_label(self, gen: int, max_gen: int) -> str:
        stage = self._get_stage(gen, max_gen)
        labels = ["moon", "mid", "earth"]
        # Extend for custom stage counts
        while len(labels) < len(self.stage_gravities):
            labels.append(f"s{len(labels)}")
        return labels[stage]

    def describe(self) -> str:
        stages = " → ".join(
            f"{g} ({int(f * 100)}%)"
            for g, f in zip(self.stage_gravities, self.stage_fracs)
        )
        return f"StagedEvolution  {stages}"

    # -------------------------------------------------- stage transition hook
    def at_stage_boundary(self, gen: int, max_gen: int) -> bool:
        """Returns True once at each stage transition, then False until next."""
        current = self._get_stage(gen, max_gen)
        prev = self._last_stage
        self._last_stage = current
        return current != prev and prev >= 0
