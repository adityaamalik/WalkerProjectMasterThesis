"""
Adaptive Progression curriculum strategy.

Gravity advances through discrete levels automatically when the population
fitness plateaus. A plateau is declared when the best fitness seen at the
current gravity level fails to improve by at least `min_improvement` for
`plateau_threshold` consecutive generations. On plateau, gravity steps to
the next level and CMA-ES is restarted (via `at_stage_boundary()`) centred
on the best solution found so far.

This lets evolution "choose" when it is ready for a harder challenge rather
than following a fixed generational schedule.

Default gravity levels
-----------------------
  level 0 : -1.6   (Moon)
  level 1 : -3.0
  level 2 : -5.0
  level 3 : -7.0
  level 4 : -9.81  (Earth) — final level, no further advance

Parameters
----------
gravity_levels     : tuple[float, ...]  default (-1.6, -3.0, -5.0, -7.0, -9.81)
    Ordered list of gravity values (all negative). Must be strictly decreasing
    (more negative = harder).
plateau_threshold  : int    default 30
    Consecutive generations without min_improvement before advancing.
min_improvement    : float  default 5.0
    Minimum fitness gain (at current level) to reset the plateau counter.
seed               : int    default 0  (unused; kept for API consistency)
"""

from __future__ import annotations

from .base import CurriculumBase


class AdaptiveProgression(CurriculumBase):
    """Gravity advances when fitness plateaus; ES is restarted at each advance."""

    name = "adaptive_progression"

    def __init__(
        self,
        gravity_levels: tuple[float, ...] = (-1.6, -3.0, -5.0, -7.0, -9.81),
        plateau_threshold: int = 30,
        min_improvement: float = 5.0,
        seed: int = 0,
    ):
        assert len(gravity_levels) >= 2, "Need at least 2 gravity levels"
        assert all(g < 0 for g in gravity_levels), "All gravity levels must be negative"
        self.gravity_levels = [float(g) for g in gravity_levels]
        self.plateau_threshold = int(plateau_threshold)
        self.min_improvement = float(min_improvement)

        # Mutable state — updated by notify()
        self._level: int = 0
        self._plateau_counter: int = 0
        self._best_at_level: float = -float("inf")
        self._just_advanced: bool = False

    # ------------------------------------------------------------------ core
    def gravity(self, gen: int, max_gen: int) -> float:
        return self.gravity_levels[self._level]

    def phase_label(self, gen: int, max_gen: int) -> str:
        g = self.gravity_levels[self._level]
        return f"lv{self._level}({g:.1f})"

    def describe(self) -> str:
        levels = "→".join(str(g) for g in self.gravity_levels)
        return (
            f"AdaptiveProgression  [{levels}]  "
            f"plateau={self.plateau_threshold}gen  "
            f"min_improve={self.min_improvement}"
        )

    # ------------------------------------------------------------ notify hook
    def notify(
        self,
        gen: int,
        max_gen: int,
        best_score: float,
        mean_score: float,
        sigma: float,
    ) -> None:
        """Update plateau counter; advance gravity level when plateau detected."""
        self._just_advanced = False

        if best_score > self._best_at_level + self.min_improvement:
            self._best_at_level = best_score
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1

        if (
            self._plateau_counter >= self.plateau_threshold
            and self._level < len(self.gravity_levels) - 1
        ):
            self._level += 1
            self._plateau_counter = 0
            self._best_at_level = -float("inf")
            self._just_advanced = True

    # -------------------------------------------------- stage transition hook
    def at_stage_boundary(self, gen: int, max_gen: int) -> bool:
        """True immediately after notify() advanced the gravity level."""
        return self._just_advanced
