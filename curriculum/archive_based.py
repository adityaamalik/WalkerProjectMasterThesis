"""
Archive-Based Curriculum strategy.

Maintains a solution archive — the top-K parameter vectors found at each
gravity level. When fitness plateaus and gravity advances, CMA-ES is
restarted ("seeded") from the best archived solution at the previous level
rather than the global best_params. This preserves solutions optimised for
easier gravities and uses them as a scaffold for harder conditions, reducing
catastrophic forgetting.

Mechanics
---------
  on_new_best()        — called every generation; stores candidate in the
                         current level's archive when it improves the
                         per-level best.
  notify()             — plateau detection identical to AdaptiveProgression.
  at_stage_boundary()  — returns True immediately after gravity advances.
  get_seed_params()    — returns the top archived solution from the PREVIOUS
                         gravity level to warm-start the new CMA-ES.

Default gravity levels
-----------------------
  level 0 : -1.6  (Moon)
  level 1 : -3.0
  level 2 : -5.0
  level 3 : -7.0
  level 4 : -9.81 (Earth)

Parameters
----------
gravity_levels    : tuple[float, ...]  default (-1.6, -3.0, -5.0, -7.0, -9.81)
plateau_threshold : int    default 30
min_improvement   : float  default 5.0
archive_size      : int    default 5  (top-K per gravity level)
seed              : int    default 0  (unused)
"""

from __future__ import annotations

import numpy as np

from .base import CurriculumBase


class ArchiveBased(CurriculumBase):
    """Archive-seeded curriculum with plateau-triggered gravity advances."""

    name = "archive_based"

    def __init__(
        self,
        gravity_levels: tuple[float, ...] = (-1.6, -3.0, -5.0, -7.0, -9.81),
        plateau_threshold: int = 30,
        min_improvement: float = 5.0,
        archive_size: int = 5,
        seed: int = 0,
    ):
        assert len(gravity_levels) >= 2, "Need at least 2 gravity levels"
        assert all(g < 0 for g in gravity_levels), "All gravity levels must be negative"
        self.gravity_levels = [float(g) for g in gravity_levels]
        self.plateau_threshold = int(plateau_threshold)
        self.min_improvement = float(min_improvement)
        self.archive_size = int(archive_size)

        # Mutable state — updated by notify() and on_new_best()
        self._level: int = 0
        self._plateau_counter: int = 0
        self._best_at_level: float = -float("inf")
        self._just_advanced: bool = False

        # Archive: level_idx -> list of (score, params) sorted descending by score
        self._archive: dict[int, list[tuple[float, np.ndarray]]] = {
            i: [] for i in range(len(gravity_levels))
        }

    # ------------------------------------------------------------------ core
    def gravity(self, gen: int, max_gen: int) -> float:
        return self.gravity_levels[self._level]

    def phase_label(self, gen: int, max_gen: int) -> str:
        arch_n = len(self._archive.get(self._level, []))
        return f"lv{self._level}(a={arch_n})"

    def describe(self) -> str:
        levels = "→".join(str(g) for g in self.gravity_levels)
        return (
            f"ArchiveBased  [{levels}]  "
            f"plateau={self.plateau_threshold}gen  "
            f"archive_size={self.archive_size}"
        )

    # ------------------------------------------------------- archive hook
    def on_new_best(self, params: np.ndarray, score: float, gen: int) -> None:
        """Store candidate in archive if it belongs to the top-K at this level."""
        level = self._level
        archive = self._archive[level]
        # Add when archive is not yet full, or score beats the current worst entry
        if len(archive) < self.archive_size or score > archive[-1][0]:
            archive.append((float(score), params.copy()))
            archive.sort(key=lambda x: x[0], reverse=True)
            self._archive[level] = archive[: self.archive_size]

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

    # -------------------------------------------------- stage transition hooks
    def at_stage_boundary(self, gen: int, max_gen: int) -> bool:
        """True immediately after notify() advanced the gravity level."""
        return self._just_advanced

    def get_seed_params(self) -> np.ndarray | None:
        """Return the best archived solution from the previous gravity level."""
        prev = self._level - 1
        if prev >= 0 and self._archive[prev]:
            return self._archive[prev][0][1].copy()
        return None
