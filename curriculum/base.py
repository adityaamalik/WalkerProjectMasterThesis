"""
Base class for all curriculum learning strategies.

Extended interface
------------------
Beyond the original gravity()/phase_label()/describe() trio, strategies may
optionally override:

  eval_gravities(gen, max_gen) -> list[float]
      Gravities at which each candidate is evaluated this generation.
      Default: [self.gravity(gen, max_gen)]  — single gravity, no extra cost.

  combine_fitness(scores: list[float]) -> float
      How to aggregate per-gravity fitness values into one scalar.
      Default: arithmetic mean.

  notify(gen, max_gen, best_score, mean_score, sigma) -> None
      Called once per generation with population statistics.
      Adaptive strategies use this to update their internal state.

  at_stage_boundary(gen, max_gen) -> bool
      Returns True on the first call after a stage transition.
      Triggers an ES restart in the training loop.
      Default: always False (no restarts).

  get_seed_params() -> np.ndarray | None
      Called immediately after at_stage_boundary() returns True.
      Returns parameters to warm-start the new ES from.
      Default: None (training loop falls back to best_params).

  on_new_best(params, score, gen) -> None
      Called every generation with the generation's best candidate.
      Archive-based strategies store solutions here.
      Default: no-op.
"""

from __future__ import annotations

import numpy as np


class CurriculumBase:
    """Abstract base for curriculum strategies with extended hook interface."""

    # ------------------------------------------------------------------ core
    def gravity(self, gen: int, max_gen: int) -> float:
        raise NotImplementedError

    def phase_label(self, gen: int, max_gen: int) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    # ---------------------------------------------------------- multi-gravity
    def eval_gravities(self, gen: int, max_gen: int) -> list[float]:
        """Gravities to evaluate each candidate at. Default: single gravity."""
        return [self.gravity(gen, max_gen)]

    def combine_fitness(self, scores: list[float]) -> float:
        """Aggregate per-gravity fitness values. Default: arithmetic mean."""
        return float(np.mean(scores))

    # ------------------------------------------------------------ adaptive
    def notify(
        self,
        gen: int,
        max_gen: int,
        best_score: float,
        mean_score: float,
        sigma: float,
    ) -> None:
        """Called after each generation with population statistics. No-op by default."""

    # -------------------------------------------------- stage transitions
    def at_stage_boundary(self, gen: int, max_gen: int) -> bool:
        """Returns True when a stage transition just occurred. No restarts by default."""
        return False

    def get_seed_params(self) -> np.ndarray | None:
        """Warm-start params for ES restart. None → training loop uses best_params."""
        return None

    # ------------------------------------------------------- archive hook
    def on_new_best(
        self, params: np.ndarray, score: float, gen: int
    ) -> None:
        """Called every generation with the best candidate. No-op by default."""
