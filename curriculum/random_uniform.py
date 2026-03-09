"""
Random Uniform curriculum strategy.

This is a non-curriculum variable-gravity control arm:
gravity is sampled independently each generation from a fixed range.
"""

import numpy as np

from .base import CurriculumBase


class RandomUniform(CurriculumBase):
    """Uniform random gravity schedule with deterministic per-generation sampling."""

    name = "random_uniform"
    description = "Gravity sampled uniformly each generation (no curriculum ordering)"

    def __init__(
        self,
        gravity_min: float = -12.0,
        gravity_max: float = -6.0,
        seed: int = 0,
    ):
        assert gravity_min < gravity_max, "gravity_min must be < gravity_max"
        assert gravity_max < 0.0, "Gravity values must be negative"
        self.gravity_min = float(gravity_min)
        self.gravity_max = float(gravity_max)
        self.seed = int(seed)

    def gravity(self, generation: int, max_generations: int) -> float:
        """
        Deterministic random draw for generation index.

        Uses a generation-specific RNG seed so results are reproducible and
        independent of call order.
        """
        local_seed = self.seed + (generation * 100_003)
        rng = np.random.default_rng(local_seed)
        return float(rng.uniform(self.gravity_min, self.gravity_max))

    def phase_label(self, generation: int, max_generations: int) -> str:
        return "random"

    def describe(self) -> str:
        return (
            f"RandomUniform  gravity U[{self.gravity_min}, {self.gravity_max}]  "
            f"seed={self.seed}"
        )
