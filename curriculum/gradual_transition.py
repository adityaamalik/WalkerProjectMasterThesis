"""
Gradual Transition curriculum strategy.

Gravity starts easy (low magnitude) and linearly increases toward earth gravity
over the course of training.

Rationale
---------
In low gravity the walker falls more slowly, forces are gentler, and the
fitness landscape is smoother — easier for CMA-ES to find a stable gait.
As gravity increases the gait is progressively stressed until the creature
must handle full earth-gravity locomotion.

Schedule
--------
  Phase 1 (0 → warmup_frac × max_gen):
      gravity = gravity_start   (hold at easy gravity while a basic gait forms)

  Phase 2 (warmup_frac → 1.0 of max_gen):
      gravity linearly interpolates from gravity_start → gravity_end

  Example (defaults, 500 generations):
      gen   0 –  99  : gravity = -2.00  (warmup)
      gen 100 – 499  : gravity -2.00 → -9.81  (linear ramp)
      gen 499        : gravity = -9.81  (full earth)

Parameters
----------
  gravity_start : float, default -2.00   easy gravity (Moon-ish)
  gravity_end   : float, default -9.81   target gravity (Earth)
  warmup_frac   : float, default 0.20    fraction of training spent at easy gravity
"""


from .base import CurriculumBase  # noqa: E402 (placed here to avoid shadowing docstring)


class GradualTransition(CurriculumBase):
    """
    Linear gravity curriculum from easy → earth.

    Attributes
    ----------
    gravity_start : float   Starting (easy) gravity value  (negative, e.g. -2.0)
    gravity_end   : float   Final (target) gravity value   (negative, e.g. -9.81)
    warmup_frac   : float   Fraction of max_generations held at gravity_start
    """

    name        = "gradual_transition"
    description = "Gravity ramps linearly from easy (-2.0) to earth (-9.81)"

    def __init__(
        self,
        gravity_start: float = -2.0,
        gravity_end:   float = -9.81,
        warmup_frac:   float = 0.20,
    ):
        assert gravity_start < 0 and gravity_end < 0, "Gravity must be negative"
        assert 0.0 <= warmup_frac < 1.0, "warmup_frac must be in [0, 1)"
        self.gravity_start = gravity_start
        self.gravity_end   = gravity_end
        self.warmup_frac   = warmup_frac

    def gravity(self, generation: int, max_generations: int) -> float:
        """
        Return the gravity value for this generation.

        Parameters
        ----------
        generation      : current generation index (0-based)
        max_generations : total number of generations planned

        Returns
        -------
        float : gravity value to use (always negative)
        """
        warmup_end = int(self.warmup_frac * max_generations)

        if generation <= warmup_end:
            return self.gravity_start

        # Linear interpolation from gravity_start → gravity_end
        ramp_gen   = generation - warmup_end
        ramp_total = max(1, max_generations - warmup_end)
        t = min(1.0, ramp_gen / ramp_total)
        return self.gravity_start + t * (self.gravity_end - self.gravity_start)

    def phase_label(self, generation: int, max_generations: int) -> str:
        """Human-readable phase name for logging."""
        warmup_end = int(self.warmup_frac * max_generations)
        if generation <= warmup_end:
            return "warmup"
        t = (generation - warmup_end) / max(1, max_generations - warmup_end)
        pct = int(min(100, t * 100))
        return f"ramp {pct}%"

    def describe(self) -> str:
        return (
            f"GradualTransition  gravity {self.gravity_start} → {self.gravity_end}  "
            f"warmup={int(self.warmup_frac * 100)}%"
        )
