"""
Curriculum learning strategies for Walker2d-v5 CMA-ES training.

Every strategy implements the CurriculumBase interface:

  Core (required)
  ---------------
  gravity(generation, max_generations) -> float
  phase_label(generation, max_generations) -> str
  describe() -> str

  Extended (optional, default no-ops in CurriculumBase)
  ------------------------------------------------------
  eval_gravities(gen, max_gen) -> list[float]
      Gravities at which each candidate is evaluated. Default: [gravity()].
  combine_fitness(scores) -> float
      Aggregate per-gravity scores. Default: arithmetic mean.
  notify(gen, max_gen, best_score, mean_score, sigma) -> None
      Receive per-generation stats. Used by adaptive strategies.
  at_stage_boundary(gen, max_gen) -> bool
      Signal a stage transition that triggers an ES restart.
  get_seed_params() -> np.ndarray | None
      Warm-start params for the restarted ES. None → use best_params.
  on_new_best(params, score, gen) -> None
      Called every generation with the best candidate. Used by archive.

Strategies
----------
  gradual_transition   : linear gravity ramp from easy to earth (Moon → Earth)
  random_uniform       : gravity sampled uniformly each generation (control arm)
  staged_evolution     : discrete stages with ES restart at transitions
  multi_environment    : simultaneous multi-gravity evaluation per candidate
  adaptive_progression : gravity advances when fitness plateaus
  archive_based        : archive seeds new levels from previous-level solutions

Usage
-----
  from curriculum import get_strategy
  strategy = get_strategy("staged_evolution", seed=42)
  g = strategy.gravity(gen, max_gen)
"""

import inspect

from curriculum.base import CurriculumBase
from curriculum.gradual_transition import GradualTransition
from curriculum.random_uniform import RandomUniform
from curriculum.staged_evolution import StagedEvolution
from curriculum.multi_environment import MultiEnvironment
from curriculum.adaptive_progression import AdaptiveProgression
from curriculum.archive_based import ArchiveBased

_REGISTRY: dict[str, type] = {
    "gradual_transition": GradualTransition,
    "random_uniform": RandomUniform,
    "staged_evolution": StagedEvolution,
    "multi_environment": MultiEnvironment,
    "adaptive_progression": AdaptiveProgression,
    "archive_based": ArchiveBased,
}


def get_strategy(name: str, **kwargs) -> CurriculumBase:
    """Return an instantiated curriculum strategy by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown curriculum strategy '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    strategy_cls = _REGISTRY[name]
    sig = inspect.signature(strategy_cls.__init__)
    accepted = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != "self"
    }
    return strategy_cls(**accepted)


def list_strategies() -> list[str]:
    return list(_REGISTRY.keys())
