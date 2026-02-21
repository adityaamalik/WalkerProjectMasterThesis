"""
Curriculum learning strategies for Walker2d-v5 CMA-ES training.

Each strategy is a class with a single interface:
    gravity(generation: int, max_generations: int) -> float

Strategies
----------
  gradual_transition  : gravity linearly interpolates from easy → earth

Usage in train_curriculum.py:
    from curriculum import get_strategy
    schedule = get_strategy("gradual_transition")
    g = schedule.gravity(gen, max_gen)
"""

from curriculum.gradual_transition import GradualTransition

_REGISTRY = {
    "gradual_transition": GradualTransition,
}


def get_strategy(name: str):
    """Return an instantiated curriculum strategy by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown curriculum strategy '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def list_strategies() -> list:
    return list(_REGISTRY.keys())
