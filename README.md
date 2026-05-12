# Walker2D — Curriculum Learning for Neuro-Morphological Evolution

A simulation-based platform for evolving bipedal locomotion controllers under variable gravitational conditions. Implements **CMA-ES** neuroevolution of a **CPG–MLP hybrid controller** with **co-evolved morphology**, evaluated across **seven gravity strategies** (one fixed baseline, one random control, five curriculum learning variants).

This repository accompanies the master's thesis *"Assessing Curriculum Learning Effects on Evolutionary Algorithms via Neuro-Morphological Evolution of Artificial Creatures in Variable-Gravity Environments."*

---

## Overview

- **Environment:** Walker2D-v5 (Gymnasium / MuJoCo), gravity programmatically modified via XML at evaluation time.
- **Genome (508-dim):** 18 CPG parameters (6 oscillators × {amplitude, frequency, phase}) + 486 MLP parameters (23→16→6 with `tanh`) + 4 morphological multipliers (torso density, leg density, motor gear, foot friction).
- **Optimizer:** CMA-ES via Hansen's `cma` library (population 40, σ₀ = 0.5, 500 generations).
- **Fitness:** Multi-objective, gravity-agnostic — 5 reward terms (forward distance, net progress, speed tracking, upright bonus, time alive) and 7 penalty terms with a 200-generation warmup schedule.

---

## Installation

Requires Python 3.10+ and macOS / Linux. MuJoCo is bundled with `gymnasium[mujoco]`.

```bash
git clone <repo-url>
cd walker2d-evo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### Train the fixed-gravity baseline

```bash
python train.py --seed 0
```

### Train with a curriculum strategy

```bash
python train_curriculum.py --curriculum gradual_transition --seed 0
```

Available strategies: `gradual_transition`, `staged_evolution`, `adaptive_progression`, `archive_based`, `multi_environment`, `random_uniform`.

### Render a trained agent

```bash
python render_agent.py checkpoints/cpg/best_params.npy --episodes 3
```

Pass `--gravity -6.0` (or any value) to test the agent under different gravitational conditions.

### Interactive viewer (recommended)

A GUI is provided to browse trained agents without using the terminal:

```bash
python viewer_app.py
```

The viewer lets you select any of the 7 strategies and 10 seeds, highlights the best-performing seed for each strategy, displays per-run statistics, and provides one-click access to: render the agent in MuJoCo, plot the learning curve, plot the gravity robustness profile, compare all strategies side-by-side, and run a **time-lapse evolution** that replays the walker at generations 0, 50, 100, 200, 350, and 490 in a single window — visualizing how the controller learned to walk. Built with Tkinter — no extra dependencies beyond what is already in `requirements.txt`.

### Reproduce the full thesis experiment (7 arms × 10 seeds = 70 runs)

```bash
python scripts/run_thesis_batch.py --output-dir experiments/thesis_morph
```

Approximate wall-clock time on an Apple M1 (8 cores): ~31 hours total. Six arms take ~20 min/run; the multi-environment arm takes ~1 h/run due to its 3× evaluation cost.

---

## Repository structure

```
walker2d-evo/
├── train.py                   # Fixed-gravity CMA-ES training loop
├── train_curriculum.py        # Curriculum-aware training loop
├── evaluate.py                # Episode rollout, fitness function, custom-XML builder
├── cpg_policy.py              # CPG–MLP hybrid policy (508-dim genome)
├── render_agent.py            # MuJoCo viewer for trained agents
├── debug_fitness.py           # Inspect per-term fitness contributions
├── curriculum/                # Six curriculum strategy implementations
├── scripts/                   # Batch runner, plotting, statistical analysis
├── experiments/               # Run outputs (logs, probes, sweeps, summaries)
└── checkpoints/               # Saved parameter vectors (.npy)
```

---

## Key results

Across 70 evolutionary runs (10 seeds × 7 strategies, 500 generations each):

| Strategy            | Mean final score | Improvement vs. baseline |
|---------------------|------------------|--------------------------|
| Staged evolution    | 1370.8           | +48%                     |
| Gradual transition  | 1328.9           | +44%                     |
| Archive-based       | 1311.4           | +42%                     |
| Multi-environment   | 1225.0           | +33%                     |
| Fixed gravity       | 923.9            | (baseline)               |
| Random variable     | 858.6            | −7%                      |
| Adaptive progression| 833.6            | −10%                     |

Three of five curricula improve substantially over the baseline; adaptive progression and random gravity variation do not. Differences are not statistically significant at *n* = 10 due to high seed-to-seed variance, but effect sizes are medium-to-large. Strategy rankings are robust under ±50% fitness-weight perturbation (88% ranking stability).

---

## Outputs per run

Each run produces, under `experiments/<exp>/<arm>/seed_<NN>/`:

- `training_log.csv` — per-generation training statistics
- `earth_probe.csv` — Earth-gravity probes every 10 generations (50 points)
- `gravity_sweep.csv` — final evaluation at five gravities (-6.0, -7.5, -9.81, -11.0, -12.0)
- `summary.json` — hyperparameters, best scores, time-to-effect, robustness metrics
- `checkpoints/best_params.npy` — best parameter vector found

---

## License & attribution

CMA-ES via the [`cma`](https://github.com/CMA-ES/pycma) Python library (Hansen et al.).
Walker2D-v5 environment from [Gymnasium](https://gymnasium.farama.org/) (Towers et al., 2024).
MuJoCo physics engine (Todorov et al., 2012).
