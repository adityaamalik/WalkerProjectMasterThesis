# Walker2D — Evolutionary Algorithm (CMA-ES)

Train a 2D biped walker using **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
on OpenAI Gymnasium's `Walker2d-v5` (MuJoCo).

## Architecture

```
observations (17) → MLP(64) → MLP(64) → actions (6)   tanh activations
```

The entire MLP is encoded as a flat parameter vector (~5k floats) that CMA-ES
evolves to maximise cumulative reward.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--pop` | 40 | CMA-ES population size (lambda) |
| `--sigma` | 0.5 | Initial step-size σ₀ |
| `--generations` | 500 | Max CMA-ES generations |
| `--hidden` | 64 | Hidden layer width |
| `--checkpoint PATH` | — | Resume / load existing params |
| `--render` | off | Visualise the loaded checkpoint |

### 3. Visualise the best policy

```bash
python train.py --render --checkpoint checkpoints/best_params.npy
```

### 4. Plot training curve

The file `fitness_curve.png` is updated every 10 generations automatically.
You can also run:

```bash
python -c "
import numpy as np, matplotlib.pyplot as plt, glob, json
# fitness_curve.png is already generated during training
print('See fitness_curve.png')
"
```

## File structure

```
walker2d-evo/
├── policy.py          # MLP policy (flat-param interface)
├── evaluate.py        # Rollout & fitness evaluation
├── train.py           # CMA-ES training loop
├── requirements.txt
├── checkpoints/       # Saved parameter snapshots (created at runtime)
└── fitness_curve.png  # Generated training plot
```

## Background

CMA-ES treats the neural-network weights as a continuous optimisation problem.
At each generation it:
1. Samples `popsize` weight vectors from a multivariate Gaussian.
2. Evaluates each candidate by running MuJoCo rollouts.
3. Updates the Gaussian mean and covariance towards high-reward regions.

This approach requires no gradient information and scales well to ~10 000
parameters, making it ideal for locomotion tasks.

## Expected performance

With the default settings you can expect:
- ~50 reward after 20–30 generations (agent starts walking)
- ~200–400 reward after 100–200 generations (steady gait)
- ~500+ reward after 300+ generations (smooth locomotion)

Training time is ~5–10 min/gen on a modern laptop CPU.
