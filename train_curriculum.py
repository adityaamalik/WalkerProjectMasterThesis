"""
Curriculum learning training loop for Walker2d-v5.

Policy: hybrid CPG + feedback MLP (504 parameters).
Gravity is varied per-generation according to the chosen curriculum strategy.

Usage
-----
# Train with gradual transition curriculum (fresh start)
python train_curriculum.py --curriculum gradual_transition

# Warm-start from an existing checkpoint
python train_curriculum.py --curriculum gradual_transition \\
    --checkpoint checkpoints/cpg/best_params.npy

# Render the best saved curriculum policy
python train_curriculum.py --curriculum gradual_transition --render \\
    --checkpoint experiments/curriculum_gradual_transition/checkpoints/best_params.npy

Available curriculum strategies
--------------------------------
  gradual_transition  : gravity ramps from -2.0 (easy) → -9.81 (earth) over training
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import numpy as np
import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpg_policy import CPGPolicy, N_PARAMS as CPG_N_PARAMS
from evaluate import evaluate_policy, evaluate_parallel
from curriculum import get_strategy, list_strategies


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_EPISODES      = 3
MAX_STEPS       = 1000
POP_SIZE        = 40
SIGMA0          = 0.5
MAX_GEN         = 500
SAVE_EVERY      = 10
PLOT_EVERY      = 25
EXPERIMENTS_DIR = "experiments"
N_WORKERS       = mp.cpu_count()


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Walker2d with a curriculum learning strategy."
    )
    p.add_argument(
        "--curriculum", type=str, required=True,
        choices=list_strategies(),
        help="Curriculum strategy to use. Available: " + ", ".join(list_strategies()),
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved .npy params file to warm-start from")
    p.add_argument("--render", action="store_true",
                   help="Render the policy — requires --checkpoint")
    p.add_argument("--pop",         type=int,   default=POP_SIZE)
    p.add_argument("--sigma",       type=float, default=SIGMA0)
    p.add_argument("--generations", type=int,   default=MAX_GEN)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Experiment directory helpers
# ---------------------------------------------------------------------------

def experiment_dir(curriculum_name: str) -> str:
    """e.g.  experiments/curriculum_gradual_transition/"""
    return os.path.join(EXPERIMENTS_DIR, f"curriculum_{curriculum_name}")


def checkpoint_dir(curriculum_name: str) -> str:
    return os.path.join(experiment_dir(curriculum_name), "checkpoints")


def plot_path(curriculum_name: str) -> str:
    return os.path.join(experiment_dir(curriculum_name), "fitness_curve.png")


def gravity_plot_path(curriculum_name: str) -> str:
    return os.path.join(experiment_dir(curriculum_name), "gravity_curve.png")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fitness(history: list, path: str, curriculum_name: str) -> None:
    gens    = [h["gen"]     for h in history]
    best    = [h["best"]    for h in history]
    mean    = [h["mean"]    for h in history]
    gravity = [h["gravity"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(gens, best, label="Best fitness",  color="green")
    ax1.plot(gens, mean, label="Mean fitness",  color="blue",  linestyle="--")
    ax1.set_ylabel("Cumulative reward")
    ax1.set_title(f"CMA-ES — Walker2d-v5  [{curriculum_name}]")
    ax1.legend()

    ax2.plot(gens, gravity, label="Gravity (m/s²)", color="red")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Gravity (m/s²)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  [plot] Saved fitness curve → {path}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(params: np.ndarray, path: str,
                    meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    if meta is not None:
        with open(path.replace(".npy", ".json"), "w") as f:
            json.dump(meta, f, indent=2)


def write_experiment_json(curriculum_name: str, strategy,
                          best_score: float, best_gen: int,
                          max_gen: int, pop: int, sigma: float) -> None:
    """Write / update experiment.json in the curriculum experiment dir."""
    exp_dir   = experiment_dir(curriculum_name)
    meta_path = os.path.join(exp_dir, "experiment.json")

    data = {
        "id":          f"curriculum_{curriculum_name}",
        "name":        f"Curriculum — {strategy.describe()}",
        "curriculum":  curriculum_name,
        "policy":      "cpg",
        "n_params":    CPG_N_PARAMS,
        "generations": max_gen,
        "population":  pop,
        "sigma0":      sigma,
        "n_episodes":  N_EPISODES,
        "max_steps":   MAX_STEPS,
        "gravity_schedule": strategy.describe(),
        "result": {
            "best_score":      round(best_score, 2),
            "best_generation": best_gen,
            "outcome":         "complete",
        },
        "artifacts": {
            "fitness_curve": "fitness_curve.png",
            "gravity_curve": "gravity_curve.png",
            "best_params":   "checkpoints/best_params.npy",
        },
    }
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [exp] Wrote {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    strategy = get_strategy(args.curriculum)

    exp_dir  = experiment_dir(args.curriculum)
    ckpt_dir = checkpoint_dir(args.curriculum)
    os.makedirs(ckpt_dir, exist_ok=True)

    n_params = CPG_N_PARAMS
    print(f"Walker2d-v5 | policy=CPG  params={n_params}")
    print(f"Curriculum   : {strategy.describe()}")
    print()

    # ------------------------------------------------------------------ render
    if args.render:
        if args.checkpoint is None:
            print("ERROR: --render requires --checkpoint")
            return
        params = np.load(args.checkpoint)
        meta_path = args.checkpoint.replace(".npy", ".json")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Checkpoint: {meta.get('label')}  |  "
                  f"gen={meta.get('generation')}  |  "
                  f"score={meta.get('score')}")

        # Render at earth gravity (final target)
        score = evaluate_policy(
            params=params,
            n_episodes=5,
            max_steps=MAX_STEPS,
            render=True,
            meta=meta,
            gravity=-9.81,
            phase="earth (final)",
        )
        print(f"Rendered policy score: {score:.1f}")
        return

    # ------------------------------------------------------------------ train
    if args.checkpoint and os.path.exists(args.checkpoint):
        x0 = np.load(args.checkpoint)
        if x0.shape[0] != n_params:
            print(f"WARNING: checkpoint has {x0.shape[0]} params, "
                  f"expected {n_params}. Starting fresh.")
            x0 = np.random.randn(n_params) * 0.1
        else:
            print(f"Warm-starting from {args.checkpoint}")
    else:
        x0 = np.random.randn(n_params) * 0.1

    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma,
        {
            "popsize":        args.pop,
            "maxiter":        args.generations,
            "verbose":        -9,
            "tolx":           1e-12,
            "tolfun":         1e-12,
            "tolstagnation":  args.generations,
        },
    )

    history     = []
    best_score  = -np.inf
    best_params = x0.copy()
    fit_plot    = plot_path(args.curriculum)

    print(f"\nStarting CMA-ES | pop={args.pop} | sigma0={args.sigma} | "
          f"max_gen={args.generations} | workers={N_WORKERS}")
    print(f"Checkpoints → {ckpt_dir}/")
    print()
    print(f"{'Gen':>5}  {'Best':>10}  {'Mean':>10}  {'Sigma':>8}  "
          f"{'Gravity':>9}  {'Phase':<14}  {'Time':>7}")
    print("-" * 72)

    ctx = mp.get_context("spawn")
    gen = 0
    with ctx.Pool(N_WORKERS) as pool:
        while not es.stop():
            t0 = time.time()

            # Ask curriculum for this generation's gravity
            g     = strategy.gravity(gen, args.generations)
            phase = strategy.phase_label(gen, args.generations)

            solutions = es.ask()

            fitness = evaluate_parallel(
                population  = solutions,
                n_episodes  = N_EPISODES,
                max_steps   = MAX_STEPS,
                pool        = pool,
                gravity     = g,
                phase       = phase,
            )
            es.tell(solutions, [-f for f in fitness])

            best_gen  = max(fitness)
            mean_gen  = float(np.mean(fitness))
            elapsed   = time.time() - t0

            history.append({
                "gen":     gen,
                "best":    best_gen,
                "mean":    mean_gen,
                "gravity": g,
                "phase":   phase,
            })

            if best_gen > best_score:
                best_score  = best_gen
                best_params = solutions[int(np.argmax(fitness))].copy()
                save_checkpoint(
                    best_params,
                    os.path.join(ckpt_dir, "best_params.npy"),
                    meta={
                        "score":      round(best_score, 2),
                        "generation": gen,
                        "policy":     "cpg",
                        "curriculum": args.curriculum,
                        "gravity":    round(g, 4),
                        "phase":      phase,
                        "label":      "all-time best",
                    },
                )

            if gen % SAVE_EVERY == 0:
                save_checkpoint(
                    es.result.xbest,
                    os.path.join(ckpt_dir, f"gen_{gen:04d}.npy"),
                    meta={
                        "score":      round(best_gen, 2),
                        "generation": gen,
                        "policy":     "cpg",
                        "curriculum": args.curriculum,
                        "gravity":    round(g, 4),
                        "phase":      phase,
                        "label":      f"gen {gen} snapshot",
                    },
                )

            if gen % PLOT_EVERY == 0:
                plot_fitness(history, fit_plot, args.curriculum)

            print(f"{gen:>5}  {best_gen:>10.1f}  {mean_gen:>10.1f}  "
                  f"{es.sigma:>8.4f}  {g:>9.4f}  {phase:<14}  {elapsed:>6.1f}s")
            gen += 1

    # Final save + plot
    save_checkpoint(
        best_params,
        os.path.join(ckpt_dir, "final_params.npy"),
        meta={
            "score":      round(best_score, 2),
            "generation": gen - 1,
            "policy":     "cpg",
            "curriculum": args.curriculum,
            "label":      "final (all-time best)",
        },
    )
    plot_fitness(history, fit_plot, args.curriculum)

    print(f"\nTraining complete. Best fitness: {best_score:.1f}")
    print(f"Best params → {ckpt_dir}/best_params.npy")
    print(f"Fitness curve → {fit_plot}")

    write_experiment_json(
        curriculum_name = args.curriculum,
        strategy        = strategy,
        best_score      = best_score,
        best_gen        = gen - 1,
        max_gen         = args.generations,
        pop             = args.pop,
        sigma           = args.sigma,
    )


if __name__ == "__main__":
    main()
