"""
CMA-ES training loop for Walker2D-v5.

Policy: hybrid CPG + feedback MLP (504 parameters).

Usage
-----
# Train
python train.py

# Resume from a checkpoint
python train.py --checkpoint checkpoints/cpg/best_params.npy

# Visualise the best saved policy
python train.py --render --checkpoint checkpoints/cpg/best_params.npy
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import numpy as np
import cma
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on macOS / headless)
import matplotlib.pyplot as plt

from cpg_policy import CPGPolicy, N_JOINTS, N_PARAMS as CPG_N_PARAMS
from evaluate import evaluate_policy, evaluate_parallel


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
ENV_ID          = "Walker2d-v5"
N_EPISODES      = 3        # rollouts per individual
MAX_STEPS       = 1000     # max timesteps per episode
POP_SIZE        = 40       # CMA-ES population (lambda)
SIGMA0          = 0.5      # initial step-size
MAX_GEN         = 500      # maximum generations
SAVE_EVERY      = 10       # checkpoint every N generations
PLOT_EVERY      = 25       # plot fitness curve every N generations (matplotlib is slow)
CHECKPOINT_DIR  = "checkpoints"
EXPERIMENTS_DIR = "experiments"
N_WORKERS       = mp.cpu_count()  # use all available cores
PENALTY_SCALE_MIN = 0.2    # early generations: allow locomotion exploration
PENALTY_SCALE_RAMP_GEN = 200  # ramp penalties to full strength over this many generations
CORR_SCALE_MIN = 0.0       # start CPG-dominant; suppress random MLP destabilisation
CORR_SCALE_MAX = 0.5       # match cpg_policy default
CORR_SCALE_RAMP_GEN = 200  # ramp feedback authority gradually


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved .npy params file to load / resume from")
    p.add_argument("--render", action="store_true",
                   help="Render the policy — requires --checkpoint")
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--sigma", type=float, default=SIGMA0)
    p.add_argument("--generations", type=int, default=MAX_GEN)
    p.add_argument("--exp", type=str, default=None,
                   help="Experiment ID (e.g. exp006_cpg_v6_...). "
                        "If provided, fitness curve and best params are also "
                        "copied into experiments/<exp>/ at the end of training.")
    p.add_argument("--penalty-min", type=float, default=None,
                   help="Initial penalty scale at generation 0. "
                        "Default: scratch=0.2, warm-start=1.0")
    p.add_argument("--penalty-ramp-gen", type=int, default=None,
                   help="Generations to ramp penalty scale to 1.0. "
                        "Default: scratch=200, warm-start=1")
    p.add_argument("--corr-min", type=float, default=None,
                   help="Initial feedback correction scale at generation 0. "
                        "Default: scratch=0.0, warm-start=0.5")
    p.add_argument("--corr-max", type=float, default=None,
                   help="Maximum feedback correction scale. Default: 0.5")
    p.add_argument("--corr-ramp-gen", type=int, default=None,
                   help="Generations to ramp correction scale to corr-max. "
                        "Default: scratch=200, warm-start=1")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_path() -> str:
    return "fitness_curve_cpg.png"


def make_structured_scratch_init(seed: int = 42) -> np.ndarray:
    """
    Build a scratch initial mean that is dynamically gentle.

    Reason:
    Fully random CPG logits around 0 imply amplitude ~0.5 and moderate
    oscillator speed on all joints, which tends to cause immediate falls.
    A low-amplitude/low-frequency prior keeps early exploration near
    stabilisable motions while still being from scratch (no checkpoint).
    """
    rng = np.random.default_rng(seed)
    x0 = np.zeros(CPG_N_PARAMS, dtype=np.float64)

    # CPG block layout: [A×6 | w×6 | phi×6 | ...]
    cpg_amp_start = 0
    cpg_amp_end = N_JOINTS
    cpg_w_end = cpg_amp_end + N_JOINTS
    cpg_phi_end = cpg_w_end + N_JOINTS

    # Amplitude sigmoid(-1.9) ≈ 0.13 (gentle but not frozen)
    x0[cpg_amp_start:cpg_amp_end] = -1.9
    # Frequency softplus(-0.2) ≈ 0.60 rad/step before DT scaling in policy.
    # With DT*10 scaling in cpg_policy.py, this is near a walk-like cadence.
    x0[cpg_amp_end:cpg_w_end] = -0.2
    # Phase offsets start neutral; left/right anti-phase comes from theta reset
    x0[cpg_w_end:cpg_phi_end] = 0.0

    # Small exploration noise, smaller on MLP than on CPG core.
    x0[cpg_amp_start:cpg_phi_end] += rng.normal(0.0, 0.12, cpg_phi_end - cpg_amp_start)
    x0[cpg_phi_end:] += rng.normal(0.0, 0.02, CPG_N_PARAMS - cpg_phi_end)
    return x0


def resolve_schedule_args(args, warm_start: bool) -> dict:
    """
    Resolve penalty/correction schedules.

    Warm-start default is fine-tuning mode (full-strength schedules from gen 0).
    Scratch default is exploratory ramps.
    """
    penalty_min = args.penalty_min
    if penalty_min is None:
        penalty_min = 1.0 if warm_start else PENALTY_SCALE_MIN
    penalty_min = float(np.clip(penalty_min, 0.0, 1.0))

    penalty_ramp_gen = args.penalty_ramp_gen
    if penalty_ramp_gen is None:
        penalty_ramp_gen = 1 if warm_start else PENALTY_SCALE_RAMP_GEN
    penalty_ramp_gen = max(1, int(penalty_ramp_gen))

    corr_min = args.corr_min
    if corr_min is None:
        corr_min = CORR_SCALE_MAX if warm_start else CORR_SCALE_MIN

    corr_max = args.corr_max
    if corr_max is None:
        corr_max = CORR_SCALE_MAX

    corr_min = float(np.clip(corr_min, 0.0, 1.0))
    corr_max = float(np.clip(corr_max, 0.0, 1.0))
    if corr_max < corr_min:
        corr_max = corr_min

    corr_ramp_gen = args.corr_ramp_gen
    if corr_ramp_gen is None:
        corr_ramp_gen = 1 if warm_start else CORR_SCALE_RAMP_GEN
    corr_ramp_gen = max(1, int(corr_ramp_gen))

    return {
        "penalty_min": penalty_min,
        "penalty_ramp_gen": penalty_ramp_gen,
        "corr_min": corr_min,
        "corr_max": corr_max,
        "corr_ramp_gen": corr_ramp_gen,
    }


def archive_experiment(exp_id: str,
                       best_score: float, best_gen: int,
                       best_params: np.ndarray, plot_file: str,
                       ckpt_dir: str) -> None:
    """
    Copy best checkpoint + fitness curve into experiments/<exp_id>/.
    Also writes a minimal experiment.json if one doesn't already exist.
    """
    import shutil
    exp_dir   = os.path.join(EXPERIMENTS_DIR, exp_id)
    ckpt_copy = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_copy, exist_ok=True)

    # Fitness curve
    if os.path.exists(plot_file):
        shutil.copy2(plot_file, os.path.join(exp_dir, "fitness_curve.png"))

    # Best params + metadata
    src_npy  = os.path.join(ckpt_dir, "best_params.npy")
    src_json = os.path.join(ckpt_dir, "best_params.json")
    if os.path.exists(src_npy):
        shutil.copy2(src_npy,  os.path.join(ckpt_copy, "best_params.npy"))
    if os.path.exists(src_json):
        shutil.copy2(src_json, os.path.join(ckpt_copy, "best_params.json"))

    # Write experiment.json stub if absent
    meta_path = os.path.join(exp_dir, "experiment.json")
    if not os.path.exists(meta_path):
        stub = {
            "id":     exp_id,
            "policy": "cpg",
            "result": {
                "best_score":      round(best_score, 2),
                "best_generation": best_gen,
            },
            "artifacts": {
                "fitness_curve": "fitness_curve.png",
                "best_params":   "checkpoints/best_params.npy",
            },
        }
        with open(meta_path, "w") as f:
            json.dump(stub, f, indent=2)

    print(f"  [exp] Archived → {exp_dir}/")


def save_checkpoint(params: np.ndarray, path: str,
                    meta: dict | None = None) -> None:
    """Save params as .npy and write a companion .json with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    if meta is not None:
        meta_path = path.replace(".npy", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def plot_fitness(history: list, path: str) -> None:
    gens = [h["gen"] for h in history]
    best = [h["best"] for h in history]
    mean = [h["mean"] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(gens, best, label="Best fitness", color="green")
    plt.plot(gens, mean, label="Mean fitness", color="blue", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Cumulative reward")
    plt.title("CMA-ES on Walker2d-v5  [CPG hybrid]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  [plot] Saved fitness curve → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    n_params = CPG_N_PARAMS
    print(f"Walker2d-v5 | policy=CPG  params={n_params}")
    print(CPGPolicy().describe())
    print()

    # ------------------------------------------------------------------ render
    if args.render:
        if args.checkpoint is None:
            print("ERROR: --render requires --checkpoint")
            return
        params = np.load(args.checkpoint)

        # Load companion metadata if it exists
        meta_path = args.checkpoint.replace(".npy", ".json")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Checkpoint: {meta.get('label')}  |  "
                  f"gen={meta.get('generation')}  |  "
                  f"train_score={meta.get('score')}")

        score = evaluate_policy(
            params=params,
            n_episodes=5,
            max_steps=MAX_STEPS,
            render=True,
            meta=meta,
        )
        print(f"Rendered policy score: {score:.1f}")
        return

    # ------------------------------------------------------------------ train
    ckpt_dir = os.path.join(CHECKPOINT_DIR, "cpg")
    os.makedirs(ckpt_dir, exist_ok=True)

    warm_start = bool(args.checkpoint and os.path.exists(args.checkpoint))
    if warm_start:
        x0 = np.load(args.checkpoint)
        print(f"Resuming from {args.checkpoint}")
    else:
        x0 = make_structured_scratch_init(seed=42)
        print("Starting from structured scratch prior (no warm-start checkpoint).")

    schedule = resolve_schedule_args(args, warm_start=warm_start)
    print(
        "Schedule | "
        f"penalty_min={schedule['penalty_min']:.2f} "
        f"penalty_ramp_gen={schedule['penalty_ramp_gen']} "
        f"corr_min={schedule['corr_min']:.2f} "
        f"corr_max={schedule['corr_max']:.2f} "
        f"corr_ramp_gen={schedule['corr_ramp_gen']}"
    )

    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma,
        {
            "popsize": args.pop,
            "maxiter": args.generations,
            "verbose": -9,
            # Disable tight convergence checks — Walker2D reward is noisy
            "tolx": 1e-12,
            "tolfun": 1e-12,
            "tolstagnation": args.generations,
            # Prevent immediate stop when early generations are nearly flat.
            "tolflatfitness": args.generations,
        },
    )

    history    = []
    best_score = -np.inf
    best_params = x0.copy()
    plot_file  = plot_path()

    print(f"\nStarting CMA-ES | pop={args.pop} | "
          f"sigma0={args.sigma} | max_gen={args.generations} | workers={N_WORKERS}\n")
    print(f"{'Gen':>5}  {'Best':>10}  {'Mean':>10}  {'Sigma':>8}  {'Pen':>5}  {'Corr':>5}  {'Time':>7}")
    print("-" * 66)

    # Persistent pool — created once, avoids per-generation spawn overhead
    ctx = mp.get_context("spawn")
    gen = 0
    with ctx.Pool(N_WORKERS) as pool:
        while not es.stop():
            t0 = time.time()
            solutions = es.ask()
            penalty_scale = min(
                1.0,
                schedule["penalty_min"] +
                (1.0 - schedule["penalty_min"]) * (gen / schedule["penalty_ramp_gen"])
            )
            correction_scale = min(
                schedule["corr_max"],
                schedule["corr_min"] +
                (schedule["corr_max"] - schedule["corr_min"]) * (gen / schedule["corr_ramp_gen"])
            )

            # Negate reward — CMA-ES minimises
            fitness = evaluate_parallel(
                population=solutions,
                n_episodes=N_EPISODES,
                max_steps=MAX_STEPS,
                pool=pool,
                penalty_scale=penalty_scale,
                correction_scale=correction_scale,
            )
            es.tell(solutions, [-f for f in fitness])

            best_gen  = max(fitness)
            mean_gen  = float(np.mean(fitness))
            elapsed   = time.time() - t0

            history.append({"gen": gen, "best": best_gen, "mean": mean_gen})

            if best_gen > best_score:
                best_score  = best_gen
                best_params = solutions[int(np.argmax(fitness))].copy()
                save_checkpoint(best_params,
                                os.path.join(ckpt_dir, "best_params.npy"),
                                meta={"score": round(best_score, 2),
                                      "generation": gen,
                                      "policy": "cpg",
                                      "label": "all-time best"})

            if gen % SAVE_EVERY == 0:
                save_checkpoint(es.result.xbest,
                                os.path.join(ckpt_dir, f"gen_{gen:04d}.npy"),
                                meta={"score": round(best_gen, 2),
                                      "generation": gen,
                                      "policy": "cpg",
                                      "label": f"gen {gen} snapshot"})

            # Plot less frequently — matplotlib adds ~0.5s per call
            if gen % PLOT_EVERY == 0:
                plot_fitness(history, plot_file)

            print(f"{gen:>5}  {best_gen:>10.1f}  {mean_gen:>10.1f}  "
                  f"{es.sigma:>8.4f}  {penalty_scale:>5.2f}  {correction_scale:>5.2f}  {elapsed:>6.1f}s")
            gen += 1

    stop_reasons = es.stop()
    if stop_reasons:
        print(f"CMA-ES stop reasons: {stop_reasons}")

    # Final save + plot
    save_checkpoint(best_params,
                    os.path.join(ckpt_dir, "final_params.npy"),
                    meta={"score": round(best_score, 2),
                          "generation": gen - 1,
                          "policy": "cpg",
                          "label": "final (all-time best)"})
    plot_fitness(history, plot_file)

    print(f"\nTraining complete. Best fitness: {best_score:.1f}")
    print(f"Best params → {ckpt_dir}/best_params.npy")
    print(f"Fitness curve → {plot_file}")

    # Auto-archive into experiments/ if --exp was specified
    if args.exp:
        archive_experiment(
            exp_id      = args.exp,
            best_score  = best_score,
            best_gen    = gen - 1,
            best_params = best_params,
            plot_file   = plot_file,
            ckpt_dir    = ckpt_dir,
        )


if __name__ == "__main__":
    main()
