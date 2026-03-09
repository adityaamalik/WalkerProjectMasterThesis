"""
Curriculum learning training loop for Walker2d-v5.

Policy: hybrid CPG + feedback MLP (504 parameters).
Gravity is varied per-generation according to the chosen curriculum strategy.

Usage
-----
# Train with gradual transition curriculum (fresh start)
python train_curriculum.py --curriculum gradual_transition

# Warm-start from an existing checkpoint
python train_curriculum.py --curriculum gradual_transition \
    --checkpoint checkpoints/cpg/best_params.npy

# Render the best saved curriculum policy
python train_curriculum.py --curriculum gradual_transition --render \
    --checkpoint experiments/curriculum_gradual_transition/checkpoints/best_params.npy

Available curriculum strategies
--------------------------------
  gradual_transition  : gravity ramps from -2.0 (easy) → -9.81 (earth) over training
  random_uniform      : gravity sampled from a fixed range each generation
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time

import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpg_policy import CPGPolicy, N_JOINTS, N_PARAMS as CPG_N_PARAMS
from evaluate import evaluate_policy, evaluate_parallel
from curriculum import get_strategy, list_strategies


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_EPISODES = 3
MAX_STEPS = 1000
POP_SIZE = 40
SIGMA0 = 0.5
MAX_GEN = 500
SAVE_EVERY = 10
PLOT_EVERY = 25
EXPERIMENTS_DIR = "experiments"
N_WORKERS = mp.cpu_count()

TTE_NET_PROGRESS_MIN = 4.0
TTE_FELL_MAX = 0.20
TTE_BACKWARD_MAX = 1.0
TTE_CENSOR_OFFSET = 10
DEFAULT_GRAVITY_SWEEP = "-6.0,-7.5,-9.81,-11.0,-12.0"

# Penalty / correction schedules — must match train.py defaults so all arms
# are evaluated under identical conditions throughout training.
PENALTY_SCALE_MIN = 0.2       # penalty scale at gen 0 (scratch)
PENALTY_SCALE_RAMP_GEN = 200  # generations to ramp penalty to 1.0
CORR_SCALE_MIN = 0.0          # correction scale at gen 0 (scratch)
CORR_SCALE_MAX = 0.5          # maximum correction scale
CORR_SCALE_RAMP_GEN = 200     # generations to ramp correction to CORR_SCALE_MAX
THESIS_EVAL_CORR_SCALE = CORR_SCALE_MAX


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
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--sigma", type=float, default=SIGMA0)
    p.add_argument("--generations", type=int, default=MAX_GEN)
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for init, CMA-ES and rollout evaluation.")
    p.add_argument("--arm", type=str, default=None,
                   help="Optional experiment arm label.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Optional run identifier.")
    p.add_argument("--exp", type=str, default=None,
                   help="Experiment output path under experiments/. "
                        "Example: thesis/curriculum_variable_gravity/seed_00")

    p.add_argument("--probe-every", type=int, default=10,
                   help="Earth probe cadence in generations (0 disables).")
    p.add_argument("--probe-episodes", type=int, default=10,
                   help="Episodes per Earth probe.")
    p.add_argument("--probe-gravity", type=float, default=-9.81,
                   help="Gravity used for periodic Earth probes.")

    p.add_argument("--final-eval-episodes", type=int, default=20,
                   help="Episodes for final Earth and gravity-sweep evaluations.")
    p.add_argument("--gravity-sweep", type=str, default=DEFAULT_GRAVITY_SWEEP,
                   help="Comma-separated gravities for final sweep. Empty disables sweep.")

    # GradualTransition parameters
    p.add_argument("--gravity-start", type=float, default=-2.0,
                   help="Start gravity for gradual_transition strategy.")
    p.add_argument("--gravity-end", type=float, default=-9.81,
                   help="End gravity for gradual_transition strategy.")
    p.add_argument("--warmup-frac", type=float, default=0.20,
                   help="Warmup fraction for gradual_transition strategy.")

    # RandomUniform parameters
    p.add_argument("--random-gravity-min", type=float, default=-12.0,
                   help="Min gravity for random_uniform strategy.")
    p.add_argument("--random-gravity-max", type=float, default=-6.0,
                   help="Max gravity for random_uniform strategy.")

    # Penalty / correction scheduling (mirrors train.py for controlled comparisons)
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

def make_structured_scratch_init(seed: int = 42) -> np.ndarray:
    """
    Build a scratch initial mean that is dynamically gentle.

    Identical to the function in train.py so that all three arms start from
    the same CPG prior: low amplitude (sigmoid(-1.9) ≈ 0.13), moderate
    frequency (softplus(-0.2)), near-zero MLP weights.
    """
    rng = np.random.default_rng(seed)
    x0 = np.zeros(CPG_N_PARAMS, dtype=np.float64)

    cpg_amp_start = 0
    cpg_amp_end = N_JOINTS
    cpg_w_end = cpg_amp_end + N_JOINTS
    cpg_phi_end = cpg_w_end + N_JOINTS

    x0[cpg_amp_start:cpg_amp_end] = -1.9
    x0[cpg_amp_end:cpg_w_end] = -0.2
    x0[cpg_w_end:cpg_phi_end] = 0.0

    x0[cpg_amp_start:cpg_phi_end] += rng.normal(0.0, 0.12, cpg_phi_end - cpg_amp_start)
    x0[cpg_phi_end:] += rng.normal(0.0, 0.02, CPG_N_PARAMS - cpg_phi_end)
    return x0


def parse_gravity_sweep(spec: str) -> list[float]:
    text = (spec or "").strip()
    if not text:
        return []
    out = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def make_run_context(args) -> dict:
    ctx = {"seed": int(args.seed)}
    if args.arm:
        ctx["arm"] = args.arm
    if args.run_id:
        ctx["run_id"] = args.run_id
    return ctx


def resolve_schedule_args(args, warm_start: bool) -> dict:
    """
    Resolve penalty/correction schedule parameters.

    Mirrors the identical function in train.py so that all three arms
    (fixed_gravity, random_variable_gravity, curriculum_variable_gravity)
    use the same penalty_scale and correction_scale ramps during training.

    Scratch defaults: penalty_scale ramps 0.2 → 1.0 over 200 gen,
                      correction_scale ramps 0.0 → 0.5 over 200 gen.
    Warm-start defaults: both start at full strength from gen 0.
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


def compute_tte(earth_probes: list[dict], max_generations: int) -> int:
    for p in earth_probes:
        if (
            p["net_progress_m"] >= TTE_NET_PROGRESS_MIN and
            p["fell"] <= TTE_FELL_MAX and
            p["backward_distance_m"] <= TTE_BACKWARD_MAX
        ):
            return int(p["gen"])
    return int(max_generations + TTE_CENSOR_OFFSET)


def experiment_dir(curriculum_name: str, exp_override: str | None = None) -> str:
    """Default: experiments/curriculum_<name>/, override: experiments/<exp_override>/"""
    if exp_override:
        return os.path.join(EXPERIMENTS_DIR, exp_override)
    return os.path.join(EXPERIMENTS_DIR, f"curriculum_{curriculum_name}")


def checkpoint_dir(curriculum_name: str, exp_override: str | None = None) -> str:
    return os.path.join(experiment_dir(curriculum_name, exp_override), "checkpoints")


def plot_path(curriculum_name: str, exp_override: str | None = None) -> str:
    return os.path.join(experiment_dir(curriculum_name, exp_override), "fitness_curve.png")


def save_checkpoint(params: np.ndarray, path: str,
                    meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    if meta is not None:
        with open(path.replace(".npy", ".json"), "w") as f:
            json.dump(meta, f, indent=2)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_fitness(history: list[dict], path: str, curriculum_name: str) -> None:
    gens = [h["gen"] for h in history]
    best = [h["best"] for h in history]
    mean = [h["mean"] for h in history]
    gravity = [h["gravity"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(gens, best, label="Best fitness", color="green")
    ax1.plot(gens, mean, label="Mean fitness", color="blue", linestyle="--")
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


def evaluate_gravity_sweep(
    params: np.ndarray,
    gravities: list[float],
    n_episodes: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rows = []
    for idx, g in enumerate(gravities):
        eval_seed = int(seed * 1_000_000 + 700_000 + (idx * 10_000))
        score, diag = evaluate_policy(
            params=params,
            n_episodes=n_episodes,
            max_steps=MAX_STEPS,
            seed=eval_seed,
            gravity=float(g),
            phase=f"sweep {g:.2f}",
            return_diagnostics=True,
            penalty_scale=1.0,
            correction_scale=THESIS_EVAL_CORR_SCALE,
        )
        stats = diag["mean_stats"]
        rows.append({
            "gravity": float(g),
            "score": float(score),
            "net_progress_m": float(stats.get("net_progress_m", 0.0)),
            "fell": float(stats.get("fell", 0.0)),
            "backward_distance_m": float(stats.get("backward_distance_m", 0.0)),
            "mean_forward_speed_mps": float(stats.get("mean_forward_speed_mps", 0.0)),
        })

    if rows:
        net_vals = [r["net_progress_m"] for r in rows]
        fell_vals = [r["fell"] for r in rows]
        robustness = {
            "mean_net_progress_m": float(np.mean(net_vals)),
            "worst_net_progress_m": float(np.min(net_vals)),
            "mean_fall_rate": float(np.mean(fell_vals)),
        }
    else:
        robustness = {
            "mean_net_progress_m": None,
            "worst_net_progress_m": None,
            "mean_fall_rate": None,
        }
    return rows, robustness


def write_experiment_json(
    exp_dir: str,
    curriculum_name: str,
    strategy,
    best_score: float,
    best_gen: int,
    max_gen: int,
    pop: int,
    sigma: float,
    run_context: dict,
) -> None:
    """Write / update experiment.json in the curriculum experiment dir."""
    meta_path = os.path.join(exp_dir, "experiment.json")

    data = {
        "id": os.path.basename(exp_dir),
        "name": f"Curriculum — {strategy.describe()}",
        "curriculum": curriculum_name,
        "policy": "cpg",
        "n_params": CPG_N_PARAMS,
        "generations": max_gen,
        "population": pop,
        "sigma0": sigma,
        "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "gravity_schedule": strategy.describe(),
        "run": run_context,
        "result": {
            "best_score": round(best_score, 2),
            "best_generation": int(best_gen),
            "outcome": "complete",
        },
        "artifacts": {
            "fitness_curve": "fitness_curve.png",
            "best_params": "checkpoints/best_params.npy",
            "training_log": "training_log.csv",
            "earth_probe": "earth_probe.csv",
            "gravity_sweep": "gravity_sweep.csv",
            "summary": "summary.json",
        },
    }
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [exp] Wrote {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_context = make_run_context(args)
    strategy = get_strategy(
        args.curriculum,
        seed=args.seed,
        gravity_start=args.gravity_start,
        gravity_end=args.gravity_end,
        warmup_frac=args.warmup_frac,
        gravity_min=args.random_gravity_min,
        gravity_max=args.random_gravity_max,
    )

    exp_dir = experiment_dir(args.curriculum, args.exp)
    ckpt_dir = checkpoint_dir(args.curriculum, args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)

    n_params = CPG_N_PARAMS
    print(f"Walker2d-v5 | policy=CPG  params={n_params}")
    print(f"Curriculum   : {strategy.describe()}")
    print(
        "Run context | "
        f"seed={args.seed} "
        f"arm={args.arm or '-'} "
        f"run_id={args.run_id or '-'}"
    )
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
            print(
                f"Checkpoint: {meta.get('label')}  |  "
                f"gen={meta.get('generation')}  |  "
                f"score={meta.get('score')}"
            )

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
    warm_start = bool(args.checkpoint and os.path.exists(args.checkpoint))
    if warm_start:
        x0 = np.load(args.checkpoint)
        if x0.shape[0] != n_params:
            print(
                f"WARNING: checkpoint has {x0.shape[0]} params, "
                f"expected {n_params}. Starting fresh."
            )
            x0 = make_structured_scratch_init(seed=args.seed)
            warm_start = False
        else:
            print(f"Warm-starting from {args.checkpoint}")
    else:
        x0 = make_structured_scratch_init(seed=args.seed)

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
            "tolx": 1e-12,
            "tolfun": 1e-12,
            "tolstagnation": args.generations,
            "tolflatfitness": args.generations,
            "seed": int(args.seed),
        },
    )

    history = []
    earth_probes = []
    best_score = -np.inf
    best_params = x0.copy()
    best_earth_probe_score_so_far = -np.inf
    best_earth_params = x0.copy()
    fit_plot = plot_path(args.curriculum, args.exp)

    print(
        f"\nStarting CMA-ES | pop={args.pop} | sigma0={args.sigma} | "
        f"max_gen={args.generations} | workers={N_WORKERS} | seed={args.seed}"
    )
    print(f"Checkpoints → {ckpt_dir}/")
    print()
    print(
        f"{'Gen':>5}  {'Best':>10}  {'Mean':>10}  {'Sigma':>8}  "
        f"{'Pen':>5}  {'Corr':>5}  {'Gravity':>9}  {'Phase':<14}  {'Time':>7}"
    )
    print("-" * 84)

    ctx = mp.get_context("spawn")
    gen = 0
    with ctx.Pool(N_WORKERS) as pool:
        while not es.stop() and gen < args.generations:
            t0 = time.time()

            g = strategy.gravity(gen, args.generations)
            phase = strategy.phase_label(gen, args.generations)

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

            solutions = es.ask()
            eval_base_seed = int(args.seed * 1_000_000 + gen * 100_000)

            # --- multi-gravity evaluation (MultiEnvironment) or single-gravity ---
            eval_grav_list = strategy.eval_gravities(gen, args.generations)
            if len(eval_grav_list) > 1:
                all_scores: list[list[float]] = []
                for mv_idx, mv_g in enumerate(eval_grav_list):
                    mv_seed = eval_base_seed + mv_idx * 7919  # prime offset per gravity
                    scores = evaluate_parallel(
                        population=solutions,
                        n_episodes=N_EPISODES,
                        max_steps=MAX_STEPS,
                        pool=pool,
                        gravity=mv_g,
                        phase=phase,
                        penalty_scale=penalty_scale,
                        correction_scale=correction_scale,
                        base_seed=mv_seed,
                    )
                    all_scores.append(scores)
                # Combine: per-candidate weighted aggregation across gravities
                n_candidates = len(solutions)
                fitness = [
                    strategy.combine_fitness(
                        [all_scores[gi][ci] for gi in range(len(eval_grav_list))]
                    )
                    for ci in range(n_candidates)
                ]
            else:
                fitness = evaluate_parallel(
                    population=solutions,
                    n_episodes=N_EPISODES,
                    max_steps=MAX_STEPS,
                    pool=pool,
                    gravity=g,
                    phase=phase,
                    penalty_scale=penalty_scale,
                    correction_scale=correction_scale,
                    base_seed=eval_base_seed,
                )

            es.tell(solutions, [-f for f in fitness])

            best_idx = int(np.argmax(fitness))
            best_gen = float(fitness[best_idx])
            mean_gen = float(np.mean(fitness))
            elapsed = float(time.time() - t0)
            best_solution = solutions[best_idx].copy()

            # Notify archive-based strategy every generation (no-op for others)
            strategy.on_new_best(best_solution, best_gen, gen)

            history.append({
                "gen": gen,
                "best": best_gen,
                "mean": mean_gen,
                "sigma": float(es.sigma),
                "penalty_scale": float(penalty_scale),
                "correction_scale": float(correction_scale),
                "gravity": float(g),
                "phase": phase,
                "elapsed_s": elapsed,
            })

            if best_gen > best_score:
                best_score = best_gen
                best_params = best_solution
                save_checkpoint(
                    best_params,
                    os.path.join(ckpt_dir, "best_params.npy"),
                    meta={
                        "score": round(best_score, 2),
                        "generation": gen,
                        "policy": "cpg",
                        "curriculum": args.curriculum,
                        "gravity": round(float(g), 4),
                        "phase": phase,
                        "label": "all-time best",
                        **run_context,
                    },
                )

            if gen % SAVE_EVERY == 0:
                save_checkpoint(
                    es.result.xbest,
                    os.path.join(ckpt_dir, f"gen_{gen:04d}.npy"),
                    meta={
                        "score": round(best_gen, 2),
                        "generation": gen,
                        "policy": "cpg",
                        "curriculum": args.curriculum,
                        "gravity": round(float(g), 4),
                        "phase": phase,
                        "label": f"gen {gen} snapshot",
                        **run_context,
                    },
                )

            if args.probe_every > 0 and gen % args.probe_every == 0:
                probe_seed = int(args.seed * 1_000_000 + 900_000 + gen * 100)
                probe_score, probe_diag = evaluate_policy(
                    params=best_solution,
                    n_episodes=args.probe_episodes,
                    max_steps=MAX_STEPS,
                    seed=probe_seed,
                    gravity=args.probe_gravity,
                    phase="earth probe",
                    return_diagnostics=True,
                    penalty_scale=1.0,
                    correction_scale=THESIS_EVAL_CORR_SCALE,
                )
                probe_stats = probe_diag["mean_stats"]
                probe_row = {
                    "gen": gen,
                    "score": float(probe_score),
                    "net_progress_m": float(probe_stats.get("net_progress_m", 0.0)),
                    "fell": float(probe_stats.get("fell", 0.0)),
                    "backward_distance_m": float(probe_stats.get("backward_distance_m", 0.0)),
                    "mean_forward_speed_mps": float(probe_stats.get("mean_forward_speed_mps", 0.0)),
                }
                earth_probes.append(probe_row)
                is_new_best_earth = probe_score > best_earth_probe_score_so_far
                if is_new_best_earth:
                    best_earth_probe_score_so_far = probe_score
                    best_earth_params = best_solution.copy()
                    save_checkpoint(
                        best_earth_params,
                        os.path.join(ckpt_dir, "best_earth_params.npy"),
                        meta={
                            "score": round(float(probe_score), 2),
                            "generation": gen,
                            "policy": "cpg",
                            "curriculum": args.curriculum,
                            "gravity": round(float(args.probe_gravity), 4),
                            "label": "best earth probe",
                            **run_context,
                        },
                    )
                print(
                    "  [probe] "
                    f"gen={gen} score={probe_row['score']:.1f} "
                    f"net={probe_row['net_progress_m']:.3f} "
                    f"fell={probe_row['fell']:.3f}"
                    + (" [new best earth]" if is_new_best_earth else "")
                )

            if gen % PLOT_EVERY == 0:
                plot_fitness(history, fit_plot, args.curriculum)

            # --- adaptive/archive strategy hooks ---
            # notify() lets stateful strategies update gravity level based on
            # population fitness. Must be called BEFORE at_stage_boundary() so
            # that AdaptiveProgression/_just_advanced is set correctly.
            # Pass best_score (all-time running best, monotonically non-decreasing)
            # not best_gen (current gen best, noisy). Plateau detection requires
            # a stable signal: 30 gens without a new all-time best means the
            # algorithm has converged at the current gravity level.
            strategy.notify(gen, args.generations, best_score, mean_gen, float(es.sigma))

            if strategy.at_stage_boundary(gen, args.generations):
                seed_p = strategy.get_seed_params()
                if seed_p is None:
                    seed_p = best_params
                remaining = args.generations - gen - 1
                if remaining > 0:
                    es = cma.CMAEvolutionStrategy(
                        seed_p,
                        args.sigma,  # restart with original sigma for fresh exploration
                        {
                            "popsize": args.pop,
                            "maxiter": remaining,
                            "verbose": -9,
                            "tolx": 1e-12,
                            "tolfun": 1e-12,
                            "tolstagnation": remaining,
                            "tolflatfitness": remaining,
                            "seed": int(args.seed + gen),
                        },
                    )
                    new_g = strategy.gravity(gen + 1, args.generations)
                    print(
                        f"  [stage] gen={gen}  gravity {g:.2f} → {new_g:.2f}  "
                        f"ES restarted  remaining={remaining}"
                    )

            print(
                f"{gen:>5}  {best_gen:>10.1f}  {mean_gen:>10.1f}  "
                f"{es.sigma:>8.4f}  {penalty_scale:>5.2f}  {correction_scale:>5.2f}  "
                f"{g:>9.4f}  {phase:<14}  {elapsed:>6.1f}s"
            )
            gen += 1

    # Final save + plot
    save_checkpoint(
        best_params,
        os.path.join(ckpt_dir, "final_params.npy"),
        meta={
            "score": round(best_score, 2),
            "generation": gen - 1,
            "policy": "cpg",
            "curriculum": args.curriculum,
            "label": "final (all-time best)",
            **run_context,
        },
    )
    plot_fitness(history, fit_plot, args.curriculum)

    print(f"\nTraining complete. Best fitness: {best_score:.1f}")
    print(f"Best params → {ckpt_dir}/best_params.npy")
    print(f"Fitness curve → {fit_plot}")

    # Select the policy for final evaluation.
    # For curriculum and random arms, best_params was selected by curriculum-gravity
    # fitness, which may favour low-gravity specialists. best_earth_params is
    # selected by the best Earth probe score — always evaluated at Earth gravity
    # with full penalty_scale=1.0 — making it the correct choice for the final
    # Earth scorecard and gravity sweep. Fall back to best_params if probing was
    # disabled (probe_every=0) and no Earth probes were recorded.
    eval_params = best_earth_params if earth_probes else best_params
    earth_probe_score_str = (
        f"{best_earth_probe_score_so_far:.1f}" if earth_probes else "n/a"
    )
    print(
        f"Eval params  → {'best_earth_params' if earth_probes else 'best_params'} "
        f"(earth probe score: {earth_probe_score_str})"
    )

    # Final Earth scorecard
    final_earth = None
    if args.final_eval_episodes > 0:
        final_seed = int(args.seed * 1_000_000 + 980_000)
        final_score, final_diag = evaluate_policy(
            params=eval_params,
            n_episodes=args.final_eval_episodes,
            max_steps=MAX_STEPS,
            seed=final_seed,
            gravity=-9.81,
            phase="final earth",
            return_diagnostics=True,
            penalty_scale=1.0,
            correction_scale=THESIS_EVAL_CORR_SCALE,
        )
        final_stats = final_diag["mean_stats"]
        final_earth = {
            "score": float(final_score),
            "net_progress_m": float(final_stats.get("net_progress_m", 0.0)),
            "fell": float(final_stats.get("fell", 0.0)),
            "backward_distance_m": float(final_stats.get("backward_distance_m", 0.0)),
            "mean_forward_speed_mps": float(final_stats.get("mean_forward_speed_mps", 0.0)),
        }
        print(
            "Final Earth scorecard | "
            f"score={final_earth['score']:.1f} "
            f"net={final_earth['net_progress_m']:.3f} "
            f"fell={final_earth['fell']:.3f}"
        )

    gravity_sweep_rows = []
    robustness = None
    sweep_gravities = parse_gravity_sweep(args.gravity_sweep)
    if sweep_gravities and args.final_eval_episodes > 0:
        gravity_sweep_rows, robustness = evaluate_gravity_sweep(
            params=eval_params,
            gravities=sweep_gravities,
            n_episodes=args.final_eval_episodes,
            seed=args.seed,
        )
        print(
            "Gravity sweep | "
            f"gravities={len(gravity_sweep_rows)} "
            f"mean_net={robustness['mean_net_progress_m']:.3f} "
            f"worst_net={robustness['worst_net_progress_m']:.3f}"
        )

    # Persist thesis analysis artifacts
    write_csv(
        os.path.join(exp_dir, "training_log.csv"),
        history,
        fieldnames=[
            "gen", "best", "mean", "sigma",
            "penalty_scale", "correction_scale",
            "gravity", "phase", "elapsed_s",
        ],
    )

    if earth_probes:
        write_csv(
            os.path.join(exp_dir, "earth_probe.csv"),
            earth_probes,
            fieldnames=[
                "gen", "score", "net_progress_m", "fell",
                "backward_distance_m", "mean_forward_speed_mps",
            ],
        )

    if gravity_sweep_rows:
        write_csv(
            os.path.join(exp_dir, "gravity_sweep.csv"),
            gravity_sweep_rows,
            fieldnames=[
                "gravity", "score", "net_progress_m", "fell",
                "backward_distance_m", "mean_forward_speed_mps",
            ],
        )

    tte_gen = compute_tte(earth_probes, args.generations)
    best_earth_probe_score = (
        float(best_earth_probe_score_so_far) if earth_probes else None
    )
    summary = {
        "run": run_context,
        "curriculum": args.curriculum,
        "gravity_schedule": strategy.describe(),
        "population": int(args.pop),
        "sigma0": float(args.sigma),
        "generations": int(args.generations),
        "schedule": schedule,
        "best_score": float(best_score),
        "best_earth_probe_score": best_earth_probe_score,
        "best_generation": int(gen - 1),
        "tte_generation": int(tte_gen),
        "tte_censored": bool(tte_gen > args.generations),
        "tte_thresholds": {
            "net_progress_m_min": TTE_NET_PROGRESS_MIN,
            "fell_max": TTE_FELL_MAX,
            "backward_distance_m_max": TTE_BACKWARD_MAX,
        },
        "earth_probe": {
            "every_generations": int(args.probe_every),
            "episodes": int(args.probe_episodes),
            "gravity": float(args.probe_gravity),
            "records": int(len(earth_probes)),
        },
        "seed_stability_target_n": 10,
    }
    if final_earth is not None:
        summary["final_earth"] = final_earth
    if robustness is not None:
        summary["robustness"] = robustness
        summary["gravity_sweep"] = {
            "gravities": sweep_gravities,
            "episodes": int(args.final_eval_episodes),
        }

    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_experiment_json(
        exp_dir=exp_dir,
        curriculum_name=args.curriculum,
        strategy=strategy,
        best_score=best_score,
        best_gen=gen - 1,
        max_gen=args.generations,
        pop=args.pop,
        sigma=args.sigma,
        run_context=run_context,
    )

    print(f"  [exp] Wrote {exp_dir}/training_log.csv")
    if earth_probes:
        print(f"  [exp] Wrote {exp_dir}/earth_probe.csv")
    if gravity_sweep_rows:
        print(f"  [exp] Wrote {exp_dir}/gravity_sweep.csv")
    print(f"  [exp] Wrote {exp_dir}/summary.json")


if __name__ == "__main__":
    main()
