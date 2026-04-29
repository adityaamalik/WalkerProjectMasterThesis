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
import csv
import json
import multiprocessing as mp
import os
import time

import cma
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on macOS / headless)
import matplotlib.pyplot as plt
import numpy as np

from cpg_policy import CPGPolicy, N_JOINTS, N_PARAMS as CPG_N_PARAMS
from evaluate import evaluate_policy, evaluate_parallel, N_MORPH_PARAMS


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
ENV_ID = "Walker2d-v5"
N_EPISODES = 3
MAX_STEPS = 1000
POP_SIZE = 40
SIGMA0 = 0.5
MAX_GEN = 500
SAVE_EVERY = 10
PLOT_EVERY = 25
CHECKPOINT_DIR = "checkpoints"
EXPERIMENTS_DIR = "experiments"
N_WORKERS = mp.cpu_count()

PENALTY_SCALE_MIN = 0.2
PENALTY_SCALE_RAMP_GEN = 200
CORR_SCALE_MIN = 0.0
CORR_SCALE_MAX = 0.5
CORR_SCALE_RAMP_GEN = 200
THESIS_EVAL_CORR_SCALE = CORR_SCALE_MAX

TTE_NET_PROGRESS_MIN = 4.0
TTE_FELL_MAX = 0.20
TTE_BACKWARD_MAX = 1.0
TTE_CENSOR_OFFSET = 10

DEFAULT_GRAVITY_SWEEP = "-6.0,-7.5,-9.81,-11.0,-12.0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved .npy params file to load / resume from")
    p.add_argument("--render", action="store_true",
                   help="Render the policy — requires --checkpoint")
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--sigma", type=float, default=SIGMA0)
    p.add_argument("--generations", type=int, default=MAX_GEN)
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for init, CMA-ES and rollout evaluation.")
    p.add_argument("--arm", type=str, default=None,
                   help="Optional experiment arm label (e.g. fixed_gravity).")
    p.add_argument("--run-id", type=str, default=None,
                   help="Optional run identifier (e.g. fixed_seed_03).")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Checkpoint working directory. Default: checkpoints/cpg")
    p.add_argument("--exp", type=str, default=None,
                   help="Experiment ID (e.g. exp006_cpg_v6_...). "
                        "If provided, outputs are archived into experiments/<exp>/.")

    p.add_argument("--probe-every", type=int, default=10,
                   help="Earth probe cadence in generations (0 disables).")
    p.add_argument("--probe-episodes", type=int, default=10,
                   help="Episodes per Earth probe checkpoint evaluation.")
    p.add_argument("--probe-gravity", type=float, default=-9.81,
                   help="Gravity used for periodic Earth probes.")

    p.add_argument("--final-eval-episodes", type=int, default=20,
                   help="Episodes for final Earth and gravity-sweep evaluations.")
    p.add_argument("--gravity-sweep", type=str, default=DEFAULT_GRAVITY_SWEEP,
                   help="Comma-separated gravities for final sweep. Empty disables sweep.")

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
    p.add_argument("--evolve-morphology", action="store_true",
                   help="Enable co-evolution of scalar morphological traits "
                        "(mass, gear, friction). Expands search space by 4 params.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_path() -> str:
    return "fitness_curve_cpg.png"


def parse_gravity_sweep(spec: str) -> list[float]:
    """Parse comma-separated gravity values. Empty string -> []."""
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


def make_structured_scratch_init(seed: int = 42,
                                 evolve_morphology: bool = False) -> np.ndarray:
    """
    Build a scratch initial mean that is dynamically gentle.

    Reason:
    Fully random CPG logits around 0 imply amplitude ~0.5 and moderate
    oscillator speed on all joints, which tends to cause immediate falls.
    A low-amplitude/low-frequency prior keeps early exploration near
    stabilisable motions while still being from scratch (no checkpoint).

    If *evolve_morphology* is True, appends N_MORPH_PARAMS zeros (morphological
    logits at neutral → multiplier = 1.0 via 2^tanh(0) mapping).
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
    x0[cpg_amp_end:cpg_w_end] = -0.2
    # Phase offsets start neutral; left/right anti-phase comes from theta reset
    x0[cpg_w_end:cpg_phi_end] = 0.0

    # Small exploration noise, smaller on MLP than on CPG core.
    x0[cpg_amp_start:cpg_phi_end] += rng.normal(0.0, 0.12, cpg_phi_end - cpg_amp_start)
    x0[cpg_phi_end:] += rng.normal(0.0, 0.02, CPG_N_PARAMS - cpg_phi_end)

    if evolve_morphology:
        # Morph logits initialised to 0.0 (neutral → multiplier = 1.0)
        x0 = np.concatenate([x0, np.zeros(N_MORPH_PARAMS, dtype=np.float64)])
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


def compute_tte(earth_probes: list[dict], max_generations: int) -> int:
    """First generation meeting Earth success thresholds, else censored."""
    for p in earth_probes:
        if (
            p["net_progress_m"] >= TTE_NET_PROGRESS_MIN and
            p["fell"] <= TTE_FELL_MAX and
            p["backward_distance_m"] <= TTE_BACKWARD_MAX
        ):
            return int(p["gen"])
    return int(max_generations + TTE_CENSOR_OFFSET)


def save_checkpoint(params: np.ndarray, path: str,
                    meta: dict | None = None) -> None:
    """Save params as .npy and write a companion .json with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    if meta is not None:
        meta_path = path.replace(".npy", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def plot_fitness(history: list[dict], path: str) -> None:
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


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def archive_experiment(exp_id: str,
                       best_score: float, best_gen: int,
                       plot_file: str,
                       ckpt_dir: str,
                       run_context: dict) -> str:
    """
    Copy best checkpoint + fitness curve into experiments/<exp_id>/.
    Also writes a minimal experiment.json if one doesn't already exist.
    """
    import shutil

    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    ckpt_copy = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_copy, exist_ok=True)

    # Fitness curve
    if os.path.exists(plot_file):
        shutil.copy2(plot_file, os.path.join(exp_dir, "fitness_curve.png"))

    # Best params + metadata
    src_npy = os.path.join(ckpt_dir, "best_params.npy")
    src_json = os.path.join(ckpt_dir, "best_params.json")
    if os.path.exists(src_npy):
        shutil.copy2(src_npy, os.path.join(ckpt_copy, "best_params.npy"))
    if os.path.exists(src_json):
        shutil.copy2(src_json, os.path.join(ckpt_copy, "best_params.json"))

    # experiment.json (only if absent)
    meta_path = os.path.join(exp_dir, "experiment.json")
    if not os.path.exists(meta_path):
        stub = {
            "id": exp_id,
            "policy": "cpg",
            "result": {
                "best_score": round(best_score, 2),
                "best_generation": int(best_gen),
            },
            "run": run_context,
            "artifacts": {
                "fitness_curve": "fitness_curve.png",
                "best_params": "checkpoints/best_params.npy",
            },
        }
        with open(meta_path, "w") as f:
            json.dump(stub, f, indent=2)

    print(f"  [exp] Archived → {exp_dir}/")
    return exp_dir


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_context = make_run_context(args)

    n_params = CPG_N_PARAMS + N_MORPH_PARAMS if args.evolve_morphology else CPG_N_PARAMS
    print(f"Walker2d-v5 | policy=CPG  params={n_params}"
          + (" (morph ON)" if args.evolve_morphology else ""))
    print(CPGPolicy().describe())
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
                f"train_score={meta.get('score')}"
            )

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
    ckpt_dir = args.checkpoint_dir or os.path.join(CHECKPOINT_DIR, "cpg")
    os.makedirs(ckpt_dir, exist_ok=True)

    warm_start = bool(args.checkpoint and os.path.exists(args.checkpoint))
    if warm_start:
        x0 = np.load(args.checkpoint)
        if x0.shape[0] != n_params:
            print(
                f"WARNING: checkpoint has {x0.shape[0]} params, "
                f"expected {n_params}. Starting fresh."
            )
            x0 = make_structured_scratch_init(seed=args.seed,
                                              evolve_morphology=args.evolve_morphology)
            warm_start = False
        else:
            print(f"Resuming from {args.checkpoint}")
    else:
        x0 = make_structured_scratch_init(seed=args.seed,
                                          evolve_morphology=args.evolve_morphology)
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
    plot_file = plot_path()

    print(
        f"\nStarting CMA-ES | pop={args.pop} | sigma0={args.sigma} | "
        f"max_gen={args.generations} | workers={N_WORKERS} | seed={args.seed}\n"
    )
    print(
        f"{'Gen':>5}  {'Best':>10}  {'Mean':>10}  {'Sigma':>8}  "
        f"{'Pen':>5}  {'Corr':>5}  {'Time':>7}"
    )
    print("-" * 66)

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

            eval_base_seed = int(args.seed * 1_000_000 + gen * 100_000)
            fitness = evaluate_parallel(
                population=solutions,
                n_episodes=N_EPISODES,
                max_steps=MAX_STEPS,
                pool=pool,
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

            history.append({
                "gen": gen,
                "best": best_gen,
                "mean": mean_gen,
                "sigma": float(es.sigma),
                "penalty_scale": float(penalty_scale),
                "correction_scale": float(correction_scale),
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
                        "evolve_morphology": args.evolve_morphology,
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
                        "evolve_morphology": args.evolve_morphology,
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
                            "evolve_morphology": args.evolve_morphology,
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
                plot_fitness(history, plot_file)

            print(
                f"{gen:>5}  {best_gen:>10.1f}  {mean_gen:>10.1f}  "
                f"{es.sigma:>8.4f}  {penalty_scale:>5.2f}  {correction_scale:>5.2f}  {elapsed:>6.1f}s"
            )
            gen += 1

    stop_reasons = es.stop()
    if stop_reasons:
        print(f"CMA-ES stop reasons: {stop_reasons}")

    save_checkpoint(
        best_params,
        os.path.join(ckpt_dir, "final_params.npy"),
        meta={
            "score": round(best_score, 2),
            "generation": gen - 1,
            "policy": "cpg",
            "evolve_morphology": args.evolve_morphology,
            "label": "final (all-time best)",
            **run_context,
        },
    )
    plot_fitness(history, plot_file)

    print(f"\nTraining complete. Best fitness: {best_score:.1f}")
    print(f"Best params → {ckpt_dir}/best_params.npy")
    print(f"Fitness curve → {plot_file}")

    # Select the policy for final evaluation.
    # best_earth_params is the solution that achieved the highest Earth probe score
    # across all periodic probes — always evaluated at Earth gravity with full
    # penalty_scale=1.0. This is the correct and consistent selection criterion
    # across all three arms. Falls back to best_params if probing was disabled.
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

    # Auto-archive into experiments/ if --exp was specified
    if args.exp:
        exp_dir = archive_experiment(
            exp_id=args.exp,
            best_score=best_score,
            best_gen=gen - 1,
            plot_file=plot_file,
            ckpt_dir=ckpt_dir,
            run_context=run_context,
        )

        # Per-generation training log
        write_csv(
            os.path.join(exp_dir, "training_log.csv"),
            history,
            fieldnames=[
                "gen", "best", "mean", "sigma", "penalty_scale",
                "correction_scale", "elapsed_s",
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
            "env_id": ENV_ID,
            "policy": "cpg",
            "evolve_morphology": args.evolve_morphology,
            "population": int(args.pop),
            "sigma0": float(args.sigma),
            "generations": int(args.generations),
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

        print(f"  [exp] Wrote {exp_dir}/training_log.csv")
        if earth_probes:
            print(f"  [exp] Wrote {exp_dir}/earth_probe.csv")
        if gravity_sweep_rows:
            print(f"  [exp] Wrote {exp_dir}/gravity_sweep.csv")
        print(f"  [exp] Wrote {exp_dir}/summary.json")


if __name__ == "__main__":
    main()
