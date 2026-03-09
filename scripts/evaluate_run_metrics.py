#!/usr/bin/env python3
"""Post-hoc evaluator for Earth probes and final gravity sweep from checkpoints.

Use this when a run exists but probe/sweep artifacts are missing or need recompute.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import evaluate_policy

TTE_NET_PROGRESS_MIN = 4.0
TTE_FELL_MAX = 0.20
TTE_BACKWARD_MAX = 1.0
TTE_CENSOR_OFFSET = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved checkpoints into thesis metrics.")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Checkpoint directory containing gen_XXXX.npy and/or best_params.npy")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory where earth_probe.csv / gravity_sweep.csv / summary.json are written")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--probe-gravity", type=float, default=-9.81)

    p.add_argument("--final-checkpoint", type=str, default="best_params.npy",
                   help="File name in checkpoint-dir for final policy evaluation")
    p.add_argument("--final-eval-episodes", type=int, default=20)
    p.add_argument("--gravity-sweep", type=str, default="-6.0,-7.5,-9.81,-11.0,-12.0")

    p.add_argument("--penalty-scale", type=float, default=1.0)
    p.add_argument("--correction-scale", type=float, default=0.5)
    return p.parse_args()


def parse_gravity_sweep(spec: str) -> list[float]:
    text = (spec or "").strip()
    if not text:
        return []
    out = []
    for token in text.split(","):
        t = token.strip()
        if t:
            out.append(float(t))
    return out


def checkpoint_generation(path: Path) -> int:
    # expects gen_XXXX.npy
    stem = path.stem
    try:
        return int(stem.split("_")[1])
    except Exception:
        return 0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_tte(earth_probes: list[dict], max_gen: int) -> int:
    for p in earth_probes:
        if (
            p["net_progress_m"] >= TTE_NET_PROGRESS_MIN and
            p["fell"] <= TTE_FELL_MAX and
            p["backward_distance_m"] <= TTE_BACKWARD_MAX
        ):
            return int(p["gen"])
    return int(max_gen + TTE_CENSOR_OFFSET)


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_ckpts = sorted(ckpt_dir.glob("gen_*.npy"), key=checkpoint_generation)
    if not gen_ckpts:
        raise SystemExit(f"No generation checkpoints found in {ckpt_dir}")

    # Earth probe from selected generations
    earth_rows = []
    for path in gen_ckpts:
        gen = checkpoint_generation(path)
        if args.probe_every > 0 and gen % args.probe_every != 0:
            continue
        params = np.load(path)
        probe_seed = int(args.seed * 1_000_000 + 900_000 + gen * 100)
        score, diag = evaluate_policy(
            params=params,
            n_episodes=args.probe_episodes,
            max_steps=1000,
            seed=probe_seed,
            gravity=args.probe_gravity,
            phase="earth probe",
            return_diagnostics=True,
            penalty_scale=args.penalty_scale,
            correction_scale=args.correction_scale,
        )
        stats = diag["mean_stats"]
        earth_rows.append({
            "gen": gen,
            "score": float(score),
            "net_progress_m": float(stats.get("net_progress_m", 0.0)),
            "fell": float(stats.get("fell", 0.0)),
            "backward_distance_m": float(stats.get("backward_distance_m", 0.0)),
            "mean_forward_speed_mps": float(stats.get("mean_forward_speed_mps", 0.0)),
        })

    write_csv(
        out_dir / "earth_probe.csv",
        earth_rows,
        [
            "gen", "score", "net_progress_m", "fell",
            "backward_distance_m", "mean_forward_speed_mps",
        ],
    )

    # Final checkpoint evaluation
    final_path = ckpt_dir / args.final_checkpoint
    if not final_path.exists():
        raise SystemExit(f"Final checkpoint not found: {final_path}")
    final_params = np.load(final_path)

    final_score, final_diag = evaluate_policy(
        params=final_params,
        n_episodes=args.final_eval_episodes,
        max_steps=1000,
        seed=int(args.seed * 1_000_000 + 980_000),
        gravity=-9.81,
        phase="final earth",
        return_diagnostics=True,
        penalty_scale=args.penalty_scale,
        correction_scale=args.correction_scale,
    )
    final_stats = final_diag["mean_stats"]
    final_earth = {
        "score": float(final_score),
        "net_progress_m": float(final_stats.get("net_progress_m", 0.0)),
        "fell": float(final_stats.get("fell", 0.0)),
        "backward_distance_m": float(final_stats.get("backward_distance_m", 0.0)),
        "mean_forward_speed_mps": float(final_stats.get("mean_forward_speed_mps", 0.0)),
    }

    # Gravity sweep
    sweep_rows = []
    for idx, g in enumerate(parse_gravity_sweep(args.gravity_sweep)):
        score, diag = evaluate_policy(
            params=final_params,
            n_episodes=args.final_eval_episodes,
            max_steps=1000,
            seed=int(args.seed * 1_000_000 + 700_000 + idx * 10_000),
            gravity=g,
            phase=f"sweep {g:.2f}",
            return_diagnostics=True,
            penalty_scale=args.penalty_scale,
            correction_scale=args.correction_scale,
        )
        stats = diag["mean_stats"]
        sweep_rows.append({
            "gravity": float(g),
            "score": float(score),
            "net_progress_m": float(stats.get("net_progress_m", 0.0)),
            "fell": float(stats.get("fell", 0.0)),
            "backward_distance_m": float(stats.get("backward_distance_m", 0.0)),
            "mean_forward_speed_mps": float(stats.get("mean_forward_speed_mps", 0.0)),
        })

    write_csv(
        out_dir / "gravity_sweep.csv",
        sweep_rows,
        [
            "gravity", "score", "net_progress_m", "fell",
            "backward_distance_m", "mean_forward_speed_mps",
        ],
    )

    robustness = None
    if sweep_rows:
        robustness = {
            "mean_net_progress_m": float(np.mean([r["net_progress_m"] for r in sweep_rows])),
            "worst_net_progress_m": float(np.min([r["net_progress_m"] for r in sweep_rows])),
            "mean_fall_rate": float(np.mean([r["fell"] for r in sweep_rows])),
        }

    max_gen = max(checkpoint_generation(p) for p in gen_ckpts)
    tte_gen = compute_tte(earth_rows, max_gen)

    summary = {
        "seed": int(args.seed),
        "best_score": float(final_earth["score"]),
        "best_generation": int(max_gen),
        "tte_generation": int(tte_gen),
        "tte_censored": bool(tte_gen > max_gen),
        "tte_thresholds": {
            "net_progress_m_min": TTE_NET_PROGRESS_MIN,
            "fell_max": TTE_FELL_MAX,
            "backward_distance_m_max": TTE_BACKWARD_MAX,
        },
        "final_earth": final_earth,
        "seed_stability_target_n": 10,
    }
    if robustness is not None:
        summary["robustness"] = robustness

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_dir / 'earth_probe.csv'}")
    print(f"Wrote {out_dir / 'gravity_sweep.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
