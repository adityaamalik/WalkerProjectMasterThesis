#!/usr/bin/env python3
"""Batch runner for thesis experiments (fixed, random-variable, curriculum).

Default protocol:
- 3 arms
- 10 seeds per arm (0..9)
- standardized outputs in experiments/thesis/<arm>/seed_<NN>/
- run manifest in experiments/thesis/run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS_DEFAULT = [
    "fixed_gravity",
    "random_variable_gravity",
    "gradual_transition",
    "staged_evolution",
    "multi_environment",
    "adaptive_progression",
    "archive_based",
]

# Legacy arm kept for backward compatibility with the original 30-run dataset.
_LEGACY_ARMS = {"curriculum_variable_gravity"}
_ALL_KNOWN_ARMS = set(ARMS_DEFAULT) | _LEGACY_ARMS


def parse_seed_spec(seed_spec: str | None, num_seeds: int) -> list[int]:
    if seed_spec is None:
        return list(range(num_seeds))
    out = []
    for chunk in seed_spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run thesis training batch across arms/seeds.")
    p.add_argument("--num-seeds", type=int, default=10,
                   help="Number of seeds when --seeds is not provided.")
    p.add_argument("--seeds", type=str, default=None,
                   help="Explicit seed list/ranges (e.g. 0,1,2 or 0-9).")
    p.add_argument("--arms", type=str, default=",".join(ARMS_DEFAULT),
                   help="Comma-separated arms subset.")

    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--sigma", type=float, default=0.5)

    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--probe-episodes", type=int, default=10)
    p.add_argument("--probe-gravity", type=float, default=-9.81)
    p.add_argument("--final-eval-episodes", type=int, default=20)
    p.add_argument("--gravity-sweep", type=str, default="-6.0,-7.5,-9.81,-11.0,-12.0")

    p.add_argument("--curriculum", type=str, default="gradual_transition",
                   help="Ordered curriculum strategy for curriculum_variable_gravity arm (legacy).")
    p.add_argument("--random-gravity-min", type=float, default=-12.0)
    p.add_argument("--random-gravity-max", type=float, default=-6.0)
    # gradual_transition arm parameters (Moon → Earth by default)
    p.add_argument("--gravity-start", type=float, default=-1.6,
                   help="Start gravity for the gradual_transition arm (default: Moon -1.6).")
    p.add_argument("--gravity-end", type=float, default=-9.81,
                   help="End gravity for the gradual_transition arm (default: Earth -9.81).")

    p.add_argument("--evolve-morphology", action="store_true",
                   help="Enable co-evolution of scalar morphological traits "
                        "(mass, gear, friction). Passed through to train.py / train_curriculum.py.")
    p.add_argument("--exp-root", type=str, default="thesis",
                   help="Subdirectory under experiments/ for all outputs (default: thesis).")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python executable used for child training runs.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs that already have summary.json")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop batch on first failed run.")
    return p.parse_args()


def build_command(args: argparse.Namespace, arm: str, seed: int) -> tuple[list[str], str, Path]:
    root = args.exp_root
    exp_rel = f"{root}/{arm}/seed_{seed:02d}"
    run_id = f"{arm}_seed_{seed:02d}"
    ckpt_work_rel = f"experiments/{root}/{arm}/seed_{seed:02d}/working_checkpoints"

    common = [
        "--seed", str(seed),
        "--arm", arm,
        "--run-id", run_id,
        "--exp", exp_rel,
        "--pop", str(args.pop),
        "--sigma", str(args.sigma),
        "--generations", str(args.generations),
        "--probe-every", str(args.probe_every),
        "--probe-episodes", str(args.probe_episodes),
        "--probe-gravity", str(args.probe_gravity),
        "--final-eval-episodes", str(args.final_eval_episodes),
        f"--gravity-sweep={args.gravity_sweep}",
    ]
    if args.evolve_morphology:
        common.append("--evolve-morphology")

    if arm == "fixed_gravity":
        cmd = [args.python, "train.py", "--checkpoint-dir", ckpt_work_rel, *common]

    elif arm == "random_variable_gravity":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "random_uniform",
            "--random-gravity-min", str(args.random_gravity_min),
            "--random-gravity-max", str(args.random_gravity_max),
            *common,
        ]

    # Legacy arm: gradual_transition with -2.0 warmup start (pre-Phase-20 default)
    elif arm == "curriculum_variable_gravity":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "gradual_transition",
            *common,
        ]

    # New arm: gradual_transition starting at Moon gravity (-1.6)
    elif arm == "gradual_transition":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "gradual_transition",
            "--gravity-start", str(args.gravity_start),
            "--gravity-end", str(args.gravity_end),
            *common,
        ]

    elif arm == "staged_evolution":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "staged_evolution",
            *common,
        ]

    elif arm == "multi_environment":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "multi_environment",
            *common,
        ]

    elif arm == "adaptive_progression":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "adaptive_progression",
            *common,
        ]

    elif arm == "archive_based":
        cmd = [
            args.python, "train_curriculum.py",
            "--curriculum", "archive_based",
            *common,
        ]

    else:
        raise ValueError(f"Unknown arm '{arm}'")

    return cmd, exp_rel, Path("experiments") / root / arm / f"seed_{seed:02d}"


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    seeds = parse_seed_spec(args.seeds, args.num_seeds)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    invalid = [a for a in arms if a not in _ALL_KNOWN_ARMS]
    if invalid:
        raise ValueError(f"Unknown arms: {invalid}. Allowed: {sorted(_ALL_KNOWN_ARMS)}")

    thesis_root = repo_root / "experiments" / args.exp_root
    thesis_root.mkdir(parents=True, exist_ok=True)

    manifest_runs = []
    failures = 0

    print(f"Batch config | arms={arms} seeds={seeds} (n={len(seeds)})")

    for arm in arms:
        for seed in seeds:
            cmd, exp_rel, seed_dir_rel = build_command(args, arm, seed)
            seed_dir = repo_root / seed_dir_rel
            seed_dir.mkdir(parents=True, exist_ok=True)
            summary_path = seed_dir / "summary.json"
            log_path = seed_dir / "run.log"

            if args.resume and summary_path.exists():
                entry = {
                    "arm": arm,
                    "seed": seed,
                    "exp": exp_rel,
                    "status": "skipped",
                    "reason": "summary_exists",
                    "log": str(log_path),
                }
                manifest_runs.append(entry)
                print(f"[skip] {arm} seed={seed:02d} (summary exists)")
                continue

            print(f"[run] {arm} seed={seed:02d}")
            t0 = time.time()
            if args.dry_run:
                status = "dry_run"
                returncode = 0
            else:
                with open(log_path, "w") as logf:
                    proc = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                returncode = proc.returncode
                status = "completed" if returncode == 0 else "failed"

            elapsed_s = time.time() - t0
            entry = {
                "arm": arm,
                "seed": seed,
                "exp": exp_rel,
                "status": status,
                "returncode": returncode,
                "elapsed_s": round(elapsed_s, 2),
                "command": cmd,
                "log": str(log_path),
                "summary": str(summary_path),
            }

            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    entry["best_score"] = summary.get("best_score")
                    entry["tte_generation"] = summary.get("tte_generation")
                except (json.JSONDecodeError, OSError):
                    pass

            manifest_runs.append(entry)
            print(
                f"[done] {arm} seed={seed:02d} status={status} "
                f"elapsed={elapsed_s:.1f}s"
            )

            if status == "failed":
                failures += 1
                if args.fail_fast:
                    print("Fail-fast enabled. Stopping batch.")
                    break
        if failures and args.fail_fast:
            break

    manifest = {
        "created_at_epoch": int(time.time()),
        "config": {
            "arms": arms,
            "seeds": seeds,
            "num_seeds": len(seeds),
            "generations": args.generations,
            "pop": args.pop,
            "sigma": args.sigma,
            "evolve_morphology": args.evolve_morphology,
            "probe_every": args.probe_every,
            "probe_episodes": args.probe_episodes,
            "probe_gravity": args.probe_gravity,
            "final_eval_episodes": args.final_eval_episodes,
            "gravity_sweep": args.gravity_sweep,
            "curriculum": args.curriculum,
            "random_gravity_min": args.random_gravity_min,
            "random_gravity_max": args.random_gravity_max,
            "dry_run": args.dry_run,
            "resume": args.resume,
        },
        "summary": {
            "runs_total": len(manifest_runs),
            "runs_failed": sum(1 for r in manifest_runs if r.get("status") == "failed"),
            "runs_completed": sum(1 for r in manifest_runs if r.get("status") == "completed"),
            "runs_skipped": sum(1 for r in manifest_runs if r.get("status") == "skipped"),
            "runs_dry_run": sum(1 for r in manifest_runs if r.get("status") == "dry_run"),
        },
        "runs": manifest_runs,
    }

    manifest_path = thesis_root / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    if failures > 0 and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
