#!/usr/bin/env python3
"""Plot CMA-ES sigma (step-size) evolution for all thesis arms.

Sigma is CMA-ES's internal step size — a direct window into the optimizer's
state. Curriculum-driven ES restarts appear as visible upward jumps; gradual
gravity changes appear as altered convergence rates vs the fixed baseline.

Reads training_log.csv from every seed directory and aggregates per (arm, gen).

Usage:
    python scripts/plot_sigma_evolution.py
    python scripts/plot_sigma_evolution.py --thesis-root experiments/thesis_val100 --output sigma.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_COLORS = {
    "fixed_gravity":               "#1f77b4",
    "random_variable_gravity":     "#ff7f0e",
    "curriculum_variable_gravity": "#2ca02c",
    "gradual_transition":          "#d62728",
    "staged_evolution":            "#9467bd",
    "multi_environment":           "#8c564b",
    "adaptive_progression":        "#e377c2",
    "archive_based":               "#7f7f7f",
}

ARM_LABELS = {
    "fixed_gravity":               "Fixed Gravity (baseline)",
    "random_variable_gravity":     "Random Gravity (control)",
    "curriculum_variable_gravity": "Gradual (legacy)",
    "gradual_transition":          "Gradual Transition",
    "staged_evolution":            "Staged Evolution",
    "multi_environment":           "Multi-Environment",
    "adaptive_progression":        "Adaptive Progression",
    "archive_based":               "Archive-Based",
}

ARMS_DEFAULT = [
    "fixed_gravity",
    "random_variable_gravity",
    "curriculum_variable_gravity",
    "gradual_transition",
    "staged_evolution",
    "multi_environment",
    "adaptive_progression",
    "archive_based",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CMA-ES sigma evolution per arm.")
    p.add_argument("--thesis-root", default="experiments/thesis")
    p.add_argument("--arms", default=None,
                   help="Comma-separated arm subset. Default: all present.")
    p.add_argument("--output", default=None)
    p.add_argument("--figsize", default="12x6")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_sigma_curves(thesis_root: Path, arms: list[str]) -> dict:
    """Return {arm: {gen: [sigma_values_across_seeds]}}."""
    data: dict = {}
    for arm in arms:
        arm_dir = thesis_root / arm
        if not arm_dir.exists():
            continue
        gen_sigmas: dict = defaultdict(list)
        seed_dirs = sorted([p for p in arm_dir.glob("seed_*") if p.is_dir()])
        for sd in seed_dirs:
            log = sd / "training_log.csv"
            if not log.exists():
                continue
            with open(log, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        gen = int(float(row["gen"]))
                        sigma = float(row["sigma"])
                    except (KeyError, ValueError):
                        continue
                    gen_sigmas[gen].append(sigma)
        if gen_sigmas:
            data[arm] = gen_sigmas
    return data


def main() -> int:
    args = parse_args()
    thesis_root = Path(args.thesis_root)

    arms = ARMS_DEFAULT
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    sigma_data = load_sigma_curves(thesis_root, arms)
    arms_present = [a for a in arms if a in sigma_data]

    if not arms_present:
        raise SystemExit(f"No training_log.csv files found under {thesis_root}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    try:
        w, h = (float(v) for v in args.figsize.lower().split("x"))
    except ValueError:
        w, h = 12.0, 6.0

    fig, ax = plt.subplots(figsize=(w, h))

    for arm in arms_present:
        gen_sigmas = sigma_data[arm]
        gens_sorted = sorted(gen_sigmas)
        medians, q1s, q3s, ns = [], [], [], []

        for g in gens_sorted:
            arr = np.array(gen_sigmas[g])
            medians.append(float(np.median(arr)))
            q1s.append(float(np.percentile(arr, 25)))
            q3s.append(float(np.percentile(arr, 75)))
            ns.append(len(arr))

        gens = np.array(gens_sorted)
        medians = np.array(medians)
        q1s = np.array(q1s)
        q3s = np.array(q3s)
        n_label = max(ns)

        color = ARM_COLORS.get(arm)
        label = ARM_LABELS.get(arm, arm)

        ax.plot(gens, medians, label=f"{label} (n={n_label})",
                color=color, linewidth=1.8)
        ax.fill_between(gens, q1s, q3s, alpha=0.12, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("CMA-ES Step Size σ", fontsize=12)
    ax.set_title(
        "CMA-ES Step Size (σ) Evolution by Training Arm\n"
        "Upward jumps = explicit ES restarts at curriculum stage boundaries",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    output = args.output or str(thesis_root / "sigma_evolution.png")
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    print(f"Saved: {output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
