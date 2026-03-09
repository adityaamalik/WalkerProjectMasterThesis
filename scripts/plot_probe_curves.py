#!/usr/bin/env python3
"""Plot earth probe learning curves for all thesis experiment arms.

Reads aggregated_probe_curves.csv produced by aggregate_thesis_results.py and
generates a figure showing the median earth probe score over training generations
for each arm, with IQR shading.

Usage:
    python scripts/plot_probe_curves.py
    python scripts/plot_probe_curves.py --thesis-root experiments/thesis --output results.png
    python scripts/plot_probe_curves.py --arms fixed_gravity,random_variable_gravity
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# fmt: off
ARM_COLORS = {
    "fixed_gravity":           "#1f77b4",
    "random_variable_gravity": "#ff7f0e",
    "curriculum_variable_gravity": "#2ca02c",
    "gradual_transition":      "#d62728",
    "staged_evolution":        "#9467bd",
    "multi_environment":       "#8c564b",
    "adaptive_progression":    "#e377c2",
    "archive_based":           "#7f7f7f",
}

ARM_LABELS = {
    "fixed_gravity":           "Fixed Gravity",
    "random_variable_gravity": "Random Gravity",
    "curriculum_variable_gravity": "Gradual (legacy, -2.0 start)",
    "gradual_transition":      "Gradual Transition (Moon→Earth)",
    "staged_evolution":        "Staged Evolution",
    "multi_environment":       "Multi-Environment",
    "adaptive_progression":    "Adaptive Progression",
    "archive_based":           "Archive-Based",
}
# fmt: on


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot earth probe learning curves.")
    p.add_argument("--thesis-root", default="experiments/thesis",
                   help="Path to thesis experiment root directory.")
    p.add_argument("--output", default=None,
                   help="Output image path. Default: <thesis-root>/probe_learning_curves.png")
    p.add_argument("--arms", default=None,
                   help="Comma-separated subset of arm names to plot. Default: all arms in CSV.")
    p.add_argument("--figsize", default="12x6",
                   help="Figure size as WxH in inches (default: 12x6).")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI (default: 150).")
    return p.parse_args()


def read_probe_curves(path: Path) -> dict:
    """Load aggregated_probe_curves.csv, return {arm: {gen: [median, q1, q3, n]}}."""
    arm_data: dict = defaultdict(lambda: {"gen": [], "median": [], "q1": [], "q3": [], "n": []})
    if not path.exists():
        return arm_data

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                arm = row["arm"]
                gen = int(row["gen"])
                median = float(row["median_score"])
                q1 = float(row["q1_score"])
                q3 = float(row["q3_score"])
                n = int(row["n_seeds"])
            except (KeyError, ValueError, TypeError):
                continue
            arm_data[arm]["gen"].append(gen)
            arm_data[arm]["median"].append(median)
            arm_data[arm]["q1"].append(q1)
            arm_data[arm]["q3"].append(q3)
            arm_data[arm]["n"].append(n)

    return arm_data


def main() -> int:
    args = parse_args()
    thesis_root = Path(args.thesis_root)
    curves_path = thesis_root / "aggregated_probe_curves.csv"

    if not curves_path.exists():
        raise SystemExit(
            f"Not found: {curves_path}\n"
            "Run scripts/aggregate_thesis_results.py first."
        )

    arm_data = read_probe_curves(curves_path)
    if not arm_data:
        raise SystemExit("No data found in probe curves CSV.")

    arms_to_plot = list(arm_data.keys())
    if args.arms:
        requested = [a.strip() for a in args.arms.split(",") if a.strip()]
        missing = [a for a in requested if a not in arm_data]
        if missing:
            print(f"Warning: arms not found in CSV: {missing}")
        arms_to_plot = [a for a in requested if a in arm_data]

    if not arms_to_plot:
        raise SystemExit("No arms to plot.")

    # Parse figure size.
    try:
        w, h = (float(v) for v in args.figsize.lower().split("x"))
    except ValueError:
        w, h = 12.0, 6.0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(w, h))

    for arm in arms_to_plot:
        data = arm_data[arm]
        gen = np.array(data["gen"])
        median = np.array(data["median"])
        q1 = np.array(data["q1"])
        q3 = np.array(data["q3"])

        # Sort by generation in case CSV rows aren't ordered.
        order = np.argsort(gen)
        gen = gen[order]
        median = median[order]
        q1 = q1[order]
        q3 = q3[order]

        color = ARM_COLORS.get(arm)
        label = ARM_LABELS.get(arm, arm)
        n_label = max(data["n"]) if data["n"] else 0

        ax.plot(gen, median, label=f"{label} (n={n_label})",
                color=color, linewidth=1.8)
        ax.fill_between(gen, q1, q3, alpha=0.15, color=color)

    # Horizontal reference at score=0 (agent barely moves / falls immediately).
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35,
               label="_zero reference")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Earth Probe Score", fontsize=12)
    ax.set_title("Earth Probe Learning Curves by Training Arm\n"
                 "(median ± IQR across seeds, evaluated at g = −9.81 m/s²)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    output = args.output or str(thesis_root / "probe_learning_curves.png")
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    print(f"Saved: {output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
