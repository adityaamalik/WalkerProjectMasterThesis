#!/usr/bin/env python3
"""Aggregated earth-probe learning curves with confidence bands.

Loads earth_probe.csv from all seeds of each arm, computes mean ± SEM,
and plots them on a single figure for thesis comparison.

Outputs:
  - experiments/learning_curves.png         (main figure)
  - experiments/learning_curves_data.csv    (aggregated data for reproducibility)
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


ARM_COLORS = {
    "fixed_gravity":            "#2c3e50",
    "random_variable_gravity":  "#7f8c8d",
    "gradual_transition":       "#2980b9",
    "staged_evolution":         "#27ae60",
    "adaptive_progression":     "#e67e22",
    "archive_based":            "#8e44ad",
    "multi_environment":        "#c0392b",
}

ARM_LABELS = {
    "fixed_gravity":            "Fixed Gravity (baseline)",
    "random_variable_gravity":  "Random Variable (control)",
    "gradual_transition":       "Gradual Transition",
    "staged_evolution":         "Staged Evolution",
    "adaptive_progression":     "Adaptive Progression",
    "archive_based":            "Archive-Based",
    "multi_environment":        "Multi-Environment",
}


def load_probe_curves(exp_root, arm, n_seeds=10):
    """Load earth_probe.csv for all seeds. Returns dict: gen -> [scores]."""
    gen_scores = {}
    gen_progress = {}
    loaded = 0
    for seed in range(n_seeds):
        path = f"experiments/{exp_root}/{arm}/seed_{seed:02d}/earth_probe.csv"
        if not os.path.exists(path):
            continue
        loaded += 1
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                g = int(row["gen"])
                score = float(row["score"])
                progress = float(row["net_progress_m"])
                gen_scores.setdefault(g, []).append(score)
                gen_progress.setdefault(g, []).append(progress)
    return gen_scores, gen_progress, loaded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root", default="thesis_morph")
    p.add_argument("--arms", type=str,
                   default="fixed_gravity,random_variable_gravity,gradual_transition,"
                           "staged_evolution,adaptive_progression,archive_based,multi_environment")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output-png", default="experiments/learning_curves.png")
    p.add_argument("--output-csv", default="experiments/learning_curves_data.csv")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_score = axes[0]
    ax_progress = axes[1]

    csv_rows = []

    for arm in arms:
        gen_scores, gen_progress, n_loaded = load_probe_curves(args.exp_root, arm, args.seeds)
        if not gen_scores:
            print(f"  SKIP {arm} — no probe data")
            continue
        print(f"  {arm}: {n_loaded} seeds loaded")

        # Sort by generation
        gens = sorted(gen_scores.keys())
        # Only include generations where all seeds have data
        gens = [g for g in gens if len(gen_scores[g]) >= n_loaded]

        means_score = [np.mean(gen_scores[g]) for g in gens]
        sems_score = [np.std(gen_scores[g]) / np.sqrt(len(gen_scores[g])) for g in gens]
        means_progress = [np.mean(gen_progress[g]) for g in gens]
        sems_progress = [np.std(gen_progress[g]) / np.sqrt(len(gen_progress[g])) for g in gens]

        color = ARM_COLORS.get(arm, "#333333")
        label = ARM_LABELS.get(arm, arm)

        # Score plot
        ax_score.plot(gens, means_score, color=color, label=label, linewidth=1.5)
        ax_score.fill_between(
            gens,
            [m - s for m, s in zip(means_score, sems_score)],
            [m + s for m, s in zip(means_score, sems_score)],
            alpha=0.15, color=color,
        )

        # Progress plot
        ax_progress.plot(gens, means_progress, color=color, label=label, linewidth=1.5)
        ax_progress.fill_between(
            gens,
            [m - s for m, s in zip(means_progress, sems_progress)],
            [m + s for m, s in zip(means_progress, sems_progress)],
            alpha=0.15, color=color,
        )

        # CSV data
        for i, g in enumerate(gens):
            csv_rows.append({
                "arm": arm,
                "generation": g,
                "mean_score": round(means_score[i], 2),
                "sem_score": round(sems_score[i], 2),
                "mean_progress_m": round(means_progress[i], 3),
                "sem_progress_m": round(sems_progress[i], 3),
                "n_seeds": len(gen_scores[g]),
            })

    # Format score plot
    ax_score.set_ylabel("Earth Probe Score", fontsize=12)
    ax_score.set_title("Learning Curves: Earth Probe Performance (mean ± SEM, n=10 seeds)", fontsize=13)
    ax_score.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_score.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax_score.grid(True, alpha=0.2)

    # Format progress plot
    ax_progress.set_xlabel("Generation", fontsize=12)
    ax_progress.set_ylabel("Net Progress on Earth (m)", fontsize=12)
    ax_progress.set_title("Learning Curves: Earth Walking Distance (mean ± SEM, n=10 seeds)", fontsize=13)
    ax_progress.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_progress.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax_progress.axhline(y=4.0, color="green", linestyle=":", alpha=0.4, label="TTE threshold (4m)")
    ax_progress.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure → {args.output_png}")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "arm", "generation", "mean_score", "sem_score",
            "mean_progress_m", "sem_progress_m", "n_seeds",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved data → {args.output_csv}")


if __name__ == "__main__":
    main()
