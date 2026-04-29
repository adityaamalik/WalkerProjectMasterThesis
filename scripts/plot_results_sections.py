#!/usr/bin/env python3
"""Generate figures for Results sections 4.5-4.7.

Produces:
  1. Morphology heatmap (section 4.5)
  2. Gravity robustness profiles (section 4.6)
  3. Fitness sensitivity tornado chart (section 4.7)
"""

import json
import csv
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiments" / "thesis_morph"
OUT_DIR = ROOT / "thesis_text" / "thesis_template" / "images"

ARMS = [
    "staged_evolution", "gradual_transition", "archive_based",
    "multi_environment", "fixed_gravity", "random_variable_gravity",
    "adaptive_progression",
]
LABELS = {
    "staged_evolution":        "Staged evolution",
    "gradual_transition":      "Gradual transition",
    "archive_based":           "Archive-based",
    "multi_environment":       "Multi-environment",
    "fixed_gravity":           "Fixed gravity",
    "random_variable_gravity": "Random variable",
    "adaptive_progression":    "Adaptive progression",
}
COLORS = {
    "staged_evolution":        "#2166ac",
    "gradual_transition":      "#4393c3",
    "archive_based":           "#92c5de",
    "multi_environment":       "#d1e5f0",
    "fixed_gravity":           "#878787",
    "random_variable_gravity": "#bababa",
    "adaptive_progression":    "#e08214",
}


# ── Figure: Morphology heatmap ───────────────────────────────────────────────
def plot_morphology_heatmap():
    morph_csv = ROOT / "experiments" / "thesis_morph" / "morphology_analysis.csv"
    rows = []
    with open(morph_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    param_names = ["torso_mass_mult", "leg_mass_mult", "motor_gear_mult", "friction_mult"]
    param_labels = ["Torso mass", "Leg mass", "Motor gear", "Foot friction"]

    # Build matrix: arms × params (mean values)
    matrix = np.zeros((len(ARMS), len(param_names)))
    std_matrix = np.zeros((len(ARMS), len(param_names)))
    for i, arm in enumerate(ARMS):
        arm_rows = [r for r in rows if r["arm"] == arm]
        for j, pname in enumerate(param_names):
            vals = [float(r[pname]) for r in arm_rows]
            matrix[i, j] = np.mean(vals)
            std_matrix[i, j] = np.std(vals)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0.5, vmax=2.0)

    # Annotate cells
    for i in range(len(ARMS)):
        for j in range(len(param_names)):
            val = matrix[i, j]
            std = std_matrix[i, j]
            color = "white" if val > 1.5 or val < 0.65 else "black"
            ax.text(j, i, f"{val:.2f}\n±{std:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_labels, fontsize=10)
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([LABELS[a] for a in ARMS], fontsize=10)

    # Tier divider
    ax.axhline(y=3.5, color="black", linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Morphology multiplier")
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="-")
    ax.set_title("Evolved Morphology by Strategy (mean ± std, 10 seeds)", fontsize=12)

    fig.tight_layout()
    out = OUT_DIR / "morphology_heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure: Gravity robustness profiles ──────────────────────────────────────
def plot_gravity_robustness():
    gravities = [-6.0, -7.5, -9.81, -11.0, -12.0]
    grav_labels = ["6.0", "7.5", "9.81", "11.0", "12.0"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for arm in ARMS:
        means = []
        sems = []
        for g in gravities:
            vals = []
            for seed in range(10):
                path = DATA_DIR / arm / f"seed_{seed:02d}" / "gravity_sweep.csv"
                with open(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if abs(float(row["gravity"]) - g) < 0.01:
                            vals.append(float(row["net_progress_m"]))
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))

        ax.plot(range(len(gravities)), means, color=COLORS[arm],
                label=LABELS[arm], linewidth=2, marker="o", markersize=6)
        ax.fill_between(range(len(gravities)),
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.15, color=COLORS[arm])

    # Tier divider annotation
    ax.axvline(x=2, color="green", linestyle=":", alpha=0.4, linewidth=1.5)
    ax.annotate("Earth gravity", xy=(2, ax.get_ylim()[0]),
                xytext=(2.15, 0.5), fontsize=9, fontstyle="italic", color="green")

    ax.set_xticks(range(len(gravities)))
    ax.set_xticklabels([f"−{g}" for g in grav_labels], fontsize=10)
    ax.set_xlabel("Gravitational acceleration (m/s²)", fontsize=11)
    ax.set_ylabel("Net forward progress (m)", fontsize=11)
    ax.set_title("Gravity Robustness Profiles (mean ± SEM, 10 seeds × 20 episodes)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / "gravity_robustness.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure: Sensitivity tornado ──────────────────────────────────────────────
def plot_sensitivity_tornado():
    csv_path = ROOT / "experiments" / "thesis_morph" / "sensitivity_analysis_full.csv"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Get all unique weights
    weights = []
    seen = set()
    for r in rows:
        w = r["weight"]
        if w not in seen:
            weights.append(w)
            seen.add(w)

    # For each weight, compute the average delta_pct across all arms at 0.5x and 1.5x
    weight_effects = {}
    for w in weights:
        low_deltas = [float(r["delta_pct"]) for r in rows if r["weight"] == w and float(r["scale"]) == 0.5]
        high_deltas = [float(r["delta_pct"]) for r in rows if r["weight"] == w and float(r["scale"]) == 1.5]
        avg_low = np.mean(low_deltas) if low_deltas else 0
        avg_high = np.mean(high_deltas) if high_deltas else 0
        weight_effects[w] = (avg_low, avg_high)

    # Sort by total range (most sensitive first)
    sorted_weights = sorted(weights, key=lambda w: abs(weight_effects[w][1] - weight_effects[w][0]))

    # Pretty labels
    wlabels = {
        "W_DISTANCE": "Distance reward",
        "W_NET_PROGRESS": "Net progress reward",
        "W_SPEED_TRACK": "Speed tracking",
        "W_UPRIGHT": "Upright reward",
        "W_BACKWARD": "Backward penalty",
        "W_Z_VELOCITY": "Z-velocity penalty",
        "W_PITCH_RATE": "Pitch rate penalty",
        "W_VEL_CHANGE": "Velocity change penalty",
        "W_JOINT_PENALTY": "Joint limit penalty",
        "W_IMPACT_PENALTY": "Impact penalty",
        "FALL_PENALTY": "Fall penalty",
        "W_LOW_PROGRESS": "Low progress penalty",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(sorted_weights))

    for i, w in enumerate(sorted_weights):
        low, high = weight_effects[w]
        color = "#c0392b" if abs(high - low) > 20 else "#2980b9" if abs(high - low) > 5 else "#95a5a6"
        ax.barh(i, high, height=0.4, color=color, alpha=0.8, label="×1.5" if i == 0 else "")
        ax.barh(i, low, height=0.4, color=color, alpha=0.5, label="×0.5" if i == 0 else "")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([wlabels.get(w, w) for w in sorted_weights], fontsize=10)
    ax.set_xlabel("Mean score change (%)", fontsize=11)
    ax.set_title("Fitness Weight Sensitivity (±50% perturbation, averaged across 7 strategies)", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.25)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c0392b", alpha=0.7, label="Dominant (>20%)"),
        Patch(facecolor="#2980b9", alpha=0.7, label="Moderate (5-20%)"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="Negligible (<5%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "sensitivity_tornado.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_morphology_heatmap()
    plot_gravity_robustness()
    plot_sensitivity_tornado()
    print("Done.")
