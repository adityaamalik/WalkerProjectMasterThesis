#!/usr/bin/env python3
"""Generate figures for Results section 4.2 (Overall Performance Comparison).

Produces:
  1. Box plot of final Earth scores across 7 strategies
  2. Multi-panel bar chart of 4 primary metrics with per-seed scatter
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiments" / "thesis_morph"
OUT_DIR = ROOT / "thesis_text" / "thesis_template" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy definitions (display order: upper tier first, then lower) ───────
ARMS = [
    "staged_evolution",
    "gradual_transition",
    "archive_based",
    "multi_environment",
    "fixed_gravity",
    "random_variable_gravity",
    "adaptive_progression",
]
LABELS = {
    "staged_evolution":        "Staged\nevolution",
    "gradual_transition":      "Gradual\ntransition",
    "archive_based":           "Archive-\nbased",
    "multi_environment":       "Multi-\nenvironment",
    "fixed_gravity":           "Fixed gravity\n(baseline)",
    "random_variable_gravity": "Random variable\n(control)",
    "adaptive_progression":    "Adaptive\nprogression",
}
# Colours: curricula = blues/teals, baseline/control = greys, adaptive = orange
COLORS = {
    "staged_evolution":        "#2166ac",
    "gradual_transition":      "#4393c3",
    "archive_based":           "#92c5de",
    "multi_environment":       "#d1e5f0",
    "fixed_gravity":           "#878787",
    "random_variable_gravity": "#bababa",
    "adaptive_progression":    "#e08214",
}

SEEDS = list(range(10))


# ── Load data ────────────────────────────────────────────────────────────────
def load_all():
    data = {}
    for arm in ARMS:
        records = []
        for seed in SEEDS:
            path = DATA_DIR / arm / f"seed_{seed:02d}" / "summary.json"
            with open(path) as f:
                s = json.load(f)
            fe = s["final_earth"]
            rob = s["robustness"]
            records.append({
                "score": fe["score"],
                "net_progress_m": fe["net_progress_m"],
                "fell": fe["fell"],
                "robust_net": rob["mean_net_progress_m"],
            })
        data[arm] = records
    return data


# ── Figure 1: Box plot of final scores ───────────────────────────────────────
def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    box_data = [np.array([r["score"] for r in data[arm]]) for arm in ARMS]
    positions = np.arange(len(ARMS))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                       markersize=6, zorder=5),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="grey",
                        markersize=5, alpha=0.7),
    )

    for patch, arm in zip(bp["boxes"], ARMS):
        patch.set_facecolor(COLORS[arm])
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    # Overlay individual seed points (jittered)
    rng = np.random.default_rng(42)
    for i, arm in enumerate(ARMS):
        scores = np.array([r["score"] for r in data[arm]])
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(positions[i] + jitter, scores, color="black", s=22, alpha=0.5,
                   zorder=4, edgecolors="none")

    # Tier divider
    ax.axvline(x=3.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.5, ax.get_ylim()[1] * 0.02 + 2850, "Upper tier", ha="center",
            fontsize=9, fontstyle="italic", color="#444444")
    ax.text(5.0, ax.get_ylim()[1] * 0.02 + 2850, "Lower tier", ha="center",
            fontsize=9, fontstyle="italic", color="#444444")

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("Final Earth fitness score", fontsize=11)
    ax.set_title("Final Evaluation Scores by Strategy (20 episodes, Earth gravity)", fontsize=12)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(250))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)

    # Legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.5, label="Individual seeds"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    outpath = OUT_DIR / "score_boxplot.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ── Figure 2: Multi-panel bar chart of 4 metrics ────────────────────────────
def plot_metric_panels(data):
    metrics = [
        ("score",          "Final score",              False),
        ("net_progress_m", "Net forward progress (m)", False),
        ("fell",           "Fall rate",                True),   # lower is better
        ("robust_net",     "Robust progress (m)",      False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    positions = np.arange(len(ARMS))
    bar_width = 0.6

    for ax, (key, ylabel, invert) in zip(axes, metrics):
        means = []
        stds = []
        for arm in ARMS:
            vals = np.array([r[key] for r in data[arm]])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        bars = ax.bar(positions, means, bar_width, yerr=stds,
                      color=[COLORS[a] for a in ARMS],
                      edgecolor="black", linewidth=0.6,
                      capsize=3, error_kw=dict(linewidth=1.0, capthick=1.0),
                      alpha=0.85, zorder=3)

        # Overlay individual seeds
        rng = np.random.default_rng(42)
        for i, arm in enumerate(ARMS):
            vals = np.array([r[key] for r in data[arm]])
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(positions[i] + jitter, vals, color="black", s=14, alpha=0.45,
                       zorder=4, edgecolors="none")

        # Tier divider
        ax.axvline(x=3.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

        if invert:
            # For fall rate, add an arrow/note that lower is better
            ax.annotate("← lower is better", xy=(0.98, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8, fontstyle="italic", color="#666666")

    fig.suptitle("Primary Metrics by Strategy (mean ± std, 10 seeds)", fontsize=13, y=1.01)
    fig.tight_layout()
    outpath = OUT_DIR / "metrics_overview.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all()
    plot_boxplot(data)
    plot_metric_panels(data)
    print("Done.")
