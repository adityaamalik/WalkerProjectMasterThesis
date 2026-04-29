#!/usr/bin/env python3
"""Pairwise statistical comparisons across curriculum arms.

For each pair of arms, runs a Mann-Whitney U test on four key metrics
(10 seeds each), then applies Benjamini-Hochberg FDR correction.

Outputs:
  - experiments/statistical_tests.csv       (full pairwise table)
  - experiments/statistical_summary.txt     (human-readable summary)
"""

import argparse
import csv
import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_arm_metrics(exp_root, arm, n_seeds=10):
    """Load per-seed metrics for one arm."""
    metrics = {
        "best_earth_probe_score": [],
        "net_progress_m": [],
        "fell": [],
        "robustness_mean": [],
    }
    for seed in range(n_seeds):
        path = f"experiments/{exp_root}/{arm}/seed_{seed:02d}/summary.json"
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        metrics["best_earth_probe_score"].append(s.get("best_earth_probe_score", 0) or 0)
        fe = s.get("final_earth", {})
        metrics["net_progress_m"].append(fe.get("net_progress_m", 0))
        metrics["fell"].append(fe.get("fell", 1.0))
        rob = s.get("robustness", {})
        metrics["robustness_mean"].append(rob.get("mean_net_progress_m", 0) or 0)
    return metrics


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj = min(prev, p * n / rank)
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def effect_size_rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root", default="thesis_morph")
    p.add_argument("--arms", type=str,
                   default="fixed_gravity,random_variable_gravity,gradual_transition,"
                           "staged_evolution,adaptive_progression,archive_based,multi_environment")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-csv", default="experiments/statistical_tests.csv")
    p.add_argument("--output-txt", default="experiments/statistical_summary.txt")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    metric_names = ["best_earth_probe_score", "net_progress_m", "fell", "robustness_mean"]
    metric_labels = {
        "best_earth_probe_score": "Earth Probe Score",
        "net_progress_m": "Earth Net Progress (m)",
        "fell": "Fall Rate",
        "robustness_mean": "Robustness Mean Progress (m)",
    }
    # For fall rate, lower is better
    higher_is_better = {
        "best_earth_probe_score": True,
        "net_progress_m": True,
        "fell": False,
        "robustness_mean": True,
    }

    # Load all data
    print("Loading metrics...")
    arm_data = {}
    for arm in arms:
        arm_data[arm] = load_arm_metrics(args.exp_root, arm, args.seeds)
        n = len(arm_data[arm]["net_progress_m"])
        mean_prog = np.mean(arm_data[arm]["net_progress_m"]) if n else 0
        print(f"  {arm}: {n} seeds, mean progress={mean_prog:.2f}m")

    # Pairwise Mann-Whitney U tests
    print("\nRunning pairwise Mann-Whitney U tests...")
    rows = []
    all_p_values = []

    for metric in metric_names:
        for arm_a, arm_b in combinations(arms, 2):
            x = np.array(arm_data[arm_a][metric])
            y = np.array(arm_data[arm_b][metric])
            if len(x) < 3 or len(y) < 3:
                continue
            # Two-sided test
            u_stat, p_value = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
            r_effect = effect_size_rank_biserial(u_stat, len(x), len(y))

            mean_a = float(np.mean(x))
            mean_b = float(np.mean(y))
            median_a = float(np.median(x))
            median_b = float(np.median(y))

            if higher_is_better[metric]:
                winner = arm_a if mean_a > mean_b else arm_b
            else:
                winner = arm_a if mean_a < mean_b else arm_b

            rows.append({
                "metric": metric,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "mean_a": round(mean_a, 3),
                "mean_b": round(mean_b, 3),
                "median_a": round(median_a, 3),
                "median_b": round(median_b, 3),
                "u_statistic": round(u_stat, 1),
                "p_value": p_value,
                "effect_size_r": round(r_effect, 3),
                "winner": winner,
            })
            all_p_values.append(p_value)

    # FDR correction
    adjusted_p = benjamini_hochberg(all_p_values, args.alpha)
    for i, row in enumerate(rows):
        row["p_adjusted"] = adjusted_p[i]
        row["significant"] = adjusted_p[i] < args.alpha

    # Write CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = [
        "metric", "arm_a", "arm_b", "mean_a", "mean_b",
        "median_a", "median_b", "u_statistic", "p_value",
        "p_adjusted", "effect_size_r", "significant", "winner",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["p_value"] = f"{out['p_value']:.6f}"
            out["p_adjusted"] = f"{out['p_adjusted']:.6f}"
            writer.writerow(out)
    print(f"\nWrote {len(rows)} comparisons → {args.output_csv}")

    # Summary report
    lines = []
    lines.append("=" * 90)
    lines.append("STATISTICAL SIGNIFICANCE REPORT")
    lines.append(f"Test: Mann-Whitney U (two-sided), FDR correction: Benjamini-Hochberg (α={args.alpha})")
    lines.append(f"Arms: {len(arms)}, Seeds per arm: {args.seeds}")
    lines.append("=" * 90)

    for metric in metric_names:
        label = metric_labels[metric]
        lines.append(f"\n{'─' * 90}")
        lines.append(f"METRIC: {label}")
        lines.append(f"{'─' * 90}")

        # Arm means sorted
        arm_means = [(arm, float(np.mean(arm_data[arm][metric]))) for arm in arms]
        if higher_is_better[metric]:
            arm_means.sort(key=lambda x: x[1], reverse=True)
        else:
            arm_means.sort(key=lambda x: x[1])
        lines.append(f"\n  Ranking ({'higher' if higher_is_better[metric] else 'lower'} is better):")
        for rank, (arm, mean) in enumerate(arm_means, 1):
            lines.append(f"    {rank}. {arm:<30s} {mean:.3f}")

        # Significant pairs
        sig_pairs = [r for r in rows if r["metric"] == metric and r["significant"]]
        nonsig_pairs = [r for r in rows if r["metric"] == metric and not r["significant"]]

        lines.append(f"\n  Significant differences ({len(sig_pairs)}/{len(sig_pairs) + len(nonsig_pairs)} pairs):")
        if sig_pairs:
            for r in sorted(sig_pairs, key=lambda x: x["p_adjusted"]):
                lines.append(
                    f"    {r['arm_a']} vs {r['arm_b']}: "
                    f"p={r['p_adjusted']:.4f}, r={r['effect_size_r']:.2f}, "
                    f"winner={r['winner']}"
                )
        else:
            lines.append("    (none)")

        lines.append(f"\n  Non-significant differences ({len(nonsig_pairs)} pairs):")
        if nonsig_pairs:
            for r in sorted(nonsig_pairs, key=lambda x: x["p_adjusted"]):
                lines.append(
                    f"    {r['arm_a']} vs {r['arm_b']}: "
                    f"p={r['p_adjusted']:.4f} (n.s.)"
                )
        else:
            lines.append("    (none)")

    # Key findings
    lines.append(f"\n{'=' * 90}")
    lines.append("KEY FINDINGS: Baseline comparisons")
    lines.append("=" * 90)
    baseline = "fixed_gravity"
    for metric in metric_names:
        label = metric_labels[metric]
        lines.append(f"\n  {label}:")
        for r in rows:
            if r["metric"] == metric and baseline in (r["arm_a"], r["arm_b"]):
                other = r["arm_b"] if r["arm_a"] == baseline else r["arm_a"]
                sig_str = f"p={r['p_adjusted']:.4f} {'***' if r['p_adjusted'] < 0.001 else '**' if r['p_adjusted'] < 0.01 else '*' if r['p_adjusted'] < 0.05 else 'n.s.'}"
                lines.append(f"    {baseline} vs {other}: {sig_str}, winner={r['winner']}")

    report = "\n".join(lines)
    with open(args.output_txt, "w") as f:
        f.write(report)
    print(f"Wrote summary → {args.output_txt}")
    print("\n" + report)


if __name__ == "__main__":
    main()
