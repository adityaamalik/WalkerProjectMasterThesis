#!/usr/bin/env python3
"""Morphology convergence analysis across arms and seeds.

For each arm, loads the evolved morphological parameters from all 10 seeds
and analyzes:
  1. Mean ± std of each morphological multiplier per arm
  2. Within-arm convergence (low std = consistent body plan)
  3. Between-arm differences (do different strategies evolve different bodies?)
  4. Correlation between morphology and performance

Outputs:
  - experiments/morphology_analysis.csv     (per-seed raw data)
  - experiments/morphology_summary.txt      (human-readable analysis)
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import decode_morphology, CPG_N_PARAMS, N_MORPH_PARAMS, MORPH_PARAM_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root", default="thesis_morph")
    p.add_argument("--arms", type=str,
                   default="fixed_gravity,random_variable_gravity,gradual_transition,"
                           "staged_evolution,adaptive_progression,archive_based,multi_environment")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output-csv", default="experiments/morphology_analysis.csv")
    p.add_argument("--output-txt", default="experiments/morphology_summary.txt")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    morph_names = list(MORPH_PARAM_NAMES)

    # Collect per-seed data
    rows = []
    arm_morph = {}  # arm -> {param_name: [values across seeds]}
    arm_scores = {}  # arm -> [earth_probe_scores]

    for arm in arms:
        arm_morph[arm] = {name: [] for name in morph_names}
        arm_scores[arm] = []
        for seed in range(args.seeds):
            params_path = f"experiments/{args.exp_root}/{arm}/seed_{seed:02d}/checkpoints/best_params.npy"
            summary_path = f"experiments/{args.exp_root}/{arm}/seed_{seed:02d}/summary.json"
            if not os.path.exists(params_path):
                continue

            params = np.load(params_path)
            if len(params) != CPG_N_PARAMS + N_MORPH_PARAMS:
                continue

            raw_morph = params[CPG_N_PARAMS:]
            decoded = decode_morphology(raw_morph)

            # Load performance
            score = 0
            if os.path.exists(summary_path):
                s = json.load(open(summary_path))
                score = s.get("best_earth_probe_score", 0) or 0

            row = {"arm": arm, "seed": seed, "earth_probe_score": round(score, 2)}
            for name in morph_names:
                row[name] = round(decoded[name], 4)
                row[f"{name}_raw"] = round(float(raw_morph[morph_names.index(name)]), 4)
                arm_morph[arm][name].append(decoded[name])
            arm_scores[arm].append(score)
            rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = ["arm", "seed", "earth_probe_score"]
    for name in morph_names:
        fieldnames.extend([name, f"{name}_raw"])
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {args.output_csv}")

    # Analysis
    lines = []
    lines.append("=" * 90)
    lines.append("MORPHOLOGY CONVERGENCE ANALYSIS")
    lines.append("=" * 90)

    # 1. Per-arm summary
    lines.append("\n1. EVOLVED MORPHOLOGY PER ARM (mean ± std across 10 seeds)")
    lines.append("   Neutral baseline = 1.00x for all parameters")
    lines.append("   Range: (0.5x, 2.0x)")
    lines.append("")
    header = f"  {'Arm':<28s}"
    for name in morph_names:
        short = name.replace("_mult", "").replace("_", " ").title()
        header += f"  {short:>16s}"
    lines.append(header)
    lines.append("  " + "-" * (28 + 18 * len(morph_names)))

    for arm in arms:
        row_str = f"  {arm:<28s}"
        for name in morph_names:
            vals = arm_morph[arm][name]
            if vals:
                row_str += f"  {np.mean(vals):>6.2f} ± {np.std(vals):>5.2f}"
            else:
                row_str += f"  {'N/A':>16s}"
        lines.append(row_str)

    # 2. Within-arm convergence
    lines.append(f"\n\n2. WITHIN-ARM CONVERGENCE (coefficient of variation = std/mean)")
    lines.append("   Low CV (<10%) = seeds converge to similar morphology")
    lines.append("   High CV (>20%) = morphology varies widely across seeds")
    lines.append("")
    header = f"  {'Arm':<28s}"
    for name in morph_names:
        short = name.replace("_mult", "").replace("_", " ").title()
        header += f"  {short:>12s}"
    header += f"  {'Mean CV':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (28 + 14 * len(morph_names) + 12))

    arm_mean_cvs = {}
    for arm in arms:
        row_str = f"  {arm:<28s}"
        cvs = []
        for name in morph_names:
            vals = arm_morph[arm][name]
            if vals and np.mean(vals) > 0:
                cv = np.std(vals) / np.mean(vals) * 100
                cvs.append(cv)
                row_str += f"  {cv:>10.1f}%"
            else:
                row_str += f"  {'N/A':>12s}"
        mean_cv = np.mean(cvs) if cvs else 0
        arm_mean_cvs[arm] = mean_cv
        row_str += f"  {mean_cv:>8.1f}%"
        lines.append(row_str)

    # 3. Between-arm Kruskal-Wallis test
    lines.append(f"\n\n3. BETWEEN-ARM DIFFERENCES (Kruskal-Wallis H-test)")
    lines.append("   Tests whether different curriculum strategies evolve different morphologies")
    lines.append("")
    for name in morph_names:
        groups = [arm_morph[arm][name] for arm in arms if len(arm_morph[arm][name]) >= 3]
        if len(groups) >= 2:
            h_stat, p_val = scipy_stats.kruskal(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            short = name.replace("_mult", "").replace("_", " ").title()
            lines.append(f"  {short:<20s}: H={h_stat:.2f}, p={p_val:.4f} {sig}")

    # 4. Correlation: morphology vs performance
    lines.append(f"\n\n4. MORPHOLOGY-PERFORMANCE CORRELATION (Spearman rank)")
    lines.append("   Pooled across all arms and seeds (n=70)")
    lines.append("")
    all_scores = []
    all_morph = {name: [] for name in morph_names}
    for arm in arms:
        for i, score in enumerate(arm_scores[arm]):
            all_scores.append(score)
            for name in morph_names:
                all_morph[name].append(arm_morph[arm][name][i])

    for name in morph_names:
        rho, p_val = scipy_stats.spearmanr(all_morph[name], all_scores)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        short = name.replace("_mult", "").replace("_", " ").title()
        direction = "higher → better" if rho > 0 else "lower → better"
        lines.append(f"  {short:<20s}: ρ={rho:+.3f}, p={p_val:.4f} {sig}  ({direction})")

    # 5. Key observations
    lines.append(f"\n\n{'=' * 90}")
    lines.append("KEY OBSERVATIONS")
    lines.append("=" * 90)

    # Find most/least converged arms
    most_converged = min(arm_mean_cvs, key=arm_mean_cvs.get)
    least_converged = max(arm_mean_cvs, key=arm_mean_cvs.get)
    lines.append(f"\n  Most converged arm:  {most_converged} (mean CV = {arm_mean_cvs[most_converged]:.1f}%)")
    lines.append(f"  Least converged arm: {least_converged} (mean CV = {arm_mean_cvs[least_converged]:.1f}%)")

    # Overall morphology trend
    lines.append(f"\n  Overall evolved morphology (mean across all 70 runs):")
    for name in morph_names:
        all_vals = all_morph[name]
        short = name.replace("_mult", "").replace("_", " ").title()
        mean_val = np.mean(all_vals)
        direction = "↑ increased" if mean_val > 1.05 else "↓ decreased" if mean_val < 0.95 else "≈ neutral"
        lines.append(f"    {short:<20s}: {mean_val:.3f}x ({direction} from 1.0x baseline)")

    report = "\n".join(lines)
    with open(args.output_txt, "w") as f:
        f.write(report)
    print(f"Wrote summary → {args.output_txt}")
    print("\n" + report)


if __name__ == "__main__":
    main()
