#!/usr/bin/env python3
"""Fitness weight sensitivity analysis.

Takes a trained agent and evaluates it under perturbed fitness weights to show
that the final ranking of strategies is robust to moderate weight changes.

Produces a CSV and summary showing how each weight perturbation affects the
fitness decomposition — proving the weights are not fragile magic constants.
"""

import argparse
import csv
import json
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import (
    evaluate_policy, FitnessTracker,
    W_DISTANCE, W_NET_PROGRESS, W_SPEED_TRACK, W_UPRIGHT, W_TIME_ALIVE,
    W_BACKWARD, W_Z_VELOCITY, W_PITCH_RATE, W_VEL_CHANGE,
    W_JOINT_PENALTY, W_IMPACT_PENALTY, FALL_PENALTY, W_LOW_PROGRESS,
)

# The active weights and their baseline values
ACTIVE_WEIGHTS = {
    "W_DISTANCE":     W_DISTANCE,
    "W_NET_PROGRESS": W_NET_PROGRESS,
    "W_SPEED_TRACK":  W_SPEED_TRACK,
    "W_UPRIGHT":      W_UPRIGHT,
    "W_BACKWARD":     W_BACKWARD,
    "W_Z_VELOCITY":   W_Z_VELOCITY,
    "W_PITCH_RATE":   W_PITCH_RATE,
    "W_VEL_CHANGE":   W_VEL_CHANGE,
    "W_JOINT_PENALTY": W_JOINT_PENALTY,
    "W_IMPACT_PENALTY": W_IMPACT_PENALTY,
    "FALL_PENALTY":   FALL_PENALTY,
    "W_LOW_PROGRESS": W_LOW_PROGRESS,
}


def evaluate_with_weights(params_path, n_episodes=10, seed=0):
    """Evaluate a trained agent and return per-term raw (unweighted) values."""
    params = np.load(params_path)
    score, diag = evaluate_policy(
        params, n_episodes=n_episodes, seed=seed,
        return_diagnostics=True, penalty_scale=1.0, correction_scale=0.5,
    )
    return score, diag["mean_terms"], diag["mean_stats"]


def reweight_score(mean_terms, scale_factors):
    """Recompute fitness score with perturbed weights.

    mean_terms: dict from evaluate_policy (already weighted by baseline).
    scale_factors: dict mapping weight name -> multiplier (e.g. 0.5, 1.5).

    We divide each term by its baseline weight to get the raw value,
    then multiply by (baseline_weight * scale_factor).
    """
    # Map term keys to weight names
    term_to_weight = {
        "distance_reward":       "W_DISTANCE",
        "net_progress_reward":   "W_NET_PROGRESS",
        "speed_track_reward":    "W_SPEED_TRACK",
        "upright_reward":        "W_UPRIGHT",
        "time_alive_reward":     "W_TIME_ALIVE",
        "backward_penalty":      "W_BACKWARD",
        "z_velocity_penalty":    "W_Z_VELOCITY",
        "pitch_rate_penalty":    "W_PITCH_RATE",
        "velocity_change_penalty": "W_VEL_CHANGE",
        "joint_penalty":         "W_JOINT_PENALTY",
        "impact_penalty":        "W_IMPACT_PENALTY",
        "fall_penalty":          "FALL_PENALTY",
        "low_progress_penalty":  "W_LOW_PROGRESS",
    }
    # Terms with zero weight (disabled) — pass through unchanged
    total = 0.0
    for term_key, value in mean_terms.items():
        wname = term_to_weight.get(term_key)
        if wname and wname in scale_factors:
            total += value * scale_factors[wname]
        else:
            total += value
    return total


def main():
    p = argparse.ArgumentParser(description="Fitness weight sensitivity analysis")
    p.add_argument("--arms", type=str,
                   default="fixed_gravity,gradual_transition,staged_evolution",
                   help="Comma-separated arms to analyze")
    p.add_argument("--seed", type=int, default=0,
                   help="Which seed to use (default: 0)")
    p.add_argument("--exp-root", type=str, default="thesis_morph",
                   help="Experiment root directory")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--perturbations", type=str, default="0.5,0.75,1.0,1.25,1.5",
                   help="Comma-separated scale factors to test")
    p.add_argument("--output", type=str, default="experiments/sensitivity_analysis.csv")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    perturbations = [float(x) for x in args.perturbations.split(",")]

    # Step 1: Evaluate each arm once to get raw term values
    print("Step 1: Evaluating agents under baseline weights...")
    arm_data = {}
    for arm in arms:
        params_path = f"experiments/{args.exp_root}/{arm}/seed_{args.seed:02d}/checkpoints/best_params.npy"
        if not os.path.exists(params_path):
            print(f"  SKIP {arm} — {params_path} not found")
            continue
        print(f"  Evaluating {arm}...")
        score, terms, stats = evaluate_with_weights(
            params_path, n_episodes=args.episodes, seed=args.seed
        )
        arm_data[arm] = {
            "baseline_score": score,
            "terms": terms,
            "stats": stats,
        }
        print(f"    baseline score: {score:.1f}, progress: {stats.get('net_progress_m', 0):.2f}m")

    # Step 2: Perturb each weight and recompute scores
    print("\nStep 2: Perturbing weights...")
    rows = []
    weight_names = [w for w in ACTIVE_WEIGHTS.keys() if w != "W_TIME_ALIVE"]

    for arm, data in arm_data.items():
        terms = data["terms"]
        baseline = data["baseline_score"]

        for wname in weight_names:
            for scale in perturbations:
                factors = {w: 1.0 for w in ACTIVE_WEIGHTS}
                factors[wname] = scale
                new_score = reweight_score(terms, factors)
                rows.append({
                    "arm": arm,
                    "weight": wname,
                    "scale": scale,
                    "baseline_score": round(baseline, 2),
                    "perturbed_score": round(new_score, 2),
                    "delta": round(new_score - baseline, 2),
                    "delta_pct": round(100 * (new_score - baseline) / max(1, abs(baseline)), 1),
                })

    # Step 3: Check if rankings change
    print("\nStep 3: Checking ranking stability...")
    ranking_changes = 0
    total_comparisons = 0
    baseline_ranking = sorted(arm_data.keys(), key=lambda a: arm_data[a]["baseline_score"], reverse=True)
    print(f"  Baseline ranking: {' > '.join(baseline_ranking)}")

    for wname in weight_names:
        for scale in perturbations:
            if scale == 1.0:
                continue
            factors = {w: 1.0 for w in ACTIVE_WEIGHTS}
            factors[wname] = scale
            perturbed_scores = {
                arm: reweight_score(arm_data[arm]["terms"], factors)
                for arm in arm_data
            }
            perturbed_ranking = sorted(perturbed_scores.keys(),
                                       key=lambda a: perturbed_scores[a], reverse=True)
            total_comparisons += 1
            if perturbed_ranking != baseline_ranking:
                ranking_changes += 1
                print(f"  RANK CHANGE: {wname}×{scale} → {' > '.join(perturbed_ranking)}")

    stability_pct = 100 * (1 - ranking_changes / max(1, total_comparisons))
    print(f"\n  Ranking stability: {stability_pct:.0f}% "
          f"({total_comparisons - ranking_changes}/{total_comparisons} perturbations preserved ranking)")

    # Step 4: Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "arm", "weight", "scale", "baseline_score",
            "perturbed_score", "delta", "delta_pct",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.output}")

    # Step 5: Print summary table
    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY: max |Δ%| per weight when scaled ±50%")
    print("=" * 80)
    print(f"  {'Weight':<22s} {'Max |Δ%|':>10s}  Interpretation")
    print("-" * 80)
    for wname in weight_names:
        max_delta_pct = max(
            abs(r["delta_pct"]) for r in rows
            if r["weight"] == wname and r["scale"] in (0.5, 1.5)
        )
        if max_delta_pct < 5:
            interp = "negligible — weight is not critical"
        elif max_delta_pct < 15:
            interp = "moderate — weight shapes behavior"
        elif max_delta_pct < 30:
            interp = "significant — key design parameter"
        else:
            interp = "dominant — primary fitness driver"
        print(f"  {wname:<22s} {max_delta_pct:>9.1f}%  {interp}")


if __name__ == "__main__":
    main()
