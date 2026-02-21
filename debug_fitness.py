"""
Fitness diagnostics utility for Walker2d CPG checkpoints.

Usage
-----
python3 debug_fitness.py --checkpoint checkpoints/cpg/best_params.npy
python3 debug_fitness.py --checkpoint a.npy --checkpoint b.npy --episodes 3 --steps 1000
"""

import argparse
import os
import numpy as np

from evaluate import evaluate_policy


def parse_args():
    p = argparse.ArgumentParser(description="Inspect weighted fitness terms for checkpoints.")
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Path to a .npy checkpoint. Repeat to compare multiple.",
    )
    p.add_argument("--episodes", type=int, default=3, help="Episodes per checkpoint.")
    p.add_argument("--steps", type=int, default=1000, help="Max steps per episode.")
    p.add_argument("--seed", type=int, default=0, help="Base random seed.")
    p.add_argument(
        "--penalty-scale",
        type=float,
        default=1.0,
        help="Penalty scale used during evaluation (match training phase if needed).",
    )
    p.add_argument(
        "--correction-scale",
        type=float,
        default=0.5,
        help="Feedback MLP correction scale used during evaluation.",
    )
    return p.parse_args()


def _print_terms(terms: dict) -> None:
    ordered = [
        "distance_reward",
        "net_progress_reward",
        "speed_track_reward",
        "upright_reward",
        "alternation_reward",
        "overtake_reward",
        "front_timer_reward",
        "front_timer_penalty",
        "step_length_reward",
        "single_support_reward",
        "time_alive_reward",
        "backward_penalty",
        "flight_penalty",
        "sym_contact_penalty",
        "z_velocity_penalty",
        "pitch_rate_penalty",
        "velocity_change_penalty",
        "joint_penalty",
        "impact_penalty",
        "fall_penalty",
        "low_progress_penalty",
    ]
    for k in ordered:
        v = terms.get(k, 0.0)
        print(f"    {k:<22} {v:>10.2f}")


def _print_stats(stats: dict) -> None:
    keys = [
        "forward_distance_m",
        "backward_distance_m",
        "net_progress_m",
        "mean_forward_speed_mps",
        "valid_switches",
        "overtake_events",
        "step_length_bonus_m",
        "front_timer_reward_steps",
        "front_timer_penalty_steps",
        "single_support_ratio",
        "double_support_ratio",
        "flight_ratio",
        "episode_steps",
        "fell",
    ]
    for k in keys:
        v = stats.get(k, 0.0)
        print(f"    {k:<22} {v:>10.3f}")


def main():
    args = parse_args()
    for path in args.checkpoint:
        if not os.path.exists(path):
            print(f"\n[missing] {path}")
            continue
        params = np.load(path)
        score, diag = evaluate_policy(
            params=params,
            n_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            return_diagnostics=True,
            penalty_scale=args.penalty_scale,
            correction_scale=args.correction_scale,
        )
        print(f"\n=== {path} ===")
        print(
            f"  mean_score: {score:.2f}  "
            f"(penalty_scale={args.penalty_scale:.2f}, correction_scale={args.correction_scale:.2f})"
        )
        print("  mean_terms:")
        _print_terms(diag["mean_terms"])
        print("  mean_stats:")
        _print_stats(diag["mean_stats"])


if __name__ == "__main__":
    main()
