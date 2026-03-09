#!/usr/bin/env python3
"""Render a trained agent in the MuJoCo viewer."""
import argparse
import numpy as np
from evaluate import evaluate_policy


def main():
    p = argparse.ArgumentParser(description="Visualize a trained Walker2D agent.")
    p.add_argument("params", help="Path to .npy params file")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--gravity", type=float, default=-9.81)
    args = p.parse_args()

    params = np.load(args.params)
    result = evaluate_policy(params, render=True, n_episodes=args.episodes, gravity=args.gravity)
    print(f"Score: {result['score']:.1f}, Progress: {result['net_progress_m']:.2f}m")


if __name__ == "__main__":
    main()
