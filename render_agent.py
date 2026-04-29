#!/usr/bin/env python3
"""Render a trained agent in the MuJoCo viewer."""
import argparse
import signal
import sys
import warnings

import numpy as np

from evaluate import evaluate_policy


def main():
    p = argparse.ArgumentParser(description="Visualize a trained Walker2D agent.")
    p.add_argument("params", help="Path to .npy params file")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--gravity", type=float, default=-9.81)
    p.add_argument("--fullscreen", action="store_true",
                   help="Open the MuJoCo window covering the full screen.")
    p.add_argument("--no-fullscreen", dest="fullscreen", action="store_false",
                   help="Disable fullscreen mode.")
    p.set_defaults(fullscreen=True)

    p.add_argument("--window-x", type=int, default=None,
                   help="Place the viewer window at this x position (overrides fullscreen).")
    p.add_argument("--window-y", type=int, default=None,
                   help="Place the viewer window at this y position (overrides fullscreen).")
    p.add_argument("--window-width", type=int, default=None,
                   help="Viewer window width in pixels.")
    p.add_argument("--window-height", type=int, default=None,
                   help="Viewer window height in pixels.")
    p.add_argument("--window-title", type=str, default=None,
                   help="Custom title for the viewer window.")
    p.add_argument("--no-hud", dest="show_hud", action="store_false",
                   help="Hide the on-screen HUD overlay (cleaner for tiled views).")
    p.set_defaults(show_hud=True)

    args = p.parse_args()

    # Suppress benign GLFW shutdown warnings printed when the window is closed
    warnings.filterwarnings("ignore", module="glfw")

    # Translate SIGTERM (used by the GUI Stop button) into a clean exit
    def _terminate(_signum, _frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _terminate)

    # If explicit window geometry was given, use it instead of fullscreen
    window_rect = None
    if all(v is not None for v in (args.window_x, args.window_y,
                                    args.window_width, args.window_height)):
        window_rect = (args.window_x, args.window_y,
                       args.window_width, args.window_height)
        fullscreen = False
    else:
        fullscreen = args.fullscreen

    params = np.load(args.params)
    try:
        mean_score, diag = evaluate_policy(
            params,
            render=True,
            n_episodes=args.episodes,
            gravity=args.gravity,
            fullscreen=fullscreen,
            window_rect=window_rect,
            window_title=args.window_title,
            show_hud=args.show_hud,
            return_diagnostics=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return

    mean_progress = diag["mean_stats"].get("net_progress_m", float("nan"))
    print(f"Mean score: {mean_score:.1f}, Mean progress: {mean_progress:.2f}m "
          f"(over {args.episodes} episodes)")


if __name__ == "__main__":
    main()
