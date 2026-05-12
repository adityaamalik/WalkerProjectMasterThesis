#!/usr/bin/env python3
"""Render a time-lapse evolution of the walker across generations.

Loads per-generation parameter snapshots (gen_0000.npy, gen_0010.npy, ..., gen_0490.npy)
from a single training run and replays them in a single MuJoCo viewer window. The body
morphology is fixed to the final-generation evolved body, so the viewer sees how the
controller learned to walk on the same body across generations.

Usage:
    python render_timelapse.py --run-dir experiments/thesis_morph/staged_evolution/seed_00
    python render_timelapse.py --run-dir <dir> --strategy-name "Staged Evolution"
    python render_timelapse.py --run-dir <dir> --generations 0,50,100,200,350,490

Snapshots are searched first in <run-dir>/checkpoints/ then in <run-dir>/working_checkpoints/.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import warnings

import gymnasium as gym
import numpy as np

from cpg_policy import CPGPolicy, N_PARAMS as CPG_N_PARAMS
from evaluate import (
    FitnessTracker,
    N_MORPH_PARAMS,
    _hide_viewer_menu,
    _make_custom_env_xml,
    _maximize_viewer_window,
    _patch_hud,
    _position_viewer_window,
    _set_viewer_window_title,
    _setup_tracking_camera,
    _update_hud_state,
    _viewer_should_close,
    decode_morphology,
    make_policy,
)


# Default key generations chosen to show clear progression: random → twitching →
# rhythmic → coordinated → walking
DEFAULT_KEY_GENERATIONS = [0, 50, 100, 200, 350, 490]


def find_snapshot(run_dir: str, generation: int) -> str | None:
    """Locate gen_NNNN.npy under <run_dir>/checkpoints/ or working_checkpoints/."""
    fname = f"gen_{generation:04d}.npy"
    for sub in ("checkpoints", "working_checkpoints"):
        path = os.path.join(run_dir, sub, fname)
        if os.path.exists(path):
            return path
    return None


def split_genome(params: np.ndarray):
    """Split a stored genome into (cpg_params, morph_dict_or_None)."""
    n = len(params)
    if n == CPG_N_PARAMS:
        return params, None
    if n == CPG_N_PARAMS + N_MORPH_PARAMS:
        cpg = params[:CPG_N_PARAMS]
        morph = decode_morphology(params[CPG_N_PARAMS:])
        return cpg, morph
    raise ValueError(
        f"Unexpected genome length {n}; expected {CPG_N_PARAMS} (neural only) "
        f"or {CPG_N_PARAMS + N_MORPH_PARAMS} (neural+morph)."
    )


def run_one_generation(
    env,
    policy: CPGPolicy,
    gen_label: int,
    n_episodes: int,
    max_steps: int,
    seed_base: int,
    hud_meta: dict,
    hud_state: dict,
    gravity: float,
    morph_dict: dict | None,
    show_hud: bool,
) -> bool:
    """Run n_episodes in the given env with the given policy. Returns True if user closed window."""

    # Update HUD label so the on-screen overlay shows the current generation
    hud_meta["label"] = f"gen {gen_label:>4d}"
    hud_meta["generation"] = str(gen_label)
    hud_state["gravity"] = gravity
    hud_state["morph"] = morph_dict
    hud_state["phase"] = f"timelapse — gen {gen_label}"

    user_closed = False
    for ep in range(n_episodes):
        if user_closed:
            break

        obs, _ = env.reset(seed=seed_base + ep)
        policy.reset()
        tracker = FitnessTracker(max_steps=max_steps, penalty_scale=1.0)

        for _ in range(max_steps):
            action = policy(obs)
            obs, _gym_reward, terminated, truncated, info = env.step(action)
            tracker.update(obs, action, info, env, terminated, truncated)

            try:
                env.render()
            except Exception:
                user_closed = True
                break

            # Detect user closing the GLFW window
            try:
                viewer = env.unwrapped.mujoco_renderer.viewer
                if viewer is not None and _viewer_should_close(viewer):
                    user_closed = True
                    break
            except AttributeError:
                pass

            if show_hud:
                _update_hud_state(
                    hud_state, ep, tracker.compute(),
                    tracker.current_x, tracker.current_velocity,
                    gravity=gravity, phase=hud_state["phase"],
                    tracker=tracker,
                )

            if terminated or truncated:
                break

    return user_closed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Path to a seed directory (e.g., experiments/thesis_morph/staged_evolution/seed_00)")
    p.add_argument("--strategy-name", default="",
                   help="Display name shown in window title and HUD")
    p.add_argument("--generations", default=",".join(str(g) for g in DEFAULT_KEY_GENERATIONS),
                   help="Comma-separated generation indices to play in sequence")
    p.add_argument("--episodes-per-gen", type=int, default=1,
                   help="Number of episodes to play at each generation (default 1)")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Maximum steps per episode (default 1000)")
    p.add_argument("--gravity", type=float, default=-9.81,
                   help="Gravity to use throughout the time-lapse (default Earth = -9.81)")
    p.add_argument("--seed", type=int, default=42,
                   help="Episode seed offset (changes the rollout but not the policy)")

    p.add_argument("--fullscreen", action="store_true",
                   help="Open the MuJoCo window covering the full screen.")
    p.add_argument("--no-fullscreen", dest="fullscreen", action="store_false",
                   help="Disable fullscreen mode.")
    p.set_defaults(fullscreen=True)

    p.add_argument("--window-x", type=int, default=None)
    p.add_argument("--window-y", type=int, default=None)
    p.add_argument("--window-width", type=int, default=None)
    p.add_argument("--window-height", type=int, default=None)
    p.add_argument("--window-title", default=None)
    p.add_argument("--no-hud", dest="show_hud", action="store_false",
                   help="Hide the on-screen HUD overlay.")
    p.set_defaults(show_hud=True)
    args = p.parse_args()

    warnings.filterwarnings("ignore", module="glfw")

    def _terminate(_signum, _frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _terminate)

    # ---- Resolve snapshot paths ---------------------------------------------
    requested_gens = [int(g.strip()) for g in args.generations.split(",") if g.strip()]
    snapshots: list[tuple[int, str]] = []
    missing: list[int] = []
    for g in requested_gens:
        path = find_snapshot(args.run_dir, g)
        if path is None:
            missing.append(g)
        else:
            snapshots.append((g, path))

    if missing:
        print(f"[timelapse] Warning: snapshots missing for generations {missing}", file=sys.stderr)
    if not snapshots:
        print(f"[timelapse] No snapshots found under {args.run_dir}; aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"[timelapse] Will play {len(snapshots)} generations: "
          f"{[g for g, _ in snapshots]}")

    # ---- Determine the body to use (final snapshot's morphology) ------------
    # Holding the body fixed across generations gives the viewer a clean visual
    # of the controller learning to walk on a stable target body.
    final_params = np.load(snapshots[-1][1])
    _, final_morph = split_genome(final_params)

    # ---- Build env once with the chosen body + gravity ----------------------
    need_custom_xml = abs(args.gravity - (-9.81)) > 1e-6 or final_morph is not None
    tmp_xml_path = None
    if not need_custom_xml:
        env = gym.make("Walker2d-v5", render_mode="human")
    else:
        tmp_xml_path = _make_custom_env_xml(args.gravity, final_morph)
        env = gym.make("Walker2d-v5", render_mode="human", xml_file=tmp_xml_path)

    # ---- Determine window-rect override ------------------------------------
    window_rect = None
    if all(v is not None for v in (args.window_x, args.window_y,
                                    args.window_width, args.window_height)):
        window_rect = (args.window_x, args.window_y,
                       args.window_width, args.window_height)

    # ---- HUD setup ---------------------------------------------------------
    hud_meta = {
        "policy": "timelapse",
        "label": "init",
        "score": "?",
        "generation": "?",
        "strategy": args.strategy_name or "Time-lapse",
    }
    hud_state = {
        "ep": 0, "n_episodes": args.episodes_per_gen, "fitness": 0.0,
        "x_pos": 0.0, "velocity": 0.0, "gravity": args.gravity,
        "phase": "starting...", "morph": final_morph,
        "steps": 0, "strides": 0, "fell": False,
        "forward_distance": 0.0, "backward_distance": 0.0,
        "flight_rate": 0.0,
    }

    # We need to render once to make the viewer object exist before we can patch it
    # Trigger a render via a no-op step + reset
    env.reset(seed=args.seed)
    env.render()

    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
        if args.show_hud:
            _patch_hud(viewer, hud_meta, hud_state)
        else:
            _hide_viewer_menu(viewer)
        _setup_tracking_camera(env)
        if window_rect is not None:
            _position_viewer_window(viewer, *window_rect)
        elif args.fullscreen:
            _maximize_viewer_window(viewer)
        title = args.window_title or f"Time-lapse evolution: {args.strategy_name or 'walker'}"
        _set_viewer_window_title(viewer, title)
    except AttributeError:
        # Couldn't get viewer; carry on without window adjustments
        pass

    # ---- Replay each generation in sequence --------------------------------
    user_closed = False
    try:
        for gen, path in snapshots:
            if user_closed:
                break
            params = np.load(path)
            cpg_params, _gen_morph = split_genome(params)
            policy = make_policy(cpg_params, correction_scale=0.5)

            print(f"[timelapse] Playing gen {gen:>4d}  ({path})")

            user_closed = run_one_generation(
                env=env,
                policy=policy,
                gen_label=gen,
                n_episodes=args.episodes_per_gen,
                max_steps=args.max_steps,
                seed_base=args.seed,
                hud_meta=hud_meta,
                hud_state=hud_state,
                gravity=args.gravity,
                morph_dict=final_morph,
                show_hud=args.show_hud,
            )
    except KeyboardInterrupt:
        print("\n[timelapse] Interrupted.")
    finally:
        env.close()
        if tmp_xml_path:
            try:
                os.unlink(tmp_xml_path)
            except OSError:
                pass

    print("[timelapse] Finished.")


if __name__ == "__main__":
    main()
