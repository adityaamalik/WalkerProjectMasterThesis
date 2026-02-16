"""
Rollout evaluation helpers for the CPG hybrid policy.

Custom fitness function replaces Gymnasium's built-in reward with a
multi-objective signal designed to produce robust forward locomotion.
"""

import numpy as np
import gymnasium as gym
import mujoco

from cpg_policy import CPGPolicy

# ---------------------------------------------------------------------------
# Fitness weights  — tweak these to change behaviour priority
#
# v4 redesign: exp003 converged to hopping (2 hops, then falls).
# Root cause: speed_bonus rewards any motion >1 m/s; the cheapest CPG solution
# is a ballistic hop (high amplitude, both feet push simultaneously).
# After 2–3 hops, torso rotational momentum accumulates → pitch > 1.0 rad → fall.
#
# Fix: two anti-hopping penalties:
#   1. pitch_rate_penalty — obs[2] is torso pitch angular velocity.
#      A hopper builds up large pitch rate each landing; a walker keeps it near 0.
#   2. airborne_penalty — both feet off the ground simultaneously is a hop.
#      Penalise steps where both foot contact forces are below a floor threshold.
#
# Calibration notes (1000-step episode, 0.008 s DT, target ~1.5 m/s walk):
#   cumulative_distance ~ 12 m              → ×30   = 360  (dominant)
#   speed_bonus         ~ 0.5/step × 1000   → ×2.0  = 1000 (strong)
#   velocity_bonus      ~ 375/ep            → ×0.5  = 188
#   upright_bonus       ~ 900/ep            → ×0.2  = 180
#   pitch_rate_penalty  ~ hopper: |dθ/dt| ~ 3–8 rad/s per step
#                         walker: |dθ/dt| ~ 0.1–0.5 rad/s per step
#                         hopper sum ~ 3000–8000  → ×0.05 gives 150–400 penalty
#   airborne_penalty    ~ hopper: ~300 airborne steps/ep  → ×0.5 gives 150 penalty
#                         walker: ~0 (always one foot on ground)
#   fall_penalty        = 200 (increased — falling is now very costly)
# ---------------------------------------------------------------------------
W_DISTANCE          =   30.0   # cumulative forward distance (metres)
W_UPRIGHT           =    0.2   # bonus for staying upright
W_VELOCITY          =    0.0   # removed in v5 — absorbed into speed_bonus signal
W_TIME_ALIVE        =    0.05  # bonus per timestep survived
W_SPEED_BONUS       =    2.0   # bonus for velocity above MIN_SPEED threshold
W_ALTERNATION       =    1.5   # bonus per foot-contact switch (left↔right alternation)
W_JOINT_PENALTY     =    0.01  # penalty for torques above 0.8 threshold
W_IMPACT_PENALTY    =    0.005 # penalty for foot forces above 500 N threshold
W_PITCH_RATE        =    0.05  # penalty for torso pitch angular velocity (anti-hop)
W_AIRBORNE          =    1.0   # penalty per step both feet are off the ground (increased)
FALL_PENALTY        =  200.0   # flat penalty if episode ends by falling

# Speed bonus threshold: only reward velocity above this (m/s).
MIN_SPEED           =    1.0   # m/s

# Foot contact threshold: force below this means the foot is airborne
FOOT_CONTACT_FLOOR  =   20.0   # Newtons — below this the foot isn't bearing weight

# Timestep used for distance integration (must match MuJoCo dt)
_DT = 0.008

# Body ID of the Walker2d torso — camera will track this body.
_TORSO_BODY_ID = 1


# ---------------------------------------------------------------------------
# Camera + HUD
# ---------------------------------------------------------------------------

def _setup_tracking_camera(env) -> None:
    """Lock the viewer camera onto the torso body."""
    try:
        cam = env.unwrapped.mujoco_renderer.viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = _TORSO_BODY_ID
        cam.distance = 4.0
        cam.elevation = -10
        cam.azimuth = 90
    except AttributeError:
        pass


def _patch_hud(viewer, meta: dict, state: dict) -> None:
    """
    Monkey-patch the viewer's _create_overlay so our TOPRIGHT text is
    injected every frame at the same time as the viewer's own overlays.

    Root cause of flicker: render() calls _create_overlay() then renders
    then calls _overlays.clear(). Any text we write outside that window
    either gets cleared before rendering or renders on alternating frames.

    Fix: wrap _create_overlay so it also writes our TOPRIGHT text every
    call — guaranteeing our text is present in the same dict the renderer
    reads, every single frame, with no clearing race condition.
    """
    original_create_overlay = viewer._create_overlay.__func__

    def patched_create_overlay(self):
        # Run the original (writes TOPLEFT + BOTTOMLEFT)
        original_create_overlay(self)
        # Append our TOPRIGHT text in the same call
        tr = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        ep        = state.get("ep", 0)
        fitness   = state.get("fitness", 0.0)
        x_pos     = state.get("x_pos", 0.0)
        velocity  = state.get("velocity", 0.0)
        score     = meta.get("score", "?")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)

        self.add_overlay(tr, f"Ep {ep + 1}  Fitness", f"{fitness:.1f}")
        self.add_overlay(tr, "Distance",               f"{x_pos:.2f} m")
        self.add_overlay(tr, "Velocity",               f"{velocity:.2f} m/s")
        self.add_overlay(tr, "Policy",                 meta.get("policy", "?").upper())
        self.add_overlay(tr, "Checkpoint",             meta.get("label", "?"))
        self.add_overlay(tr, "Train score",            score_str)
        self.add_overlay(tr, "Gen",                    str(meta.get("generation", "?")))

    import types
    viewer._create_overlay = types.MethodType(patched_create_overlay, viewer)


def _update_hud_state(state: dict, ep: int, fitness: float,
                      x_pos: float, velocity: float) -> None:
    """Update the shared state dict that the patched overlay reads."""
    state["ep"]       = ep
    state["fitness"]  = fitness
    state["x_pos"]    = x_pos
    state["velocity"] = velocity


# ---------------------------------------------------------------------------
# Custom fitness accumulator
# ---------------------------------------------------------------------------

class FitnessTracker:
    """
    Accumulates per-timestep statistics and computes the final fitness score.

    Components
    ----------
    cumulative_distance : integral of positive x-velocity × DT  (metres travelled forward)
                          Replaces max_distance from v1 — no plateau after one lunge.
    upright_bonus       : sum of (1 - |torso_pitch| / (π/2)) over all steps
    velocity_bonus      : sum of smooth forward velocity (clipped to avoid noise)
    time_alive          : number of steps survived
    joint_violations    : sum of squared joint torques exceeding a threshold
    impact_penalty      : sum of large foot contact force magnitudes
    fell                : True if episode ended by health termination (not timeout)
    """

    JOINT_TORQUE_LIMIT = 0.8   # normalised torque above which we penalise
    IMPACT_FORCE_LIMIT = 500.0 # Newtons above which we penalise foot impact
                               # (normal walking: 90–1100 N; only extreme stomping penalised)

    def __init__(self):
        self.cumulative_distance = 0.0   # v2: replaces max_distance
        self.upright_bonus       = 0.0
        self.velocity_bonus      = 0.0
        self.speed_bonus         = 0.0   # v3: excess speed above MIN_SPEED threshold
        self.pitch_rate_penalty  = 0.0   # v4: sum of |torso pitch rate| (anti-hop)
        self.airborne_penalty    = 0.0   # v4: steps where both feet off ground (anti-hop)
        self.alternation_bonus   = 0.0   # v5: count of foot-contact switches (left↔right)
        self.time_alive          = 0
        self.joint_violations    = 0.0
        self.impact_penalty      = 0.0
        self.fell                = False
        self._last_x             = None  # for display property
        self._last_contact       = None  # v5: tracks which foot was last dominant

    def update(self, obs: np.ndarray, action: np.ndarray,
               info: dict, env, terminated: bool, truncated: bool) -> None:
        """Call once per timestep after env.step()."""
        x_pos    = info.get("x_position", 0.0)
        x_vel    = info.get("x_velocity", 0.0)
        # obs[1] = torso pitch angle in radians
        pitch    = obs[1]

        # Cumulative distance: integrate positive forward velocity over time.
        # max(0, v) means backward movement earns nothing (no negative contribution).
        # Clipped at 10 m/s to ignore physics glitches / contact explosions.
        # This keeps growing every forward step — no plateau after a single lunge.
        self._last_x = float(x_pos)
        self.cumulative_distance += max(0.0, min(float(x_vel), 10.0)) * _DT

        # Upright bonus: 1.0 when perfectly upright, 0.0 when at ±90°
        self.upright_bonus += max(0.0, 1.0 - abs(pitch) / (np.pi / 2))

        # Velocity bonus: forward velocity, clipped to avoid rewarding
        # brief lunges. Negative velocity (falling backward) gives 0.
        self.velocity_bonus += max(0.0, min(float(x_vel), 5.0))

        # Speed bonus (v3): only reward velocity above MIN_SPEED threshold.
        # This creates a hard floor: shuffling at 0.3 m/s earns nothing here;
        # striding at 1.5 m/s earns 0.5 m/s of bonus every step.
        # Pushes the optimiser away from the safe shuffle local minimum.
        self.speed_bonus += max(0.0, float(x_vel) - MIN_SPEED)

        # Pitch rate penalty (v4): obs[2] is torso pitch angular velocity (rad/s).
        # A hopper accumulates large pitch rate at each landing (3–8 rad/s).
        # A walker keeps pitch rate near 0. Penalise the absolute value every step.
        pitch_rate = obs[2]
        self.pitch_rate_penalty += abs(float(pitch_rate))

        # Time alive: every step survived counts
        self.time_alive += 1

        # Joint violation penalty: penalise torques that slam against limits
        excess = np.abs(action) - self.JOINT_TORQUE_LIMIT
        self.joint_violations += float(np.sum(np.maximum(0.0, excess) ** 2))

        # Contact forces + impact + airborne penalties
        # cfrc_ext[4] = right foot, cfrc_ext[7] = left foot (body indices)
        try:
            data = env.unwrapped.data
            # cfrc_ext is (nbody, 6) — take the force magnitude (last 3 cols)
            right_foot_force = np.linalg.norm(data.cfrc_ext[4, 3:])
            left_foot_force  = np.linalg.norm(data.cfrc_ext[7, 3:])
            excess_r = max(0.0, right_foot_force - self.IMPACT_FORCE_LIMIT)
            excess_l = max(0.0, left_foot_force  - self.IMPACT_FORCE_LIMIT)
            self.impact_penalty += excess_r + excess_l

            # Airborne penalty (v4, strengthened in v5):
            # Both feet off ground simultaneously = hopping.
            both_airborne = (right_foot_force < FOOT_CONTACT_FLOOR and
                             left_foot_force  < FOOT_CONTACT_FLOOR)
            if both_airborne:
                self.airborne_penalty += 1.0

            # Alternation bonus (v5): reward left↔right foot contact switches.
            # Determines which foot is currently dominant (higher contact force).
            # When the dominant foot changes, that's one step of alternating gait.
            # A symmetric hopper contacts both feet simultaneously — never switches.
            # A walker switches ~2× per gait cycle (~120 times over 1000 steps).
            if not both_airborne:
                current_contact = "right" if right_foot_force >= left_foot_force else "left"
                if self._last_contact is not None and current_contact != self._last_contact:
                    self.alternation_bonus += 1.0
                self._last_contact = current_contact
        except (AttributeError, IndexError):
            pass

        # Fell: episode ended due to health termination (not natural timeout)
        if terminated and not truncated:
            self.fell = True

    def compute(self) -> float:
        """Compute the final scalar fitness from accumulated statistics."""
        fitness = (
            self.cumulative_distance * W_DISTANCE       # metres walked forward (v2)
            + self.upright_bonus     * W_UPRIGHT
            + self.velocity_bonus    * W_VELOCITY       # 0.0 in v5 — kept for compat
            + self.speed_bonus       * W_SPEED_BONUS    # v3: only pays above MIN_SPEED
            + self.alternation_bonus * W_ALTERNATION    # v5: left↔right foot switches
            + self.time_alive        * W_TIME_ALIVE
            - self.pitch_rate_penalty * W_PITCH_RATE    # v4: anti-hop (pitch stability)
            - self.airborne_penalty  * W_AIRBORNE       # v4/v5: anti-hop (both feet off)
            - self.joint_violations  * W_JOINT_PENALTY
            - self.impact_penalty    * W_IMPACT_PENALTY
            - (FALL_PENALTY if self.fell else 0.0)
        )
        return float(fitness)

    @property
    def current_x(self) -> float:
        """Current x-position for HUD display."""
        return self._last_x if self._last_x is not None else 0.0

    @property
    def current_velocity(self) -> float:
        """Average forward velocity over the episode so far (m/s)."""
        if self.time_alive == 0:
            return 0.0
        # cumulative_distance / elapsed_time gives mean speed
        elapsed = self.time_alive * _DT
        return self.cumulative_distance / elapsed


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def make_policy(params: np.ndarray) -> CPGPolicy:
    return CPGPolicy.from_flat_params(params)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_policy(
    params: np.ndarray,
    n_episodes: int = 3,
    max_steps: int = 1000,
    seed: int = 0,
    render: bool = False,
    meta: dict | None = None,
) -> float:
    """
    Run n_episodes rollouts and return the mean fitness across episodes.

    Uses the custom FitnessTracker instead of Gymnasium's built-in reward.
    The environment is still created with the default reward for compatibility
    (we simply ignore env's reward and compute our own).
    """
    render_mode = "human" if render else None
    # Use default gymnasium env — we ignore its reward and compute our own
    env = gym.make("Walker2d-v5", render_mode=render_mode)

    policy = make_policy(params)
    hud_meta = meta or {"policy": "cpg", "label": "live",
                        "score": "?", "generation": "?"}

    # Shared mutable state dict read by the patched overlay every frame
    hud_state = {"ep": 0, "fitness": 0.0, "x_pos": 0.0, "velocity": 0.0}
    hud_patched = False

    total_fitness = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)

        policy.reset()

        tracker = FitnessTracker()

        for _ in range(max_steps):
            action = policy(obs)
            obs, _gym_reward, terminated, truncated, info = env.step(action)
            tracker.update(obs, action, info, env, terminated, truncated)

            if render:
                env.render()
                # Patch once — after first render so viewer object exists
                if not hud_patched:
                    try:
                        viewer = env.unwrapped.mujoco_renderer.viewer
                        _patch_hud(viewer, hud_meta, hud_state)
                        _setup_tracking_camera(env)
                        hud_patched = True
                    except AttributeError:
                        pass
                # Update state dict — patch reads this on next frame
                _update_hud_state(hud_state, ep,
                                  tracker.compute(),
                                  tracker.current_x,
                                  tracker.current_velocity)

            if terminated or truncated:
                break

        total_fitness += tracker.compute()

    env.close()
    return total_fitness / n_episodes


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def evaluate_parallel(
    population: list,
    n_episodes: int = 3,
    max_steps: int = 1000,
    pool=None,
) -> list:
    """
    Evaluate every individual in the population.
    Pass a persistent multiprocessing.Pool for best performance.
    """
    args_list = [
        (p, n_episodes, max_steps, i * 10)
        for i, p in enumerate(population)
    ]
    if pool is not None:
        return pool.starmap(evaluate_policy, args_list)
    return [evaluate_policy(*a) for a in args_list]
