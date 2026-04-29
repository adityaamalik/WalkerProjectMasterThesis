"""
Rollout evaluation helpers for the CPG hybrid policy.

Custom fitness function replaces Gymnasium's built-in reward with a
multi-objective signal designed to produce robust forward locomotion.
"""

import numpy as np
import gymnasium as gym
import mujoco

from cpg_policy import CPGPolicy, N_PARAMS as CPG_N_PARAMS

# ---------------------------------------------------------------------------
# Fitness design (thesis mode: gravity-agnostic locomotion objective)
# ---------------------------------------------------------------------------
W_DISTANCE          =  55.0    # forward-only distance reward
W_NET_PROGRESS      =  95.0    # reward for net forward displacement
W_SPEED_TRACK       =   0.4    # reward for staying near target speed
W_UPRIGHT           =   0.20   # torso stays upright
W_TIME_ALIVE        =   0.01   # mild survival reward

# Gait-style shaping terms are disabled for thesis runs to preserve generality
W_ALTERNATION       =   0.0
W_OVERTAKE          =   0.0
W_FRONT_TIMER_REWARD = 0.0
W_FRONT_TIMER_PENALTY = 0.0
W_STEP_LENGTH       =   0.0
W_SINGLE_SUPPORT    =   0.0
W_FLIGHT            =   0.0
W_SYM_CONTACT       =   0.0

# Penalties are normalized by episode duration (see weighted_terms()).
W_BACKWARD          =  55.0    # average backward speed penalty
W_Z_VELOCITY        =   3.0    # average vertical velocity magnitude penalty
W_PITCH_RATE        =   2.0    # average pitch-rate magnitude penalty
W_VEL_CHANGE        =   8.0    # average frame-to-frame velocity change penalty
W_JOINT_PENALTY     =   0.5    # average torque-limit violation penalty
W_IMPACT_PENALTY    =   0.006  # average impact penalty
FALL_PENALTY        = 120.0    # episode terminated by unhealthy state
MIN_NET_PROGRESS_M  =   2.0    # desired minimum forward displacement per episode
W_LOW_PROGRESS      =  80.0    # penalty per missing metre below MIN_NET_PROGRESS_M

TARGET_SPEED        =   1.2    # m/s target walking speed
SPEED_TOLERANCE     =   0.45   # Gaussian tolerance around target speed

# Minimum steps a foot must stay dominant before a contact switch counts as a stride.
# Prevents high-frequency vibration/flickering from generating fake alternation points.
# At DT=0.008s: 30 steps = 0.24s minimum contact time per foot.
# A real walking stride at 1.5 m/s has ~0.3s per foot → ~37 steps. 30 is conservative.
# A vibrating hopper switches every 1–3 steps → nearly all switches filtered out.
MIN_CONTACT_STEPS = 30         # steps — minimum stance duration for a valid stride switch
MIN_LEAD_STEPS = 10            # steps — minimum stable lead before an overtake counts
LEAD_DEADBAND_M = 0.02         # metres — ignore near-equal foot positions
MIN_OVERTAKE_SEP_M = 0.08      # metres — minimum lead distance for a valid overtake
TARGET_OVERTAKE_SEP_M = 0.22   # metres — full reward lead distance at switch
MIN_OVERTAKE_FORWARD_VEL = 0.35  # m/s — only reward overtakes during forward locomotion
FRONT_LEAD_TIMEOUT_S = 2.0     # seconds — reset when lead foot changes

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
        n_eps     = state.get("n_episodes", 1)
        fitness   = state.get("fitness", 0.0)
        x_pos     = state.get("x_pos", 0.0)
        velocity  = state.get("velocity", 0.0)
        gravity   = state.get("gravity", -9.81)
        phase     = state.get("phase", "")

        # Real-time gait metrics from tracker
        steps         = state.get("steps", 0)
        strides       = state.get("strides", 0)
        fell          = state.get("fell", False)
        fwd_dist      = state.get("forward_distance", 0.0)
        bwd_dist      = state.get("backward_distance", 0.0)
        net_progress  = fwd_dist - bwd_dist
        flight_rate   = state.get("flight_rate", 0.0)

        # Morphology
        morph         = state.get("morph", None)

        # --- Simulation section ---
        self.add_overlay(tr, "", "--- SIMULATION ---")
        self.add_overlay(tr, f"Episode",   f"{ep + 1} / {n_eps}")
        grav_str = f"{gravity:.2f} m/s\u00b2"
        if phase:
            grav_str += f"  ({phase})"
        self.add_overlay(tr, "Gravity",    grav_str)

        # --- Locomotion section ---
        self.add_overlay(tr, "", "--- LOCOMOTION ---")
        self.add_overlay(tr, "Position",   f"{x_pos:.2f} m")
        self.add_overlay(tr, "Net progress", f"{net_progress:.2f} m")
        self.add_overlay(tr, "Speed",      f"{velocity:.2f} m/s  (target {TARGET_SPEED:.1f})")

        # --- Gait section ---
        self.add_overlay(tr, "", "--- GAIT QUALITY ---")
        self.add_overlay(tr, "Steps alive", str(steps))
        self.add_overlay(tr, "Strides",    str(int(strides)))
        self.add_overlay(tr, "Airborne",   f"{flight_rate:.0%}")
        status = "FALLEN" if fell else "WALKING" if velocity > 0.1 else "STANDING"
        self.add_overlay(tr, "Status",     status)

        # --- Fitness section ---
        self.add_overlay(tr, "", "--- FITNESS ---")
        self.add_overlay(tr, "Live score", f"{fitness:.1f}")

        # --- Morphology section (only when evolved) ---
        if morph is not None:
            self.add_overlay(tr, "", "--- MORPHOLOGY ---")
            self.add_overlay(tr, "Torso mass",  f"{morph.get('torso_mass_mult', 1.0):.2f}x")
            self.add_overlay(tr, "Leg mass",    f"{morph.get('leg_mass_mult', 1.0):.2f}x")
            self.add_overlay(tr, "Motor gear",  f"{morph.get('motor_gear_mult', 1.0):.2f}x")
            self.add_overlay(tr, "Foot grip",   f"{morph.get('friction_mult', 1.0):.2f}x")

    import types
    viewer._create_overlay = types.MethodType(patched_create_overlay, viewer)


def _update_hud_state(state: dict, ep: int, fitness: float,
                      x_pos: float, velocity: float,
                      gravity: float = -9.81, phase: str = "",
                      tracker: "FitnessTracker | None" = None) -> None:
    """Update the shared state dict that the patched overlay reads."""
    state["ep"]       = ep
    state["fitness"]  = fitness
    state["x_pos"]    = x_pos
    state["velocity"] = velocity
    state["gravity"]  = gravity
    state["phase"]    = phase
    if tracker is not None:
        samples = max(1, tracker._contact_samples)
        state["steps"]            = tracker.time_alive
        state["strides"]          = tracker.alternation_bonus
        state["fell"]             = tracker.fell
        state["forward_distance"] = tracker.cumulative_distance
        state["backward_distance"] = tracker.backward_distance
        state["flight_rate"]      = tracker._flight_steps / samples


# ---------------------------------------------------------------------------
# Custom fitness accumulator
# ---------------------------------------------------------------------------

class FitnessTracker:
    """
    Accumulates per-timestep statistics and computes the final fitness score.

    Components
    ----------
    cumulative_distance : integral of positive x-velocity × DT
    backward_distance   : integral of backward x-velocity × DT
    speed_track_bonus   : Gaussian tracking reward centered at TARGET_SPEED
    flight_penalty      : both-feet-airborne count (diagnostic only in thesis mode)
    sym_contact_penalty : bilateral loading symmetry (diagnostic only in thesis mode)
    fell                : True if episode ended by health termination
    """

    JOINT_TORQUE_LIMIT = 0.8   # normalised torque above which we penalise
    IMPACT_FORCE_LIMIT = 500.0 # Newtons above which we penalise foot impact
                               # (normal walking: 90–1100 N; only extreme stomping penalised)

    def __init__(self, max_steps: int = 1000, penalty_scale: float = 1.0):
        self.cumulative_distance = 0.0   # forward-only distance
        self.backward_distance   = 0.0   # backward-only distance
        self.speed_track_bonus   = 0.0   # closeness to target speed
        self.upright_bonus       = 0.0
        self.pitch_rate_penalty  = 0.0
        self.vertical_velocity_penalty = 0.0
        self.flight_penalty      = 0.0
        self.sym_contact_penalty = 0.0
        self.alternation_bonus   = 0.0
        self.overtake_bonus      = 0.0
        self.step_length_bonus   = 0.0
        self.front_timer_reward_steps = 0.0
        self.front_timer_penalty_steps = 0.0
        self.overtake_events     = 0
        self.time_alive          = 0
        self.joint_violations    = 0.0
        self.impact_penalty      = 0.0
        self.velocity_change_penalty = 0.0
        self.fell                = False
        self._last_x             = None  # for display property
        self._last_contact       = None  # tracks which foot was last dominant
        self._contact_steps      = 0     # steps current foot has been dominant (stride filter)
        self._contact_samples    = 0     # steps where contact force data was available
        self._flight_steps       = 0     # both feet airborne
        self._single_support     = 0     # exactly one foot loaded
        self._double_support     = 0     # both feet loaded
        self._right_dominant     = 0     # right foot force >= left
        self._left_dominant      = 0     # left foot force > right
        self._lead_side          = 0     # +1 right ahead, -1 left ahead, 0 unknown/deadband
        self._lead_steps         = 0     # how long current lead_side has been stable
        self._front_timer_side   = 0     # current lead side for timer logic
        self._front_timer_s      = FRONT_LEAD_TIMEOUT_S
        self._last_front_foot_x  = None  # front-foot x at last lead switch (step length)
        self._prev_body_velocity = None  # previous frame torso velocity [x, z]
        self.max_steps           = int(max_steps)
        self.penalty_scale       = float(np.clip(penalty_scale, 0.0, 1.0))

    def update(self, obs: np.ndarray, action: np.ndarray,
               info: dict, env, terminated: bool, truncated: bool) -> None:
        """Call once per timestep after env.step()."""
        x_pos    = info.get("x_position", 0.0)
        x_vel    = info.get("x_velocity", 0.0)
        # obs[1] = torso pitch angle in radians
        pitch    = obs[1]

        # Integrate forward and backward displacement separately.
        # Clipped to suppress occasional physics spikes.
        self._last_x = float(x_pos)
        v = float(np.clip(x_vel, -10.0, 10.0))
        self.cumulative_distance += max(0.0, v) * _DT
        self.backward_distance += max(0.0, -v) * _DT

        # Upright bonus: 1.0 when perfectly upright, 0.0 when at ±90°
        self.upright_bonus += max(0.0, 1.0 - abs(pitch) / (np.pi / 2))

        # Dense walking-speed tracking term: peaks near TARGET_SPEED, decays away.
        speed_err = (v - TARGET_SPEED) / SPEED_TOLERANCE
        self.speed_track_bonus += float(np.exp(-0.5 * speed_err * speed_err))

        # Pitch rate penalty: obs[10] is torso angular velocity (rad/s)
        # obs[2] is a joint angle; using obs[10] aligns with Walker2d-v5 docs.
        pitch_rate = obs[10]
        self.pitch_rate_penalty += abs(float(pitch_rate))
        self.vertical_velocity_penalty += abs(float(obs[9]))
        body_velocity = np.array([float(obs[8]), float(obs[9])], dtype=np.float64)
        if self._prev_body_velocity is not None:
            dv = body_velocity - self._prev_body_velocity
            self.velocity_change_penalty += float(np.linalg.norm(dv))
        self._prev_body_velocity = body_velocity

        # Time alive: every step survived
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

            # Symmetric contact penalty (v6): replaces airborne penalty.
            # Detects hopping via bilateral force symmetry at landing.
            # When both feet are on the ground with similar forces → hopping.
            # Formula: penalty = 1 - |Fr - Fl| / (Fr + Fl + ε)
            #   → 1.0 when perfectly symmetric (hopping landing)
            #   → ~0.0 when one foot dominates (walking stance)
            # Only fires when both feet are actually loaded (both > FLOOR).
            both_loaded = (right_foot_force >= FOOT_CONTACT_FLOOR and
                           left_foot_force  >= FOOT_CONTACT_FLOOR)
            both_airborne = (right_foot_force < FOOT_CONTACT_FLOOR and
                             left_foot_force  < FOOT_CONTACT_FLOOR)
            self._contact_samples += 1
            if both_airborne:
                self._flight_steps += 1
                self.flight_penalty += 1.0
            elif both_loaded:
                self._double_support += 1
            else:
                self._single_support += 1

            if right_foot_force >= left_foot_force:
                self._right_dominant += 1
            else:
                self._left_dominant += 1

            if both_loaded:
                total_force = right_foot_force + left_foot_force + 1e-6
                symmetry = 1.0 - abs(right_foot_force - left_foot_force) / total_force
                self.sym_contact_penalty += symmetry  # 0=asymmetric(walker), 1=symmetric(hopper)

            # Overtake reward:
            # reward when the trailing foot truly passes the lead foot by a
            # meaningful distance while moving forward (discourages catch-up shuffle).
            right_foot_x = float(data.xpos[4, 0])
            left_foot_x = float(data.xpos[7, 0])
            lead_delta = right_foot_x - left_foot_x
            if lead_delta > LEAD_DEADBAND_M:
                lead_side = 1
            elif lead_delta < -LEAD_DEADBAND_M:
                lead_side = -1
            else:
                lead_side = 0

            if lead_side == self._lead_side:
                if lead_side != 0:
                    self._lead_steps += 1
            elif lead_side != 0:
                was_stable = (self._lead_side != 0 and self._lead_steps >= MIN_LEAD_STEPS)
                moving_forward = (v >= MIN_OVERTAKE_FORWARD_VEL)
                lead_sep = abs(lead_delta)
                sep_quality = (lead_sep - MIN_OVERTAKE_SEP_M) / max(
                    1e-6, (TARGET_OVERTAKE_SEP_M - MIN_OVERTAKE_SEP_M)
                )
                sep_quality = float(np.clip(sep_quality, 0.0, 1.0))
                if was_stable and moving_forward and sep_quality > 0.0:
                    self.overtake_bonus += sep_quality
                    self.overtake_events += 1
                self._lead_side = lead_side
                self._lead_steps = 1

            # 2-second lead-foot timer and step-length reward.
            # Reset timer when lead foot changes. Reward frames with timer >= 0,
            # penalise frames with timer < 0.
            if lead_side != 0:
                if lead_side == self._front_timer_side:
                    self._front_timer_s -= _DT
                else:
                    current_front_x = right_foot_x if lead_side == 1 else left_foot_x
                    if self._last_front_foot_x is not None:
                        step_len = current_front_x - self._last_front_foot_x
                        if step_len > 0.0:
                            self.step_length_bonus += step_len
                    self._last_front_foot_x = current_front_x
                    self._front_timer_side = lead_side
                    self._front_timer_s = FRONT_LEAD_TIMEOUT_S

                if self._front_timer_s >= 0.0:
                    self.front_timer_reward_steps += 1.0
                else:
                    self.front_timer_penalty_steps += 1.0

            # Alternation bonus — stride-filtered (v8 fix).
            # Only counts a left↔right switch as a valid stride if the current foot
            # has been dominant for at least MIN_CONTACT_STEPS steps.
            # Kills high-frequency vibration exploit (switches every 1–3 steps)
            # while rewarding genuine strides (switches every 30–150 steps).
            if not both_airborne:
                current_contact = "right" if right_foot_force >= left_foot_force else "left"
                if current_contact == self._last_contact:
                    self._contact_steps += 1
                else:
                    # Foot switched — only count if previous stance was long enough
                    if (self._last_contact is not None and
                            self._contact_steps >= MIN_CONTACT_STEPS):
                        self.alternation_bonus += 1.0
                    self._last_contact = current_contact
                    self._contact_steps = 1
        except (AttributeError, IndexError):
            pass

        # Fell: episode ended due to health termination (not natural timeout)
        if terminated and not truncated:
            self.fell = True

    def weighted_terms(self) -> dict:
        """Return weighted reward/penalty contributions used in final fitness."""
        net_progress = self.cumulative_distance - self.backward_distance
        low_progress_gap = max(0.0, MIN_NET_PROGRESS_M - net_progress)
        steps = max(1, self.time_alive)
        elapsed_s = max(_DT, steps * _DT)

        # Gravity-agnostic normalization: compare average penalties per unit time/step
        # rather than totals that scale with episode length.
        backward_speed = self.backward_distance / elapsed_s
        flight_rate = self.flight_penalty / steps
        sym_contact_rate = self.sym_contact_penalty / steps
        z_velocity_mean = self.vertical_velocity_penalty / steps
        pitch_rate_mean = self.pitch_rate_penalty / steps
        velocity_change_mean = self.velocity_change_penalty / steps
        joint_violation_mean = self.joint_violations / steps
        impact_mean = self.impact_penalty / steps

        if self.fell:
            fall_scale = max(0.0, 1.0 - (self.time_alive / max(1, self.max_steps)))
            fall_pen = FALL_PENALTY * fall_scale
        else:
            fall_pen = 0.0
        return {
            "distance_reward": self.cumulative_distance * W_DISTANCE,
            "net_progress_reward": net_progress * W_NET_PROGRESS,
            "speed_track_reward": self.speed_track_bonus * W_SPEED_TRACK,
            "upright_reward": self.upright_bonus * W_UPRIGHT,
            "alternation_reward": self.alternation_bonus * W_ALTERNATION,
            "overtake_reward": self.overtake_bonus * W_OVERTAKE,
            "front_timer_reward": self.front_timer_reward_steps * W_FRONT_TIMER_REWARD,
            "front_timer_penalty": -self.front_timer_penalty_steps * W_FRONT_TIMER_PENALTY,
            "step_length_reward": self.step_length_bonus * W_STEP_LENGTH,
            "single_support_reward": self._single_support * W_SINGLE_SUPPORT,
            "time_alive_reward": self.time_alive * W_TIME_ALIVE,
            "backward_penalty": -backward_speed * W_BACKWARD * self.penalty_scale,
            "flight_penalty": -flight_rate * W_FLIGHT * self.penalty_scale,
            "sym_contact_penalty": -sym_contact_rate * W_SYM_CONTACT * self.penalty_scale,
            "z_velocity_penalty": -z_velocity_mean * W_Z_VELOCITY * self.penalty_scale,
            "pitch_rate_penalty": -pitch_rate_mean * W_PITCH_RATE * self.penalty_scale,
            "velocity_change_penalty": -velocity_change_mean * W_VEL_CHANGE * self.penalty_scale,
            "joint_penalty": -joint_violation_mean * W_JOINT_PENALTY * self.penalty_scale,
            "impact_penalty": -impact_mean * W_IMPACT_PENALTY * self.penalty_scale,
            "fall_penalty": -fall_pen * self.penalty_scale,
            "low_progress_penalty": -low_progress_gap * W_LOW_PROGRESS * self.penalty_scale,
        }

    def diagnostics(self) -> dict:
        """Return unweighted behavioural stats useful for debugging."""
        samples = max(1, self._contact_samples)
        return {
            "forward_distance_m": self.cumulative_distance,
            "backward_distance_m": self.backward_distance,
            "backward_speed_mps": self.backward_distance / max(_DT, self.time_alive * _DT),
            "net_progress_m": self.cumulative_distance - self.backward_distance,
            "mean_forward_speed_mps": self.current_velocity,
            "speed_track_accum": self.speed_track_bonus,
            "upright_accum": self.upright_bonus,
            "valid_switches": self.alternation_bonus,
            "overtake_bonus": self.overtake_bonus,
            "overtake_events": float(self.overtake_events),
            "step_length_bonus_m": self.step_length_bonus,
            "front_timer_reward_steps": self.front_timer_reward_steps,
            "front_timer_penalty_steps": self.front_timer_penalty_steps,
            "single_support_steps": self._single_support,
            "double_support_steps": self._double_support,
            "flight_steps": self._flight_steps,
            "single_support_ratio": self._single_support / samples,
            "double_support_ratio": self._double_support / samples,
            "flight_ratio": self._flight_steps / samples,
            "symmetry_accum": self.sym_contact_penalty,
            "vertical_velocity_accum": self.vertical_velocity_penalty,
            "mean_abs_z_velocity": self.vertical_velocity_penalty / max(1, self.time_alive),
            "pitch_rate_accum": self.pitch_rate_penalty,
            "mean_abs_pitch_rate": self.pitch_rate_penalty / max(1, self.time_alive),
            "velocity_change_accum": self.velocity_change_penalty,
            "mean_velocity_change": self.velocity_change_penalty / max(1, self.time_alive),
            "joint_violation_accum": self.joint_violations,
            "mean_joint_violation": self.joint_violations / max(1, self.time_alive),
            "impact_accum": self.impact_penalty,
            "mean_impact": self.impact_penalty / max(1, self.time_alive),
            "episode_steps": self.time_alive,
            "fell": float(self.fell),
        }

    def compute(self) -> float:
        """
        Compute the final scalar fitness from accumulated statistics.

        Dense additive objective:
        reward robust forward locomotion and stability across gravities.
        """
        return float(sum(self.weighted_terms().values()))

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

def make_policy(params: np.ndarray, correction_scale: float = 0.5) -> CPGPolicy:
    return CPGPolicy.from_flat_params(params, correction_scale=correction_scale)


# ---------------------------------------------------------------------------
# Morphological decoder
# ---------------------------------------------------------------------------

# Number of morphological parameters evolved alongside the neural controller.
N_MORPH_PARAMS = 4
# Names in the order they appear in the raw parameter vector.
MORPH_PARAM_NAMES = (
    "torso_mass_mult",
    "leg_mass_mult",
    "motor_gear_mult",
    "friction_mult",
)


def decode_morphology(raw_morph_params: np.ndarray) -> dict:
    """Map 4 unbounded CMA-ES logits to safe physical multiplier ranges.

    Uses ``2 ** tanh(x)`` so that:
      - x = 0  → multiplier = 1.0  (neutral / default body)
      - x → +∞ → multiplier → 2.0
      - x → -∞ → multiplier → 0.5

    This ensures the scratch init (all zeros) produces the stock Walker2D
    body, so any performance difference is attributable to evolved morphology.

    Parameters
    ----------
    raw_morph_params : np.ndarray
        Array of shape (4,) with unbounded floats from the CMA-ES vector.

    Returns
    -------
    dict with keys: torso_mass_mult, leg_mass_mult, motor_gear_mult,
    friction_mult — each a float in (0.5, 2.0).
    """
    raw = np.asarray(raw_morph_params, dtype=np.float64).ravel()
    assert len(raw) == N_MORPH_PARAMS, (
        f"Expected {N_MORPH_PARAMS} morph params, got {len(raw)}"
    )
    scaled = np.power(2.0, np.tanh(raw))    # (0.5, 2.0), neutral at 1.0
    return dict(zip(MORPH_PARAM_NAMES, scaled.tolist()))


# ---------------------------------------------------------------------------
# Viewer window helpers (fullscreen + close detection)
# ---------------------------------------------------------------------------

def _maximize_viewer_window(viewer) -> None:
    """Resize the GLFW viewer window to cover the full primary monitor."""
    try:
        import glfw
    except ImportError:
        return
    win = getattr(viewer, "window", None)
    if win is None:
        return
    try:
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height
        glfw.set_window_pos(win, 0, 0)
        glfw.set_window_size(win, width, height)
    except Exception:
        # Fallback: just maximize, keep decorations
        try:
            glfw.maximize_window(win)
        except Exception:
            pass


def _position_viewer_window(viewer, x: int, y: int, width: int, height: int) -> None:
    """Place the GLFW viewer window at (x, y) with the given size."""
    try:
        import glfw
    except ImportError:
        return
    win = getattr(viewer, "window", None)
    if win is None:
        return
    try:
        glfw.set_window_pos(win, int(x), int(y))
        glfw.set_window_size(win, int(width), int(height))
    except Exception:
        pass


def _set_viewer_window_title(viewer, title: str) -> None:
    """Set the GLFW viewer window title."""
    try:
        import glfw
    except ImportError:
        return
    win = getattr(viewer, "window", None)
    if win is None:
        return
    try:
        glfw.set_window_title(win, title)
    except Exception:
        pass


def _hide_viewer_menu(viewer) -> None:
    """Suppress MuJoCo's default on-screen overlay (render-frame info, etc.).

    The Gymnasium MuJoCo viewer draws an info overlay via ``_create_overlay()``
    every frame and exposes a ``_hide_menu`` flag toggled by the 'H' key. We
    set that flag and additionally replace ``_create_overlay`` with a no-op so
    no overlay text is drawn at all.
    """
    try:
        viewer._hide_menu = True
    except Exception:
        pass
    try:
        import types
        viewer._create_overlay = types.MethodType(lambda self: None, viewer)
    except Exception:
        pass


def _viewer_should_close(viewer) -> bool:
    """Return True if the user has closed the GLFW viewer window."""
    try:
        import glfw
        win = getattr(viewer, "window", None)
        if win is None:
            return False
        return bool(glfw.window_should_close(win))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Custom environment XML builder (gravity + optional morphology)
# ---------------------------------------------------------------------------

# Default density in the stock Walker2d-v5 XML (<default> section).
_DEFAULT_DENSITY = 1000

# Geom names grouped by morphological role.
_TORSO_GEOMS = {"torso_geom"}
_LEG_GEOMS = {
    "thigh_geom", "leg_geom", "foot_geom",
    "thigh_left_geom", "leg_left_geom", "foot_left_geom",
}
_FOOT_GEOMS = {"foot_geom", "foot_left_geom"}


def _make_custom_env_xml(
    gravity: float,
    morph_params: dict | None = None,
) -> str:
    """Write a Walker2d XML with custom gravity and optional morphology.

    Uses ``xml.etree.ElementTree`` for robust, structured XML editing
    instead of fragile regex substitution.

    Parameters
    ----------
    gravity : float
        Gravitational acceleration (negative = downward, e.g. -9.81).
    morph_params : dict or None
        If provided, a dict with keys ``torso_mass_mult``,
        ``leg_mass_mult``, ``motor_gear_mult``, ``friction_mult``
        (each a float, typically in [0.5, 2.0]).

    Returns
    -------
    str — path to a temporary XML file.  Caller must ``os.unlink()`` it.
    """
    import os
    import tempfile
    import xml.etree.ElementTree as ET
    import gymnasium.envs.mujoco as _mj_envs

    # ---- Locate and parse the stock XML --------------------------------
    assets_dir = os.path.join(os.path.dirname(_mj_envs.__file__), "assets")
    stock_path = os.path.join(assets_dir, "walker2d_v5.xml")
    tree = ET.parse(stock_path)
    root = tree.getroot()

    # ---- Gravity -------------------------------------------------------
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", f"0 0 {gravity:.6f}")

    # ---- Morphological modifications -----------------------------------
    if morph_params is not None:
        torso_m = morph_params.get("torso_mass_mult", 1.0)
        leg_m = morph_params.get("leg_mass_mult", 1.0)
        gear_m = morph_params.get("motor_gear_mult", 1.0)
        fric_m = morph_params.get("friction_mult", 1.0)

        # -- Mass scaling (via density, since inertiafromgeom="true") ----
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if name in _TORSO_GEOMS:
                base = float(geom.get("density", _DEFAULT_DENSITY))
                geom.set("density", f"{base * torso_m:.6f}")
            elif name in _LEG_GEOMS:
                base = float(geom.get("density", _DEFAULT_DENSITY))
                geom.set("density", f"{base * leg_m:.6f}")

        # -- Motor gear scaling ------------------------------------------
        for motor in root.iter("motor"):
            base_gear = float(motor.get("gear", "100"))
            motor.set("gear", f"{base_gear * gear_m:.4f}")

        # -- Foot friction scaling ---------------------------------------
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if name in _FOOT_GEOMS:
                fric_str = geom.get("friction", "")
                if fric_str:
                    parts = fric_str.split()
                    # Scale the sliding (first) component; keep others
                    parts[0] = f"{float(parts[0]) * fric_m:.6f}"
                    geom.set("friction", " ".join(parts))

    # ---- Write to temp file and return path ----------------------------
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", delete=False, prefix="walker2d_custom_"
    )
    tree.write(tmp, xml_declaration=True, encoding="utf-8")
    tmp.close()
    return tmp.name


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
    gravity: float = -9.81,
    phase: str = "",
    return_diagnostics: bool = False,
    penalty_scale: float = 1.0,
    correction_scale: float = 0.5,
    fullscreen: bool = False,
    window_rect: tuple[int, int, int, int] | None = None,
    window_title: str | None = None,
    show_hud: bool = True,
) -> float | tuple[float, dict]:
    """
    Run n_episodes rollouts and return the mean fitness across episodes.

    Uses the custom FitnessTracker instead of Gymnasium's built-in reward.

    Parameters
    ----------
    gravity : float
        Gravity to use for the simulation (default -9.81, earth).
        For curriculum learning, pass the scheduled gravity value each generation.
        Non-default gravity requires writing a custom XML to a temp file.
    phase : str
        Optional label shown on the HUD (e.g. "warmup", "ramp 42%").
    """
    import os, tempfile

    # ---- Slice params: 504 = neural only, 508 = neural + 4 morph --------
    n = len(params)
    if n == CPG_N_PARAMS:
        cpg_params = params
        morph_dict = None
    elif n == CPG_N_PARAMS + N_MORPH_PARAMS:
        cpg_params = params[:CPG_N_PARAMS]
        raw_morph = params[CPG_N_PARAMS:]
        morph_dict = decode_morphology(raw_morph)
    else:
        raise ValueError(
            f"params length {n} is neither {CPG_N_PARAMS} (neural) "
            f"nor {CPG_N_PARAMS + N_MORPH_PARAMS} (neural+morph)"
        )

    render_mode = "human" if render else None

    # Build env — custom XML needed when gravity is non-default or morphology is set
    need_custom_xml = abs(gravity - (-9.81)) > 1e-6 or morph_dict is not None
    if not need_custom_xml:
        env = gym.make("Walker2d-v5", render_mode=render_mode)
        _tmp_xml_path = None
    else:
        _tmp_xml_path = _make_custom_env_xml(gravity, morph_dict)
        env = gym.make("Walker2d-v5", render_mode=render_mode,
                       xml_file=_tmp_xml_path)

    policy = make_policy(cpg_params, correction_scale=correction_scale)
    hud_meta = meta or {"policy": "cpg", "label": "live",
                        "score": "?", "generation": "?"}

    # Shared mutable state dict read by the patched overlay every frame
    hud_state = {"ep": 0, "n_episodes": n_episodes, "fitness": 0.0,
                 "x_pos": 0.0, "velocity": 0.0, "gravity": gravity,
                 "phase": phase, "morph": morph_dict,
                 "steps": 0, "strides": 0, "fell": False,
                 "forward_distance": 0.0, "backward_distance": 0.0,
                 "flight_rate": 0.0}
    hud_patched = False

    total_fitness = 0.0
    diagnostics = []
    user_closed = False

    for ep in range(n_episodes):
        if user_closed:
            break

        obs, _ = env.reset(seed=seed + ep)

        policy.reset()

        tracker = FitnessTracker(max_steps=max_steps, penalty_scale=penalty_scale)

        for _ in range(max_steps):
            action = policy(obs)
            obs, _gym_reward, terminated, truncated, info = env.step(action)
            tracker.update(obs, action, info, env, terminated, truncated)

            if render:
                try:
                    env.render()
                except Exception:
                    user_closed = True
                    break
                # Patch once — after first render so viewer object exists
                if not hud_patched:
                    try:
                        viewer = env.unwrapped.mujoco_renderer.viewer
                        if show_hud:
                            _patch_hud(viewer, hud_meta, hud_state)
                        else:
                            _hide_viewer_menu(viewer)
                        _setup_tracking_camera(env)
                        if window_rect is not None:
                            _position_viewer_window(viewer, *window_rect)
                        elif fullscreen:
                            _maximize_viewer_window(viewer)
                        if window_title is not None:
                            _set_viewer_window_title(viewer, window_title)
                        hud_patched = True
                    except AttributeError:
                        pass
                # Detect window close (X button) — exit gracefully
                if hud_patched and _viewer_should_close(viewer):
                    user_closed = True
                    break
                # Update state dict — patch reads this on next frame
                if show_hud:
                    _update_hud_state(hud_state, ep,
                                      tracker.compute(),
                                      tracker.current_x,
                                      tracker.current_velocity,
                                      gravity=gravity,
                                      phase=phase,
                                      tracker=tracker)

            if terminated or truncated:
                break

        ep_score = tracker.compute()
        total_fitness += ep_score
        if return_diagnostics:
            diagnostics.append({
                "score": ep_score,
                "terms": tracker.weighted_terms(),
                "stats": tracker.diagnostics(),
            })

    env.close()
    if _tmp_xml_path:
        try:
            os.unlink(_tmp_xml_path)
        except OSError:
            pass
    mean_score = total_fitness / n_episodes
    if not return_diagnostics:
        return mean_score

    # Aggregate means for quick tuning across episodes.
    if diagnostics:
        keys_terms = diagnostics[0]["terms"].keys()
        keys_stats = diagnostics[0]["stats"].keys()
        mean_terms = {
            k: float(np.mean([d["terms"][k] for d in diagnostics])) for k in keys_terms
        }
        mean_stats = {
            k: float(np.mean([d["stats"][k] for d in diagnostics])) for k in keys_stats
        }
    else:
        mean_terms = {}
        mean_stats = {}

    return mean_score, {
        "episodes": diagnostics,
        "mean_terms": mean_terms,
        "mean_stats": mean_stats,
    }


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def evaluate_parallel(
    population: list,
    n_episodes: int = 3,
    max_steps: int = 1000,
    pool=None,
    gravity: float = -9.81,
    phase: str = "",
    penalty_scale: float = 1.0,
    correction_scale: float = 0.5,
    base_seed: int | None = None,
) -> list:
    """
    Evaluate every individual in the population.
    Pass a persistent multiprocessing.Pool for best performance.

    Parameters
    ----------
    gravity : float
        Gravity value passed to every evaluate_policy call this generation.
        Controlled by the curriculum schedule in train_curriculum.py.
    phase : str
        Phase label for HUD / logging (e.g. "warmup", "ramp 42%").
    base_seed : int | None
        Optional deterministic seed base for the whole population evaluation.
        If omitted, preserves legacy behaviour (candidate seeds 0, 10, 20, ...).
    """
    args_list = []
    for i, p in enumerate(population):
        if base_seed is None:
            seed = i * 10
        else:
            # Keep candidate seed streams disjoint within a generation.
            seed = int(base_seed) + (i * 1000)
        args_list.append(
            (
                p, n_episodes, max_steps, seed, False, None,
                gravity, phase, False, penalty_scale, correction_scale
            )
        )
    if pool is not None:
        return pool.starmap(evaluate_policy, args_list)
    return [evaluate_policy(*a) for a in args_list]
