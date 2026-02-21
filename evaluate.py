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
# Fitness design (dense, anti-hop, walking-oriented)
# ---------------------------------------------------------------------------
W_DISTANCE          =  55.0    # forward distance reward (metres)
W_NET_PROGRESS      =  90.0    # reward for net forward displacement
W_SPEED_TRACK       =   0.4    # reward for staying near target walking speed
W_UPRIGHT           =   0.20   # torso stays upright
W_ALTERNATION       =   5.0    # valid left↔right stride switches
W_OVERTAKE          =  18.0    # reward true foot-over-foot overtakes (stride quality)
W_FRONT_TIMER_REWARD = 0.15    # reward per frame while lead-foot timer is still positive
W_FRONT_TIMER_PENALTY = 0.30   # penalty per frame once lead-foot timer is negative
W_STEP_LENGTH       = 120.0    # reward for larger forward steps at lead-foot switches
W_SINGLE_SUPPORT    =   0.05   # reward steps with exactly one stance foot
W_TIME_ALIVE        =   0.01   # mild survival reward

W_BACKWARD          =  50.0    # penalise backward travel
W_FLIGHT            =   1.8    # penalise both-feet-airborne steps (hopping signal)
W_SYM_CONTACT       =   1.0    # penalise highly symmetric bilateral loading
W_Z_VELOCITY        =   0.18   # penalise vertical COM velocity (bounce)
W_PITCH_RATE        =   0.08   # penalise torso angular velocity
W_VEL_CHANGE        =   0.80   # penalise rapid frame-to-frame body velocity changes
W_JOINT_PENALTY     =   0.01   # penalty for torques above 0.8 threshold
W_IMPACT_PENALTY    =   0.0015 # penalty for heavy impacts
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
        fitness   = state.get("fitness", 0.0)
        x_pos     = state.get("x_pos", 0.0)
        velocity  = state.get("velocity", 0.0)
        score     = meta.get("score", "?")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)

        gravity   = state.get("gravity", -9.81)
        phase     = state.get("phase", "")

        self.add_overlay(tr, f"Ep {ep + 1}  Fitness", f"{fitness:.1f}")
        self.add_overlay(tr, "Distance",               f"{x_pos:.2f} m")
        self.add_overlay(tr, "Velocity",               f"{velocity:.2f} m/s")
        self.add_overlay(tr, "Gravity",                f"{gravity:.2f} m/s²  {phase}")
        self.add_overlay(tr, "Policy",                 meta.get("policy", "?").upper())
        self.add_overlay(tr, "Checkpoint",             meta.get("label", "?"))
        self.add_overlay(tr, "Train score",            score_str)
        self.add_overlay(tr, "Gen",                    str(meta.get("generation", "?")))

    import types
    viewer._create_overlay = types.MethodType(patched_create_overlay, viewer)


def _update_hud_state(state: dict, ep: int, fitness: float,
                      x_pos: float, velocity: float,
                      gravity: float = -9.81, phase: str = "") -> None:
    """Update the shared state dict that the patched overlay reads."""
    state["ep"]       = ep
    state["fitness"]  = fitness
    state["x_pos"]    = x_pos
    state["velocity"] = velocity
    state["gravity"]  = gravity
    state["phase"]    = phase


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
    alternation_bonus   : valid stride-filtered left↔right switches
    overtake_bonus      : valid foot-lead overtakes with minimum lead separation
    step_length_bonus   : reward for larger forward step lengths
    flight_penalty      : count of both-feet-airborne steps
    sym_contact_penalty : bilateral loading symmetry (hopper signature)
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
            "backward_penalty": -self.backward_distance * W_BACKWARD * self.penalty_scale,
            "flight_penalty": -self.flight_penalty * W_FLIGHT * self.penalty_scale,
            "sym_contact_penalty": -self.sym_contact_penalty * W_SYM_CONTACT * self.penalty_scale,
            "z_velocity_penalty": -self.vertical_velocity_penalty * W_Z_VELOCITY * self.penalty_scale,
            "pitch_rate_penalty": -self.pitch_rate_penalty * W_PITCH_RATE * self.penalty_scale,
            "velocity_change_penalty": -self.velocity_change_penalty * W_VEL_CHANGE * self.penalty_scale,
            "joint_penalty": -self.joint_violations * W_JOINT_PENALTY * self.penalty_scale,
            "impact_penalty": -self.impact_penalty * W_IMPACT_PENALTY * self.penalty_scale,
            "fall_penalty": -fall_pen * self.penalty_scale,
            "low_progress_penalty": -low_progress_gap * W_LOW_PROGRESS * self.penalty_scale,
        }

    def diagnostics(self) -> dict:
        """Return unweighted behavioural stats useful for debugging."""
        samples = max(1, self._contact_samples)
        return {
            "forward_distance_m": self.cumulative_distance,
            "backward_distance_m": self.backward_distance,
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
            "pitch_rate_accum": self.pitch_rate_penalty,
            "velocity_change_accum": self.velocity_change_penalty,
            "joint_violation_accum": self.joint_violations,
            "impact_accum": self.impact_penalty,
            "episode_steps": self.time_alive,
            "fell": float(self.fell),
        }

    def compute(self) -> float:
        """
        Compute the final scalar fitness from accumulated statistics.

        Dense additive objective:
        reward forward walking and valid alternation, penalise hopping cues.
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
# Gravity XML helper
# ---------------------------------------------------------------------------

def _make_gravity_env_xml(gravity: float) -> str:
    """
    Write a Walker2d XML file with a custom gravity value to a temp file.
    Returns the path of the temp file (caller must delete it).

    Strategy: find the stock walker2d_v5.xml bundled with Gymnasium and
    patch the <option> line to replace the default gravity.
    """
    import tempfile, re
    import gymnasium.envs.mujoco as _mj_envs
    import os

    # Locate the bundled XML
    assets_dir = os.path.join(os.path.dirname(_mj_envs.__file__), "assets")
    stock_path = os.path.join(assets_dir, "walker2d_v5.xml")
    with open(stock_path) as f:
        xml = f.read()

    # Replace gravity in the <option ...> tag
    xml = re.sub(
        r'(<option\b[^>]*\bgravity\s*=\s*")[^"]*(")',
        lambda m: f'{m.group(1)}0 0 {gravity:.6f}{m.group(2)}',
        xml,
    )
    # If gravity attr not present in option tag, inject it
    if f"{gravity:.6f}" not in xml:
        xml = re.sub(
            r'(<option\b)',
            f'<option gravity="0 0 {gravity:.6f}" ',
            xml,
            count=1,
        )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, prefix="walker2d_grav_"
    )
    tmp.write(xml)
    tmp.flush()
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

    render_mode = "human" if render else None

    # Build env — if gravity is non-default, inject it via a temp XML
    if abs(gravity - (-9.81)) < 1e-6:
        env = gym.make("Walker2d-v5", render_mode=render_mode)
        _tmp_xml_path = None
    else:
        # Write a minimal gravity-override XML by patching the standard model
        _tmp_xml_path = _make_gravity_env_xml(gravity)
        env = gym.make("Walker2d-v5", render_mode=render_mode,
                       xml_file=_tmp_xml_path)

    policy = make_policy(params, correction_scale=correction_scale)
    hud_meta = meta or {"policy": "cpg", "label": "live",
                        "score": "?", "generation": "?"}

    # Shared mutable state dict read by the patched overlay every frame
    hud_state = {"ep": 0, "fitness": 0.0, "x_pos": 0.0,
                 "velocity": 0.0, "gravity": gravity, "phase": phase}
    hud_patched = False

    total_fitness = 0.0
    diagnostics = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)

        policy.reset()

        tracker = FitnessTracker(max_steps=max_steps, penalty_scale=penalty_scale)

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
                                  tracker.current_velocity,
                                  gravity=gravity,
                                  phase=phase)

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
    """
    args_list = [
        (
            p, n_episodes, max_steps, i * 10, False, None,
            gravity, phase, False, penalty_scale, correction_scale
        )
        for i, p in enumerate(population)
    ]
    if pool is not None:
        return pool.starmap(evaluate_policy, args_list)
    return [evaluate_policy(*a) for a in args_list]
