"""
Hybrid CPG + Feedback MLP policy for Walker2d-v5.

Architecture
------------

  CPG layer  (18 params)
  ────────────────────────────────────────────────────────
  6 oscillators, one per actuator:
      [right thigh, right leg, right foot,
        left thigh,  left leg,  left foot]

  Each oscillator i has 3 parameters:
      Aᵢ  ∈ (0, 1]   amplitude   (sigmoid-mapped)
      ωᵢ  ∈ (0, 4]   frequency   (softplus-mapped, rad/step)
      φᵢ  ∈ [-π, π]  phase offset (tanh-mapped)

  Internal phase θᵢ is integrated every timestep:
      θᵢ ← θᵢ + ωᵢ · dt

  CPG output for joint i:
      u_cpg[i] = Aᵢ · sin(θᵢ + φᵢ)

  Right/left legs are initialised with φ offset of π so they
  naturally start in anti-phase — a good prior for walking.

  Feedback MLP  (small correction network, ~230 params)
  ────────────────────────────────────────────────────────
  Inputs  : 17 proprioceptive observations
            + 6 CPG outputs  (so the corrector knows what the CPG is doing)
            = 23 inputs total
  Hidden  : 1 layer of 16 neurons (tanh)
  Output  : 6 correction values  (tanh → scaled by correction_scale)

  Final action
  ────────────────────────────────────────────────────────
      action[i] = clip(u_cpg[i] + correction_scale · u_fb[i], -1, 1)

  correction_scale = 0.3  (feedback can adjust ±30% of full torque range)

Parameter count
────────────────────────────────────────────────────────
  CPG          :  6 × 3       =  18
  MLP W1 (23×16):            = 368
  MLP b1 (16)  :             =  16
  MLP W2 (16×6):             =  96
  MLP b2 (6)   :             =   6
  ─────────────────────────────────
  Total        :             = 504   (vs 5702 for pure MLP)
"""

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

N_JOINTS          = 6        # Walker2d actuators
DT                = 0.008    # MuJoCo default timestep for Walker2d (125 Hz)
CORRECTION_SCALE_DEFAULT = 0.5  # default max fraction of torque range the MLP can add
                                # v3: increased from 0.3 — gives MLP more authority to
                                # push legs into a full stride rather than a shuffle
FB_HIDDEN         = 16       # feedback MLP hidden layer width
FB_INPUT          = N_JOINTS + 17   # CPG outputs + proprioception

# ── parameter layout (flat vector) ───────────────────────────────────────────
# [A×6 | ω×6 | φ×6 | W1(23×16) | b1(16) | W2(16×6) | b2(6)]

_CPG_PARAMS   = N_JOINTS * 3                         # 18
_W1_SIZE      = FB_INPUT  * FB_HIDDEN                # 368
_B1_SIZE      = FB_HIDDEN                            # 16
_W2_SIZE      = FB_HIDDEN * N_JOINTS                 # 96
_B2_SIZE      = N_JOINTS                             # 6
N_PARAMS      = _CPG_PARAMS + _W1_SIZE + _B1_SIZE + _W2_SIZE + _B2_SIZE  # 504


# ── helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _softplus(x: np.ndarray) -> np.ndarray:
    """Smooth, always-positive mapping. Output ∈ (0, ∞)."""
    return np.log1p(np.exp(x))


# ── main class ────────────────────────────────────────────────────────────────

class CPGPolicy:
    """
    Hybrid CPG + feedback MLP controller.

    Usage
    -----
    policy = CPGPolicy.from_flat_params(params)
    policy.reset()                      # call at episode start
    action = policy(obs)                # call every timestep
    """

    def __init__(self, correction_scale: float = CORRECTION_SCALE_DEFAULT):
        # CPG parameters (raw, before activation mapping)
        self._raw_A  = np.zeros(N_JOINTS)   # amplitude logits
        self._raw_w  = np.zeros(N_JOINTS)   # frequency logits
        self._raw_phi = np.zeros(N_JOINTS)  # phase offset logits

        # Feedback MLP weights
        self.W1 = np.zeros((FB_INPUT,  FB_HIDDEN))
        self.b1 = np.zeros(FB_HIDDEN)
        self.W2 = np.zeros((FB_HIDDEN, N_JOINTS))
        self.b2 = np.zeros(N_JOINTS)

        # Internal oscillator phase (reset each episode)
        self._theta = np.zeros(N_JOINTS)
        self.correction_scale = float(np.clip(correction_scale, 0.0, 1.0))

    # ── parameter interface ───────────────────────────────────────────────────

    def set_flat_params(self, params: np.ndarray) -> None:
        assert len(params) == N_PARAMS, f"Expected {N_PARAMS} params, got {len(params)}"
        idx = 0

        self._raw_A   = params[idx: idx + N_JOINTS];        idx += N_JOINTS
        self._raw_w   = params[idx: idx + N_JOINTS];        idx += N_JOINTS
        self._raw_phi = params[idx: idx + N_JOINTS];        idx += N_JOINTS

        self.W1 = params[idx: idx + _W1_SIZE].reshape(FB_INPUT,  FB_HIDDEN); idx += _W1_SIZE
        self.b1 = params[idx: idx + _B1_SIZE];                                idx += _B1_SIZE
        self.W2 = params[idx: idx + _W2_SIZE].reshape(FB_HIDDEN, N_JOINTS);  idx += _W2_SIZE
        self.b2 = params[idx: idx + _B2_SIZE]

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([
            self._raw_A, self._raw_w, self._raw_phi,
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
        ])

    @classmethod
    def from_flat_params(
        cls,
        params: np.ndarray,
        correction_scale: float = CORRECTION_SCALE_DEFAULT,
    ) -> "CPGPolicy":
        policy = cls(correction_scale=correction_scale)
        policy.set_flat_params(params)
        return policy

    # ── decoded CPG parameters (read-only properties) ─────────────────────────

    @property
    def amplitude(self) -> np.ndarray:
        """Aᵢ ∈ (0, 1] via sigmoid."""
        return _sigmoid(self._raw_A)

    @property
    def frequency(self) -> np.ndarray:
        """ωᵢ ∈ (0, ~4] rad/step via softplus, capped at 4."""
        return np.minimum(_softplus(self._raw_w), 4.0)

    @property
    def phase_offset(self) -> np.ndarray:
        """φᵢ ∈ [-π, π] via tanh × π."""
        return np.tanh(self._raw_phi) * np.pi

    # ── episode reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset internal oscillator phases.
        Right/left legs start π apart — natural anti-phase walking prior.
        joints: [R-thigh, R-leg, R-foot, L-thigh, L-leg, L-foot]
        """
        self._theta = np.array([0.0, 0.0, 0.0,      # right leg group
                                 np.pi, np.pi, np.pi]) # left leg group — anti-phase

    # ── forward pass ─────────────────────────────────────────────────────────

    def _cpg_step(self) -> np.ndarray:
        """Advance oscillators one timestep and return CPG outputs."""
        self._theta += self.frequency * DT * 10.0   # ×10 keeps freq in human-readable Hz
        self._theta  = (self._theta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        return self.amplitude * np.sin(self._theta + self.phase_offset)

    def _feedback(self, obs: np.ndarray, cpg_out: np.ndarray) -> np.ndarray:
        """Small MLP correction given proprioception + current CPG signal."""
        x = np.concatenate([obs, cpg_out])           # (23,)
        x = np.tanh(x @ self.W1 + self.b1)           # (16,)
        return np.tanh(x @ self.W2 + self.b2)        # (6,)  ∈ [-1, 1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        cpg_out  = self._cpg_step()
        fb_out   = self._feedback(obs, cpg_out)
        action   = cpg_out + self.correction_scale * fb_out
        return np.clip(action, -1.0, 1.0)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def describe(self) -> str:
        joint_names = ["R-thigh", "R-leg", "R-foot", "L-thigh", "L-leg", "L-foot"]
        lines = [
            f"CPGPolicy  |  total params = {N_PARAMS}",
            f"  {'Joint':<10} {'Amplitude':>10} {'Freq(rad/step)':>15} {'Phase(deg)':>12}",
            "  " + "-" * 50,
        ]
        for i, name in enumerate(joint_names):
            lines.append(
                f"  {name:<10} {self.amplitude[i]:>10.3f} "
                f"{self.frequency[i]:>15.4f} "
                f"{np.degrees(self.phase_offset[i]):>12.1f}°"
            )
        lines.append(f"\n  Feedback MLP hidden={FB_HIDDEN}  "
                     f"correction_scale={self.correction_scale}")
        return "\n".join(lines)
