# Walker2D Evolutionary Locomotion — Development Log

**Project:** 2D Biped Walker trained via Evolutionary Algorithms
**Environment:** OpenAI Gymnasium `Walker2d-v5` (MuJoCo physics)
**Platform:** Apple M1, 8 cores, 8 GB RAM, macOS
**Python:** 3.10 | **Key libraries:** gymnasium 1.2.3, mujoco 3.4.0, cma 4.4.2

---

## Purpose of This Document

This log records every design decision, implementation choice, problem encountered, and fix applied throughout the project. It is intended to serve two purposes:

1. A human-readable research diary for the author
2. A complete technical context document for an AI assistant to generate the **Methodology** chapter of a master thesis from

---

## Phase 1 — Project Setup and Environment

### What Walker2d-v5 provides

The Gymnasium `Walker2d-v5` environment is a MuJoCo-based simulation of a planar (2D) biped robot. It provides three things that we build on top of:

**The physical body.** A rigid-body model defined in XML with a torso, two thighs, two shins, and two feet. All joint limits, masses, inertia tensors, friction coefficients, and contact geometry are pre-configured. As of v5 both feet have identical friction (1.9), fixing an asymmetry present in earlier versions.

**The observation vector (17 floats per timestep).** The environment automatically extracts and returns sensor readings at every step:

| Index | Quantity | Unit |
|-------|----------|------|
| 0 | Torso height (z) | metres |
| 1 | Torso pitch angle | radians |
| 2 | Right thigh joint angle | radians |
| 3 | Right shin joint angle | radians |
| 4 | Right foot joint angle | radians |
| 5 | Left thigh joint angle | radians |
| 6 | Left shin joint angle | radians |
| 7 | Left foot joint angle | radians |
| 8 | Torso x-velocity | m/s |
| 9 | Torso z-velocity | m/s |
| 10 | Torso angular velocity | rad/s |
| 11–16 | Angular velocities of the 6 joints | rad/s |

The x-position of the torso is intentionally excluded from observations (the walker must learn a gait that works regardless of absolute position).

**The reward signal.** At each timestep the environment computes:

```
reward = forward_reward_weight × (Δx / dt)
       + healthy_reward
       - ctrl_cost_weight × Σ(action_i²)
```

Default values: `forward_reward_weight=1.0`, `healthy_reward=1.0`, `ctrl_cost_weight=0.001`, `dt=0.008s`.

The walker is considered **healthy** (and the episode continues) as long as torso height ∈ [0.8, 2.0] metres and torso pitch ∈ [-1.0, 1.0] radians. Episodes terminate when unhealthy or after 1000 steps.

**What the environment does NOT provide:** any controller, any learning algorithm, any training loop. Those are entirely our contribution.

### What one timestep means

One call to `env.step()` advances 4 MuJoCo substeps of 0.002 s each = **0.008 s of simulated time**. At 1000 steps per episode this represents **8 seconds of simulated walking**. The action at each step is a 6-dimensional vector of normalised joint torques, one per actuator, clamped to [-1, 1].

### The 6 actuators (action dimensions)

| Index | Joint | Leg |
|-------|-------|-----|
| 0 | Hip (thigh) | Right |
| 1 | Knee (shin) | Right |
| 2 | Ankle (foot) | Right |
| 3 | Hip (thigh) | Left |
| 4 | Knee (shin) | Left |
| 5 | Ankle (foot) | Left |

### Dependencies installed

```
gymnasium[mujoco] >= 1.2.0   # environment + MuJoCo bindings
mujoco >= 3.4.0               # physics engine
cma >= 4.4.0                  # CMA-ES optimiser
numpy >= 1.24.0               # array operations
matplotlib >= 3.7.0           # fitness curve plotting
```

---

## Phase 2 — CMA-ES with a Pure MLP Policy

### Why evolutionary algorithms instead of gradient-based RL

Standard deep RL algorithms (PPO, SAC, TD3) require differentiating through the policy network and accumulating experience in replay buffers. They are sample-inefficient and sensitive to hyperparameters. For a research project focusing on the evolutionary approach, we chose **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), which:

- Requires no gradients — treats the problem as black-box optimisation
- Naturally handles the non-smooth, episodic nature of locomotion fitness
- Has well-understood convergence behaviour in continuous search spaces
- Is implemented in the mature `cma` Python library requiring minimal boilerplate

### What CMA-ES does (algorithm overview)

CMA-ES maintains a multivariate Gaussian distribution N(m, σ²C) over the parameter space, where:
- **m** is the current mean (best estimate of optimal parameters)
- **σ** is the global step size (how widely to sample)
- **C** is the covariance matrix (captures correlations between parameter dimensions)

Each generation proceeds as:
1. **Sample** λ candidate solutions: xᵢ ~ m + σ · N(0, C)
2. **Evaluate** each candidate by running MuJoCo rollouts and computing cumulative reward
3. **Rank** candidates by fitness (cumulative reward)
4. **Update mean** m toward the weighted centroid of the top μ = λ/2 candidates
5. **Adapt C** using two complementary mechanisms:
   - Rank-1 update (cumulative path — remembers recent step directions)
   - Rank-μ update (spread of current top candidates)
6. **Adapt σ** via a conjugate evolution path (shrinks if steps correlate, grows if random)

The covariance adaptation is the key innovation: C learns which directions in weight space lead to fitness improvement, making sampling progressively more efficient. The computational cost is O(n²) per generation where n is the parameter count (eigendecomposition of C).

### The MLP policy (`policy.py`)

The first controller was a standard **Multi-Layer Perceptron** — the simplest feedforward neural network. Architecture:

```
Input layer:   17 neurons  (observation vector)
Hidden layer 1: 64 neurons  (tanh activation)
Hidden layer 2: 64 neurons  (tanh activation)
Output layer:   6 neurons   (tanh activation → joint torques in [-1, 1])
```

**Why tanh activations?** The Walker2d actuators expect normalised torques in [-1, 1]. Tanh naturally clamps outputs to this range without any additional clipping.

**Why this architecture is a feedforward network (MLP):** Information flows strictly forward — input → hidden → output — with no recurrent connections or memory. The network is purely reactive: given the current body state, it outputs torques. It has no notion of time or history.

**Parameter count:**

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| W1: 17→64 | 1088 | 64 | 1152 |
| W2: 64→64 | 4096 | 64 | 4160 |
| W3: 64→6  | 384  | 6  | 390  |
| **Total** | | | **5702** |

The entire network is stored as a **flat parameter vector of 5702 floats**, which is what CMA-ES operates on directly.

**Why MLP is appropriate here:** The input is a flat 17-number vector with no spatial structure. CNN or transformer architectures would add complexity without benefit. The MLP is expressive enough to represent walking policies while being compact enough for CMA-ES to optimise in a reasonable number of generations.

### CMA-ES hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population size λ | 40 | Standard recommendation for n~5000 params; literature uses 40 for Walker2d |
| Initial sigma σ₀ | 0.5 | Wide enough to explore the weight space initially |
| Max generations | 500 | Empirically sufficient for convergence |
| `tolx` | 1e-12 | Disabled (set to library minimum) — Walker2d reward is noisy; default 1e-11 caused premature stopping |
| `tolfun` | 1e-12 | Same rationale |
| `tolstagnation` | 500 | Disabled — stagnation checks triggered false positives on noisy rewards |

**Important:** CMA-ES minimises by convention. We negate the reward (`neg_fitness = -reward`) before passing to `es.tell()`, so CMA-ES minimising -reward is equivalent to maximising reward.

### Expected performance (pure MLP)

With the default settings one can expect:
- Generations 0–30: agent discovers how to stay upright
- Generations 30–100: basic walking motion emerges
- Generations 100–300: gait smooths out, forward velocity increases
- Generations 300–500: refinement; improvements become marginal

---

## Phase 3 — Visualisation Improvements

### Tracking camera

The default MuJoCo viewer uses a fixed camera. Once a trained walker moves forward, it quickly walks out of the camera's view. We configured the camera to track the torso body automatically by setting:

```python
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = 1   # torso is body index 1 in Walker2d
cam.distance = 4.0    # metres
cam.elevation = -10   # degrees below horizon
cam.azimuth = 90      # side-on view
```

This is applied once after the first `env.render()` call (the viewer object doesn't exist until then).

### On-screen HUD overlay

To know which checkpoint is being visualised, we use MuJoCo's `viewer.add_overlay()` to display:
- Policy type (MLP or CPG)
- Checkpoint label (e.g. "all-time best", "gen 0250 snapshot")
- Training score (from the companion `.json` file)
- Generation number
- Live cumulative reward (updates every frame)

### Checkpoint metadata

Every saved `.npy` parameter file now has a companion `.json` with the same stem:

```json
{
  "score": 1262.07,
  "generation": 249,
  "policy": "cpg",
  "label": "all-time best"
}
```

This allows identifying any checkpoint without re-running it.

---

## Phase 4 — Hybrid CPG + Feedback MLP Policy

### Motivation: the parameter count problem

The pure MLP approach has 5702 parameters to optimise. CMA-ES's covariance matrix C is 5702×5702 — a matrix of ~32 million entries. Updating and eigendecomposing this every generation is the primary computational bottleneck and also slows convergence because the search space is very high-dimensional.

Biological locomotion is not learned from scratch on a blank neural substrate. Animals have **Central Pattern Generators (CPGs)** — oscillatory neural circuits in the spinal cord that produce rhythmic motor patterns without requiring sensory feedback on every cycle. Walking, swimming, and running all emerge from CPG activity that is then modulated by sensory feedback.

### What a CPG is

A CPG is a network of coupled oscillators. Each oscillator produces a periodic signal:

```
u_i(t) = A_i · sin(θ_i(t) + φ_i)
```

where:
- **A_i** ∈ (0, 1] — amplitude (how much torque this joint produces)
- **θ_i(t)** — internal phase, integrated over time: θ_i ← θ_i + ω_i · Δt
- **ω_i** — frequency (how fast the oscillator cycles, in rad/step)
- **φ_i** — phase offset (when in the cycle this joint peaks)

For bipedal walking, the key biological insight is that the left and right legs should oscillate **180° out of phase** — when one leg swings forward, the other pushes back. This is encoded directly in the initial phase state.

### The hybrid architecture (`cpg_policy.py`)

Rather than using a pure CPG (which ignores sensor feedback entirely) or a pure MLP (which must discover rhythmic patterns from scratch), we use a **hybrid architecture**:

```
CPG layer  →  u_cpg[i] = A_i · sin(θ_i + φ_i)       (6 values)
                                    +
Feedback MLP  →  u_fb = MLP([obs(17), u_cpg(6)])      (6 values)
                                    ↓
Final action[i] = clip(u_cpg[i] + 0.3 · u_fb[i], -1, 1)
```

The CPG provides a rhythmic baseline torque signal. The feedback MLP observes the current body state plus the CPG's current output and adds a small correction (scaled to 30% of the torque range). This correction handles balance perturbations and asymmetries that a pure oscillator cannot react to.

**Parameter layout (flat vector, 504 total):**

| Component | Params | Description |
|-----------|--------|-------------|
| A_i × 6 | 6 | Amplitude logits (sigmoid-mapped to (0,1]) |
| ω_i × 6 | 6 | Frequency logits (softplus-mapped to (0,4]) |
| φ_i × 6 | 6 | Phase offset logits (tanh-mapped to [-π, π]) |
| W1: 23→16 | 368 | Feedback MLP input weights |
| b1: 16 | 16 | Feedback MLP input biases |
| W2: 16→6 | 96 | Feedback MLP output weights |
| b2: 6 | 6 | Feedback MLP output biases |
| **Total** | **504** | **11.3× fewer than pure MLP** |

**Why sigmoid/softplus/tanh mappings?**
Raw parameters from CMA-ES can take any real value. We apply smooth monotone functions to map them into physically meaningful ranges:
- Sigmoid maps any real → (0, 1]: amplitude is always positive and bounded
- Softplus maps any real → (0, ∞): frequency is always positive (oscillators only go forward)
- Tanh × π maps any real → (-π, π]: phase offset covers the full cycle

**Biological prior — anti-phase initialisation:**
At the start of each episode, the right leg oscillators begin at phase θ = 0 and left leg oscillators at θ = π. This encodes the biological expectation that legs alternate, giving CMA-ES a strong starting prior that significantly accelerates early learning.

**Why 23 inputs to the feedback MLP?**
The feedback MLP receives both the 17 proprioceptive observations and the 6 current CPG outputs. By seeing what the CPG is "planning to do", the corrector can compute a more targeted adjustment rather than having to infer the CPG's state indirectly.

**Why correction_scale = 0.3?**
This limits the feedback MLP's authority to ±30% of the full torque range. This prevents the feedback path from overwhelming the CPG's rhythmic structure (which would degrade it back to an unconstrained MLP) while still allowing meaningful reactive corrections.

### CPG vs MLP: direct comparison

| Property | Pure MLP | Hybrid CPG+MLP |
|----------|----------|----------------|
| Parameter count | 5702 | 504 |
| CMA-ES search space | Very high-dimensional | Low-dimensional |
| Biological plausibility | None | High (CPG matches animal spinal circuits) |
| Convergence speed | Hundreds of generations | Tens of generations |
| Rhythmicity | Must be learned from scratch | Structurally guaranteed |
| Reactivity to perturbations | Full | Partial (30% correction budget) |
| Interpretability | Opaque weights | CPG params directly readable (frequency, amplitude, phase) |

---

## Phase 5 — Parallelisation

### Problem

Sequential evaluation of 40 individuals per generation at 1000 steps each (3 episodes) took approximately **10 seconds per generation** once the policy was trained (because a skilled walker survives all 1000 steps, unlike random policies which fall in ~10 steps). At 500 generations this is ~83 minutes.

### Why GPU acceleration does not apply

A common misconception is that neural network training should use GPU. This does not apply here because:

1. **The network is tiny.** A 23→16→6 MLP does ~500 multiply-adds per forward pass. GPU parallelism is only beneficial for operations involving thousands of concurrent operations (e.g. batching 10,000 images through a 50-layer ResNet).

2. **The bottleneck is physics simulation.** MuJoCo's rigid-body physics engine runs sequentially on CPU — each timestep depends on the previous one. It cannot be parallelised across timesteps.

3. **Apple M1 unified memory.** Even if GPU use were beneficial, MuJoCo cannot use the M1 GPU (Metal backend). The unified memory architecture means there is no bus transfer cost, but the GPU still cannot execute MuJoCo's physics code.

The correct parallelisation strategy is to evaluate **multiple individuals concurrently**, since different individuals' rollouts are completely independent.

### Solution: persistent multiprocessing pool

Python's `multiprocessing` module with the `spawn` start context (required on macOS — `fork` conflicts with MuJoCo's OpenGL context) distributes individuals across CPU cores.

**Critical implementation detail — persistent pool:**
A naive implementation creates a new pool each generation:
```python
# BAD: ~0.3–0.5s overhead per generation from pool creation
with Pool(8) as pool:
    fitness = pool.starmap(evaluate_policy, args)
```

We instead create the pool **once before training** and reuse it every generation:
```python
# GOOD: pool created once, negligible per-generation overhead
with ctx.Pool(N_WORKERS) as pool:
    while not es.stop():
        fitness = evaluate_parallel(..., pool=pool)
```

**Benchmarks (trained CPG policy, pop=40, 3ep×1000steps):**

| Strategy | Time/gen | 500 gens |
|----------|----------|----------|
| Sequential | ~10s | ~83 min |
| Persistent pool (8 workers) | ~1–2s | ~10–17 min |
| Speedup | ~5–8× | — |

**Why not more than 8 workers?** The machine has 8 cores (4 performance + 4 efficiency on M1). Using more workers than physical cores causes context-switching overhead that negates the benefit.

### Plot frequency reduction

`matplotlib.savefig()` was called every 10 generations inside the training loop, adding ~0.5s per call. Reduced to every 25 generations (`PLOT_EVERY = 25`), saving ~2 minutes over a 500-generation run.

---

## Phase 6 — Reward Shaping: Fixing the Standing Still Exploit

### Problem discovered

After 500 generations of training (best score ~1262), behavioural analysis revealed the walker had learned to **exploit the healthy reward**:

```
Steps   0-100:  Δx = +0.427 m   (takes one good step forward)
Steps 100-500:  Δx ≈  0.000 m   (stands completely still)
```

The episode reward at convergence was approximately 1.26 per step, almost entirely from the `healthy_reward = 1.0` component. The walker discovered that:
- Walking is risky: it might fall and lose all future healthy reward
- Standing in a stable pose collects +1.0 reward indefinitely at zero risk
- The forward velocity reward of ~0.26 was not sufficient incentive to walk

This is a well-known problem in reward-shaping for locomotion: the **healthy reward exploit**. It is a form of reward hacking where the agent finds an unintended policy that achieves high reward without fulfilling the intended behaviour.

### Fix applied

Increase `forward_reward_weight` from 1.0 to **2.0**:

```python
env = gym.make("Walker2d-v5", forward_reward_weight=2.0)
```

**Why this works:**
With `forward_reward_weight=2.0`, the reward structure becomes:
```
reward = 2.0 × velocity + 1.0 (healthy) - 0.001 × ctrl_cost
```

Standing still now yields exactly **1.0/step** (healthy reward only). Even modest walking at 0.5 m/s yields `2.0 × 0.5/0.008 + 1.0 = 126/step`. The forward velocity signal now dominates, making it strictly better to walk than to stand. The risk of falling is outweighed by the reward of moving.

**Why not remove the healthy reward entirely?**
Setting `healthy_reward=0.0` would make the reward purely velocity-based, which is harder to learn: the agent receives zero reward for staying upright during early training when it cannot yet walk, which destabilises learning. The healthy reward provides a curriculum signal that first teaches balance, then rewards walking.

**Why not a custom reward wrapper?**
A velocity threshold wrapper (e.g. "only give healthy reward if velocity > 0.1 m/s") would be more principled but adds code complexity and introduces a new hyperparameter. The `forward_reward_weight` fix is a single parameter change to a well-understood existing term.

---

## File Structure Summary

```
walker2d-evo/
├── cpg_policy.py      # Hybrid CPG + feedback MLP (504 params)
│                      # 6 oscillators + 23→16→6 correction network
│
├── evaluate.py        # Rollout evaluation (CPG only)
│                      # - Custom FitnessTracker
│                      # - Tracking camera setup
│                      # - On-screen HUD overlay
│                      # - Pool-aware parallel evaluation
│
├── train.py           # CMA-ES training loop (CPG only)
│                      # - Persistent multiprocessing pool
│                      # - Checkpoint + metadata saving
│                      # - Fitness curve plotting
│
├── checkpoints/
│   └── cpg/           # CPG checkpoints
│       ├── best_params.npy / .json
│       ├── final_params.npy / .json
│       └── gen_XXXX.npy / .json   (periodic snapshots)
│
├── experiments/       # Permanent per-run archives
│   ├── README.md
│   └── expXXX_*/
│       ├── experiment.json
│       ├── fitness_curve.png
│       └── checkpoints/best_params.{npy,json}
│
├── fitness_curve_cpg.png   # Latest training plot (working copy)
├── requirements.txt
├── README.md
└── DEVLOG.md          # This file
```

---

## Key Commands Reference

```bash
# Train CPG hybrid
python3 train.py

# Resume from checkpoint
python3 train.py --checkpoint checkpoints/cpg/best_params.npy

# Warm-start from an experiment checkpoint with tighter sigma
python3 train.py --checkpoint experiments/expXXX_.../checkpoints/best_params.npy --sigma 0.2

# Visualise best checkpoint
python3 train.py --render --checkpoint checkpoints/cpg/best_params.npy

# Tune population size or step size
python3 train.py --pop 60 --sigma 0.3

# Archive results to an experiment folder
python3 train.py --exp exp006_cpg_v6_...

# Inspect a checkpoint's metadata
cat checkpoints/cpg/best_params.json
```

---

## Phase 7 — HUD Flicker Fix

### Problem
After implementing the on-screen overlay (Phase 3), the displayed text flickered rapidly during rendering.

### Root cause
MuJoCo's `viewer.add_overlay(gridpos, text1, text2)` **appends** to internal string buffers — it does not replace them. Calling it every frame (60 fps) caused text to accumulate indefinitely in memory and re-render repeatedly, producing the flicker effect.

### Fix
Clear the `_overlays` dictionary before writing each frame's values:
```python
viewer._overlays.clear()   # wipe previous frame's text
viewer.add_overlay(...)    # write fresh values
```
This is now done in `_add_hud()` in `evaluate.py` on every render call.

### HUD improvements
The overlay now shows 4 live values on the right side of the screen (in addition to checkpoint metadata on the left):
- **Episode number**
- **Current fitness** (custom score, updates every step)
- **Max distance reached** (metres)
- **Average velocity** (m/s)

---

## Phase 8 — Custom Multi-Objective Fitness Function

### Motivation: two failed reward approaches

**Attempt 1 — Gymnasium default reward** (`forward_reward_weight=1.0`)
The default reward is `velocity + 1.0 (healthy) - 0.001 × ctrl_cost`. The walker learned to exploit the healthy reward by finding a stable standing pose and collecting +1.0/step indefinitely. Behavioural analysis confirmed:
```
Steps   0–100:  Δx = +0.427 m  (one step forward)
Steps 100–500:  Δx ≈  0.000 m  (stands completely still)
```
After 500 generations, best score ≈ 1262/ep but the walker did not walk.

**Attempt 2 — Increased forward weight** (`forward_reward_weight=2.0`)
Doubling the forward weight was intended to make walking worth more than standing. This partially worked but introduced a new failure mode: the walker learned to lunge forward with one leg (hop on one foot), since a single large forward displacement gives a big reward spike without the sustained gait cost. The walker lifted one leg and bounced rather than walking.

### The custom fitness function (`FitnessTracker` class in `evaluate.py`)

We replaced the Gymnasium reward entirely with a custom multi-objective fitness function that accumulates statistics over the full episode and computes a scalar at the end:

```
fitness = max_distance      × 100.0    (dominant — ~90% of signal)
        + upright_bonus     × 0.1      (secondary — rewards balanced posture)
        + velocity_bonus    × 0.05     (tertiary — rewards smooth movement)
        + time_alive        × 0.01     (small — rewards survival)
        - joint_violations  × 0.01     (penalty — discourages joint stress)
        - impact_penalty    × 0.005    (penalty — discourages stomping)
        - fall_penalty      × 100.0    (flat — episode ended by falling)
```

### Component definitions

**max_distance** — the furthest x-position reached during the episode (not the final position). Using the maximum rather than the final position prevents penalising a walker that overshoots and comes back, and rewards exploratory forward motion.

**upright_bonus** — accumulated each step as `max(0, 1 - |torso_pitch| / (π/2))`. Equals 1.0 when perfectly upright, 0.5 at 45°, 0.0 at 90° or beyond. Integrated over 1000 steps, a perfectly upright walker accumulates 1000, which at ×0.1 contributes ~100 to fitness.

**velocity_bonus** — accumulated each step as `clip(x_velocity, 0, 5)`. Negative velocity (falling backwards) contributes zero. Clipped at 5 m/s to avoid rewarding brief uncontrolled lunges. Integrated over steps, this rewards sustained smooth walking over short sprints.

**time_alive** — simply the number of steps the episode lasted. At ×0.01 this contributes 0–10 to fitness, providing a small signal to survive longer even when not walking well.

**joint_violations** — sum of `max(0, |action_i| - 0.8)²` over all joints and all steps. Only activates when a joint torque exceeds 80% of its range. At ×0.01 the contribution is negligible (~0–0.2) but provides a tie-breaker between equally good gaits.

**impact_penalty** — uses MuJoCo's `data.cfrc_ext` (external contact forces on each body). We extract foot contact forces (body indices 4 and 7 for right and left feet). Only forces above 500 N are penalised — this threshold was calibrated against observed walking forces (mean ~90–170 N, peak ~1100 N during normal walking). At ×0.005, contribution is -0 to -25.

**fall_penalty** — a flat -100 if the episode terminated due to health failure (torso height or angle out of range), as opposed to natural timeout. This distinguishes a walker that falls at step 100 from one that walks steadily for 1000 steps even if both reach the same x-position.

### Weight calibration

The weights were calibrated by running an instrumented rollout of the previously trained CPG policy and measuring the raw magnitudes of each component:

| Component | Raw magnitude (per episode) | Weight | Contribution |
|-----------|----------------------------|--------|--------------|
| max_distance | ~2 m | 100.0 | ~200 |
| upright_bonus | ~370 | 0.1 | ~37 |
| velocity_bonus | ~300 | 0.05 | ~15 |
| time_alive | ~400 steps | 0.01 | ~4 |
| joint_violations | ~8 | 0.01 | ~-0.08 |
| impact_penalty | ~3700 N excess | 0.005 | ~-18 |
| fall_penalty | 1 (fell) | 100.0 | -100 |

The distance term dominates (~80% of signal for a 2 m walk), upright is meaningful (~15%), and penalties are informative without being catastrophic.

### Why `max_distance` not `final_distance`

Using the maximum x-position reached rather than the final position has two advantages:
1. The walker is not penalised for small backward corrections during a gait cycle
2. Early in training when the walker falls forward, the fall still registers as distance covered — providing a gradient signal toward falling forward rather than backward

### Numerical verification (old trained CPG, new fitness function)
```
max_distance:     2.288 m  × 100.0  =  228.8
upright_bonus:  367.6      × 0.1    =   36.8
velocity_bonus: 301.5      × 0.05   =   15.1
time_alive:     402        × 0.01   =    4.0
joint_violations: 8.3      × 0.01   =   -0.08
impact_penalty: 3704.6     × 0.005  =  -18.5
fall_penalty:   (fell)     × 100.0  = -100.0
TOTAL FITNESS:                         166.1
```

---

## Phase 9 — Fitness Function v2: Cumulative Distance (replacing max_distance)

### Problem diagnosed

After training to generation 449 (best score 307.97) the walker exhibited a clear failure
mode: it would **hop once, land slightly forward, then stop completely**.

Root cause: `max_distance × 100` creates a **plateau effect**.  Once the walker reaches
its furthest x-position (even by a single lunge or fall), `max_distance` stops increasing.
The survival terms (`time_alive × 0.01`, `velocity_bonus × 0.05`) are too small (~15–30)
to outweigh holding the peak position.  The evolutionary signal becomes: *"lunge far, then
stand still forever"*.

### Fix: cumulative_distance

Replaced `max_distance` (one-time peak) with **cumulative forward distance**:

```python
# v1 — plateau effect
self.max_distance = max(self.max_distance, x_pos)

# v2 — keeps growing every forward step
self.cumulative_distance += max(0.0, min(x_vel, 10.0)) * _DT   # DT = 0.008 s
```

This is the integral of clamped forward velocity over time (metres actually walked forward).
Backwards movement contributes 0.  Physics glitch spikes are capped at 10 m/s.

Every step the walker moves forward the fitness increases — there is no longer any
incentive to stand still after an initial lunge.

### Recalibrated weights

The dominant term now grows continuously at ~0.016 m/step at 2 m/s, so a lower weight
(30 vs. 100) still makes it dominant over a full 1000-step episode (~16 m × 30 = 480).
The secondary terms were also scaled up to create stronger pressure for uprightness and
sustained speed:

| Component           | v1 weight | v2 weight | Reason for change                              |
|---------------------|-----------|-----------|------------------------------------------------|
| distance            | 100.0     | 30.0      | Metric changed (cumulative m, not peak m)      |
| upright_bonus       | 0.1       | 0.3       | Stronger pressure to stay upright while walking|
| velocity_bonus      | 0.05      | 0.5       | 10× increase — drives sustained speed          |
| time_alive          | 0.01      | 0.1       | Stronger survival reward                       |
| joint_penalty       | 0.01      | 0.01      | Unchanged                                      |
| impact_penalty      | 0.005     | 0.005     | Unchanged                                      |
| fall_penalty        | 100.0     | 100.0     | Unchanged                                      |

### Expected fitness range (v2, 1000-step episode, 2 m/s target gait)

```
cumulative_distance : 2 m/s × 1000 × 0.008 = 16.0 m  × 30    = 480
upright_bonus       : ~900 steps upright              × 0.3   = 270
velocity_bonus      : ~450 avg vel units              × 0.5   = 225
time_alive          : 1000 steps                      × 0.1   = 100
joint_violations    : ~10 units                       × 0.01  =  -0.1
impact_penalty      : ~0 (below 500 N threshold)      × 0.005 =   0
fall_penalty        : (didn't fall)                   × 100   =   0
TOTAL                                                          ≈ 1075
```

Scores above ~400 indicate sustained forward locomotion; early-generation individuals
standing still will score ~0–100.

### current_velocity HUD fix

`current_velocity` now reports mean speed (`cumulative_distance / elapsed_time`) instead
of the previous proxy (`velocity_bonus / time_alive`), giving a physically meaningful m/s
reading on the HUD.

---

## Phase 10 — Experiment Tracking + Fitness Function v3 (speed bonus)

### Experiment archive structure

All training runs are now stored under `experiments/`:

```
experiments/
  README.md                         ← master experiment log table
  exp001_cpg_v1_maxdist/
    experiment.json                 ← config, weights, result, diagnosis
    fitness_curve.png
  exp002_cpg_v2_cumdist/
    experiment.json
    fitness_curve.png
    checkpoints/best_params.{npy,json}
  exp003_cpg_v3_stride/
    experiment.json
    fitness_curve.png               ← populated after training completes
    checkpoints/best_params.{npy,json}
```

`train.py` now accepts `--exp <id>` and automatically archives the fitness curve
and best checkpoint into `experiments/<id>/` at the end of training.

### exp002 result analysis

exp002 achieved 539.6 (vs predicted ~1075 for a 2 m/s gait).

Observed gait: **shuffle-walk** — places front foot forward, brings rear foot to
same position, repeats. Feet never overtake each other.  Estimated speed: ~0.5 m/s.

Why the gap from prediction:
- At 0.5 m/s: `cumulative_distance = 0.5 × 1000 × 0.008 = 4 m × 30 = 120`
  vs predicted `16 m × 30 = 480` at 2 m/s.
- `speed_bonus` didn't exist in v2 — shuffling and striding earned the same fitness
  per step of forward movement.
- `CORRECTION_SCALE = 0.3` limited the feedback MLP to ±0.3 torque adjustment;
  the CPG alone may not produce enough leg lift for a real stride.

### Fitness function v3 changes

**Added: speed bonus** — only rewards velocity *above* a threshold (1.0 m/s):

```python
self.speed_bonus += max(0.0, float(x_vel) - MIN_SPEED)  # MIN_SPEED = 1.0 m/s
# In compute():
+ self.speed_bonus * W_SPEED_BONUS   # W_SPEED_BONUS = 2.0
```

At 1.5 m/s: `0.5 excess/step × 1000 steps × 2.0 = 1000 pts` — larger than all
other terms combined.  At 0.5 m/s shuffle: `0 pts`.  This makes striding strictly
better than shuffling in the fitness landscape.

**Increased: `CORRECTION_SCALE`** (`cpg_policy.py`): 0.3 → 0.5

Gives the feedback MLP more authority to override the CPG and push legs through
a fuller range of motion.

**Adjusted secondary weights** to prevent upright_bonus dominating over speed:

| Weight | v2 | v3 |
|---|---|---|
| W_SPEED_BONUS | — | **2.0** (new) |
| W_UPRIGHT | 0.3 | 0.2 |
| W_TIME_ALIVE | 0.1 | 0.05 |

### Predicted score for exp003

At 1.5 m/s striding, 1000 steps, no falls:
```
cumulative_distance : 1.5×1000×0.008 = 12 m × 30     = 360
upright_bonus       : ~900             × 0.2           = 180
velocity_bonus      : ~375             × 0.5           = 188
speed_bonus         : 0.5 excess×1000 × 2.0            = 1000
time_alive          : 1000             × 0.05          =  50
penalties           :                                  ≈ -25
TOTAL                                                  ≈ 1753
```

---

## Phase 11 — exp003 Result + Fitness Function v4 (anti-hop penalties)

### exp003 result: 985.43 @ gen 490

The speed bonus (+232% over exp002) successfully broke the shuffle local minimum.
However the optimiser found a new exploit: **ballistic hopping**.

**Observed gait:** 2 successful hops forward, falls on 3rd hop.

**Why hopping?** The `speed_bonus` threshold (1.0 m/s) is satisfied most cheaply
by a ballistic hop: both CPG oscillators fire at high amplitude simultaneously,
launching the body forward. A genuine walking stride requires the legs to alternate
and stay grounded, which is harder to learn.

After 2–3 hops, torso rotational (pitch) momentum accumulates. Walker2d's health
termination fires when `|pitch| > 1.0 rad` — the episode ends with FALL_PENALTY.

**Pattern:** Each experiment so far has found a local optimum:
- v1: stand still (max_distance plateaus)
- v2: shuffle (speed_bonus didn't exist)
- v3: hop (speed_bonus satisfied by ballistic launch)
- v4: target is true alternating-leg walk

### Fitness function v4 — anti-hop penalties

Two new penalty terms directly targeting the hopping exploit:

**1. Pitch rate penalty** (per step)

```python
pitch_rate = obs[2]   # torso pitch angular velocity (rad/s)
self.pitch_rate_penalty += abs(pitch_rate)
# In compute: - self.pitch_rate_penalty * 0.05
```

A hopper builds up large pitch rate at each launch and landing (~3–8 rad/s).
A walker keeps pitch rate near zero (~0.1–0.5 rad/s) throughout the stride cycle.
At 5 rad/s average, 300 hopping steps: `300 × 5 × 0.05 = 75 pts penalty`.

**2. Airborne penalty** (per step, both feet off ground)

```python
both_airborne = (right_foot_force < 20.0 and left_foot_force < 20.0)
if both_airborne:
    self.airborne_penalty += 1.0
# In compute: - self.airborne_penalty * 0.5
```

During a hop, both feet leave the ground simultaneously (flight phase).
During walking, at least one foot is always in contact with the ground.
A hopper with 300 airborne steps loses `300 × 0.5 = 150 pts`.

**3. Increased FALL_PENALTY: 100 → 200**

Falling was previously an acceptable outcome (100 pts, small vs ~1000 speed_bonus).
At 200, falling costs more than an entire episode's time_alive bonus.

**4. Warm-start σ=0.2 from exp003 best_params**

Rather than starting fresh (σ=0.5), CMA-ES is initialised at exp003's best solution
with a tighter σ. This refines the existing (good-but-unstable) motion rather than
searching from scratch, saving ~100–200 generations of convergence time.

```bash
python3 train.py --policy cpg \
    --checkpoint experiments/exp003_cpg_v3_stride/checkpoints/best_params.npy \
    --sigma 0.2 \
    --exp exp004_cpg_v4_antihop
```

### Expected penalty budget (hopping vs walking, 1000 steps)

| Behaviour | pitch_rate_penalty | airborne_penalty | fall_penalty | Net anti-hop cost |
|---|---|---|---|---|
| Hopping (exp003 quality) | ~75 | ~150 | 200 | **~425 pts** |
| Walking at 1.5 m/s | ~3 | 0 | 0 | **~3 pts** |

The anti-hop terms make walking strictly better than hopping by ~422 pts,
larger than any single bonus term for the hopper.

---

## Phase 12 — exp004 Result + Fitness Function v5 (gait alternation)

### exp004 result: 2028.1 @ gen 308

The anti-hop penalties worked — score jumped from 985 to 2028, nearly double.
Warm-start (σ=0.2) converged in 308 generations vs 490 in exp003.

**Observed gait:** Sustained stable hopping for a long time. Much more controlled than
exp003's 2-hop-then-fall. The creature learned to land with low pitch rate and maintain
balance through multiple hops.

**New problem:** Still hopping. The airborne penalty only fires when both feet are
simultaneously below 20N — a clever hopper can satisfy this by keeping one foot
lightly grazing the ground during the push-off phase. The optimiser found this loophole.

**Pattern of local optima so far:**
| Exp | Local optimum found | Intervention |
|-----|---------------------|--------------|
| v1 | Stand still | cumulative distance |
| v2 | Shuffle slowly | speed bonus threshold |
| v3 | Ballistic hop (falls fast) | pitch rate + airborne penalty |
| v4 | Sustained hop (stable) | → need alternation reward |
| v5 | Target: alternating-leg walk | |

### Fitness function v5 — gait alternation reward

**Core idea:** A hopper always contacts the ground with both feet together.
A walker alternates: right foot down → left foot down → right foot down.
Rewarding the *switch event* directly incentivises alternation without
needing to specify what the gait should look like.

```python
# Track which foot is dominant (higher contact force)
current_contact = "right" if right_foot_force >= left_foot_force else "left"
# Count switches
if self._last_contact is not None and current_contact != self._last_contact:
    self.alternation_bonus += 1.0
self._last_contact = current_contact
# In compute():
+ self.alternation_bonus * W_ALTERNATION   # W_ALTERNATION = 1.5
```

**Why this works:**
- Symmetric hopper: both feet contact simultaneously → dominant foot never switches
  while grounded → 0 alternation bonus
- Walker at 1.5 m/s: ~2 switches per stride cycle, ~100–150 strides in 1000 steps
  → ~200–300 switches × 1.5 = **300–450 pts** in alternation bonus alone
- Even a slow alternating walk at 0.5 m/s earns more alternation bonus than a hopper

**Other v5 changes:**
- `W_AIRBORNE`: 0.5 → 1.0 (closes the "lightly graze" loophole further)
- `W_VELOCITY`: 0.0 (removed — redundant with speed_bonus)

### Command to start exp005

```bash
python3 train.py --policy cpg \
    --checkpoint experiments/exp004_cpg_v4_antihop/checkpoints/best_params.npy \
    --sigma 0.2 \
    --exp exp005_cpg_v5_alternation
```

---

## Phase 13 — Dropping the Pure MLP Policy

### Decision

After running the pure MLP policy for ~130 generations under the v5 fitness function,
we retired it completely. `policy.py` was deleted and all MLP references were removed
from `train.py`, `evaluate.py`, and the experiment records.

The project is now **CPG-only**.

### Why MLP was tried

Phase 2 established the MLP as the baseline.  It had two roles:
1. A sanity check that CMA-ES can learn anything at all on Walker2d-v5.
2. A comparison baseline to measure how much the CPG structural prior is actually
   worth.

After 5 CPG experiments reaching a best score of 2028.1, it seemed worth checking
whether the CPG's advantage was real or artefactual.

### Why MLP failed with the current setup

Three compounding reasons made the MLP incompatible with the current code as-is:

**1. Parameter count mismatch (5702 vs 504)**

CMA-ES maintains an n×n covariance matrix.  For the MLP that is 5702×5702 ≈ 32 million
entries.  Adapting this matrix requires far more evaluations than the 504-dim CPG case.
The recommended minimum population for reliable convergence scales as `4 + 3·ln(n)`:
~29 for CPG, ~37 for MLP — but in practice the MLP needs 3–5× more generations to
reach comparable fitness because the covariance matrix update carries less information
per generation when the search space is 11× larger.

**2. Fitness function calibrated for CPG behaviour**

`FALL_PENALTY = 200` was set assuming the agent would survive at least briefly.
A random MLP (initial σ=0.5) produces incoherent joint torques — the walker
immediately pitches out of the healthy range and the episode terminates at step 1–5,
incurring the full 200-point penalty every evaluation.  The CPG's anti-phase
initialisation and amplitude/frequency structure means a random CPG parameter vector
still produces a rough rhythmic motion and the walker survives 50–200 steps.

The speed bonus threshold (1.0 m/s) and alternation bonus are similarly calibrated
around a CPG that already has some locomotion structure.  An MLP would need the
equivalent of the v1–v3 fitness function progression (which took ~1500 CPG generations
to traverse) before the v5 penalties become useful rather than catastrophic.

**3. No structural prior**

The CPG encodes three things before training even begins:
- Rhythmicity: outputs are always periodic (sin-based)
- Bounded authority: feedback MLP is capped at ±50% torque (`CORRECTION_SCALE=0.5`)
- Anti-phase bias: right and left legs start 180° out of phase

A pure MLP has none of these.  It must discover periodicity, balance, and alternation
from a completely flat prior.  With CMA-ES (black-box, no gradient information) this
means thousands more generations before the first useful behaviours emerge.

### What was observed at 130 generations

Scores were consistently negative (−200 to −50), indicating nearly every individual
in every generation fell within the first few steps and collected the full FALL_PENALTY.
The mean fitness was not converging toward zero — the population had not yet found even
a "stand still" policy, which is the first local optimum a random search usually finds.

This is not a bug.  It is the expected behaviour of CMA-ES on a 5702-dim space with
a fitness function that punishes falling hard.  Given enough generations (estimated
800–2000) the MLP would likely converge to a standing policy, and given further
generations to a shuffle, and so on — retracing the entire v1→v5 development arc.

### Why we dropped it rather than fixing it

The comparison has already been answered conceptually:
- CPG: 504 params, structural priors, first useful gait within 100 generations.
- MLP: 5702 params, no priors, likely 500+ generations just to stand.

Replicating the v1→v5 fitness evolution for MLP would consume significant compute
and deliver a well-understood result.  The CPG architecture is the intended focus of
this research and produces better gaits with far fewer evaluations.

The thesis comparison between MLP and CPG is made at the architectural level (Phase 4)
rather than requiring a fully trained MLP baseline.

### Codebase changes

| File | Action |
|------|--------|
| `policy.py` | Deleted |
| `train.py` | Removed `--policy`/`--hidden` args, `MLP` import, `HIDDEN_SIZE`, MLP branches |
| `evaluate.py` | Removed `MLP` import, MLP branch in `make_policy()`, `obs_dim`/`act_dim`/`hidden_size` params |
| `experiments/mlp001_mlp_v5_baseline/` | Deleted (130-gen abandoned run) |
| `checkpoints/mlp/` | Deleted |
| `experiments/README.md` | Removed MLP experiments table |

`train.py` is now CPG-only: `python3 train.py` always trains the CPG hybrid.

---

## Open Questions / Future Work

1. **Custom fitness tuning** — The current weight values are initial calibrated estimates. Systematic ablation (training with each component removed or scaled) would identify which terms most influence gait quality.

2. **Oscillator coupling** — Currently the 6 CPG oscillators are independent (only linked by the shared anti-phase initialisation). Biological CPGs have explicit coupling terms between oscillators. Adding Kuramoto-style coupling (oscillators pull toward each other's phase) might produce more stable gaits.

3. **Adaptive correction scale** — The correction budget is now fixed at 0.5. A learnable scalar per joint could allow the optimiser to decide how much feedback vs. CPG to use for each joint independently.

4. **Comparison study** — A systematic comparison of pure MLP vs. pure CPG vs. hybrid across multiple random seeds and training budgets would quantify the benefit of the hybrid approach rigorously.

5. **Larger population** — `--pop 40` is the minimum recommended for n=504. Increasing to `--pop 80` or `--pop 100` would improve exploration at the cost of more evaluations per generation.

6. **Neuroevolution alternatives** — NEAT (NeuroEvolution of Augmenting Topologies) evolves both weights and network structure simultaneously and may find more compact solutions than a fixed architecture.
