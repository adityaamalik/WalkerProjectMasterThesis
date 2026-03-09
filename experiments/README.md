# Experiment Log — Walker2D-v5 CMA-ES

Each subdirectory is one training run.  Every experiment folder contains:

| File | Contents |
|---|---|
| `experiment.json` | Config, fitness weights, result, diagnosis |
| `fitness_curve.png` | Best + mean fitness over generations |
| `checkpoints/best_params.npy` | Best policy weights |
| `checkpoints/best_params.json` | Score + generation metadata |

To replay any experiment:
```bash
python3 train.py --render \
    --checkpoint experiments/expXXX_.../checkpoints/best_params.npy
```

---

## CPG hybrid experiments

Architecture: 6 CPG oscillators + 23→16→6 feedback MLP = **504 parameters**

| ID | Name | Best score | Predicted | Gait | Key change |
|----|------|-----------|-----------|------|-----------|
| [exp001](exp001_cpg_v1_maxdist/) | CPG v1 — max_distance | 307.97 | — | Hops once, stops | Baseline CPG hybrid |
| [exp002](exp002_cpg_v2_cumdist/) | CPG v2 — cumulative_distance | 539.6 | ~1075 | Shuffle-walk | Replaced max_dist with cumulative dist |
| [exp003](exp003_cpg_v3_stride/) | CPG v3 — speed bonus | 985.43 | ~1753 | 2 hops, falls 3rd | Speed bonus broke shuffle; triggered hopping exploit |
| [exp004](exp004_cpg_v4_antihop/) | CPG v4 — anti-hop penalties | **2028.1** | ~1750 | Sustained stable hops | Pitch-rate + airborne penalty made hopping controlled |
| [exp005](exp005_cpg_v5_alternation/) | CPG v5 — gait alternation reward | 1064.78 | ~1742 | Regression — alternation conflicted with speed | Alternation bonus + stronger airborne penalty |
| [exp006](exp006_cpg_v6_walking/) | CPG v6 — symmetric contact penalty | TBD | ~2178 | One step then stood still (lazy balancer exploit) | Bilateral symmetry penalty + W_ALTERNATION 1.5→8.0 + lower speed threshold |
| [exp007](exp007_cpg_v7_forward_drive/) | CPG v7 — forward-gated bonuses | TBD | ~1813 | Still stood still (wobble dodged stagnation counter) | Motion-gated upright/progress + quadratic stagnation penalty |
| [exp008](exp008_cpg_v8_threshold/) | CPG v8 — distance threshold gate | 4107 | ~1768 | Vibration exploit: 360 fake alternation switches → +1800 pts | fitness=0 if distance < 1.5m — no gradient from partial solutions |
| [exp009](exp009_cpg_v8_stride_filter/) | CPG v8+filter — stride-filtered alternation | TBD | ~2140 | TBD | MIN_CONTACT_STEPS=30 — foot must hold stance 0.24s before switch counts |
| [exp010](exp010_cpg_v9_softgate_gaitfactor/) | CPG v9 — soft distance gate + gait multiplier | 418.16 | — | Low-progress locomotion | Replaced hard gate with soft gate and multiplicative gait quality |
| [exp011](exp011_cpg_v9_scratch_walk/) | CPG v9 scratch | 0.0 | — | Immediate collapse | Scratch run exposed flat-zero fitness regime |
| [exp012](exp012_cpg_v9_scratch_lowdist_penalty/) | CPG v9+ — low-distance penalty | -160.01 | — | Survives briefly, then falls | Penalised low-distance attractor to avoid neutral zero solutions |
| [exp013](exp013_cpg_v10_densewalk_scratch/) | CPG v10 — dense anti-hop objective | -130.34 | — | Immediate-fall local optimum | Dense terms but penalties dominated early search |
| [exp014](exp014_cpg_v11_staged_scratch/) | CPG v11 — staged penalty/correction schedules | 953.61 | — | Stable forward locomotion | Ramped penalty_scale and correction_scale from scratch |
| [exp015](exp015_cpg_v12_overtake_stride/) | CPG v12 — overtake stride term | 1074.56 | — | Improved stepping, limited overtakes | Added explicit lead-foot overtake reward |
| [exp016](exp016_cpg_v12_overtake_warmstart/) | CPG v12 warm-start fine-tune | 1226.38 | — | Better stability + speed | Warm-start from exp015 with full-strength schedules |
| [exp017](exp017_cpg_v13_timer_step_smooth_warmstart/) | CPG v13 — timer + step + smoothness (warm-start) | 1502.77 | — | Stable walking, still catch-up style | Added 2s lead-foot timer shaping, step-length reward, velocity-change penalty |
| [exp018](exp018_cpg_v13_timer_step_smooth_scratch/) | CPG v13 scratch | **1596.79** | — | Best current stable walking (not fully human-like) | Same v13 fitness as exp017, trained from scratch |

---

## Experiment details

### exp001 — CPG hybrid, fitness v1 (max_distance)
**Score: 307.97 @ gen 449**

First full CPG run.  Fitness dominated by `max_distance × 100`.

**Problem:** `max_distance` only records the furthest x-position ever reached.
Once the walker lunges forward once (even by falling), the fitness stops increasing —
no evolutionary pressure to keep walking.

**Observed gait:** Hops when it falls, takes one step forward, then stands still.

**Fix:** Replace with cumulative distance → exp002.

---

### exp002 — CPG hybrid, fitness v2 (cumulative_distance)
**Score: 539.6 @ gen 460**

Replaced `max_distance` (peak) with `cumulative_distance` (integral of forward velocity × dt).
Fitness now grows continuously as long as the walker moves forward.
Also scaled up `W_UPRIGHT` (0.1→0.3), `W_VELOCITY` (0.05→0.5), `W_TIME_ALIVE` (0.01→0.1).

**Improvement:** +232 points (+75%) over exp001.  No more "hop and stop".

**Observed gait:** Places front foot forward, brings back foot to same position, repeats.
Shuffle-walk — feet never overtake each other.  Sustained but slow.

**Why score is ~540 not ~1075:**
The walker shuffles slowly (~0.5–0.8 m/s) instead of striding at 2 m/s.
Cumulative distance over 1000 steps is ~4–6 m instead of predicted 16 m.
Root cause: the CPG has no explicit reward for *stride length* — the walker learned
the safest motion (shuffle) rather than an efficient stride.

**Fix:** Add a stride-length bonus → exp003.

---

### exp003 — CPG hybrid, fitness v3 (speed bonus)
**Score: 985.43 @ gen 490**

Added `speed_bonus = max(0, x_vel - 1.0) × 2.0` per step. This successfully broke
the shuffle local minimum from exp002.  Also increased `CORRECTION_SCALE` 0.3 → 0.5.

**Observed gait:** 2 successful hops forward, falls on 3rd hop.

**Problem:** Speed bonus rewards *any* motion above 1.0 m/s. The cheapest CPG solution
is a ballistic hop: both oscillators fire simultaneously at high amplitude, launching
the body.  After 2–3 hops the torso pitch momentum accumulates past the health
termination threshold (>1.0 rad) → fall + 100pt penalty.

**Fix:** Add anti-hop penalties → exp004.

---

### exp004 — CPG hybrid, fitness v4 (anti-hop penalties + warm-start)
**Score: 2028.1 @ gen 308**

Warm-started from exp003 best params with σ=0.2. Added pitch_rate_penalty (obs[2] ×
0.05/step) and airborne_penalty (both feet < 20N → +0.5/step), increased FALL_PENALTY
100 → 200.

**Observed gait:** Sustained stable hopping for a long time. Much more controlled than
exp003's crash-after-2-hops. Anti-hop penalties taught the creature to land stably
and maintain pitch control through multiple hops.

**Problem:** Still hopping — not alternating-leg walking. The airborne penalty only fires
when both feet are simultaneously off the ground. A clever hopper can satisfy this by
keeping one foot barely touching while still launching. The optimiser found this loophole.

**Fix:** Reward explicit left-right foot contact alternation → exp005.

---

### exp005 — CPG hybrid, fitness v5 (gait alternation reward)
**Score: 1064.78 @ gen 428**

See [exp005/experiment.json](exp005_cpg_v5_alternation/experiment.json) for full config.

Key change: add an **alternation bonus** that fires whenever the foot contact pattern
switches from one foot to the other (right→left or left→right). A walker collects this
every stride (~2 switches per full gait cycle). A symmetric hopper (both feet together)
never collects it.

---

### exp018 — Current reference fitness snapshot (v13)
**Score: 1596.79 @ gen 499**

The exact currently active fitness logic (all weights, constants, formula and schedule)
is manually documented in:

- [exp018/experiment.json](exp018_cpg_v13_timer_step_smooth_scratch/experiment.json)

This is the canonical reference for the latest “stable walking” configuration.

---

## Curriculum learning experiments

Architecture: CPG hybrid (504 params). Gravity varies per-generation according to the curriculum strategy.

Training entry point: `train_curriculum.py`

```bash
# Train
python3 train_curriculum.py --curriculum gradual_transition

# Warm-start from best CPG checkpoint
python3 train_curriculum.py --curriculum gradual_transition \
    --checkpoint checkpoints/cpg/best_params.npy

# Render the trained policy
python3 train_curriculum.py --curriculum gradual_transition --render \
    --checkpoint experiments/curriculum_gradual_transition/checkpoints/best_params.npy
```

### Available strategies

| Strategy flag | Description |
|---|---|
| `gradual_transition` | Gravity ramps linearly from -2.0 m/s² (easy) → -9.81 m/s² (earth) over training |
| `random_uniform` | Gravity is sampled each generation from a fixed range (default `[-12.0, -6.0]`) |

### Results

| ID | Strategy | Best score | Baseline (exp004) | Verdict |
|----|----------|-----------|-------------------|---------|
| [curriculum_gradual_transition](curriculum_gradual_transition/) | Gradual Transition | TBD | 2028.1 | pending |

---

## Thesis protocol (10 seeds per arm)

The thesis comparison now runs with **10 seeds per arm** (`0..9`) for:

- `fixed_gravity`
- `random_variable_gravity` (non-curriculum control)
- `curriculum_variable_gravity`

Standardized output layout:

`experiments/thesis/<arm>/seed_<NN>/`

Each seed directory stores:

- `training_log.csv`
- `earth_probe.csv`
- `gravity_sweep.csv`
- `summary.json`

Batch runner and aggregator:

```bash
# Run full 3-arm batch (default: 10 seeds)
python3 scripts/run_thesis_batch.py

# Aggregate across seeds (requires 10 completed seeds/arm by default)
python3 scripts/aggregate_thesis_results.py

# Recompute probe/sweep metrics from saved checkpoints for one run
python3 scripts/evaluate_run_metrics.py \
    --checkpoint-dir experiments/thesis/<arm>/seed_<NN>/checkpoints \
    --output-dir experiments/thesis/<arm>/seed_<NN>
```

Aggregation outputs:

- `experiments/thesis/aggregated_metrics.csv`
- `experiments/thesis/aggregated_summary.json`
- `experiments/thesis/stats_report.md`
