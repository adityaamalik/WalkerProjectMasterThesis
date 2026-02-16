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
