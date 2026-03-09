# Thesis Action Plan: Curriculum vs Fixed Gravity Using a Gravity-Agnostic Objective

## Summary
This plan evaluates whether curriculum learning improves learning speed and quality versus fixed-gravity training, without forcing a specific gait style. We freeze one gravity-agnostic fitness objective, run a 3-arm controlled study, and compare arms on predefined metrics with statistical tests.

## Parameters That Define “Better or Worse”
1. **Primary metric: Time-to-Earth-Success (TTE)**
Earth probe at `g=-9.81` every 10 generations.
Success at a probe if all hold over 10 eval episodes:
- `mean net_progress_m >= 4.0`
- `mean fell <= 0.20`
- `mean backward_distance_m <= 1.0`
TTE = first generation that meets success. If never reached by gen 500, mark as censored (`510`).

2. **Sample-efficiency metric: Earth Learning AUC**
Area under Earth probe curve (`net_progress_m` vs generation), computed from gen 0..500.

3. **Final-performance metric: Final Earth Scorecard (gen 500)**
At Earth gravity, evaluate 20 episodes and report:
- `mean net_progress_m`
- `mean fell`
- `mean forward speed`
- mean objective score

4. **Robustness metric: Gravity Sweep Scorecard**
Evaluate final policy on gravity set `{-6.0, -7.5, -9.81, -11.0, -12.0}` for 20 episodes each:
- mean `net_progress_m` across gravities
- worst-gravity `net_progress_m`
- mean fall rate across gravities

5. **Consistency metric: Seed Stability**
Across **10 seeds** per arm, report seed-level:
- median
- Q1
- Q3
- IQR (`Q3 - Q1`)
for TTE and final Earth `net_progress_m`.

## Decision Rule for Thesis Claim
Curriculum is considered better only if all are true:
1. Median TTE is at least 20% lower than both controls.
2. Final Earth `net_progress_m` is non-inferior to Fixed Earth (>=95% of Fixed median).
3. Gravity-sweep robustness is not worse (higher/equal mean progress and lower/equal fall rate).

Use bootstrap 95% CIs and Mann-Whitney U for pairwise comparisons (Curriculum vs Fixed, Curriculum vs Random-Variable).

Implementation note:
- Statistical tests are run on **seed-level aggregates** (not per-episode values).

## Experiment Design (Locked)
1. **Arm A: Fixed Earth**
Train with constant `g=-9.81`.

2. **Arm B: Random Variable Gravity (no curriculum)**
Each generation uses randomly sampled gravity in `[-6.0, -12.0]`.

3. **Arm C: Curriculum Variable Gravity**
Ordered schedule from easier to harder gravity (same bounds, ending near Earth phase for target task evaluation).

4. **Shared controls**
- Scratch training only (no warm starts)
- Same policy architecture (CPG hybrid, 504 params)
- Same CMA-ES budget: 500 generations, pop 40, sigma fixed per protocol
- **10 seeds per arm**

## Fitness Function Plan (Gravity-Agnostic)
Replace gait-style shaping with task-level locomotion objective in `/Users/adityamalik/Developer/walker2d-evo/evaluate.py`:
1. Keep: forward/net progress, survival, uprightness, backward penalty, fall penalty, control/impact regularization.
2. Remove for thesis runs: alternation/overtake/front-timer/step-length terms and hop-specific style constraints.
3. Normalize per-step penalty terms by episode length so scales stay comparable across gravity.

Freeze this objective after one pilot calibration pass and do not retune during the main 3-arm study.

## Implementation Plan
1. Add reproducibility and logging hooks in `/Users/adityamalik/Developer/walker2d-evo/train.py` and `/Users/adityamalik/Developer/walker2d-evo/train_curriculum.py`:
- explicit `--seed`
- structured per-generation log file
- periodic Earth probes from saved checkpoints

2. Add non-curriculum variable-gravity strategy in `/Users/adityamalik/Developer/walker2d-evo/curriculum/` (new strategy file + registry update).

3. Add evaluation/analysis scripts:
- checkpoint-to-Earth-probe curve extraction
- final gravity sweep evaluation
- aggregation across seeds and statistical summary tables

4. Store outputs per experiment:
- `earth_probe.csv`
- `gravity_sweep.csv`
- `summary.json`
- `stats_report.md`

5. Update documentation:
- `/Users/adityamalik/Developer/walker2d-evo/experiments/README.md`
- `/Users/adityamalik/Developer/walker2d-evo/DEVLOG.md`

## Important API / Interface Changes
1. `train.py` new CLI:
- `--seed`
- `--probe-every`
- `--probe-episodes`
- `--probe-gravity`

2. `train_curriculum.py` new CLI:
- `--seed`
- `--probe-every`
- `--probe-episodes`
- `--probe-gravity`
- curriculum bounds parameters for reproducible schedules

3. New curriculum strategy interface addition:
- support deterministic seeded random-gravity schedule.

4. New output schema additions in experiment artifacts:
- machine-readable probe and sweep files for thesis tables/plots.

## Test Cases and Validation Scenarios
1. Smoke run each arm for 5 generations verifies logs/probe files are created and parseable.
2. Reproducibility test: same seed/args produces identical generation-0 checkpoint metadata and matching probe outputs.
3. Earth-probe evaluator test: detects threshold crossing correctly on synthetic checkpoint series.
4. Gravity-sweep evaluator test: outputs complete rows for all configured gravities and required metrics.
5. Statistical aggregation test: handles censored TTE runs (no threshold hit by gen 500).

## Assumptions and Defaults Chosen
1. Primary thesis target is **faster Earth learning**.
2. Gravity evaluation range is **narrow** (`-6` to `-12`).
3. Main study uses **10 seeds x 500 generations**.
4. Main comparison uses **3 arms** (Fixed, Random-Variable, Curriculum).
5. Morphology stays fixed; no co-evolution in this phase.
