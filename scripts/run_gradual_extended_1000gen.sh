#!/usr/bin/env bash
# Launch a fully-isolated 1000-generation, 30-seed experiment for the
# gradual_transition curriculum, written into a separate experiment tree so
# nothing in the existing thesis_morph dataset can be affected.
#
# Output goes to: experiments/thesis_morph_extended/gradual_transition_extended/seed_NN/
# Existing data at experiments/thesis_morph/ is NOT touched.
#
# Estimated wall-clock: ~60 hours sequential (~2 hours per seed × 30 seeds).
# Run with `caffeinate -dimsu -w <PID>` attached.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REPO_ROOT="/Users/adityamalik/Developer/walker2d-evo"
EXP_ROOT="thesis_morph_extended"           # SEPARATE from thesis_morph
ARM="gradual_transition_extended"
SEEDS="0-29"                               # full 30 seeds
GENERATIONS=1000
GRAVITY_START="-1.6"                       # same ramp as the 500-gen gradual arm
GRAVITY_END="-9.81"

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "================================================================"
echo "PRE-FLIGHT CHECKS"
echo "================================================================"

# 1. Verify the existing thesis_morph tree exists but won't be touched.
if [[ ! -d "experiments/thesis_morph" ]]; then
  echo "⚠️  experiments/thesis_morph not found — expected the existing dataset to be present"
  echo "   (continuing anyway; this is not a blocker for the extended run)"
else
  existing_count=$(find experiments/thesis_morph -mindepth 3 -name "summary.json" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
  echo "✓ Existing dataset intact: $existing_count summary.json files in experiments/thesis_morph/"
fi

# 2. Confirm the extended tree is either empty or matches the planned launch.
ext_root="experiments/${EXP_ROOT}/${ARM}"
if [[ -d "$ext_root" ]]; then
  collisions=$(find "$ext_root" -mindepth 2 -name "summary.json" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
  if [[ "$collisions" -gt 0 ]]; then
    echo "⚠️  Found $collisions completed summary.json under $ext_root"
    echo "   --resume will skip these. Verify this is intentional."
    read -p "Continue? (y/N) " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  else
    echo "✓ Extended arm directory exists but contains no completed runs"
  fi
else
  echo "✓ Extended arm directory does not exist yet (will be created on launch)"
fi

# 3. Git cleanliness — code freeze check for training-relevant files.
if [[ -n "$(git status --porcelain evaluate.py cpg_policy.py train.py train_curriculum.py curriculum/ 2>/dev/null)" ]]; then
  echo "⚠️  Uncommitted changes in training-relevant files:"
  git status --short evaluate.py cpg_policy.py train.py train_curriculum.py curriculum/
  read -p "Continue anyway? (y/N) " ans
  [[ "$ans" =~ ^[Yy]$ ]] || exit 1
else
  echo "✓ Training-relevant files have no uncommitted changes"
fi

# -----------------------------------------------------------------------------
# Dry-run preview
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "DRY RUN — preview what will be launched"
echo "================================================================"
echo "  Arm:          $ARM"
echo "  Seeds:        $SEEDS"
echo "  Generations:  $GENERATIONS"
echo "  Gravity ramp: $GRAVITY_START → $GRAVITY_END (20% warmup ⇒ gen 0-199)"
echo "  Output root:  experiments/$EXP_ROOT/$ARM/seed_NN/"
echo "  Existing data at experiments/thesis_morph/ will NOT be touched."
echo ""

python3 -m scripts.run_thesis_batch \
  --arms "$ARM" \
  --seeds "$SEEDS" \
  --generations "$GENERATIONS" \
  --exp-root "$EXP_ROOT" \
  --gravity-start "$GRAVITY_START" \
  --gravity-end "$GRAVITY_END" \
  --evolve-morphology \
  --resume \
  --dry-run 2>&1 | tail -40

echo ""
read -p "Launch the real batch? (y/N) " ans
[[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# -----------------------------------------------------------------------------
# Real launch
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "LAUNCHING EXTENDED RUN — gradual_transition @ 1000 generations × 30 seeds"
echo "Started at: $(date)"
echo "Output logs: experiments/${EXP_ROOT}/${ARM}/seed_<NN>/run.log"
echo "================================================================"
echo ""
echo "Estimated duration: ~60 hours sequential."
echo "Recommended: in a new terminal, run:"
echo "    caffeinate -dimsu -w \$(pgrep -f 'run_thesis_batch') &"
echo ""

python3 -m scripts.run_thesis_batch \
  --arms "$ARM" \
  --seeds "$SEEDS" \
  --generations "$GENERATIONS" \
  --exp-root "$EXP_ROOT" \
  --gravity-start "$GRAVITY_START" \
  --gravity-end "$GRAVITY_END" \
  --evolve-morphology \
  --resume

echo ""
echo "Finished at: $(date)"
