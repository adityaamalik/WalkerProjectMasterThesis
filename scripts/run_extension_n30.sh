#!/usr/bin/env bash
# Launch extension runs (seeds 10-29) for staged_evolution and fixed_gravity.
# Existing seeds 00-09 are NOT touched. Uses --resume as a belt-and-suspenders
# safety net: even if a target dir somehow exists, an existing summary.json
# causes the batch to skip rather than overwrite.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (edit ARMS to add gradual_transition later if time permits)
# -----------------------------------------------------------------------------
REPO_ROOT="/Users/adityamalik/Developer/walker2d-evo"
EXP_ROOT="thesis_morph"           # MUST match existing data root
ARMS="staged_evolution,fixed_gravity"
SEEDS="10-29"                     # 20 new seeds per arm, disjoint from 00-09

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "================================================================"
echo "PRE-FLIGHT CHECKS"
echo "================================================================"

# 1. Verify existing seeds are intact (sample a couple)
for arm in staged_evolution fixed_gravity; do
  for s in 00 09; do
    p="experiments/${EXP_ROOT}/${arm}/seed_${s}/summary.json"
    if [[ ! -f "$p" ]]; then
      echo "❌ MISSING existing data: $p"
      echo "   This script expects the original n=10 dataset to be in place."
      exit 1
    fi
  done
done
echo "✓ Existing seeds 00 and 09 intact for both arms"

# 2. Verify new seed slots are empty (no collisions)
collision=0
for arm in staged_evolution fixed_gravity; do
  for s in $(seq -w 10 29); do
    p="experiments/${EXP_ROOT}/${arm}/seed_${s}/summary.json"
    if [[ -f "$p" ]]; then
      echo "⚠️  Already exists: $p"
      collision=$((collision + 1))
    fi
  done
done
if [[ $collision -gt 0 ]]; then
  echo "❌ Found $collision pre-existing summary files in the target seed range."
  echo "   --resume will skip these but please verify this is intentional."
  read -p "Continue anyway? (y/N) " ans
  [[ "$ans" =~ ^[Yy]$ ]] || exit 1
else
  echo "✓ Target seed range 10-29 is empty for both arms"
fi

# 3. Git cleanliness — code freeze check
if [[ -n "$(git status --porcelain evaluate.py cpg_policy.py train.py train_curriculum.py curriculum/ 2>/dev/null)" ]]; then
  echo "⚠️  Uncommitted changes in training-relevant files:"
  git status --short evaluate.py cpg_policy.py train.py train_curriculum.py curriculum/
  echo "   New seeds 10-29 will use this code, which differs from seeds 00-09."
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
python3 -m scripts.run_thesis_batch \
  --arms "$ARMS" \
  --seeds "$SEEDS" \
  --exp-root "$EXP_ROOT" \
  --evolve-morphology \
  --resume \
  --dry-run 2>&1 | tail -50

echo ""
read -p "Launch the real batch? (y/N) " ans
[[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# -----------------------------------------------------------------------------
# Real launch
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "LAUNCHING EXTENSION RUNS"
echo "Output logs: experiments/${EXP_ROOT}/<arm>/seed_<NN>/run.log"
echo "================================================================"
echo "Started at: $(date)"
echo ""

python3 -m scripts.run_thesis_batch \
  --arms "$ARMS" \
  --seeds "$SEEDS" \
  --exp-root "$EXP_ROOT" \
  --evolve-morphology \
  --resume

echo ""
echo "Finished at: $(date)"
