# scripts/run_modified_fitness_run2.ps1
#
# Launches the modified-fitness experiment (thesis_morph_run_2) on Windows.
#
# - Modified fitness weights:
#     FITNESS_UPRIGHT_WEIGHT       = 0.05  (down from default 0.20)
#     FITNESS_KNEE_FLEXION_WEIGHT  = 0.05  (default 0.0 — disables the term)
# - 7 arms x 20 seeds x 1000 generations
# - Output tree: experiments/thesis_morph_run_2/<arm>/seed_NN/
# - The original experiments/thesis_morph/ tree is NOT touched.
#
# Usage:
#     1. Open PowerShell, cd into the repo root.
#     2. Run:  .\scripts\run_modified_fitness_run2.ps1
#
# Cross-platform note: this script uses $PSScriptRoot to locate the repo,
# so it works regardless of where the user cloned the repo on their Windows
# machine. No absolute paths are hardcoded.

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Resolve repo root from this script's location (parent of scripts/)
# -----------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repository root: $RepoRoot"

# -----------------------------------------------------------------------------
# Modified fitness weights (the whole point of run_2)
# -----------------------------------------------------------------------------
$env:FITNESS_UPRIGHT_WEIGHT       = "0.05"
$env:FITNESS_KNEE_FLEXION_WEIGHT  = "0.05"

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
$ExpRoot       = "thesis_morph_run_2"
$Seeds         = "0-19"
$Generations   = 1000
$GravityStart  = -1.6
$GravityEnd    = -9.81
# Arms left at default = all 7 standard arms (see ARMS_DEFAULT in
# scripts/run_thesis_batch.py)

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================"
Write-Host "MODIFIED FITNESS EXPERIMENT — thesis_morph_run_2"
Write-Host "================================================================"
Write-Host "  FITNESS_UPRIGHT_WEIGHT       = $env:FITNESS_UPRIGHT_WEIGHT  (default 0.20)"
Write-Host "  FITNESS_KNEE_FLEXION_WEIGHT  = $env:FITNESS_KNEE_FLEXION_WEIGHT  (default 0.0)"
Write-Host ""
Write-Host "  Arms:         all 7 standard arms"
Write-Host "  Seeds:        $Seeds (= 20 seeds per arm)"
Write-Host "  Generations:  $Generations per seed"
Write-Host "  Output root:  experiments\$ExpRoot\<arm>\seed_NN\"
Write-Host ""
Write-Host "  Existing experiments\thesis_morph\ tree will NOT be touched."
Write-Host ""

# Make sure the original tree is intact if present
$OriginalRoot = Join-Path $RepoRoot "experiments\thesis_morph"
if (Test-Path $OriginalRoot) {
    $existingCount = (Get-ChildItem -Path $OriginalRoot -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue).Count
    Write-Host "[OK] Original dataset intact: $existingCount summary.json files in experiments\thesis_morph\"
} else {
    Write-Host "[INFO] experiments\thesis_morph\ not found (this is fine if you cloned a fresh copy)"
}

# Check for any pre-existing run_2 data
$NewRoot = Join-Path $RepoRoot "experiments\$ExpRoot"
if (Test-Path $NewRoot) {
    $collisions = (Get-ChildItem -Path $NewRoot -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue).Count
    if ($collisions -gt 0) {
        Write-Host "[WARN] Found $collisions completed summary.json under experiments\$ExpRoot\"
        Write-Host "       --resume will skip these. Verify this is intentional."
        $resp = Read-Host "Continue? (y/N)"
        if ($resp -ne "y" -and $resp -ne "Y") { exit 1 }
    } else {
        Write-Host "[OK] experiments\$ExpRoot\ exists but contains no completed runs"
    }
} else {
    Write-Host "[OK] experiments\$ExpRoot\ does not exist yet (will be created on launch)"
}

# -----------------------------------------------------------------------------
# Confirm before launching
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "Estimated wall-clock: ~280 hours sequential, ~140 hours if you run two arms in parallel"
Write-Host "  (Recommended: open a second PowerShell window and follow the parallel-launch tip below.)"
Write-Host ""
$resp = Read-Host "Launch the experiment? (y/N)"
if ($resp -ne "y" -and $resp -ne "Y") {
    Write-Host "Aborted."
    exit 0
}

# -----------------------------------------------------------------------------
# Real launch
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================"
Write-Host "LAUNCHING — started at $(Get-Date)"
Write-Host "================================================================"

# Use `python` on Windows (not `python3` — Windows installs only `python.exe`)
python -m scripts.run_thesis_batch `
    --seeds $Seeds `
    --generations $Generations `
    --exp-root $ExpRoot `
    --gravity-start $GravityStart `
    --gravity-end $GravityEnd `
    --evolve-morphology `
    --resume

Write-Host ""
Write-Host "================================================================"
Write-Host "Finished at $(Get-Date)"
Write-Host "================================================================"
