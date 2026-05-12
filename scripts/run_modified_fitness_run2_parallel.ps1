# scripts/run_modified_fitness_run2_parallel.ps1
#
# Launch two parallel batches of the modified-fitness experiment, splitting
# the 7 arms across two PowerShell windows. Roughly halves wall-clock on the
# Ryzen 7700X (8 cores / 16 threads) — each batch uses ~8 threads.
#
# Usage:
#     Window 1:  .\scripts\run_modified_fitness_run2_parallel.ps1 -Group A
#     Window 2:  .\scripts\run_modified_fitness_run2_parallel.ps1 -Group B
#
# Group A runs: staged_evolution, gradual_transition, archive_based, multi_environment
# Group B runs: fixed_gravity, random_variable_gravity, adaptive_progression
#
# Both groups write into the SAME experiments\thesis_morph_run_2\ tree but
# into different arm subdirectories, so they cannot collide.

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("A", "B")]
    [string]$Group
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Modified fitness weights (identical to the single-batch script)
$env:FITNESS_UPRIGHT_WEIGHT      = "0.05"
$env:FITNESS_KNEE_FLEXION_WEIGHT = "0.05"

$ExpRoot      = "thesis_morph_run_2"
$Seeds        = "0-19"
$Generations  = 1000
$GravityStart = -1.6
$GravityEnd   = -9.81

if ($Group -eq "A") {
    $Arms = "staged_evolution,gradual_transition,archive_based,multi_environment"
} else {
    $Arms = "fixed_gravity,random_variable_gravity,adaptive_progression"
}

Write-Host "================================================================"
Write-Host "PARALLEL BATCH — Group $Group"
Write-Host "================================================================"
Write-Host "  Arms:         $Arms"
Write-Host "  Seeds:        $Seeds"
Write-Host "  Generations:  $Generations"
Write-Host "  Output root:  experiments\$ExpRoot\"
Write-Host "  Fitness:      UPRIGHT=$env:FITNESS_UPRIGHT_WEIGHT  KNEE_FLEX=$env:FITNESS_KNEE_FLEXION_WEIGHT"
Write-Host ""
$resp = Read-Host "Launch group $Group ? (y/N)"
if ($resp -ne "y" -and $resp -ne "Y") {
    Write-Host "Aborted."
    exit 0
}

python -m scripts.run_thesis_batch `
    --arms $Arms `
    --seeds $Seeds `
    --generations $Generations `
    --exp-root $ExpRoot `
    --gravity-start $GravityStart `
    --gravity-end $GravityEnd `
    --evolve-morphology `
    --resume

Write-Host ""
Write-Host "Group $Group finished at $(Get-Date)"
