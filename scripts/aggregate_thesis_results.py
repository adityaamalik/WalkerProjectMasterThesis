#!/usr/bin/env python3
"""Aggregate thesis runs across seeds/arms and compute statistics.

Outputs:
- experiments/thesis/aggregated_metrics.csv
- experiments/thesis/aggregated_summary.json
- experiments/thesis/aggregated_probe_curves.csv
- experiments/thesis/stats_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

ARMS_DEFAULT = [
    "fixed_gravity",
    "random_variable_gravity",
    "curriculum_variable_gravity",   # legacy (gradual_transition, -2.0 start)
    "gradual_transition",            # Moon → Earth (-1.6 → -9.81)
    "staged_evolution",
    "multi_environment",
    "adaptive_progression",
    "archive_based",
]

METRICS = [
    "tte_generation",
    "final_net_progress_m",
    "robustness_mean_net_progress_m",
    "best_earth_probe_score",
    "mean_earth_probe_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate thesis experiment results.")
    p.add_argument("--thesis-root", type=str, default="experiments/thesis")
    p.add_argument("--arms", type=str, default=",".join(ARMS_DEFAULT))
    p.add_argument("--expected-seeds", type=int, default=10,
                   help="Expected completed seeds per arm.")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Allow aggregation with fewer than expected seeds.")
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(x, default=None):
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def to_int(x, default=None):
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """NumPy-only average rank implementation for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + (j - 1)) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def mann_whitney_u(x_vals: list[float], y_vals: list[float]) -> dict:
    """Two-sided Mann-Whitney U with effect size r. scipy when available, NumPy fallback."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)

    if x.size < 2 or y.size < 2:
        return {
            "u": None,
            "p_value": None,
            "effect_size_r": None,
            "method": "insufficient_data",
        }

    n1 = x.size
    n2 = y.size
    combined = np.concatenate([x, y])
    ranks = rankdata_average(combined)

    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1) / 2.0)
    u2 = (n1 * n2) - u1
    u_min = min(u1, u2)
    mean_u = n1 * n2 / 2.0

    # Effect size r = |Z| / sqrt(N), Z from normal approximation (uncorrected std).
    # Computed here so it is available regardless of which p-value path is taken.
    std_u_raw = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_u_raw > 0:
        z_effect = (u1 - mean_u) / std_u_raw
        effect_r = abs(z_effect) / math.sqrt(n1 + n2)
    else:
        effect_r = 0.0

    # Try scipy for more accurate p-value.
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        sp_u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return {
            "u": float(sp_u),
            "p_value": float(p),
            "effect_size_r": float(effect_r),
            "method": "scipy",
        }
    except Exception:
        pass

    # Tie-corrected variance for numpy fallback p-value.
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts)
    n = n1 + n2
    if n <= 1:
        return {
            "u": float(u_min),
            "p_value": 1.0,
            "effect_size_r": float(effect_r),
            "method": "fallback",
        }

    var_u = (n1 * n2 / 12.0) * ((n + 1) - (tie_term / (n * (n - 1))))
    if var_u <= 0:
        p_val = 1.0
    else:
        z = (u_min - mean_u + 0.5) / math.sqrt(var_u)
        p_val = math.erfc(abs(z) / math.sqrt(2.0))

    return {
        "u": float(u_min),
        "p_value": float(p_val),
        "effect_size_r": float(effect_r),
        "method": "numpy_fallback",
    }


def bh_fdr_correct(p_values: list) -> list:
    """Benjamini-Hochberg FDR-adjusted p-values within a family of tests.

    None inputs (insufficient_data tests) are passed through unchanged.
    Returns a list of the same length as p_values.
    """
    m = len(p_values)
    if m == 0:
        return []

    valid_idx = [i for i, p in enumerate(p_values) if p is not None]
    if not valid_idx:
        return list(p_values)

    valid_p = [p_values[i] for i in valid_idx]
    n = len(valid_p)
    order = sorted(range(n), key=lambda i: valid_p[i])

    # Process from largest rank down to smallest, tracking running minimum.
    p_adj_valid = [1.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        bh_p = min(n / rank * valid_p[i], 1.0)
        running_min = min(running_min, bh_p)
        p_adj_valid[i] = running_min

    result: list = [None] * m
    for orig_i, adj_p in zip(valid_idx, p_adj_valid):
        result[orig_i] = adj_p
    return result


def metric_stats(values: list[float]) -> dict:
    if not values:
        return {
            "n_used": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "n_used": int(arr.size),
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
    }


def parse_seed_record(arm: str, seed_dir: Path) -> dict:
    seed_name = seed_dir.name
    seed = to_int(seed_name.replace("seed_", ""), default=None)

    summary_path = seed_dir / "summary.json"
    earth_probe_path = seed_dir / "earth_probe.csv"
    gravity_sweep_path = seed_dir / "gravity_sweep.csv"

    rec = {
        "arm": arm,
        "seed": seed,
        "seed_dir": str(seed_dir),
        "summary_exists": summary_path.exists(),
        "tte_generation": None,
        "final_net_progress_m": None,
        "robustness_mean_net_progress_m": None,
        "best_earth_probe_score": None,
        "mean_earth_probe_score": None,
    }

    summary = None
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = None

    if summary is not None:
        rec["tte_generation"] = to_float(summary.get("tte_generation"))
        final_earth = summary.get("final_earth") or {}
        rec["final_net_progress_m"] = to_float(final_earth.get("net_progress_m"))
        robustness = summary.get("robustness") or {}
        rec["robustness_mean_net_progress_m"] = to_float(
            robustness.get("mean_net_progress_m")
        )
        rec["best_earth_probe_score"] = to_float(
            summary.get("best_earth_probe_score")
        )

    # Read earth probe CSV once — used for mean score and all CSV fallbacks.
    earth_probe_rows = read_csv_rows(earth_probe_path)
    probe_scores: list[float] = []
    if earth_probe_rows:
        probe_scores = [to_float(r.get("score")) for r in earth_probe_rows]
        probe_scores = [s for s in probe_scores if s is not None]
        if probe_scores:
            rec["mean_earth_probe_score"] = float(np.mean(probe_scores))

    # Fallbacks from CSV artifacts.
    if rec["final_net_progress_m"] is None and earth_probe_rows:
        rec["final_net_progress_m"] = to_float(
            earth_probe_rows[-1].get("net_progress_m")
        )

    if rec["robustness_mean_net_progress_m"] is None:
        rows = read_csv_rows(gravity_sweep_path)
        vals = [to_float(r.get("net_progress_m")) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            rec["robustness_mean_net_progress_m"] = float(np.mean(vals))

    # CSV fallback for best_earth_probe_score — works for runs that pre-date
    # the best_earth_probe_score field in summary.json.
    if rec["best_earth_probe_score"] is None and probe_scores:
        rec["best_earth_probe_score"] = float(max(probe_scores))

    return rec


def write_aggregated_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "seed",
        "seed_dir",
        "summary_exists",
        "tte_generation",
        "final_net_progress_m",
        "robustness_mean_net_progress_m",
        "best_earth_probe_score",
        "mean_earth_probe_score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_probe_curves(thesis_root: Path, arms: list) -> list[dict]:
    """Aggregate per-arm, per-generation earth probe scores across seeds.

    Returns rows suitable for writing to aggregated_probe_curves.csv.
    Each row covers one (arm, generation) pair and reports median/Q1/Q3/mean
    across all seeds that have a probe at that generation.
    """
    rows = []
    for arm in arms:
        arm_dir = thesis_root / arm
        if not arm_dir.exists():
            continue
        seed_dirs = sorted(
            [p for p in arm_dir.glob("seed_*") if p.is_dir()],
            key=lambda p: p.name,
        )

        gen_scores: dict = {}
        for sd in seed_dirs:
            for pr in read_csv_rows(sd / "earth_probe.csv"):
                gen = to_int(pr.get("gen"))
                score = to_float(pr.get("score"))
                if gen is not None and score is not None:
                    gen_scores.setdefault(gen, []).append(score)

        for gen in sorted(gen_scores):
            arr = np.asarray(gen_scores[gen], dtype=np.float64)
            rows.append({
                "arm": arm,
                "gen": gen,
                "n_seeds": int(arr.size),
                "median_score": float(np.median(arr)),
                "q1_score": float(np.percentile(arr, 25)),
                "q3_score": float(np.percentile(arr, 75)),
                "mean_score": float(np.mean(arr)),
            })
    return rows


def build_markdown_report(summary: dict) -> str:
    lines = []
    lines.append("# Thesis Aggregation Report")
    lines.append("")
    lines.append(f"Generated at epoch: {summary['created_at_epoch']}")
    lines.append("")

    lines.append("## Completeness")
    lines.append("")
    lines.append("| Arm | Completed seeds | Expected |")
    lines.append("|---|---:|---:|")
    for arm, comp in summary["completeness"].items():
        lines.append(
            f"| {arm} | {comp['completed_seeds']} | {comp['expected_seeds']} |"
        )
    lines.append("")

    lines.append("## Seed-Level Stats (median / Q1 / Q3 / IQR)")
    lines.append("")
    lines.append("| Arm | Metric | n_used | Median | Q1 | Q3 | IQR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for arm, metric_map in summary["arm_stats"].items():
        for metric, st in metric_map.items():
            lines.append(
                f"| {arm} | {metric} | {st['n_used']} | {st['median']} | "
                f"{st['q1']} | {st['q3']} | {st['iqr']} |"
            )
    lines.append("")

    lines.append("## Pairwise Mann-Whitney U (two-sided, BH-FDR corrected)")
    lines.append("")
    lines.append(
        "| Metric | Arm A | Arm B | n_A | n_B | U | p-value | p_adj (BH) | r | Method |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary["pairwise_tests"]:
        p_adj = row.get("p_adj")
        r_val = row.get("effect_size_r")
        p_adj_str = f"{p_adj:.5f}" if p_adj is not None else "None"
        r_str = f"{r_val:.3f}" if r_val is not None else "None"
        lines.append(
            f"| {row['metric']} | {row['arm_a']} | {row['arm_b']} | "
            f"{row['n_a']} | {row['n_b']} | {row['u']} | {row['p_value']} | "
            f"{p_adj_str} | {r_str} | {row['method']} |"
        )
    lines.append("")
    lines.append(
        "_Effect size r: small ≥ 0.1, medium ≥ 0.3, large ≥ 0.5 (Cohen 1988). "
        "p_adj uses Benjamini-Hochberg FDR correction within each metric._"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    thesis_root = Path(args.thesis_root)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    invalid = [a for a in arms if a not in ARMS_DEFAULT]
    if invalid:
        raise ValueError(f"Unknown arms: {invalid}. Allowed: {sorted(ARMS_DEFAULT)}")

    per_seed_rows = []
    completeness = {}

    for arm in arms:
        arm_dir = thesis_root / arm
        if not arm_dir.exists():
            seed_dirs = []
        else:
            seed_dirs = sorted(
                [p for p in arm_dir.glob("seed_*") if p.is_dir()],
                key=lambda p: p.name,
            )

        records = [parse_seed_record(arm, sd) for sd in seed_dirs]
        completed = sum(1 for r in records if r["summary_exists"])
        completeness[arm] = {
            "completed_seeds": completed,
            "expected_seeds": int(args.expected_seeds),
        }

        if (not args.allow_incomplete) and completed < args.expected_seeds:
            raise SystemExit(
                f"Arm '{arm}' has {completed} completed seeds; "
                f"expected {args.expected_seeds}. Use --allow-incomplete to override."
            )

        per_seed_rows.extend(records)

    # Per-arm metric arrays.
    arm_metric_values = {arm: {m: [] for m in METRICS} for arm in arms}
    for row in per_seed_rows:
        arm = row["arm"]
        for metric in METRICS:
            val = to_float(row.get(metric))
            if val is not None:
                arm_metric_values[arm][metric].append(val)

    # Arm-level stats.
    arm_stats = {}
    for arm in arms:
        arm_stats[arm] = {
            metric: metric_stats(vals)
            for metric, vals in arm_metric_values[arm].items()
        }

    # Pairwise Mann-Whitney U tests.
    pairwise = []
    for metric in METRICS:
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a = arms[i]
                b = arms[j]
                x = arm_metric_values[a][metric]
                y = arm_metric_values[b][metric]
                test = mann_whitney_u(x, y)
                pairwise.append({
                    "metric": metric,
                    "arm_a": a,
                    "arm_b": b,
                    "n_a": len(x),
                    "n_b": len(y),
                    "u": test["u"],
                    "p_value": test["p_value"],
                    "effect_size_r": test.get("effect_size_r"),
                    "method": test["method"],
                })

    # Benjamini-Hochberg FDR correction applied within each metric family.
    metric_to_row_indices: dict = {m: [] for m in METRICS}
    for i, row in enumerate(pairwise):
        m = row["metric"]
        if m in metric_to_row_indices:
            metric_to_row_indices[m].append(i)

    for m, row_indices in metric_to_row_indices.items():
        raw_pvals = [pairwise[i]["p_value"] for i in row_indices]
        adj_pvals = bh_fdr_correct(raw_pvals)
        for i, adj_p in zip(row_indices, adj_pvals):
            pairwise[i]["p_adj"] = adj_p

    for row in pairwise:
        row.setdefault("p_adj", None)

    summary = {
        "created_at_epoch": int(time.time()),
        "config": {
            "thesis_root": str(thesis_root),
            "arms": arms,
            "expected_seeds": int(args.expected_seeds),
            "allow_incomplete": bool(args.allow_incomplete),
            "metrics": METRICS,
        },
        "completeness": completeness,
        "arm_stats": arm_stats,
        "pairwise_tests": pairwise,
    }

    csv_path = thesis_root / "aggregated_metrics.csv"
    json_path = thesis_root / "aggregated_summary.json"
    md_path = thesis_root / "stats_report.md"
    probe_curve_path = thesis_root / "aggregated_probe_curves.csv"

    write_aggregated_csv(csv_path, per_seed_rows)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(md_path, "w") as f:
        f.write(build_markdown_report(summary))

    probe_curve_rows = build_probe_curves(thesis_root, arms)
    probe_curve_fields = [
        "arm", "gen", "n_seeds",
        "median_score", "q1_score", "q3_score", "mean_score",
    ]
    with open(probe_curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=probe_curve_fields)
        writer.writeheader()
        for r in probe_curve_rows:
            writer.writerow(r)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {probe_curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
