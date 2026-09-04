#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable


V7_0_BASELINE_PER_SEED: dict[str, float] = {
    "13": 1.0,
    "42": 1.0,
    "137": 1.0,
}

PARITY_TOLERANCE = 0.02
K_RESCUE = 10
REQUIRED_COLS: frozenset[str] = frozenset({
    "probe_id", "seed", "n_slice", "route", "cue_hash",
    "pipeline_rank", "pipeline_hit_at_k",
})


def _is_hit_at_k(rank_str: str, k: int = K_RESCUE) -> bool:
    try:
        r = int(rank_str)
    except (TypeError, ValueError):
        return False
    return 0 < r <= k


def compute_per_route_rescue_at_k(
    rows: Iterable[dict[str, str]],
    k: int = K_RESCUE,
) -> dict[str, dict[str, float]]:
    attributable = [
        r for r in rows
        if r.get("route") in ("efe_real", "efe_shadow")
    ]
    by_seed_route: dict[tuple[str, str], list[bool]] = {}
    for r in attributable:
        key = (str(r.get("seed", "")), r["route"])
        by_seed_route.setdefault(key, []).append(
            _is_hit_at_k(r.get("pipeline_rank", ""), k)
        )
    out: dict[str, dict[str, float]] = {}
    for (seed, route), hits in by_seed_route.items():
        out.setdefault(seed, {})[route] = (
            sum(hits) / len(hits) if hits else 0.0
        )
    return out


def build_parity_summary(
    per_seed: dict[str, dict[str, float]],
    n_attributable: int,
    tolerance: float = PARITY_TOLERANCE,
) -> dict:
    per_seed_block: dict[str, dict[str, float]] = {}
    deltas: list[float] = []
    shadow_rescues: list[float] = []
    per_seed_pass: dict[str, bool] = {}

    for seed_key in sorted(per_seed.keys()):
        by_route = per_seed[seed_key]
        real = float(by_route.get("efe_real", 0.0))
        shadow = float(by_route.get("efe_shadow", 0.0))
        delta = real - shadow
        per_seed_block[seed_key] = {
            "efe_real_rescue": real,
            "efe_shadow_rescue": shadow,
            "delta": delta,
        }
        deltas.append(delta)
        shadow_rescues.append(shadow)

        baseline = V7_0_BASELINE_PER_SEED.get(seed_key)
        if baseline is not None:
            per_seed_pass[seed_key] = abs(shadow - baseline) <= tolerance

    mean_shadow = statistics.fmean(shadow_rescues) if shadow_rescues else 0.0
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    baseline_mean = (
        sum(V7_0_BASELINE_PER_SEED.values()) / len(V7_0_BASELINE_PER_SEED)
    )
    cross_seed_pass = abs(mean_shadow - baseline_mean) <= tolerance
    all_per_seed_pass = all(per_seed_pass.values()) if per_seed_pass else False
    verdict = (
        "PARITY_PASS"
        if (all_per_seed_pass and cross_seed_pass)
        else "PARITY_FAIL"
    )

    return {
        "backend": "leiden-custom",
        "seeds": sorted(int(s) for s in per_seed_block.keys()),
        "n_recalls": int(n_attributable),
        "per_seed": per_seed_block,
        "cross_seed_mean_rescue": mean_shadow,
        "cross_seed_mean_delta": mean_delta,
        "baseline_v7_0": dict(V7_0_BASELINE_PER_SEED),
        "parity_gate": {
            "tolerance": tolerance,
            "per_seed_pass": per_seed_pass,
            "cross_seed_pass": cross_seed_pass,
            "verdict": verdict,
        },
    }


def build_markdown(
    summary: dict, csv_name: str, bench_duration_s: float | None = None,
) -> str:
    lines = [
        "# Parity Summary: v7.1 custom_leiden vs v7.0 leidenalg",
        "",
        "**Phase:** 21 — Custom MIT-Licensed Leiden Implementation (M8)",
        "**Bench:** Contradiction-longitudinal Regime 2 "
        "(`bench/contradiction_longitudinal_claude.py`)",
        f"**CSV:** `{csv_name}`",
        "**Gate:** Rescue@10 within ±0.02 of v7.0 baseline (1.000)",
        f"**Backend measured:** `{summary['backend']}`",
        f"**Attributable recalls (n_recalls):** {summary['n_recalls']}",
        f"**Seeds:** {summary['seeds']}",
        "",
        "## Per-Seed Rescue@10",
        "",
        "| Seed | v7.0 (leidenalg) | v7.1 (custom_leiden) | Δ | Within ±0.02? |",
        "|------|------------------|----------------------|---|---------------|",
    ]
    for seed_key in sorted(summary["per_seed"].keys()):
        seed_block = summary["per_seed"][seed_key]
        v70 = V7_0_BASELINE_PER_SEED.get(seed_key, float("nan"))
        v71 = float(seed_block["efe_shadow_rescue"])
        delta = v71 - v70
        passed = summary["parity_gate"]["per_seed_pass"].get(seed_key, False)
        lines.append(
            f"| {seed_key} | {v70:.4f} | {v71:.4f} | {delta:+.4f} | "
            f"{'YES' if passed else 'NO'} |"
        )

    lines += [
        "",
        "## Cross-Seed Mean",
        "",
        "| | v7.0 | v7.1 | Δ |",
        "|---|------|------|---|",
    ]
    baseline_mean = (
        sum(V7_0_BASELINE_PER_SEED.values()) / len(V7_0_BASELINE_PER_SEED)
    )
    v71_mean = float(summary["cross_seed_mean_rescue"])
    delta_mean = v71_mean - baseline_mean
    lines += [
        f"| Rescue@10 mean | {baseline_mean:.4f} | {v71_mean:.4f} | "
        f"{delta_mean:+.4f} |",
        "",
        f"**PARITY GATE:** {summary['parity_gate']['verdict']}",
        "",
    ]

    if bench_duration_s is not None:
        baseline_duration_s = 10683.17
        delta_pct = (
            (bench_duration_s - baseline_duration_s) / baseline_duration_s * 100.0
            if baseline_duration_s > 0
            else 0.0
        )
        lines += [
            "## Wall-Time",
            "",
            "| | Seconds | Δ vs v7.0 |",
            "|---|---------|-----------|",
            f"| v7.0 (leidenalg, iteration-3) | {baseline_duration_s:.1f} | — |",
            f"| v7.1 (custom_leiden, iteration-0) | {bench_duration_s:.1f} | "
            f"{delta_pct:+.1f}% |",
            "",
            "Wall-time is informational only. Final perf disposition for LEIDEN-09 "
            "is measured AFTER `python-igraph` + `leidenalg` extras are uninstalled "
            "(GPL removal step). The bench above still has those extras installed, "
            "so `graph.py:117-126` C-backed betweenness is active.",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- v7.0 baseline source: `bench/results/v7.0/iteration-3/EFE-AB-SUMMARY.json`",
        "  (per-seed `efe_shadow_rescue`; cross-seed mean = 1.000).",
        "- Parity tolerance + Risk row "
        "'Retrieval Rescue@10 regression > 0.02'.",
        "- GPL removal is GATED on `PARITY_PASS`.",
        "",
    ]
    return "\n".join(lines)


def _find_newest_csv(results_dir: Path) -> Path:
    csvs = sorted(
        results_dir.rglob("contradiction_longitudinal_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csvs:
        raise FileNotFoundError(
            f"no contradiction_longitudinal_*.csv under {results_dir}"
        )
    return csvs[0]


def _read_bench_duration_from_json(csv_path: Path) -> float | None:
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
        env = data.get("env", {})
        return float(env.get("wall_clock_duration_seconds")) if env.get(
            "wall_clock_duration_seconds"
        ) is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parity-summary post-processor. Reads bench CSV, "
            "writes contradiction_longitudinal_custom_leiden.json + "
            "PARITY-SUMMARY.md. Exit 0 = PARITY_PASS, 1 = PARITY_FAIL, "
            "2 = setup error."
        ),
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Directory containing the bench CSV; newest wins.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Explicit CSV path (overrides results-dir search).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=PARITY_TOLERANCE,
        help=f"Parity tolerance (default {PARITY_TOLERANCE}).",
    )
    args = parser.parse_args(argv)

    try:
        csv_path = args.csv if args.csv else _find_newest_csv(args.results_dir)
    except FileNotFoundError as e:
        print(f"[make_parity_summary] setup error: {e}", file=sys.stderr)
        return 2

    if not csv_path.exists():
        print(
            f"[make_parity_summary] setup error: CSV path does not exist: {csv_path}",
            file=sys.stderr,
        )
        return 2

    print(f"[make_parity_summary] reading {csv_path}", file=sys.stderr)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - fieldnames
        if missing:
            print(
                f"[make_parity_summary] setup error: CSV missing required "
                f"columns: {sorted(missing)}",
                file=sys.stderr,
            )
            return 2
        rows = list(reader)

    per_seed = compute_per_route_rescue_at_k(rows)

    all_routes_seen = {
        route for by_route in per_seed.values() for route in by_route
    }
    for arm in ("efe_real", "efe_shadow"):
        if arm not in all_routes_seen:
            print(
                f"[make_parity_summary] setup error: missing {arm} rows",
                file=sys.stderr,
            )
            return 2

    n_attributable = sum(
        1 for r in rows if r.get("route") in ("efe_real", "efe_shadow")
    )
    bench_duration_s = _read_bench_duration_from_json(csv_path)
    summary = build_parity_summary(per_seed, n_attributable, args.tolerance)

    out_dir = csv_path.parent
    out_json = out_dir / "contradiction_longitudinal_custom_leiden.json"
    out_md = out_dir / "PARITY-SUMMARY.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(build_markdown(summary, csv_path.name, bench_duration_s))

    print(f"[make_parity_summary] wrote {out_json}", file=sys.stderr)
    print(f"[make_parity_summary] wrote {out_md}", file=sys.stderr)

    verdict = summary["parity_gate"]["verdict"]
    print(
        f"PARITY: cross_seed_mean_rescue={summary['cross_seed_mean_rescue']:.4f} "
        f"(baseline=1.000, tol=+/-{args.tolerance:.2f}) -> {verdict}"
    )
    return 0 if verdict == "PARITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
