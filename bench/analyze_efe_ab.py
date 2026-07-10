#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

SHIP_GATE_THRESHOLD = 0.10
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


def aggregate_across_seeds(
    per_seed: dict[str, dict[str, float]],
    threshold: float = SHIP_GATE_THRESHOLD,
) -> dict:
    per_seed_delta: dict[str, dict[str, float]] = {}
    deltas: list[float] = []
    for seed, by_route in per_seed.items():
        real = by_route.get("efe_real", 0.0)
        shadow = by_route.get("efe_shadow", 0.0)
        d = real - shadow
        per_seed_delta[seed] = {
            "efe_real_rescue": real,
            "efe_shadow_rescue": shadow,
            "delta": d,
        }
        deltas.append(d)
    mean_d = statistics.fmean(deltas) if deltas else 0.0
    return {
        "per_seed": per_seed_delta,
        "cross_seed_mean_delta": mean_d,
        "ship_gate_hit": mean_d >= threshold,
        "threshold": threshold,
    }


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


def _build_markdown(
    summary: dict,
    csv_name: str,
    n_attributable: int,
    n_seeds: int,
    threshold: float,
) -> str:
    lines = [
        "# EFE A/B Summary",
        "",
        f"- CSV: `{csv_name}`",
        f"- Rows (attributable): {n_attributable}",
        f"- Seeds: {n_seeds}",
        f"- Threshold: +{threshold:.2f}",
        "",
        "| Seed | efe_real Rescue@10 | efe_shadow Rescue@10 | Delta |",
        "|------|--------------------|----------------------|-------|",
    ]
    for seed in sorted(summary["per_seed"].keys()):
        d = summary["per_seed"][seed]
        lines.append(
            f"| {seed} | {d['efe_real_rescue']:.3f} | "
            f"{d['efe_shadow_rescue']:.3f} | {d['delta']:+.3f} |"
        )
    verdict = "PASS" if summary["ship_gate_hit"] else "FAIL"
    comparator = ">=" if summary["ship_gate_hit"] else "<"
    lines += [
        "",
        f"**Cross-seed mean delta:** {summary['cross_seed_mean_delta']:+.3f}  ",
        f"**Ship gate ({verdict}):** "
        f"{summary['cross_seed_mean_delta']:+.3f} "
        f"{comparator} +{threshold:.2f}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ship-gate analyzer. Reads a "
            "contradiction_longitudinal_*.csv and emits EFE-AB-SUMMARY.{json,md} "
            "next to it. Exit 0 = ship gate hit, 1 = miss, 2 = setup error."
        ),
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Directory containing bench CSV(s); newest wins.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Explicit CSV path (overrides results-dir search).",
    )
    parser.add_argument(
        "--threshold", type=float, default=SHIP_GATE_THRESHOLD,
        help=f"Ship-gate threshold (default {SHIP_GATE_THRESHOLD}).",
    )
    args = parser.parse_args(argv)

    try:
        csv_path = args.csv if args.csv else _find_newest_csv(args.results_dir)
    except FileNotFoundError as e:
        print(f"[analyze_efe_ab] setup error: {e}", file=sys.stderr)
        return 2

    if not csv_path.exists():
        print(
            f"[analyze_efe_ab] setup error: CSV path does not exist: {csv_path}",
            file=sys.stderr,
        )
        return 2

    print(f"[analyze_efe_ab] reading {csv_path}", file=sys.stderr)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - fieldnames
        if missing:
            print(
                f"[analyze_efe_ab] setup error: CSV missing required columns: "
                f"{sorted(missing)}",
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
                f"[analyze_efe_ab] setup error: missing {arm} rows",
                file=sys.stderr,
            )
            return 2

    agg = aggregate_across_seeds(per_seed, threshold=args.threshold)
    n_attributable = sum(
        1 for r in rows if r.get("route") in ("efe_real", "efe_shadow")
    )
    n_seeds = len(per_seed)

    summary: dict = {
        "csv_path": str(csv_path),
        "n_rows": n_attributable,
        "n_seeds": n_seeds,
        **agg,
    }

    out_dir = csv_path.parent
    (out_dir / "EFE-AB-SUMMARY.json").write_text(
        json.dumps(summary, indent=2)
    )
    (out_dir / "EFE-AB-SUMMARY.md").write_text(
        _build_markdown(
            summary, csv_path.name, n_attributable, n_seeds, args.threshold,
        )
    )

    verdict = "PASS" if summary["ship_gate_hit"] else "FAIL"
    real_vals = [d["efe_real_rescue"] for d in summary["per_seed"].values()]
    shadow_vals = [d["efe_shadow_rescue"] for d in summary["per_seed"].values()]
    real_mean = statistics.fmean(real_vals) if real_vals else 0.0
    shadow_mean = statistics.fmean(shadow_vals) if shadow_vals else 0.0
    print(
        f"EFE-AB: cross_seed_mean_delta={summary['cross_seed_mean_delta']:+.3f} "
        f"(real={real_mean:.3f} shadow={shadow_mean:.3f}) "
        f"{verdict} (threshold=+{args.threshold:.2f})"
    )
    return 0 if summary["ship_gate_hit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
