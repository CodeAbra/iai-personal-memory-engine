#!/usr/bin/env python3

from __future__ import annotations


import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

SHIP_GATE_THRESHOLD = 0.05
K_RESCUE = 10
REQUIRED_COLS: frozenset[str] = frozenset({
    "probe_id", "seed", "arousal_route",
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
        if r.get("arousal_route") in ("arousal_real", "arousal_shadow")
    ]
    by_seed_route: dict[tuple[str, str], list[bool]] = {}
    for r in attributable:
        key = (str(r.get("seed", "")), r["arousal_route"])
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
        real = by_route.get("arousal_real", 0.0)
        shadow = by_route.get("arousal_shadow", 0.0)
        d = real - shadow
        per_seed_delta[seed] = {
            "arousal_real_rescue": real,
            "arousal_shadow_rescue": shadow,
            "delta": d,
        }
        deltas.append(d)
    mean_d = statistics.fmean(deltas) if deltas else 0.0
    if mean_d >= threshold:
        verdict = "keep"
    elif mean_d <= -threshold:
        verdict = "remove"
    else:
        verdict = "consilium-resolve"
    return {
        "per_seed": per_seed_delta,
        "cross_seed_mean_delta": mean_d,
        "ship_gate_hit": mean_d >= threshold,
        "threshold": threshold,
        "verdict": verdict,
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
    route_counts: dict[str, int],
) -> str:
    verdict = summary["verdict"]
    verdict_label = {
        "keep": "KEEP (delta >= +threshold)",
        "remove": "REMOVE (delta <= -threshold)",
        "consilium-resolve": "CONSILIUM-RESOLVE (delta in band)",
    }[verdict]
    lines = [
        "# arousal_budget A/B Summary",
        "",
        f"- CSV: `{csv_name}`",
        f"- Rows (attributable): {n_attributable}",
        f"- Seeds: {n_seeds}",
        f"- Threshold: +/- {threshold:.2f}",
        f"- Route counts: arousal_real={route_counts.get('arousal_real', 0)}, "
        f"arousal_shadow={route_counts.get('arousal_shadow', 0)}, "
        f"arousal_skip={route_counts.get('arousal_skip', 0)}",
        "",
        "| Seed | arousal_real Rescue@10 | arousal_shadow Rescue@10 | Delta |",
        "|------|------------------------|--------------------------|-------|",
    ]
    for seed in sorted(summary["per_seed"].keys()):
        d = summary["per_seed"][seed]
        lines.append(
            f"| {seed} | {d['arousal_real_rescue']:.3f} | "
            f"{d['arousal_shadow_rescue']:.3f} | {d['delta']:+.3f} |"
        )
    lines += [
        "",
        f"**Cross-seed mean delta:** {summary['cross_seed_mean_delta']:+.3f}",
        f"**Ship gate ({'HIT' if summary['ship_gate_hit'] else 'MISS'}):** "
        f"{summary['cross_seed_mean_delta']:+.3f} vs +{threshold:.2f}",
        f"**Verdict:** {verdict_label}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ship-gate analyzer. Reads a "
            "contradiction_longitudinal_*.csv and emits AROUSAL-AB-SUMMARY.{json,md} "
            "next to it. Exit 0 = ship gate hit (keep), 1 = miss (remove or "
            "consilium-resolve), 2 = setup error."
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
        print(f"[analyze_arousal_ab] setup error: {e}", file=sys.stderr)
        return 2

    if not csv_path.exists():
        print(
            f"[analyze_arousal_ab] setup error: CSV path does not exist: {csv_path}",
            file=sys.stderr,
        )
        return 2

    print(f"[analyze_arousal_ab] reading {csv_path}", file=sys.stderr)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - fieldnames
        if missing:
            print(
                f"[analyze_arousal_ab] setup error: CSV missing required columns: "
                f"{sorted(missing)}",
                file=sys.stderr,
            )
            return 2
        rows = list(reader)

    per_seed = compute_per_route_rescue_at_k(rows)

    all_routes_seen = {
        route for by_route in per_seed.values() for route in by_route
    }
    for arm in ("arousal_real", "arousal_shadow"):
        if arm not in all_routes_seen:
            print(
                f"[analyze_arousal_ab] setup error: missing {arm} rows",
                file=sys.stderr,
            )
            return 2

    agg = aggregate_across_seeds(per_seed, threshold=args.threshold)
    route_counts: dict[str, int] = {}
    for r in rows:
        ar = r.get("arousal_route", "")
        if ar:
            route_counts[ar] = route_counts.get(ar, 0) + 1
    n_attributable = sum(
        1 for r in rows
        if r.get("arousal_route") in ("arousal_real", "arousal_shadow")
    )
    n_seeds = len(per_seed)

    summary: dict = {
        "phase": "25",
        "plan": "14-01",
        "csv_path": str(csv_path),
        "n_rows": n_attributable,
        "n_seeds": n_seeds,
        "n_attributable_rows": n_attributable,
        "route_counts": route_counts,
        **agg,
    }

    out_dir = csv_path.parent
    (out_dir / "AROUSAL-AB-SUMMARY.json").write_text(
        json.dumps(summary, indent=2)
    )
    (out_dir / "AROUSAL-AB-SUMMARY.md").write_text(
        _build_markdown(
            summary, csv_path.name, n_attributable, n_seeds, args.threshold,
            route_counts,
        )
    )

    real_vals = [d["arousal_real_rescue"] for d in summary["per_seed"].values()]
    shadow_vals = [d["arousal_shadow_rescue"] for d in summary["per_seed"].values()]
    real_mean = statistics.fmean(real_vals) if real_vals else 0.0
    shadow_mean = statistics.fmean(shadow_vals) if shadow_vals else 0.0
    verdict = summary["verdict"]
    print(
        f"AROUSAL-AB: cross_seed_mean_delta={summary['cross_seed_mean_delta']:+.3f} "
        f"(real={real_mean:.3f} shadow={shadow_mean:.3f}) "
        f"verdict={verdict} (threshold=+/-{args.threshold:.2f})"
    )
    return 0 if summary["ship_gate_hit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
