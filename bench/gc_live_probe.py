"""Live-daemon warm-repeated-call recall latency probe (read-only).

Usage (against the live daemon socket, NEVER a restart):

    source .venv/bin/activate
    python bench/gc_live_probe.py --dry-check --n 5
    python bench/gc_live_probe.py --n 500

Method: an in-process JSON-RPC round trip against the live daemon socket --
the same call ``iai recall`` makes -- over a pinned 18-cue, 3-band
natural-language cue set (specific / vague / novel), issued in randomized
round-robin order (the cue list is shuffled once per full cycle with a
fixed seed, so no contiguous run of samples is drawn from a single cue --
a cue-homogeneous window inflates the tail and can misfire a threshold
check near the pass/fail bar). Every requested sample is kept; none are
discarded for being slow. The reported confidence band comes from
bootstrap-resampling the full pooled sample (not from contiguous
sub-windows, which would reintroduce the same cue-homogeneity problem the
round-robin order avoids).

``--dry-check`` proves the probe is wired correctly (health check, N warm
read-only recalls, band + verdict printed) without ever restarting the
daemon -- this script has no code path that starts, stops, or restarts
anything, and never writes to the store.

Privacy: only cue text (authored by this probe, none of it the owner's
stored content), hit counts, and timings are ever read or printed --
recalled record content is never retained or printed.
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics
import sys
import time
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

ABSOLUTE_THRESHOLD_MS = 99.0
DEFAULT_N = 500
MIN_POWERED_N = 100
BOOTSTRAP_RESAMPLES = 1000
BAND_LOW_PCT = 5.0
BAND_HIGH_PCT = 95.0
SHUFFLE_SEED = 1337
BOOTSTRAP_SEED = 4242
SESSION_ID = "gc-live-probe"
# Matches the established live-probe methodology's confound-controlled run
# (isolates the wall-clock number from a request-size confound).
BUDGET_TOKENS = 900

# Pinned 18-cue, 3-band real-corpus cue set: specific factual recall,
# vague/thematic recall, and genuinely novel/off-topic recall. Do not
# substitute -- these correspond to resident content and were established
# for exactly this kind of live-daemon measurement.
CUE_BANDS: "dict[str, list[str]]" = {
    "specific": [
        "what did we decide about the release gate command",
        "why did the b-tree delete bug take so many nights to fix",
        "how does the generational cache avoid mutating shared state",
        "what is the sealed knob count in the profile registry",
        "why was the persist-graph caching approach abandoned",
        "what does the double empathy invariant forbid",
    ],
    "vague": [
        "recent latency work",
        "escalation profiling investigation",
        "storage engine hardening effort",
        "sleep pipeline consolidation steps",
        "memory recall ranking improvements",
        "daemon lifecycle and watchdog behavior",
    ],
    "novel": [
        "quantum error correction thresholds",
        "sourdough hydration ratio",
        "the history of byzantine architecture",
        "optimal chess opening theory for the sicilian defense",
        "migratory patterns of arctic terns",
        "the physics of neutron star mergers",
    ],
}


def _flatten_cues() -> "list[tuple[str, str]]":
    flat: "list[tuple[str, str]]" = []
    for band, cues in CUE_BANDS.items():
        for cue in cues:
            flat.append((band, cue))
    return flat


def _round_robin_order(n: int, seed: int = SHUFFLE_SEED) -> "list[tuple[str, str]]":
    """Randomized round-robin: every full pass through the cue set is
    reshuffled with a fixed, cycle-indexed seed (reproducible), so no
    sub-window of the returned order is cue-homogeneous."""
    cues = _flatten_cues()
    order: "list[tuple[str, str]]" = []
    cycle = 0
    while len(order) < n:
        rng = random.Random(seed + cycle)
        shuffled = cues[:]
        rng.shuffle(shuffled)
        order.extend(shuffled)
        cycle += 1
    return order[:n]


def _health_check() -> "dict | None":
    from iai_mcp.cli import _send_socket_request

    return _send_socket_request({"type": "status"}, timeout=10.0)


def _recall_once(cue: str) -> "tuple[float, dict | None]":
    """One warm recall call. Returns (client-side wall-clock ms, response
    dict or None on transport failure)."""
    from iai_mcp.cli import _send_jsonrpc_request

    t0 = time.perf_counter()
    resp = _send_jsonrpc_request(
        "memory_recall",
        {"cue": cue, "session_id": SESSION_ID, "budget_tokens": BUDGET_TOKENS},
        read_timeout=30.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = None
    if isinstance(resp, dict) and isinstance(resp.get("result"), dict):
        result = resp["result"]
    return elapsed_ms, result


def _p95(samples: "list[float]") -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    return statistics.quantiles(samples, n=100, method="inclusive")[94]


def _bootstrap_band(
    samples: "list[float]",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    lo_pct: float = BAND_LOW_PCT,
    hi_pct: float = BAND_HIGH_PCT,
) -> "tuple[float, float]":
    """Bootstrap the p95 statistic over the full pooled sample (with
    replacement), never discarding a sample and never slicing a contiguous
    sub-window. Returns (band_lower, band_upper) at the given percentiles
    of the resampled p95 distribution."""
    rng = random.Random(seed)
    n = len(samples)
    if n < 2:
        p = _p95(samples)
        return p, p
    boot_p95s = []
    for _ in range(resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        boot_p95s.append(_p95(resample))
    boot_p95s.sort()
    last = len(boot_p95s) - 1
    lo_idx = max(0, min(last, round((lo_pct / 100.0) * last)))
    hi_idx = max(0, min(last, round((hi_pct / 100.0) * last)))
    return boot_p95s[lo_idx], boot_p95s[hi_idx]


# The four fat candidate-assembly buckets, each defined as the cumulative-ms
# delta between two consecutive IAI_MCP_RECALL_TRACE marks (already wired,
# needs no daemon restart). "soft_gate" only fires its lower bound
# ("cleanup_attractor") on the rust-scorer live path, which is what the
# daemon dispatch always requests.
MACRO_BUCKET_MARK_PAIRS: "dict[str, tuple[str, str]]" = {
    "ann": ("encode", "ann"),
    "hops_tail": ("hop2_fetch", "hops"),
    "graph_edges": ("hops", "graph_edges"),
    "soft_gate": ("cleanup_attractor", "soft_gate"),
}

# Sub-stage keys this phase's 268-01 plan adds to IAI_MCP_STAGE_PROFILE's
# _stage_timings (pipeline.py / core/__init__.py). None of these currently
# reach the JSON-RPC response (see 268-01-SUMMARY.md's wiring-gap note) --
# aggregation degrades to "absent" until that gap is closed.
SOFT_GATE_SUBSTAGE_KEYS = ("rank", "t11_t12")
KNOWN_SUBSTAGE_KEYS = (
    "rank", "t11_t12", "ann_scan", "ann_inlist", "ann_decode",
    "ann_rows_fetched", "ann_rows_served", "ge_populate", "ge_incident",
    "ge_split", "ge_contr_fetch", "hops_snapshot",
)


def _macro_bucket_ms(trace_spans: "list | None") -> "dict[str, float]":
    """Delta-ms per fat bucket from one call's _recall_trace_ms span list.
    Empty dict when trace_spans is missing/empty or a required mark pair
    did not both fire (e.g. the reference scoring path, not the live
    dispatch's rust-scorer path)."""
    if not trace_spans:
        return {}
    cum: "dict[str, float]" = {}
    for entry in trace_spans:
        try:
            name, ms = entry[0], float(entry[1])
        except (IndexError, TypeError, ValueError):
            continue
        cum[name] = ms
    out: "dict[str, float]" = {}
    for bucket, (start_mark, end_mark) in MACRO_BUCKET_MARK_PAIRS.items():
        if start_mark in cum and end_mark in cum:
            out[bucket] = cum[end_mark] - cum[start_mark]
    return out


def _distribution_table(samples_by_key: "dict[str, list[float]]") -> "list[tuple[str, int, float, float]]":
    """(key, n, p50, p95) rows, sorted by descending p95 -- the tail that
    decides pass/fail, not the mean."""
    rows: "list[tuple[str, int, float, float]]" = []
    for key, vals in samples_by_key.items():
        if not vals:
            continue
        rows.append((key, len(vals), statistics.median(vals), _p95(vals)))
    rows.sort(key=lambda r: -r[3])
    return rows


def _gc_snapshot() -> "list[dict]":
    return gc.get_stats()


def _gc_delta(before: "list[dict]", after: "list[dict]") -> "dict[str, int]":
    """Per-generation collection-count delta -- a GC pause landing inside the
    measured batch shows up here as a nonzero count for that generation,
    correlatable against the p95/p50 tail-gap hypothesis (Q6)."""
    delta: "dict[str, int]" = {}
    for gen, (b, a) in enumerate(zip(before, after)):
        delta[f"gen{gen}_collections"] = int(a.get("collections", 0)) - int(b.get("collections", 0))
        delta[f"gen{gen}_collected"] = int(a.get("collected", 0)) - int(b.get("collected", 0))
    return delta


def _write_events_delta(before: "dict | None", after: "dict | None") -> "int | None":
    """Diffs a monotonic write/telemetry counter off two `status` calls, if
    the daemon exposes one. As of this probe's writing it does not (verified
    against the status handler) -- returns None rather than inventing a new
    daemon-side counter surface; callers must print this as a named
    limitation, not a silent zero."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    for key in ("writes_total", "write_count", "records_written_total"):
        if key in before and key in after:
            try:
                return int(after[key]) - int(before[key])
            except (TypeError, ValueError):
                return None
    return None


def _verdict(band_lower: float, band_upper: float, threshold: float = ABSOLUTE_THRESHOLD_MS) -> str:
    """Absolute-threshold rule on the arm's own band -- never a
    before/after delta. PASS requires the band entirely below the
    threshold with margin exceeding the band's own width; a band
    straddling or sitting at/above the threshold is inconclusive/fail
    respectively."""
    width = band_upper - band_lower
    if band_lower >= threshold:
        return "FAIL"
    if band_upper < threshold and (threshold - band_upper) > width:
        return "PASS"
    return "STRADDLE"


def run(n: int, dry_check: bool) -> int:
    health = _health_check()
    if not isinstance(health, dict) or not health.get("ok"):
        print("daemon not healthy (or not running) -- probe stops here, no start/restart attempted")
        return 1
    heartbeat = health.get("last_tick_at")
    print(f"health: ok=True heartbeat={heartbeat} fsm_state={health.get('fsm_state')}")

    if dry_check:
        print(f"DRY-CHECK mode: {n} warm read-only recalls, no restart")

    order = _round_robin_order(n)

    # One discarded warm-up call per cue so cold-build cost never lands on
    # a measured sample.
    for _band, cue in _flatten_cues():
        _recall_once(cue)

    write_events_before = _health_check()
    gc_before = _gc_snapshot()

    samples: "list[float]" = []
    per_cue_capture: "list[dict]" = []
    failures = 0
    hydrate_stage_available = False
    for band, cue in order:
        elapsed_ms, result = _recall_once(cue)
        if result is None:
            failures += 1
            continue
        samples.append(elapsed_ms)
        capture = {
            "band": band,
            "client_wall_ms": round(elapsed_ms, 2),
            "server_recall_latency_ms": result.get("_recall_latency_ms"),
            "ann_path_used": result.get("ann_path_used"),
            "structural_source": result.get("_structural_source"),
            "cue_mode": result.get("cue_mode"),
            "hit_count": len(result.get("hits") or []),
            "macro_buckets_ms": _macro_bucket_ms(result.get("_recall_trace_ms")),
        }
        if "_stage_timings" in result or "hydrate_stage_timings" in result:
            hydrate_stage_available = True
            capture["_stage_timings"] = result.get("_stage_timings") or result.get("hydrate_stage_timings")
        per_cue_capture.append(capture)

    gc_after = _gc_snapshot()
    write_events_after = _health_check()

    actual_n = len(samples)
    print(f"requested n={n} actual_n={actual_n} failures={failures}")
    if not hydrate_stage_available:
        print(
            "note: per-call sub-stage timing buckets (rank/t11_t12/ann_scan/ge_populate/"
            "hops_snapshot/etc, IAI_MCP_STAGE_PROFILE) are not exposed on the memory_recall "
            "JSON-RPC response surface -- this is a genuine wiring gap (the module-global "
            "_last_stage_timings_ms in pipeline.py is never attached to the response), not "
            "an env-flag issue; see 268-01-SUMMARY.md. Captured instead: server-reported "
            "_recall_latency_ms, ann_path_used, _structural_source, cue_mode, and the four "
            "fat-bucket totals derived from IAI_MCP_RECALL_TRACE (already wired, no restart "
            "needed)."
        )

    gc_delta = _gc_delta(gc_before, gc_after)
    write_delta = _write_events_delta(write_events_before, write_events_after)
    print(
        "gc pauses this batch: "
        + ", ".join(f"{k}={v}" for k, v in gc_delta.items())
    )
    if write_delta is None:
        print(
            "write-events counter: unavailable (daemon `status` response exposes no "
            "monotonic write/telemetry counter field -- named limitation, no new daemon "
            "surface added by this probe)"
        )
    else:
        print(f"write-events counter delta: {write_delta}")

    if actual_n == 0:
        print("no valid samples collected -- cannot compute a band")
        return 1

    p50 = statistics.median(samples)
    p95 = _p95(samples)
    band_lower, band_upper = _bootstrap_band(samples)
    width = band_upper - band_lower
    verdict = _verdict(band_lower, band_upper)
    if actual_n < MIN_POWERED_N:
        verdict = f"UNDEFINED (n={actual_n} below the {MIN_POWERED_N}-sample floor for a defensible band)"

    print(f"p50={p50:.1f}ms p95={p95:.1f}ms")
    print(
        f"bootstrap band [{BAND_LOW_PCT:.0f}-{BAND_HIGH_PCT:.0f} pct of "
        f"{BOOTSTRAP_RESAMPLES} resamples] = [{band_lower:.1f}, {band_upper:.1f}]ms "
        f"width={width:.1f}ms"
    )
    print(f"verdict (absolute {ABSOLUTE_THRESHOLD_MS:.0f}ms threshold): {verdict}")

    for band in CUE_BANDS:
        band_samples = [c["client_wall_ms"] for c in per_cue_capture if c["band"] == band]
        if band_samples:
            print(
                f"  band={band:8s} n={len(band_samples):4d} "
                f"p50={statistics.median(band_samples):.1f}ms p95={_p95(band_samples):.1f}ms"
            )

    macro_by_bucket: "dict[str, list[float]]" = {k: [] for k in MACRO_BUCKET_MARK_PAIRS}
    for c in per_cue_capture:
        for bucket, ms in (c.get("macro_buckets_ms") or {}).items():
            macro_by_bucket.setdefault(bucket, []).append(ms)
    macro_rows = _distribution_table(macro_by_bucket)
    if macro_rows:
        print("\nfat-bucket distributions (from IAI_MCP_RECALL_TRACE marks, ms):")
        for key, cnt, p50v, p95v in macro_rows:
            print(f"  {key:14s} n={cnt:4d} p50={p50v:7.1f}ms p95={p95v:7.1f}ms")
    else:
        print(
            "\nfat-bucket distributions: none captured (daemon is not running with "
            "IAI_MCP_RECALL_TRACE=1, or the rust-scorer live-dispatch marks did not fire)"
        )

    substage_by_key: "dict[str, list[float]]" = {k: [] for k in KNOWN_SUBSTAGE_KEYS}
    for c in per_cue_capture:
        timings = c.get("_stage_timings") or {}
        for key in KNOWN_SUBSTAGE_KEYS:
            if key in timings:
                try:
                    substage_by_key[key].append(float(timings[key]))
                except (TypeError, ValueError):
                    pass
    substage_rows = _distribution_table(substage_by_key)
    if substage_rows:
        print("\nsub-stage distributions (IAI_MCP_STAGE_PROFILE, ms):")
        for key, cnt, p50v, p95v in substage_rows:
            print(f"  {key:16s} n={cnt:4d} p50={p50v:7.2f}ms p95={p95v:7.2f}ms")
        ann_fetched = substage_by_key.get("ann_rows_fetched") or []
        ann_served = substage_by_key.get("ann_rows_served") or []
        if ann_fetched and ann_served:
            fetched_med = statistics.median(ann_fetched)
            served_med = statistics.median(ann_served)
            ratio = fetched_med / served_med if served_med else float("nan")
            print(
                f"  ann over-decode ratio (median fetched/served): "
                f"{fetched_med:.0f}/{served_med:.0f} = {ratio:.2f}x"
            )
    else:
        print(
            "\nsub-stage distributions: none captured -- see the wiring-gap note above"
        )

    residuals: "list[float]" = []
    for c in per_cue_capture:
        timings = c.get("_stage_timings") or {}
        soft_gate_total = (c.get("macro_buckets_ms") or {}).get("soft_gate")
        if soft_gate_total is None or "rank" not in timings or "t11_t12" not in timings:
            continue
        try:
            residuals.append(
                float(soft_gate_total) - float(timings["rank"]) - float(timings["t11_t12"])
            )
        except (TypeError, ValueError):
            continue
    if residuals:
        print(
            f"\nsoft_gate_residual (soft_gate_total - rank - t11_t12), "
            f"paired per-call, n={len(residuals)}: "
            f"p50={statistics.median(residuals):.2f}ms p95={_p95(residuals):.2f}ms"
        )
    else:
        print(
            "\nsoft_gate_residual: unavailable (stage timings absent -- requires "
            "IAI_MCP_STAGE_PROFILE=1 on the daemon plus the response-wiring fix noted above)"
        )

    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="warm samples to collect")
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="proof-of-wiring read-only run (health check + N warm recalls, never restarts)",
    )
    args = parser.parse_args(argv)
    return run(n=max(1, args.n), dry_check=args.dry_check)


if __name__ == "__main__":
    sys.exit(main())
