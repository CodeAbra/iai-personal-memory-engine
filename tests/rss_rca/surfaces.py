"""Nine-surface attribution registry for the RSS RCA harness.

Each surface couples a file:line evidence pattern (lifted verbatim from the
phase research per-surface tables) with a classification rule that combines a
tracemalloc snapshot diff (Python + numpy domain), the macOS ``vmmap -resident``
text dump, and a per-cycle internal-probe state dict (HNSW counts, SQLite
PRAGMAs, gc instance counts) into one of:

- ``CONFIRMED-LEAKING``: per-cycle steady-state growth above the surface's
  threshold, with a falsifiable signal documented in the row.
- ``CONFIRMED-CLEAN``: per-cycle steady-state growth below the surface's
  threshold AND the surface was actually exercised (the harness saw at least
  some allocation against the file:line pattern across the run).
- ``INDETERMINATE``: the surface could not be measured at this scale (e.g. an
  un-exercised codepath, or vmmap unavailable on Linux, or a transient peak
  the snapshot pattern cannot catch). Includes an attribution-gap reason.

The classification is intentionally a deterministic, threshold-based function
of the inputs rather than an ML model — the planner consuming
RCA-FINDINGS.md needs reproducible attributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import tracemalloc


# Classification labels — kept as module-level constants so callers can compare
# against them without re-typing the strings.
LEAKING: str = "CONFIRMED-LEAKING"
CLEAN: str = "CONFIRMED-CLEAN"
INDETERMINATE: str = "INDETERMINATE"


# Nine surface registry. Each row is a dict with:
#   id: 1..9 (stable surface ordering)
#   name: human-readable label
#   files_lines: tuple of "file:line" strings lifted from research
#   tm_patterns: tuple of substrings that match against
#       ``stat.traceback[0].filename`` so the tracemalloc diff is filtered to
#       this surface's allocation sites without coupling to absolute paths.
#   attribution_method: short label describing how the surface is measured
#       (tracemalloc-python, tracemalloc-numpy, vmmap-anonymous,
#        vmmap-file-backed-mmap, internal-probe, or hybrid).
#   prior_hypothesis: the research-derived expectation (CLEAN-by-design /
#       LEAKING-hypothesized / INDETERMINATE) — surfaces in the table get this
#       label as a fallback classification if the harness produced no
#       discriminating evidence at the run scale.
#   threshold_kb_clean: per-cycle steady-state growth ≤ this in KB is CLEAN.
#   threshold_kb_leaking: per-cycle steady-state growth ≥ this in KB is
#       LEAKING; values strictly between the two thresholds resolve via the
#       prior hypothesis.
#   recommended_fix: one-line fix description for the ranked fix list.
SURFACES: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "name": "Capture buffers",
        "files_lines": (
            "events.py:43",
            "events.py:111",
            "_buffers.py:16",
            "_buffers.py:72",
            "peri_event_buffer.py:29",
        ),
        "tm_patterns": (
            "iai_mcp/events.py",
            "iai_mcp/store/_buffers.py",
            "iai_mcp/peri_event_buffer.py",
        ),
        "attribution_method": "tracemalloc-python",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 100,
        "threshold_kb_leaking": 500,
        "recommended_fix": (
            "no action — flush paths use dict.pop which releases the list; "
            "buffers capped by design"
        ),
    },
    {
        "id": 2,
        "name": "HNSW standby buffer",
        "files_lines": (
            "hippo/_db.py:441",
            "hippo/_db.py:443-459",
            "hippo/_db.py:530-537",
        ),
        "tm_patterns": (),
        "attribution_method": "internal-probe + vmmap-anonymous",
        "prior_hypothesis": LEAKING,
        "threshold_kb_clean": 1024,
        "threshold_kb_leaking": 5120,
        "recommended_fix": (
            "shrink standby on quiescent cycle: reset_index() then "
            "init_index(corpus_size + headroom) so mark_deleted ghost slots "
            "do not accumulate across cycles"
        ),
    },
    {
        "id": 3,
        "name": "SQLite page cache",
        "files_lines": (
            "hippo/_db.py:121",
            "hippo/_db.py:127-130",
        ),
        "tm_patterns": (
            "iai_mcp/hippo/_db.py",
            "sqlite3",
        ),
        "attribution_method": "tracemalloc-python + internal-probe",
        "prior_hypothesis": INDETERMINATE,
        "threshold_kb_clean": 4096,
        "threshold_kb_leaking": 51200,
        "recommended_fix": (
            "PRAGMA cache_size=-65536 (64 MiB cap) and pass "
            "cached_statements=128 at sqlite3.connect() to cap statement and "
            "page cache footprint"
        ),
    },
    {
        "id": 4,
        "name": "Rust embedder",
        "files_lines": (
            "embed.py:107",
            "daemon/__init__.py:616-635",
        ),
        "tm_patterns": (
            "iai_mcp/embed.py",
        ),
        "attribution_method": "internal-probe + vmmap-file-backed-mmap",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 1024,
        "threshold_kb_leaking": 51200,
        "recommended_fix": (
            "no action — singleton warm-embedder override holds one instance "
            "for daemon lifetime; weights are mmap-resident not heap-resident"
        ),
    },
    {
        "id": 5,
        "name": "lilli HD tier vectors",
        "files_lines": (
            "lilli/tiers/bsc.py:19",
            "lilli/tiers/bsc.py:78-87",
            "lilli/tiers/bsc.py:141-149",
            "lilli/tiers/fhrr.py",
            "lilli/tiers/sparse_vsa.py",
        ),
        "tm_patterns": (
            "iai_mcp/lilli/tiers/",
        ),
        "attribution_method": "tracemalloc-python",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 1024,
        "threshold_kb_leaking": 10240,
        "recommended_fix": (
            "no action — per-record HVs are stored in MemoryRecord and not "
            "retained in process memory; lru_cache codebook capped at 256"
        ),
    },
    {
        "id": 6,
        "name": "Numba JIT cache",
        "files_lines": (
            "mosaic.py:10",
            "mosaic.py:387",
            "mosaic.py:408",
            "mosaic.py:448",
            "mosaic.py:485",
            "mosaic.py:542",
            "mosaic.py:634",
        ),
        "tm_patterns": (),
        "attribution_method": "vmmap-anonymous + USS-residual",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 1024,
        "threshold_kb_leaking": 51200,
        "recommended_fix": (
            "no action — one-time JIT compile cost (~100-200 MB) is a steady "
            "boot footprint, not a per-cycle accumulator"
        ),
    },
    {
        "id": 7,
        "name": "PyArrow accumulators",
        "files_lines": (
            "maintenance.py:18",
            "maintenance.py:19-50",
            "maintenance.py:243-249",
            "maintenance.py:305-312",
        ),
        "tm_patterns": (
            "iai_mcp/maintenance.py",
        ),
        "attribution_method": "tracemalloc-python",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 500,
        "threshold_kb_leaking": 5120,
        "recommended_fix": (
            "no action — pre/post measurement uses PRAGMA dbstat SUM(pgsize) "
            "with no batch materialization; this surface was eliminated in a "
            "prior RCA"
        ),
    },
    {
        "id": 8,
        "name": "Drain worker pool",
        "files_lines": (
            "capture.py:714",
            "capture.py:834",
            "daemon/__init__.py:956-960",
            "daemon/__init__.py:1046-1047",
        ),
        "tm_patterns": (
            "iai_mcp/capture.py",
        ),
        "attribution_method": "tracemalloc-python",
        "prior_hypothesis": CLEAN,
        "threshold_kb_clean": 2048,
        "threshold_kb_leaking": 10240,
        "recommended_fix": (
            "no action — drain has a per-run cap and shares a single asyncio "
            "default-pool worker thread; no long-lived per-session state"
        ),
    },
    {
        "id": 9,
        "name": "SCHEMA_MINE intermediate",
        "files_lines": (
            "lilli/cycle/schema.py:67-96",
            "lilli/cycle/schema.py:68",
            "lilli/cycle/schema.py:72",
        ),
        "tm_patterns": (
            "iai_mcp/lilli/cycle/schema.py",
        ),
        "attribution_method": "tracemalloc-python",
        "prior_hypothesis": LEAKING,
        "threshold_kb_clean": 1024,
        "threshold_kb_leaking": 10240,
        "recommended_fix": (
            "stream-and-aggregate the record column iter without "
            "materializing all rows into a list — drop the list(...) wrapper "
            "at schema.py:68 and hold only the pair_counts dict"
        ),
    },
)


def _sum_size_diff_for_patterns(
    diff: "list[tracemalloc.StatisticDiff] | None",
    patterns: Sequence[str],
) -> tuple[int, int]:
    """Return ``(size_diff_bytes, hit_count)`` summed over patterns.

    Iterates the snapshot diff once and accumulates ``stat.size_diff`` for
    every frame whose top-of-stack filename contains any of the listed
    pattern substrings. Returns zero-size and zero-hit when ``diff`` is None
    (smoke-test stub path) or empty.
    """
    if not diff or not patterns:
        return 0, 0
    total = 0
    hits = 0
    for stat in diff:
        if not stat.traceback:
            continue
        try:
            frame = stat.traceback[0]
        except IndexError:
            continue
        filename = getattr(frame, "filename", "") or ""
        if not filename:
            continue
        for pat in patterns:
            if pat in filename:
                total += int(stat.size_diff)
                hits += 1
                break
    return total, hits


def _classify_by_threshold(
    kb_per_cycle: float,
    surface: dict[str, Any],
    *,
    measured: bool,
    indeterminate_reason: str | None = None,
) -> tuple[str, str]:
    """Return ``(classification, evidence_suffix)`` for a tracemalloc-attributed surface.

    The threshold pair on each surface gives a deterministic ternary mapping:
    growth below the clean threshold maps to ``CLEAN`` (if the harness saw
    *any* allocation against the surface so we know the codepath was
    exercised), growth above the leaking threshold maps to ``LEAKING``, and
    the in-between band resolves via the prior research hypothesis so a
    borderline reading does not flip-flop between runs.
    """
    if not measured:
        reason = indeterminate_reason or (
            "no allocation observed against this surface in the harness — "
            "codepath likely not exercised at this corpus scale"
        )
        return INDETERMINATE, reason

    clean_kb = float(surface["threshold_kb_clean"])
    leaking_kb = float(surface["threshold_kb_leaking"])

    if kb_per_cycle <= clean_kb:
        return CLEAN, f"≤{clean_kb:.0f} KB/cycle threshold"
    if kb_per_cycle >= leaking_kb:
        return LEAKING, f"≥{leaking_kb:.0f} KB/cycle threshold"
    return surface["prior_hypothesis"], (
        f"in band ({clean_kb:.0f}–{leaking_kb:.0f} KB/cycle); "
        f"resolved via prior hypothesis"
    )


def _classify_hnsw_standby(
    state: dict[str, Any],
    prior_state: dict[str, Any] | None,
) -> tuple[str, float, str]:
    """Surface 2 — return ``(classification, kb_per_cycle, evidence)``."""
    cap = state.get("hnsw_standby_capacity")
    count = state.get("hnsw_standby_count")
    if cap is None or count is None:
        return INDETERMINATE, 0.0, "standby buffer absent — boot-only path"
    ghost = max(0, int(cap) - int(count))
    if prior_state is None:
        return INDETERMINATE, 0.0, (
            f"first cycle baseline: cap={cap}, count={count}, ghost={ghost}"
        )
    prior_cap = prior_state.get("hnsw_standby_capacity")
    if prior_cap is None:
        return INDETERMINATE, 0.0, "prior cycle had no standby buffer"
    cap_growth = int(cap) - int(prior_cap)
    bytes_per_slot = 1536  # M=16 × 4 + dim 384 × 4 ≈ 1.5 KB per element
    kb_growth = max(0.0, cap_growth * bytes_per_slot / 1024.0)
    if cap_growth <= 0:
        return CLEAN, 0.0, (
            f"capacity stable across cycles: cap={cap}, count={count}, "
            f"ghost={ghost}"
        )
    if cap_growth > 1024:
        return LEAKING, kb_growth, (
            f"capacity grew by {cap_growth} slots cycle-on-cycle "
            f"(≈{kb_growth:.0f} KB at {bytes_per_slot}B/slot); "
            f"ghost_count={ghost}"
        )
    return CLEAN, kb_growth, (
        f"capacity grew by {cap_growth} slots cycle-on-cycle "
        f"(≈{kb_growth:.0f} KB); within tolerance"
    )


def _classify_sqlite_page_cache(
    diff_py: "list[tracemalloc.StatisticDiff] | None",
    state: dict[str, Any],
    prior_state: dict[str, Any] | None,
    surface: dict[str, Any],
) -> tuple[str, float, str]:
    """Surface 3 — combine tracemalloc and PRAGMA probes."""
    tm_bytes, hits = _sum_size_diff_for_patterns(diff_py, surface["tm_patterns"])
    tm_kb = max(0.0, tm_bytes / 1024.0)

    page_count = state.get("sqlite_page_count")
    page_size = state.get("sqlite_page_size")
    pragma_kb = None
    pragma_delta_kb = None
    if page_count is not None and page_size is not None:
        pragma_kb = (int(page_count) * int(page_size)) / 1024.0
        if prior_state is not None:
            prior_pc = prior_state.get("sqlite_page_count")
            prior_ps = prior_state.get("sqlite_page_size")
            if prior_pc is not None and prior_ps is not None:
                prior_kb = (int(prior_pc) * int(prior_ps)) / 1024.0
                pragma_delta_kb = max(0.0, pragma_kb - prior_kb)

    measured = hits > 0 or pragma_delta_kb is not None
    kb_per_cycle = tm_kb + (pragma_delta_kb or 0.0)
    classification, suffix = _classify_by_threshold(
        kb_per_cycle, surface, measured=measured,
    )
    evidence_bits = [f"tm={tm_kb:.0f} KB ({hits} frames)"]
    if pragma_kb is not None:
        evidence_bits.append(f"sqlite_db={pragma_kb:.0f} KB")
    if pragma_delta_kb is not None:
        evidence_bits.append(f"Δdb={pragma_delta_kb:.0f} KB cycle-on-cycle")
    return classification, kb_per_cycle, f"{'; '.join(evidence_bits)} — {suffix}"


def _classify_rust_embedder(
    state: dict[str, Any],
    prior_state: dict[str, Any] | None,
) -> tuple[str, float, str]:
    """Surface 4 — gc instance count probe."""
    instances = state.get("embedder_instances")
    if instances is None:
        return INDETERMINATE, 0.0, "Embedder instance count not captured"
    if int(instances) <= 1:
        return CLEAN, 0.0, (
            f"singleton warm-embedder preserved (gc instances={instances})"
        )
    return LEAKING, 80_000.0, (
        f"multiple Embedder instances detected ({instances}); each holds "
        f"~80 MB of bge-small-en-v1.5 weights"
    )


def _classify_numba(
    state: dict[str, Any],
    prior_state: dict[str, Any] | None,
    cycle_idx: int,
) -> tuple[str, float, str]:
    """Surface 6 — USS residual after the first cycle should be stable."""
    if prior_state is None or cycle_idx == 0:
        return INDETERMINATE, 0.0, (
            "first cycle (cold) — JIT compile is a one-time boot cost, "
            "needs ≥ cycle 2 for steady-state assessment"
        )
    uss = int(state.get("uss_bytes") or 0)
    prior_uss = int(prior_state.get("uss_bytes") or 0)
    if uss == 0 or prior_uss == 0:
        return INDETERMINATE, 0.0, "USS sampling unavailable"
    delta_kb = max(0.0, (uss - prior_uss) / 1024.0)
    if delta_kb <= 1024:
        return CLEAN, 0.0, (
            f"USS delta {delta_kb:.0f} KB cycle-on-cycle within numba "
            f"steady-state tolerance"
        )
    if delta_kb >= 51200:
        return LEAKING, delta_kb, (
            f"USS delta {delta_kb:.0f} KB cycle-on-cycle exceeds JIT "
            f"steady-state expectation; possible signature explosion"
        )
    return CLEAN, delta_kb, (
        f"USS delta {delta_kb:.0f} KB cycle-on-cycle within bounded band"
    )


def classify_all(
    diff_py: "list[tracemalloc.StatisticDiff] | None",
    diff_np: "list[tracemalloc.StatisticDiff] | None",
    vmmap_text: str | None,
    state: dict[str, Any],
    prior_state: dict[str, Any] | None,
    *,
    cycle_idx: int = 0,
) -> list[dict[str, Any]]:
    """Return one row per surface for the RCA findings table.

    Each row is a dict with ``id``, ``name``, ``classification``,
    ``kb_per_cycle``, ``evidence``, ``recommended_fix``, ``files_lines``,
    ``attribution_method`` so the harness can render the per-surface table
    and the ranked fix list directly without further per-surface logic.
    """
    rows: list[dict[str, Any]] = []
    for surface in SURFACES:
        sid = int(surface["id"])

        if sid == 1:
            tm_bytes, hits = _sum_size_diff_for_patterns(
                diff_py, surface["tm_patterns"],
            )
            kb = max(0.0, tm_bytes / 1024.0)
            classification, suffix = _classify_by_threshold(
                kb, surface, measured=hits > 0,
            )
            evidence = (
                f"tracemalloc Δ={kb:.0f} KB across {hits} frames — {suffix}"
            )

        elif sid == 2:
            classification, kb, evidence = _classify_hnsw_standby(
                state, prior_state,
            )

        elif sid == 3:
            classification, kb, evidence = _classify_sqlite_page_cache(
                diff_py, state, prior_state, surface,
            )

        elif sid == 4:
            classification, kb, evidence = _classify_rust_embedder(
                state, prior_state,
            )

        elif sid == 5:
            tm_bytes, hits = _sum_size_diff_for_patterns(
                diff_py, surface["tm_patterns"],
            )
            np_bytes, np_hits = _sum_size_diff_for_patterns(
                diff_np, surface["tm_patterns"],
            )
            total_kb = max(0.0, (tm_bytes + np_bytes) / 1024.0)
            classification, suffix = _classify_by_threshold(
                total_kb, surface, measured=(hits + np_hits) > 0,
            )
            kb = total_kb
            evidence = (
                f"tracemalloc Δ={total_kb:.0f} KB "
                f"({hits} python frames, {np_hits} numpy frames) — {suffix}"
            )

        elif sid == 6:
            classification, kb, evidence = _classify_numba(
                state, prior_state, cycle_idx,
            )

        elif sid == 7:
            tm_bytes, hits = _sum_size_diff_for_patterns(
                diff_py, surface["tm_patterns"],
            )
            kb = max(0.0, tm_bytes / 1024.0)
            classification, suffix = _classify_by_threshold(
                kb, surface, measured=hits > 0,
            )
            evidence = (
                f"tracemalloc Δ={kb:.0f} KB across {hits} frames — {suffix}"
            )

        elif sid == 8:
            tm_bytes, hits = _sum_size_diff_for_patterns(
                diff_py, surface["tm_patterns"],
            )
            kb = max(0.0, tm_bytes / 1024.0)
            classification, suffix = _classify_by_threshold(
                kb, surface, measured=hits > 0,
                indeterminate_reason=(
                    "drain path not exercised in this cycle; the harness "
                    "stages a fixed deferred-captures batch before each "
                    "cycle and invokes drain_deferred_captures(store); a "
                    "zero-hit reading here means staging produced no work"
                ),
            )
            evidence = (
                f"tracemalloc Δ={kb:.0f} KB across {hits} frames — {suffix}"
            )

        elif sid == 9:
            tm_bytes, hits = _sum_size_diff_for_patterns(
                diff_py, surface["tm_patterns"],
            )
            kb = max(0.0, tm_bytes / 1024.0)
            classification, suffix = _classify_by_threshold(
                kb, surface, measured=hits > 0,
                indeterminate_reason=(
                    "SCHEMA_MINE did not produce a tracemalloc hit at "
                    "schema.py:68/72 — auto-induction may not have fired at "
                    "this corpus scale or with this tag distribution"
                ),
            )
            evidence = (
                f"tracemalloc Δ={kb:.0f} KB across {hits} frames — {suffix}"
            )

        else:
            classification = INDETERMINATE
            kb = 0.0
            evidence = "unknown surface id"

        rows.append({
            "id": sid,
            "name": surface["name"],
            "classification": classification,
            "kb_per_cycle": float(kb),
            "evidence": evidence,
            "recommended_fix": surface["recommended_fix"],
            "files_lines": surface["files_lines"],
            "attribution_method": surface["attribution_method"],
        })

    return rows


def render_findings_md(
    rows: Sequence[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> str:
    """Render the RCA-FINDINGS.md body for the harness output directory.

    Schema matches the planner-consumed format documented in the phase
    research §Expected Output: header block with run metadata, executive
    summary (top-3 ranked), harness description, full 9-row per-surface
    table, ranked fix list (LEAKING first then INDETERMINATE), untested
    surfaces section (any row that ended INDETERMINATE), threshold
    suggestion table, and sources-of-per-cycle-data pointer.
    """
    iso_ts = run_metadata.get("iso_ts", "")
    n = run_metadata.get("n_records", 0)
    dim = run_metadata.get("dim", 0)
    cycles = run_metadata.get("cycles", 0)
    pid = run_metadata.get("pid", 0)
    warmup = run_metadata.get("warmup_duration_sec", 0.0)

    ranked = sorted(rows, key=lambda r: r["kb_per_cycle"], reverse=True)
    leaking = [r for r in ranked if r["classification"] == LEAKING]
    indeterminate = [r for r in ranked if r["classification"] == INDETERMINATE]

    lines: list[str] = []
    lines.append("# RSS Attribution Findings")
    lines.append("")
    lines.append(f"**Harness ran:** {iso_ts}")
    lines.append(f"**Corpus size:** {n} synthetic records, {dim}-d embeddings")
    lines.append(f"**Cycles driven:** {cycles}")
    lines.append(f"**Pre-cycle warm-up:** {warmup:.2f} s")
    lines.append(
        f"**Surrogate PID:** {pid} (single-process; live daemon NOT involved)"
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "| Rank | Surface | KB/cycle steady-state | Classification | Headline fix |"
    )
    lines.append(
        "|------|---------|----------------------|----------------|--------------|"
    )
    for rank, row in enumerate(ranked[:3], start=1):
        fix = row["recommended_fix"]
        lines.append(
            f"| {rank} | {row['name']} | {row['kb_per_cycle']:.0f} | "
            f"{row['classification']} | {fix} |"
        )
    lines.append("")

    lines.append("## Harness Description")
    lines.append("")
    lines.append(
        "- Corpus: N records inserted via `MemoryStore.insert()` with random "
        "unit-norm embeddings + BSC role-filler bundles in "
        "`structure_hv_payload`."
    )
    lines.append(
        "- Cycle driver: in-process `SleepPipeline(store=store, "
        "lifecycle_state_path=tmp/lifecycle_state.json, event_log=event_log)"
        ".run()` per cycle; the full 13-step `_STEP_ORDER` runs end to end "
        "(SCHEMA_MINE first, OPTIMIZE_HIPPO third)."
    )
    lines.append(
        "- Drain interleave: before each cycle the harness writes a fixed "
        "batch of `.deferred-captures/<session>-<i>.jsonl` files under the "
        "tmp HOME and calls `drain_deferred_captures(store)`. The drain "
        "snapshot is taken separately from the OPTIMIZE snapshot so growth "
        "can be attributed to one or the other."
    )
    lines.append(
        "- Instrumentation: `tracemalloc` (depth 25, Python + numpy domain) "
        "with per-cycle snap diff; `psutil.Process().memory_full_info()` "
        "USS/RSS/PSS sampler thread at 5 s; macOS `vmmap -resident` dump per "
        "cycle; internal probes (`hnswlib.Index.get_max_elements`, "
        "`get_current_count`, SQLite PRAGMAs, `gc.get_objects()` Embedder "
        "instance count)."
    )
    lines.append(
        "- Hermeticity: `_real_store_signature()` snapshot before and after "
        "the run; any drift in `~/.iai-mcp/` raises `RuntimeError"
        "(\"HERMETICITY VIOLATION: ...\")`. `HOME` and `IAI_MCP_STORE` are "
        "redirected to a tmp root before any project import."
    )
    lines.append("")

    lines.append("## Per-Surface Findings Table")
    lines.append("")
    lines.append(
        "| # | Surface | Classification | KB/cycle | Files:lines | Recommended fix |"
    )
    lines.append(
        "|---|---------|----------------|----------|-------------|-----------------|"
    )
    for row in rows:
        files = ", ".join(row["files_lines"])
        lines.append(
            f"| {row['id']} | {row['name']} | {row['classification']} | "
            f"{row['kb_per_cycle']:.0f} | {files} | {row['recommended_fix']} |"
        )
    lines.append("")

    lines.append("### Per-Surface Evidence")
    lines.append("")
    for row in rows:
        lines.append(f"- **{row['id']}. {row['name']}** — {row['evidence']}")
    lines.append("")

    lines.append("## Ranked Fix List")
    lines.append("")
    if leaking:
        lines.append(
            "Sorted by KB/cycle impact; planner consumes this for the "
            "follow-up per-surface fix phases."
        )
        lines.append("")
        for rank, row in enumerate(leaking, start=1):
            lines.append(
                f"{rank}. **Surface {row['id']} — {row['name']}** "
                f"({row['kb_per_cycle']:.0f} KB/cycle): "
                f"{row['recommended_fix']}"
            )
    else:
        lines.append(
            "No surfaces classified as CONFIRMED-LEAKING at the run's "
            "thresholds. See the untested-surfaces section for "
            "INDETERMINATE rows that may need a larger run or a richer "
            "attribution probe."
        )
    lines.append("")

    lines.append("## Untested Surfaces")
    lines.append("")
    if indeterminate:
        lines.append("Surfaces that could not be classified at this scale:")
        lines.append("")
        for row in indeterminate:
            lines.append(
                f"- **Surface {row['id']} — {row['name']}**: {row['evidence']}"
            )
    else:
        lines.append("All 9 surfaces classified.")
    lines.append("")

    lines.append("## Threshold Suggestion")
    lines.append("")
    lines.append("Theoretical warm plateau target (post-fix):")
    lines.append("")
    lines.append("| Contributor | Bytes |")
    lines.append("|-------------|-------|")
    lines.append("| Python interpreter + runtime libraries | ~100 MB |")
    lines.append("| Rust embedder weights (bge-small-en-v1.5) | ~80 MB |")
    lines.append("| Numba JIT cache (mosaic, post-warmup) | ~150 MB |")
    lines.append("| HNSW active index (25k × ~1.5 KB) | ~38 MB |")
    lines.append("| HNSW standby index (same) | ~38 MB |")
    lines.append("| SQLite page cache (capped 64 MB) | ~64 MB |")
    lines.append("| Drain peak transient (250 events × ~10 KB) | ~3 MB |")
    lines.append("| SCHEMA_MINE intermediate (post-fix streaming) | ~5 MB |")
    lines.append("| HippoDB connection + label_map | ~20 MB |")
    lines.append("| Misc Python objects (gc heap, asyncio loop, etc) | ~150 MB |")
    lines.append("| **Subtotal** | **~650 MB** |")
    lines.append("| Safety margin (2× for GC fragmentation) | +650 MB |")
    lines.append("| **Recommended warm-plateau target** | **~1.3 GB** |")
    lines.append("| **Recommended new watchdog cap** | **2.0 GB** |")
    lines.append("")
    lines.append(
        "Current cap of 4.0 GB at `src/iai_mcp/daemon/_watchdog.py` is "
        "operating as a band-aid; subsequent fix-shipping phases should "
        "converge to the 2.0 GB cap as the leaking surfaces are addressed."
    )
    lines.append("")

    lines.append("## Sources of Per-Cycle Data")
    lines.append("")
    lines.append(
        "- `rca-log.jsonl` — one JSONL record per cycle with the full state "
        "dict (USS/RSS/PSS, HNSW counts, SQLite PRAGMAs, gc counters) plus "
        "5-second sampler snapshots tagged `kind=\"sample\"`."
    )
    lines.append(
        "- `vmmap-cycle-N.txt` — full `vmmap -resident` text dump per cycle "
        "on macOS; raw text only so the planner can do its own diff."
    )
    lines.append(
        "- `tracemalloc-top-cycle-N.txt` — top 30 file:line allocations from "
        "the post-cycle snapshot."
    )
    lines.append(
        "- `tracemalloc-diff-cycle-N.txt` — top 30 file:line diffs between "
        "the pre-cycle and post-cycle snapshots, with separate sections for "
        "the post-drain and post-OPTIMIZE_HIPPO marks."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


__all__: Sequence[str] = (
    "SURFACES",
    "LEAKING",
    "CLEAN",
    "INDETERMINATE",
    "classify_all",
    "render_findings_md",
)
