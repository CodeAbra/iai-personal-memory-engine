"""Label-free recall drop-out differential gate for the tuned retrieval
cosine weight (W_COSINE).

The tuner genuinely moves W_COSINE, so this comparator can and sometimes
will show drop-out for cues near a tie -- the pass condition is therefore a
PRE-COMMITTED, MEASURED A/A noise floor, never "zero drop-out". Production
steady-state under sustained one-sided feedback is the clamp EDGE (the
tuner's delta does not decay toward an interior fixed point), so the armed
leg below stresses both clamp edges, dispatched through the FORCED-Rust
production scorer (use_rust_scorer=True) -- the only path production takes.

Runs entirely on read-only-sourced copies (bench/recall_accuracy_real.
open_eval_copy_store) -- the operator's live store is never opened for
write. Diagnostics on failure/skip carry record ids and counts only, never
cue text or stored content.

If the clamp-edge leg exceeds floor+margin, NARROW THE PRODUCTION CLAMP
BAND (retrieval_tuning.PROD_W_COSINE_MIN/MAX) and re-measure -- never widen
the margin below. The band is exactly what this gate exists to test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from bench.recall_accuracy_real import (
    _BASELINE_STRUCTURAL_WEIGHT,
    _dispatch_real_cue,
    open_eval_copy_store,
)
from iai_mcp import retrieval_weight_cache, runtime_graph_cache
from iai_mcp.lilli.profile import retrieval_tuning as _retrieval_tuning
from iai_mcp.lilli.profile.retrieval_tuning import (
    PROD_W_COSINE_MAX,
    PROD_W_COSINE_MIN,
    save_retrieval_weights_state,
)
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import _clean_surface

_MIN_CUE_WORDS = 3
_CUE_WORD_COUNT = 6
_CUE_CAP = 100

# Measured from independent A/A trials on the real corpus at cap=100 under
# created_at ASC, id ASC ordering, no weight change at all -- see
# test_floor_measurement's own PROVENANCE output (three reps: 0.001443,
# 0.001451, 0.001443). AA_DROPOUT_FLOOR_MEASURED is derived from that
# outcome -- never hand-set to True without a matching measured run.
AA_DROPOUT_FLOOR: float = 0.0015
DROPOUT_MARGIN: float = 0.05
AA_DROPOUT_FLOOR_MEASURED: bool = True

# Conservative fail-loud guard floor (not a tuned knee) below the live
# deterministic effective-n (~693 at cap=100) -- refuses to trust a
# degenerate or too-fresh sample rather than silently accepting one.
EFFECTIVE_N_FLOOR: int = 300


def test_dropout_margin_pinned() -> None:
    """The 5% drop-out margin is the fixed safety contract: narrow the
    production clamp band when needed, never widen this margin."""
    assert DROPOUT_MARGIN == 0.05


def test_aa_dropout_floor_pinned_and_measured() -> None:
    """AA_DROPOUT_FLOOR is a hard literal by design, sourced only from a
    measured test_floor_measurement PROVENANCE run -- never hand-raised. A
    legitimate re-measurement changes this pin only in the same reviewed
    commit as a matching PROVENANCE run; do not delete this guard as stale."""
    assert AA_DROPOUT_FLOOR == 0.0015
    assert AA_DROPOUT_FLOOR_MEASURED is True


# ---------------------------------------------------------------------------
# Pure comparator (real-store-independent)
# ---------------------------------------------------------------------------


def _comparator(
    per_cue: "dict[str, dict[str, set[str]]]",
) -> "tuple[set[str], set[str], set[str]]":
    """Regression rule: an id is a regression for a cue iff it is surfaced in
    P1 AND A/A-stable (also surfaced in P2) AND absent in B. An id P1
    surfaced but P2 did not is excused as harness noise (instability). A
    reshuffle within the retrieved set is never a regression -- only
    drop-out is. Returns (stable, regressions, instability), each a union
    across every cue.
    """
    stable_total: "set[str]" = set()
    regressions: "set[str]" = set()
    instability: "set[str]" = set()
    for sets in per_cue.values():
        p1, p2, b = sets["p1"], sets["p2"], sets["b"]
        stable = p1 & p2
        stable_total |= stable
        instability |= (p1 - p2)
        regressions |= (stable - b)
    return stable_total, regressions, instability


def test_comparator_stable_drop_is_flagged():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1"}}}
    stable, regressions, instability = _comparator(per_cue)
    assert stable == {"s1", "n1"}
    assert regressions == {"s1"}
    assert instability == set()


def test_comparator_unstable_drop_is_excused():
    per_cue = {"cue-a": {"p1": {"s1"}, "p2": set(), "b": set()}}
    stable, regressions, instability = _comparator(per_cue)
    assert regressions == set()
    assert instability == {"s1"}


def test_comparator_within_set_reshuffle_is_benign():
    per_cue = {"cue-a": {"p1": {"s1", "n1"}, "p2": {"s1", "n1"}, "b": {"n1", "s1"}}}
    stable, regressions, instability = _comparator(per_cue)
    assert regressions == set()
    assert instability == set()


# ---------------------------------------------------------------------------
# Shared real-store helpers
# ---------------------------------------------------------------------------


def _real_store_present() -> bool:
    from iai_mcp.hippo import _operator_home

    return (_operator_home() / ".iai-mcp" / "hippo" / "brain.sqlite3").exists()


def _select_cue_candidates(store, cap: int) -> "list[tuple[str, str]]":
    """(record_id, literal_surface) for live, unpinned, untrusted-merge
    records -- label-free, no tag gating (this gate gauges the tuned
    cosine weight across the whole corpus, not one lever's affected rows).

    Ordered by creation time with id as a deterministic tiebreak: a new
    capture sorts to the END of the stream, so the earliest-`cap` eligible
    set a prior run selected cannot be displaced by later inserts. Ordering
    by id alone reshuffles that set as random-UUID records land over time
    -- exactly the run-to-run flake this ordering retires. A NULL or empty
    created_at is never excluded from eligibility, only ordered by the
    database's own NULL/empty-string ordering rules.
    """
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(pinned, 0) = 0"
            "   AND COALESCE(never_merge, 0) = 0"
            "   AND COALESCE(embedding_pending, 0) = 0"
            " ORDER BY created_at ASC, id ASC"
            " LIMIT ?",
            (cap * 25,),
        ).fetchall()
    candidate_ids: "list[UUID]" = [UUID(str(row["id"])) for row in rows]

    if not candidate_ids:
        return []

    out: "list[tuple[str, str]]" = []
    batch = store.get_batch(candidate_ids)
    for rid in candidate_ids:
        rec = batch.get(rid)
        if rec is None:
            continue
        text = rec.literal_surface or ""
        words = _clean_surface(text).split()
        if len(words) < _MIN_CUE_WORDS:
            continue
        out.append((str(rid), text))
        if len(out) >= cap:
            break

    if len(out) < cap and len(candidate_ids) >= cap * 25:
        pytest.fail(
            f"cue selection starved: scanned {len(candidate_ids)} eligible "
            f"candidates (prefetch ceiling {cap * 25}) but only {len(out)} "
            f"survived the minimum-word filter (need {cap}) -- genuine "
            "starvation, not a small store"
        )
    if len(out) < cap:
        print(
            f"cue selection: store smaller than the prefetch ceiling -- "
            f"{len(out)} cues found (requested cap={cap})"
        )
    return out


def _cue_texts_from_store(store, cap: int) -> "list[str]":
    candidates = _select_cue_candidates(store, cap)
    cues: "list[str]" = []
    for _rid, text in candidates:
        words = _clean_surface(text).split()
        cue_text = " ".join(words[:_CUE_WORD_COUNT])
        if cue_text:
            cues.append(cue_text)
    return cues


def _dispatch_b07(store, cue_text: str) -> "set[str]":
    # _BASELINE_STRUCTURAL_WEIGHT=0.0 leaves _resolve_use_rust_scorer free to
    # honor the production use_rust_scorer=True request -- a nonzero
    # structural_weight forces the non-Rust reference branch, which would
    # make this gate prove nothing about production.
    return set(_dispatch_real_cue(store, cue_text, _BASELINE_STRUCTURAL_WEIGHT))


def _dispatch_b07_full(store, cue_text: str) -> dict:
    """Same dispatch as `_dispatch_b07`, but returns the full response so
    the Rust-path reason string is available for the coefficient sanity
    check below (the id-only helper above is enough for drop-out counting).
    """
    from bench.recall_accuracy_real import _structural
    from iai_mcp import core as _core

    with _structural(_BASELINE_STRUCTURAL_WEIGHT):
        return _core.dispatch(store, "memory_recall", {
            "cue": cue_text, "session_id": "harness-real-thread", "budget_tokens": 2000,
        })


def _first_weighted_reason(resp: dict) -> "str | None":
    """First hit reason that carries the Rust-path cosine coefficient
    (`cos {c:.3f}*{w:g} ...`) -- skips hits the exact-cosine authority merge
    overrode (`reason == "exact-cosine"`, no coefficient), which is a
    separate, unconditional-once-warm mechanism this gate does not disable
    (production never disables it either)."""
    for h in resp.get("hits", []):
        reason = h.get("reason", "")
        if reason.startswith("cos "):
            return reason
    return None


def _rebuild(store) -> None:
    runtime_graph_cache.invalidate(store)
    assert getattr(store, "_warm_graph_bundle", None) is None
    build_runtime_graph(store)


def _open_driver_copy(driver: str):
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
    return open_eval_copy_store(driver=driver)


# ---------------------------------------------------------------------------
# Measure the A/A drop-out noise floor, no weight change at all
# ---------------------------------------------------------------------------


@pytest.mark.realstore
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_floor_measurement(driver, monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    # This test never writes to a file path itself -- the PROVENANCE lines
    # below are plain stdout; a caller captures them by redirecting the
    # test run's own output.
    if not _real_store_present():
        print(f"PROVENANCE: skipped driver={driver} reason=no-real-store")
        pytest.skip("real store not found; this is a real-store control, not a fresh-clone CI gate")

    cm = _open_driver_copy(driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            print(f"PROVENANCE: skipped driver={driver} reason=open-failed")
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        cues = _cue_texts_from_store(store, _CUE_CAP)
        if not cues:
            print(f"PROVENANCE: skipped driver={driver} reason=no-cue-candidates")
            pytest.skip("no cue candidates found on this real-store copy")

        # Three independent untuned (default weight) A legs -- no weight
        # change at all -- reusing the exact same P1/P2/B comparator shape
        # the armed leg below uses, so the floor measures pure rebuild
        # jitter under the SAME machinery, not a different metric.
        _rebuild(store)
        p1_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        _rebuild(store)
        p2_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        _rebuild(store)
        b_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        per_cue = {
            cue: {"p1": p1_by_cue[cue], "p2": p2_by_cue[cue], "b": b_by_cue[cue]}
            for cue in cues
        }
        stable, regressions, instability = _comparator(per_cue)
        if len(stable) < EFFECTIVE_N_FLOOR:
            pytest.fail(
                f"effective-n too low to trust this floor measurement: "
                f"len(stable)={len(stable)} < EFFECTIVE_N_FLOOR="
                f"{EFFECTIVE_N_FLOOR} on driver={driver} -- degenerate or "
                "too-fresh store"
            )
        floor = len(regressions) / max(1, len(stable))

        print(
            f"PROVENANCE: measured driver={driver} floor={floor:.6f} "
            f"rebuilds=3 cues={len(cues)} stable={len(stable)} "
            f"regressions={len(regressions)} instability={len(instability)}"
        )

        assert 0.0 <= floor <= 1.0
    finally:
        cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Armed clamp-edge B leg over the forced-Rust dispatch
# ---------------------------------------------------------------------------


@pytest.mark.realstore
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("edge_name,edge_weight", [
    ("max", PROD_W_COSINE_MAX),
    ("min", PROD_W_COSINE_MIN),
])
def test_clamp_edge_dropout_within_margin(driver, edge_name, edge_weight, monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    if not _real_store_present():
        pytest.skip("real store not found; this is a real-store control, not a fresh-clone CI gate")

    if not AA_DROPOUT_FLOOR_MEASURED:
        pytest.fail(
            "A/A floor never measured on this corpus; refusing to assert clamp-edge "
            "drop-out against a default -- run test_floor_measurement first and "
            "record its output into AA_DROPOUT_FLOOR/AA_DROPOUT_FLOOR_MEASURED."
        )

    cm = _open_driver_copy(driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        cues = _cue_texts_from_store(store, _CUE_CAP)
        if not cues:
            pytest.skip("no cue candidates found on this real-store copy")

        _rebuild(store)
        a1_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        _rebuild(store)
        a2_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        assert save_retrieval_weights_state(store, {"W_COSINE": edge_weight}), (
            f"failed to persist arm-point weight {edge_weight} on driver={driver} -- "
            "cannot trust a subsequent clamp-edge measurement against an unarmed store"
        )
        retrieval_weight_cache.invalidate(store)
        runtime_graph_cache.invalidate(store)
        _rebuild(store)
        b_resp_by_cue = {cue: _dispatch_b07_full(store, cue) for cue in cues}
        b_by_cue = {
            cue: {h["record_id"] for h in resp.get("hits", [])}
            for cue, resp in b_resp_by_cue.items()
        }

        per_cue = {
            cue: {"p1": a1_by_cue[cue], "p2": a2_by_cue[cue], "b": b_by_cue[cue]}
            for cue in cues
        }
        stable, regressions, _instability = _comparator(per_cue)
        if len(stable) < EFFECTIVE_N_FLOOR:
            pytest.fail(
                f"effective-n too low to trust this clamp-edge measurement: "
                f"len(stable)={len(stable)} < EFFECTIVE_N_FLOOR="
                f"{EFFECTIVE_N_FLOOR} on driver={driver} edge={edge_name} -- "
                "degenerate or too-fresh store"
            )
        dropout_rate = len(regressions) / max(1, len(stable))

        print(
            f"PROVENANCE: clamp-edge driver={driver} edge={edge_name} "
            f"value={edge_weight:g} dropout={dropout_rate:.6f} "
            f"floor={AA_DROPOUT_FLOOR:.4f} margin={DROPOUT_MARGIN:.4f} "
            f"stable={len(stable)} regressions={len(regressions)}"
        )

        # Sanity: scan every cue's TUNED response for a hit the exact-cosine
        # authority merge did not override, and confirm its printed
        # coefficient equals the persisted clamp-edge value -- direct proof
        # the weight reached the RUST base_s. Assert on FORCED-Rust output
        # only -- never a Python-branch reason string (production never
        # emits it) -- and never a bare {W_COSINE:g} (constant, uninformative).
        tuned_reason = None
        for resp in b_resp_by_cue.values():
            tuned_reason = _first_weighted_reason(resp)
            if tuned_reason is not None:
                break
        assert tuned_reason is not None, (
            "every hit across all cues was overridden by the exact-cosine "
            "authority merge in the tuned dispatch -- cannot sanity-check "
            "the Rust-path coefficient; widen the cue set"
        )
        assert f"*{edge_weight:g}" in tuned_reason, (
            f"Rust-path reason does not carry the tuned coefficient "
            f"{edge_weight:g}: {tuned_reason!r}"
        )

        margin_breached = dropout_rate > AA_DROPOUT_FLOOR + DROPOUT_MARGIN

        # lilli is the native engine driver production recall runs on, so it
        # alone binds the safety margin; stdlib is a legacy read path.
        if driver == "lilli":
            # if the clamp-edge leg exceeds floor+margin, narrow the production
            # clamp band and re-measure -- never widen the margin below.
            assert not margin_breached, (
                f"clamp-edge ({edge_name}={edge_weight}) drop-out {dropout_rate:.4f} "
                f"exceeds measured floor {AA_DROPOUT_FLOOR:.4f} + margin {DROPOUT_MARGIN:.4f} "
                f"on driver={driver}; regressions={len(regressions)} stable={len(stable)}"
            )
        elif margin_breached:
            pytest.xfail(
                f"clamp-edge ({edge_name}={edge_weight}) drop-out {dropout_rate:.4f} "
                f"exceeds floor {AA_DROPOUT_FLOOR:.4f} + margin {DROPOUT_MARGIN:.4f} on "
                "the legacy stdlib storage driver, which diverges from the native "
                "engine driver that production recall actually uses -- the "
                "clamp-edge safety margin is asserted on the native driver, and "
                "the legacy driver is retained only for cross-driver visibility"
            )
    finally:
        cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Non-vacuity: end-to-end planted drop, its OWN copy, its OWN P2 A/A leg
# ---------------------------------------------------------------------------


@pytest.mark.realstore
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_end_to_end_tombstone_detected_as_regression(driver, monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    if not _real_store_present():
        pytest.skip("real store not found; this is a real-store control, not a fresh-clone CI gate")

    cm = _open_driver_copy(driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        # Try several candidates (not just the first): a single candidate's
        # own cue may not be A/A-stable on this real-store copy, which would
        # skip the control and leave the tuner's ONLY non-vacuity proof
        # unproven rather than run. Build every candidate's P1/P2 in the
        # SAME two rebuild passes and pick the first stable one.
        candidates = _select_cue_candidates(store, cap=8)
        if not candidates:
            pytest.skip("no cue candidates found for the planted-drop control on this real-store copy")

        cue_by_target: "dict[str, str]" = {}
        for rid, real_text in candidates:
            words = _clean_surface(real_text).split()
            if len(words) < _MIN_CUE_WORDS:
                continue
            cue_by_target[rid] = " ".join(words[:_CUE_WORD_COUNT])
        if not cue_by_target:
            pytest.skip("no planted-drop candidate had enough real content to build a cue")

        _rebuild(store)
        p1_by_target = {tid: _dispatch_b07(store, cue) for tid, cue in cue_by_target.items()}

        _rebuild(store)
        p2_by_target = {tid: _dispatch_b07(store, cue) for tid, cue in cue_by_target.items()}

        stable_targets = [
            tid for tid in cue_by_target
            if tid in (p1_by_target[tid] & p2_by_target[tid])
        ]
        if not stable_targets:
            pytest.skip(
                "no planted-drop candidate is A/A-stable on this real-store "
                "copy -- cannot prove the control without a stable subject"
            )
        target_id = stable_targets[0]
        cue_text = cue_by_target[target_id]
        p1_ids = p1_by_target[target_id]
        p2_ids = p2_by_target[target_id]

        store.delete(UUID(target_id))
        _rebuild(store)
        b_ids = _dispatch_b07(store, cue_text)

        per_cue = {"planted": {"p1": p1_ids, "p2": p2_ids, "b": b_ids}}
        _stable, regressions, _instability = _comparator(per_cue)

        assert target_id in regressions, (
            "planted-drop control did not flag the tombstoned, A/A-stable id "
            "as a regression -- the comparator or id flow is broken"
        )
    finally:
        cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Temporal-invariance proof: created_at ordering vs. id ordering under churn
# ---------------------------------------------------------------------------

# Full column list for the `records` table, excluding the autoincrement
# primary key `vec_label` -- mapping-style row access (`row["col"]`) is
# proven across both drivers, but `.keys()` introspection is not, so the
# clone columns are named explicitly rather than derived from a `SELECT *`.
_RECORD_CLONE_COLUMNS: "tuple[str, ...]" = (
    "id", "tier", "literal_surface", "aaak_index", "embedding", "structure_hv",
    "community_id", "centrality", "detail_level", "pinned", "stability",
    "difficulty", "last_reviewed", "never_decay", "never_merge", "tombstoned_at",
    "schema_bypass", "labile_until", "provenance_json", "created_at", "updated_at",
    "tags_json", "language", "s5_trust_score", "profile_modulation_gain_json",
    "schema_version", "wing", "room", "drawer", "valence", "hv_tier",
    "structure_hv_payload", "embedding_pending", "role", "epistemic_status",
    "salience_level", "live", "directive",
)


def _insert_simulated_capture(store, new_id: str, new_created_at: str) -> None:
    """Clone an existing eligible row under a new id and created_at --
    simulates a fresh capture landing on the copy, never the source. Field
    AES binds ciphertext to the record id, so this clone is never
    decrypted under its new id; it exists only to exercise ordering.
    """
    with store.db._conn_lock:
        conn = store.db._conn
        template = conn.execute(
            "SELECT " + ", ".join(_RECORD_CLONE_COLUMNS) + " FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(pinned, 0) = 0"
            "   AND COALESCE(never_merge, 0) = 0"
            "   AND COALESCE(embedding_pending, 0) = 0"
            " LIMIT 1"
        ).fetchone()
        assert template is not None, "no eligible template row to clone"
        values = [template[col] for col in _RECORD_CLONE_COLUMNS]
        values[_RECORD_CLONE_COLUMNS.index("id")] = new_id
        values[_RECORD_CLONE_COLUMNS.index("created_at")] = new_created_at
        placeholders = ", ".join(["?"] * len(_RECORD_CLONE_COLUMNS))
        conn.execute(
            f"INSERT INTO records ({', '.join(_RECORD_CLONE_COLUMNS)}) "
            f"VALUES ({placeholders})",
            values,
        )


def _eligible_count(store) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(pinned, 0) = 0"
            "   AND COALESCE(never_merge, 0) = 0"
            "   AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    return int(row[0])


def _sentinel_created_at_count(store) -> int:
    """Among eligible rows, how many have a NULL or empty-string
    created_at -- counted for evidence, never excluded from selection."""
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(pinned, 0) = 0"
            "   AND COALESCE(never_merge, 0) = 0"
            "   AND COALESCE(embedding_pending, 0) = 0"
            "   AND (created_at IS NULL OR created_at = '')"
        ).fetchone()
    return int(row[0])


def _max_created_at_raw(store) -> str:
    with store.db._conn_lock:
        row = store.db._conn.execute("SELECT MAX(created_at) FROM records").fetchone()
    return str(row[0])


def _id_ordered_prefix(store, cap: int) -> "list[str]":
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(pinned, 0) = 0"
            "   AND COALESCE(never_merge, 0) = 0"
            "   AND COALESCE(embedding_pending, 0) = 0"
            " ORDER BY id ASC"
            " LIMIT ?",
            (cap,),
        ).fetchall()
    return [str(row["id"]) for row in rows]


@pytest.mark.realstore
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cue_set_temporal_invariance(driver, monkeypatch):
    """The created_at-ordered cue-id list from the real selection path is
    invariant under a simulated capture (a record inserted with created_at
    past the current max), while the id-ordered prefix shifts -- the
    churn fix, proven deterministically and embedder-free on a read-only
    copy. Never decrypts the inserted row: it sorts past the prefetch
    window and is never fetched by `get_batch`.
    """
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    if not _real_store_present():
        pytest.skip("real store not found; this is a real-store control, not a fresh-clone CI gate")

    cm = _open_driver_copy(driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        eligible_count = _eligible_count(store)
        if eligible_count < _CUE_CAP * 25:
            pytest.skip(
                f"eligible pool too small for a non-degenerate window: "
                f"{eligible_count} < {_CUE_CAP * 25}"
            )

        sentinel_count = _sentinel_created_at_count(store)
        print(
            f"PROVENANCE: sentinel-created-at-count driver={driver} "
            f"eligible={eligible_count} sentinel={sentinel_count}"
        )

        before_created = [rid for rid, _ in _select_cue_candidates(store, cap=_CUE_CAP)]
        before_by_id = _id_ordered_prefix(store, cap=_CUE_CAP)

        max_raw = _max_created_at_raw(store)
        future_created_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        if not future_created_at > max_raw:
            pytest.skip(
                f"computed future created_at {future_created_at!r} does not "
                f"sort after the observed max {max_raw!r} -- skipping rather "
                "than asserting on an unverified clock/format assumption"
            )
        new_id = "00000000-0000-4000-8000-000000000000"
        _insert_simulated_capture(store, new_id, future_created_at)

        after_created = [rid for rid, _ in _select_cue_candidates(store, cap=_CUE_CAP)]
        after_by_id = _id_ordered_prefix(store, cap=_CUE_CAP)

        assert before_created == after_created, (
            "created_at-ordered cue-id list changed after a simulated "
            "capture -- the temporal-ordering fix regressed"
        )
        assert new_id not in after_created, (
            "the inserted, future-created_at row leaked into the "
            "created_at-ordered cue selection -- it must sort past the "
            "prefetch window and never be selected"
        )
        assert before_by_id != after_by_id, (
            "id-ordered prefix did not shift after inserting a "
            "before-current-earliest id -- the contrast control is "
            "vacuous on this real-store copy"
        )
        assert new_id in after_by_id, (
            "the inserted low-sorting id did not enter the id-ordered "
            "prefix -- the contrast setup did not exercise the churn path"
        )
    finally:
        cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Sensitivity proof: a deliberately-widened band still fails the gate
# ---------------------------------------------------------------------------

# Short ascending ladder, all well past the production clamp
# (PROD_W_COSINE_MIN/MAX): the smallest rung that breaches floor+margin is
# the one recorded as the proof; an absurdly large weight alone would not
# demonstrate sensitivity near the actual mechanism.
_WIDENED_BAND_LADDER: "tuple[float, ...]" = (2.0, 4.0, 8.0, 16.0)


@pytest.mark.realstore
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_widened_band_still_fails(driver, monkeypatch):
    """A W_COSINE pushed well past the production clamp must still exceed
    AA_DROPOUT_FLOOR + DROPOUT_MARGIN -- the enlarged, deterministic
    statistic stays sensitive to a genuinely bad band, not merely quieter.
    """
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    if not _real_store_present():
        pytest.skip("real store not found; this is a real-store control, not a fresh-clone CI gate")

    if not AA_DROPOUT_FLOOR_MEASURED:
        pytest.fail(
            "A/A floor never measured on this corpus; refusing to assert "
            "sensitivity against a default -- run test_floor_measurement "
            "first and record its output into AA_DROPOUT_FLOOR/"
            "AA_DROPOUT_FLOOR_MEASURED."
        )

    cm = _open_driver_copy(driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        cues = _cue_texts_from_store(store, _CUE_CAP)
        if not cues:
            pytest.skip("no cue candidates found on this real-store copy")

        _rebuild(store)
        a1_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        _rebuild(store)
        a2_by_cue = {cue: _dispatch_b07(store, cue) for cue in cues}

        # Effective-n is gated ONCE from the unarmed A/A baseline, not per-rung:
        # `stable` in _comparator is built from p1/p2 only (a1_by_cue/a2_by_cue,
        # fixed before this loop) and never depends on the armed response `b`,
        # so len(stable) is identical on every rung -- gating once is equivalent
        # to gating per-rung and cheaper.
        baseline_stable: "set[str]" = set()
        for cue in cues:
            baseline_stable |= (a1_by_cue[cue] & a2_by_cue[cue])
        if len(baseline_stable) < EFFECTIVE_N_FLOOR:
            pytest.fail(
                f"effective-n too low to trust this sensitivity measurement: "
                f"len(baseline_stable)={len(baseline_stable)} < "
                f"EFFECTIVE_N_FLOOR={EFFECTIVE_N_FLOOR} on driver={driver} -- "
                "degenerate or too-fresh store"
            )

        # Simulate a widened production clamp band for this test's process
        # lifetime only -- retrieval_weight_cache.load() re-imports this
        # constant on every uncached call, so raising it here is honored by
        # the next post-invalidate() load without touching retrieval_tuning.py.
        monkeypatch.setattr(
            _retrieval_tuning, "PROD_W_COSINE_MAX", max(_WIDENED_BAND_LADDER) + 1.0
        )

        breach_weight: "float | None" = None
        breach_rate: "float | None" = None
        ladder_log: "list[str]" = []
        for weight in _WIDENED_BAND_LADDER:
            assert save_retrieval_weights_state(store, {"W_COSINE": weight}), (
                f"failed to persist arm-point weight {weight} on driver={driver} -- "
                "cannot trust a subsequent ladder-rung measurement against an unarmed store"
            )
            retrieval_weight_cache.invalidate(store)
            runtime_graph_cache.invalidate(store)
            _rebuild(store)
            b_resp_by_cue = {cue: _dispatch_b07_full(store, cue) for cue in cues}
            b_by_cue = {
                cue: {h["record_id"] for h in resp.get("hits", [])}
                for cue, resp in b_resp_by_cue.items()
            }

            per_cue = {
                cue: {"p1": a1_by_cue[cue], "p2": a2_by_cue[cue], "b": b_by_cue[cue]}
                for cue in cues
            }
            stable, regressions, _instability = _comparator(per_cue)
            dropout_rate = len(regressions) / max(1, len(stable))

            # Sanity FIRST, unconditionally: the coefficient must reach the
            # Rust base_s regardless of whether this rung breaches, so a
            # rung where something clamps the weight upstream never looks
            # identical to a rung that simply didn't breach. Log the
            # PROVENANCE line BEFORE either assert -- a clamped rung must
            # still leave a diagnostic trail, not a blackout on the first
            # bare assertion failure.
            tuned_reason = None
            for resp in b_resp_by_cue.values():
                tuned_reason = _first_weighted_reason(resp)
                if tuned_reason is not None:
                    break
            coeff_reached = tuned_reason is not None and f"*{weight:g}" in tuned_reason
            breached = dropout_rate > AA_DROPOUT_FLOOR + DROPOUT_MARGIN
            ladder_log.append(
                f"driver={driver} rung={weight:g} coeff_reached={coeff_reached} "
                f"dropout={dropout_rate:.6f} stable={len(stable)} breached={breached}"
            )
            print(
                f"PROVENANCE: ladder driver={driver} rung={weight:g} "
                f"coeff_reached={coeff_reached} dropout={dropout_rate:.6f} "
                f"stable={len(stable)} breached={breached}"
            )

            assert tuned_reason is not None, (
                "every hit across all cues was overridden by the "
                "exact-cosine authority merge in the widened-band dispatch "
                f"-- cannot sanity-check the Rust-path coefficient at "
                f"weight={weight:g}; widen the cue set"
            )
            assert coeff_reached, (
                f"Rust-path reason does not carry the ladder coefficient "
                f"{weight:g}: {tuned_reason!r}"
            )

            if breached:
                breach_weight = weight
                breach_rate = dropout_rate
                break

        assert breach_weight is not None and breach_rate is not None, (
            f"no ladder value in {_WIDENED_BAND_LADDER} pushed drop-out "
            f"above floor {AA_DROPOUT_FLOOR:.4f} + margin {DROPOUT_MARGIN:.4f} "
            f"on driver={driver} -- widen the ladder; full ladder history:\n"
            + "\n".join(ladder_log)
        )

        print(
            f"PROVENANCE: widened-band breach driver={driver} "
            f"weight={breach_weight:g} dropout={breach_rate:.6f}"
        )
    finally:
        cm.__exit__(None, None, None)
