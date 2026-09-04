"""FHRR stop-write regression: a REAL production consolidation cycle on a
seeded store leaves new-minted semantic records carrying the clean
``hv_tier='bsc'`` / empty ``structure_hv_payload`` default, measured by
record-id set-difference so backfilled/pre-existing rows can never inflate
the numerator. Covers both mint entry paths: the direct CLUSTER_SUMMARY step
and the dynamically-driven REM leg (run_heavy_consolidation).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

_MEMBER_COUNT = 6


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _distinct_embedding(i: int) -> list:
    vec = [0.1] * EMBED_DIM
    span = EMBED_DIM // (_MEMBER_COUNT + 2)
    start = i * span
    for j in range(start, start + span):
        vec[j] = 0.9
    return vec


def _seed_member(store, i: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"alice project note {i}: distinguishing detail about topic area {i}",
        aaak_index="",
        embedding=_distinct_embedding(i),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)
    return rec


def _seed_cluster(store) -> list[MemoryRecord]:
    """Distinct-embedding episodic members ring-wired with hebbian edges well
    above CLUSTER_EDGE_MIN_WEIGHT (0.05) and CLUSTER_MIN_SIZE (3), so a real
    community forms and is not dedup-folded."""
    members = [_seed_member(store, i) for i in range(_MEMBER_COUNT)]
    pairs = [
        (members[i].id, members[(i + 1) % len(members)].id)
        for i in range(len(members))
    ]
    store.boost_edges(pairs, delta=0.5, edge_type="hebbian")
    return members


def _snapshot_ids(store) -> set:
    return {r.id for r in store.all_records()}


def _measure_new_semantic_hv_default(store, ids_before: set) -> tuple[int, int, set]:
    """Returns (N, M, new_ids): N = count of newly-minted semantic records
    (fold-excluded by construction — a fold rewrites the id to the survivor,
    so it never appears in the set-difference); M = count of those N carrying
    the clean hv_tier='bsc' / empty structure_hv_payload default."""
    ids_after = _snapshot_ids(store)
    new_ids = ids_after - ids_before

    new_semantic = []
    for rid in new_ids:
        batch = store.get_batch([rid])
        rec = batch.get(rid)
        assert rec is not None, f"new id {rid} did not round-trip from the store"
        assert rec.tier == "semantic", (
            f"new id {rid} has tier {rec.tier!r}, expected 'semantic' — a "
            "non-semantic row leaked into new_ids (buffer-flush-before-"
            "snapshot invariant likely violated)"
        )
        new_semantic.append(rec)

    N = len(new_semantic)
    M = sum(
        1 for r in new_semantic
        if r.hv_tier == "bsc" and r.structure_hv_payload == b""
    )
    return N, M, new_ids


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_real_cluster_summary_cycle_mints_bsc_default_hv_tier(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    store = MemoryStore(path=tmp_path)
    _seed_cluster(store)

    # _process_cluster_summaries flushes these itself at its own start, but a
    # test must flush FIRST too — otherwise buffered seed rows land AFTER the
    # ids_before snapshot and pollute new_ids with non-semantic records.
    flush_record_buffer(store)
    flush_edge_buffer(store)

    ids_before = _snapshot_ids(store)
    created = _process_cluster_summaries(store)

    N, M, new_ids = _measure_new_semantic_hv_default(store, ids_before)

    assert N >= 1, (
        f"expected at least one new semantic summary from a store seeded "
        f"with {_MEMBER_COUNT} distinct-embedding members in a ring hebbian "
        f"cluster; got N=0 — the seeding construction is broken (this is a "
        f"hard fail, not a vacuous-retry: a store WE seed must mint)"
    )
    assert M == N, (
        f"N={N} new semantic records, M={M} carry the clean hv_tier='bsc' / "
        f"empty structure_hv_payload default — every new semantic summary "
        f"must land on the default; M < N means some new record still "
        f"carries a non-default hv_tier/payload (the stopped mint write "
        f"regressed)"
    )
    assert N == created, (
        f"id-set-difference new-semantic count N={N} does not match "
        f"_process_cluster_summaries' own fold-excluded return created={created} "
        f"— a summary arrived via some path other than the measured one"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_rem_leg_run_heavy_consolidation_mints_bsc_default_hv_tier(driver, tmp_path, monkeypatch) -> None:
    """Dynamically drives run_heavy_consolidation — exactly what
    dream.run_rem_cycle invokes (same internally-constructed
    BudgetLedger/RateLimitLedger, same llm_enabled=False + has_api_key=False
    gate-off) — on a FRESH seeded store, proving the REM entry path reaches
    the same shared mint site and lands the clean default. Not a read-only
    call-graph assertion: the cycle actually runs."""
    _select_driver(driver, monkeypatch)
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    # A FRESH store — reusing a store already summarized by a prior cycle
    # would coverage-skip its clusters (_cluster_already_summarized) and
    # legitimately mint zero new summaries, which is not what this test
    # measures.
    store = MemoryStore(path=tmp_path / "rem")
    _seed_cluster(store)

    flush_record_buffer(store)
    flush_edge_buffer(store)

    ids_before = _snapshot_ids(store)
    result = run_heavy_consolidation(
        store,
        "system",
        SleepConfig(llm_enabled=False),
        BudgetLedger(store),
        RateLimitLedger(store),
        has_api_key=False,
    )

    N, M, new_ids = _measure_new_semantic_hv_default(store, ids_before)

    assert N >= 1, (
        f"expected at least one new semantic summary from run_heavy_consolidation "
        f"on a fresh store seeded with {_MEMBER_COUNT} distinct-embedding members "
        f"in a ring hebbian cluster; got N=0 — either the seeding construction "
        f"broke or _decay_edges pruned the freshly-created hebbian edges before "
        f"clustering ran (this is a hard fail, not a vacuous-retry)"
    )
    assert M == N, (
        f"N={N} new semantic records from the REM entry path, M={M} carry the "
        f"clean hv_tier='bsc' / empty structure_hv_payload default — every new "
        f"semantic summary must land on the default at the shared mint site"
    )
    assert N == result["summaries_created"], (
        f"id-set-difference new-semantic count N={N} does not match "
        f"run_heavy_consolidation's own fold-excluded "
        f"summaries_created={result['summaries_created']}"
    )
