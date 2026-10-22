"""Internal-prompt/template-pollution records and operator-listed leak
records must be tombstonable: journaled, snapshotted, reversible, blind to
ordinary conversation, and never touch a real record's literal_surface."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

_POLLUTION_TEXT = (
    "Below is one record from a developer's personal memory system.\n\n"
    "Write the single most natural short query this person would type to "
    "pull this exact record back out months later."
)


@pytest.fixture(autouse=True)
def _passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")
    yield


def _capture(store, text, sid, directive=False):
    from iai_mcp.capture import capture_turn

    return UUID(capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=sid, role="user", live_turn=True, directive=directive,
    )["record_id"])


def _seed_pollution_record(store, text, sid, *, embed_index, never_merge=False) -> UUID:
    """Plants a pollution-prefixed row directly via store.insert(), bypassing
    capture_turn(): capture_turn()'s own noise guard now refuses this exact
    prefix (_NOISE_PATTERNS matches it), but these rows model the diagnosed
    historical burst that landed BEFORE that guard existed. A distinct
    one-hot embedding per row keeps store.insert()'s own pattern-separation
    gate from folding these near-identical texts into a single survivor."""
    dim = store._embed_dim
    embedding = [0.0] * dim
    embedding[embed_index % dim] = 1.0
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["capture", "role:user"],
    )
    store.insert(record)
    return record.id


def _tombstoned_at(store, rid) -> "str | None":
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT tombstoned_at FROM records WHERE id = ?", (str(rid),)
        ).fetchone()
    return row["tombstoned_at"]


def _live_flag(store, rid) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT live FROM records WHERE id = ?", (str(rid),)
        ).fetchone()
    return row["live"]


def _seed_corpus(store, tmp_path):
    """Plants: N=2 predicate-matching pollution rows, 2 explicit-id leak
    rows (ordinary text, targeted only by id), 2 ordinary conversational
    rows, and 1 pinned row that coincidentally matches the predicate."""
    pollution_ids = [
        _seed_pollution_record(
            store, _POLLUTION_TEXT + " variant one", "s-pollution-1", embed_index=0,
        ),
        _seed_pollution_record(
            store, _POLLUTION_TEXT + " variant two", "s-pollution-2", embed_index=1,
        ),
    ]
    explicit_ids = [
        _capture(store, "The greenhouse controller passed acceptance testing.", "s-leak-1"),
        _capture(store, "From now on reply in English please.", "s-leak-2"),
    ]
    ordinary_ids = [
        _capture(store, "Working on Zephyrbot notification handling today.", "s-real-1"),
        _capture(store, "Coral reefs host a quarter of all known marine species.", "s-real-2"),
    ]
    pinned_id = _seed_pollution_record(
        store, _POLLUTION_TEXT + " pinned variant", "s-pinned",
        embed_index=2, never_merge=True,
    )
    assert len({pollution_ids[0], pollution_ids[1], pinned_id}) == 3, (
        "distinct one-hot embeddings must prevent the pattern-separation "
        "gate from folding these near-identical texts into one survivor"
    )
    return pollution_ids, explicit_ids, ordinary_ids, pinned_id


def test_dry_run_flags_predicate_and_explicit_ids_only(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    dry = sweep_template_pollution(
        store, apply=False, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    assert dry["mode"] == "dry-run"
    assert dry["pollution_found"] == len(pollution_ids) + len(explicit_ids)
    assert dry["tombstoned"] == 0
    assert dry["skipped_pinned"] == 1, "the pinned coincidental match must be counted separately"
    assert dry["missing_explicit_ids"] == []
    assert dry["errors"] == []

    # Ordinary conversational rows and the pinned row must never be touched.
    for rid in ordinary_ids + [pinned_id]:
        assert _tombstoned_at(store, rid) is None


def test_missing_explicit_id_reported_not_silently_dropped(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    _capture(store, "ordinary text", "s-1")
    flush_record_buffer(store)

    ghost_id = str(uuid4())
    dry = sweep_template_pollution(
        store, apply=False, store_path=tmp_path / "store",
        explicit_ids=[ghost_id],
    )
    assert dry["missing_explicit_ids"] == [ghost_id]
    assert dry["pollution_found"] == 0


def test_pinned_explicit_id_reported_as_protected_not_missing(tmp_path):
    """A leak id that IS in the store but pinned (never_merge/never_decay)
    must never read as "missing" -- that reads as a typo'd id and hides a
    live, un-purged leak record from the operator."""
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pinned_leak_id = _capture(store, "ordinary text, but pinned and leaked", "s-pinned-leak")
    store.db.open_table("records").update(
        where=f"id = '{pinned_leak_id}'", values={"never_merge": 1},
    )
    flush_record_buffer(store)

    dry = sweep_template_pollution(
        store, apply=False, store_path=tmp_path / "store",
        explicit_ids=[str(pinned_leak_id)],
    )

    assert dry["missing_explicit_ids"] == [], (
        "a found-but-pinned explicit id must not be reported as missing"
    )
    assert dry["explicit_ids_protected"] == [str(pinned_leak_id)]
    assert dry["skipped_pinned"] == 1
    assert dry["pollution_found"] == 0


def test_already_tombstoned_explicit_id_reported_not_missing(tmp_path):
    """A second sweep re-listing an id this tool already tombstoned must
    report it as already-tombstoned, not missing -- an idempotent re-run
    must never make a real leak record look purely absent."""
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    leak_id = _capture(store, "The greenhouse controller passed acceptance testing.", "s-leak")
    flush_record_buffer(store)

    sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(leak_id)],
    )

    again = sweep_template_pollution(
        store, apply=False, store_path=tmp_path / "store",
        explicit_ids=[str(leak_id)],
    )
    assert again["missing_explicit_ids"] == [], (
        "an already-tombstoned explicit id must not be reported as missing"
    )
    assert again["explicit_ids_already_tombstoned"] == [str(leak_id)]
    assert again["pollution_found"] == 0


def test_apply_tombstones_targets_spares_ordinary_and_pinned(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    ordinary_surface_before = store.get(ordinary_ids[0]).literal_surface

    applied = sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    assert applied["mode"] == "apply"
    assert applied["tombstoned"] == len(pollution_ids) + len(explicit_ids)
    assert applied["skipped_pinned"] == 1
    assert applied["snapshot_dir"] and Path(applied["snapshot_dir"]).exists()
    assert applied["journal"] and Path(applied["journal"]).exists()

    journal_entries = [
        json.loads(line) for line in Path(applied["journal"]).read_text().splitlines()
    ]
    journaled_ids = {e["record_id"] for e in journal_entries}
    assert journaled_ids == {str(x) for x in pollution_ids + explicit_ids}

    for rid in pollution_ids + explicit_ids:
        assert _tombstoned_at(store, rid) is not None, f"{rid} must be tombstoned"
        assert _live_flag(store, rid) == 0, (
            "live must move with tombstoned_at in the same write"
        )

    for rid in ordinary_ids + [pinned_id]:
        assert _tombstoned_at(store, rid) is None, f"{rid} must survive"

    # literal_surface of a surviving real record is byte-identical.
    assert store.get(ordinary_ids[0]).literal_surface == ordinary_surface_before

    # Never-DELETE discipline: the row still exists in the table at all.
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT id FROM records WHERE id = ?", (str(pollution_ids[0]),)
        ).fetchone()
    assert row is not None, "tombstone must never hard-DELETE the row"

    assert applied["ann_rebuild_invoked"] is True
    assert applied["exact_index_invalidated"] is True


def test_apply_sweeps_orphan_edges_both_columns(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    now_iso = "2026-01-01T00:00:00+00:00"
    tbl = store.db.open_table(EDGES_TABLE)
    tbl.merge_insert(["src", "dst", "edge_type"]).when_matched_update_all().execute([
        {  # pollution record is the SRC endpoint
            "src": str(pollution_ids[0]), "dst": str(ordinary_ids[0]),
            "edge_type": "hebbian", "weight": 0.5, "updated_at": now_iso,
        },
        {  # explicit-id leak record is the DST endpoint
            "src": str(ordinary_ids[1]), "dst": str(explicit_ids[0]),
            "edge_type": "hebbian", "weight": 0.5, "updated_at": now_iso,
        },
        {  # a live-live edge that must survive
            "src": str(ordinary_ids[0]), "dst": str(ordinary_ids[1]),
            "edge_type": "hebbian", "weight": 0.5, "updated_at": now_iso,
        },
    ])

    applied = sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    assert applied["orphan_edge_sweep"] is not None
    assert applied["orphan_edge_sweep"]["orphan_edges_deleted"] >= 2

    remaining = store.db.open_table(EDGES_TABLE).to_pandas()
    remaining_pairs = set(zip(remaining["src"].astype(str), remaining["dst"].astype(str)))
    assert (str(pollution_ids[0]), str(ordinary_ids[0])) not in remaining_pairs
    assert (str(ordinary_ids[1]), str(explicit_ids[0])) not in remaining_pairs
    assert (str(ordinary_ids[0]), str(ordinary_ids[1])) in remaining_pairs


def test_post_apply_query_never_returns_tombstoned_pollution(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)
    embedding = store.get(pollution_ids[0]).embedding

    sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    matrix_after = {str(rid) for rid, _ in store.exact_top_k(embedding, k=10)}
    assert str(pollution_ids[0]) not in matrix_after
    assert str(pollution_ids[1]) not in matrix_after


def test_apply_never_touches_literal_surface_of_tombstoned_row(tmp_path):
    """The tombstone write sets only tombstoned_at/live -- the row's own
    literal_surface ciphertext/plaintext must be provably unchanged (a
    UPDATE that also rewrote the surface would break verbatim write-once)."""
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)
    surface_before = store.get(pollution_ids[0]).literal_surface

    sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    surface_after = store.get(pollution_ids[0]).literal_surface
    assert surface_after == surface_before


def test_apply_is_reversible_from_snapshot(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    applied = sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )
    assert _tombstoned_at(store, pollution_ids[0]) is not None
    store.close()

    hippo_dir = tmp_path / "store" / "hippo"
    shutil.rmtree(hippo_dir)
    shutil.copytree(applied["snapshot_dir"], hippo_dir)

    restored = MemoryStore(path=tmp_path / "store")
    assert _tombstoned_at(restored, pollution_ids[0]) is None, (
        "restoring the pre-apply snapshot must undo the tombstone"
    )
    restored.close()


def test_post_tombstone_maintenance_failure_preserves_rollback_pointers(tmp_path, monkeypatch):
    """A failure in the post-tombstone maintenance calls (orphan sweep,
    index rebuild) must not lose the operator's own rollback pointers --
    the snapshot/journal are already durable on disk and must still be
    reported, not swallowed by an unhandled exception."""
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer
    from iai_mcp.lilli.cycle.sleep_pipeline import _compact

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    def _boom(store):
        raise RuntimeError("simulated orphan-edge-sweep failure")

    monkeypatch.setattr(_compact, "_sweep_orphan_edges", _boom)

    applied = sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    assert applied["snapshot_dir"] and Path(applied["snapshot_dir"]).exists(), (
        "the rollback snapshot must still be reported after a maintenance failure"
    )
    assert applied["journal"] and Path(applied["journal"]).exists()
    assert applied["tombstoned"] == len(pollution_ids) + len(explicit_ids), (
        "the tombstone write itself must have already completed before "
        "the maintenance failure"
    )
    assert any("orphan_edge_sweep" in e for e in applied["errors"])


def test_apply_idempotent_second_run_finds_nothing(tmp_path):
    from iai_mcp.migrate._template_pollution import sweep_template_pollution
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    pollution_ids, explicit_ids, ordinary_ids, pinned_id = _seed_corpus(store, tmp_path)
    flush_record_buffer(store)

    sweep_template_pollution(
        store, apply=True, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )

    again = sweep_template_pollution(
        store, apply=False, store_path=tmp_path / "store",
        explicit_ids=[str(x) for x in explicit_ids],
    )
    assert again["pollution_found"] == 0, "sweep must be idempotent"
    assert again["missing_explicit_ids"] == [], (
        "explicit ids tombstoned by the prior apply must not read as missing"
    )
    assert sorted(again["explicit_ids_already_tombstoned"]) == sorted(
        str(x) for x in explicit_ids
    )
