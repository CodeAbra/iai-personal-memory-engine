from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import save_state, default_state
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _isolate_iai_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)


def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
) -> MemoryRecord:
    rng = random.Random(hash(literal_surface))
    raw = [rng.gauss(0.0, 1.0) for _ in range(embed_dim)]
    mag = math.sqrt(sum(x * x for x in raw))
    embedding = [x / mag for x in raw] if mag > 0 else raw
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


def _edge_row(*, src: str, dst: str, edge_type: str = "hebbian", weight: float = 0.5) -> dict:
    return {
        "src": src,
        "dst": dst,
        "edge_type": edge_type,
        "weight": weight,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _insert_edges(store: MemoryStore, rows: list[dict]) -> None:
    tbl = store.db.open_table(EDGES_TABLE)
    tbl.merge_insert(["src", "dst", "edge_type"]).when_matched_update_all().execute(rows)


def _make_pipeline(store: MemoryStore, tmp_path: Path) -> SleepPipeline:
    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    return SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)


class TestOrphanEdgeSweep:
    def test_orphan_src_and_dst_edges_deleted_live_edge_survives(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim

        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        rec_b = _make_record(embed_dim=embed_dim, literal_surface="record b")
        store.insert(rec_a)
        store.insert(rec_b)

        orphan_src_id = str(uuid.uuid4())
        orphan_dst_id = str(uuid.uuid4())

        _insert_edges(store, [
            _edge_row(src=str(rec_a.id), dst=str(rec_b.id)),  # live-live
            _edge_row(src=orphan_src_id, dst=str(rec_b.id)),  # orphan src
            _edge_row(src=str(rec_a.id), dst=orphan_dst_id),  # orphan dst
        ])

        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["action"] == "orphan_edge_sweep"
        assert payload["orphan_edges_deleted"] == 2
        assert payload["edges_scanned"] == 3

        remaining = store.db.open_table(EDGES_TABLE).to_pandas()
        assert len(remaining) == 1
        assert remaining.iloc[0]["src"] == str(rec_a.id)
        assert remaining.iloc[0]["dst"] == str(rec_b.id)

    def test_tombstoned_record_edges_swept(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim

        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        rec_b = _make_record(embed_dim=embed_dim, literal_surface="record b")
        store.insert(rec_a)
        store.insert(rec_b)

        _insert_edges(store, [
            _edge_row(src=str(rec_a.id), dst=str(rec_b.id)),
        ])

        tbl = store.db.open_table(RECORDS_TABLE)
        tbl.update(
            where=f"id = '{rec_b.id}'",
            values={"tombstoned_at": datetime.now(timezone.utc).isoformat()},
        )

        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["orphan_edges_deleted"] == 1

        remaining = store.db.open_table(EDGES_TABLE).to_pandas()
        assert len(remaining) == 0

    def test_empty_edges_table_zero_counts(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        pipeline = _make_pipeline(store, tmp_path)

        done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["action"] == "orphan_edge_sweep"
        assert payload["edges_scanned"] == 0
        assert payload["orphan_edges_deleted"] == 0

    def test_sweep_failure_contained_no_exception_escapes(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim
        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        store.insert(rec_a)

        pipeline = _make_pipeline(store, tmp_path)

        with patch.object(
            store.db, "open_table", side_effect=RuntimeError("boom"),
        ):
            done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["action"] == "orphan_edge_sweep"
        assert "error" in payload

    def test_event_emitted_when_deletes_greater_than_zero(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim
        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        store.insert(rec_a)

        orphan_dst_id = str(uuid.uuid4())
        _insert_edges(store, [
            _edge_row(src=str(rec_a.id), dst=orphan_dst_id),
        ])

        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["orphan_edges_deleted"] == 1

        events = query_events(store, kind="orphan_edge_sweep")
        assert len(events) == 1

    def test_live_edge_appearing_mid_sweep_not_deleted(
        self, tmp_path: Path,
    ) -> None:
        # TOCTOU regression: a record + its edge committed concurrently
        # during the sweep must NOT be swept as orphan. The sweep reads the
        # edge scan FIRST and live-ids SECOND, so a record made live between
        # the edge snapshot and the live-id read is present in that read and
        # its edge survives.
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim

        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        store.insert(rec_a)

        # This record + edge already exist in the edge snapshot, but the
        # record only becomes live partway through the sweep.
        rec_late = _make_record(embed_dim=embed_dim, literal_surface="late arrival")
        _insert_edges(store, [
            _edge_row(src=str(rec_a.id), dst=str(rec_late.id)),
        ])

        pipeline = _make_pipeline(store, tmp_path)

        import iai_mcp.lilli.cycle.sleep_pipeline._compact as _compact

        original_read = _compact._read_live_ids
        inserted = {"done": False}

        def _read_live_ids_with_race(store_arg):
            # First call is the post-edge-scan live-id read; simulate the
            # concurrent commit landing right before it so the late record is
            # now live. With the correct ordering its edge must survive.
            if not inserted["done"]:
                store_arg.insert(rec_late)
                inserted["done"] = True
            return original_read(store_arg)

        with patch.object(_compact, "_read_live_ids", _read_live_ids_with_race):
            done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["action"] == "orphan_edge_sweep"
        assert payload["orphan_edges_deleted"] == 0

        remaining = store.db.open_table(EDGES_TABLE).to_pandas()
        assert len(remaining) == 1
        assert remaining.iloc[0]["src"] == str(rec_a.id)
        assert remaining.iloc[0]["dst"] == str(rec_late.id)

    def test_orphan_count_reflects_actual_deletions(
        self, tmp_path: Path,
    ) -> None:
        # IN-06: the metric must count rows actually deleted, not triples
        # submitted. A duplicate-src orphan edge produces exactly one row.
        store = _make_store(tmp_path)
        embed_dim = store._embed_dim

        rec_a = _make_record(embed_dim=embed_dim, literal_surface="record a")
        store.insert(rec_a)

        orphan_id = str(uuid.uuid4())
        _insert_edges(store, [
            _edge_row(src=str(rec_a.id), dst=orphan_id, edge_type="hebbian"),
            _edge_row(src=str(rec_a.id), dst=orphan_id, edge_type="contradicts"),
        ])

        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_hippo_cleanup(interrupt_check=None)

        assert done is True
        assert payload["edges_scanned"] == 2
        assert payload["orphan_edges_deleted"] == 2

        remaining = store.db.open_table(EDGES_TABLE).to_pandas()
        assert len(remaining) == 0

    def test_hippo_cleanup_noop_byte_untouched(self) -> None:
        pipeline = SleepPipeline.__new__(SleepPipeline)
        done, payload = pipeline._step_hippo_cleanup_noop(interrupt_check=None)
        assert done is True
        assert payload == {"action": "hippo_cleanup_noop"}

    def test_interrupt_check_still_honored(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        pipeline = _make_pipeline(store, tmp_path)

        done, payload = pipeline._step_hippo_cleanup(
            interrupt_check=lambda: True,
        )
        assert done is False
        assert payload == {}
