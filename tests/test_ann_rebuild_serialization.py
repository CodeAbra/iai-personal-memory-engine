from __future__ import annotations

import platform
import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="threading + POSIX semantics",
)


@pytest.fixture
def iai_home_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    yield tmp_path


def _make_record(i: int):
    from iai_mcp.types import MemoryRecord, EMBED_DIM

    vec = [0.0] * EMBED_DIM
    vec[i % EMBED_DIM] = 1.0
    vec[(i * 7 + 3) % EMBED_DIM] = 0.5
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"rebuild serialization test record {i}",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "rebuild-test", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
    )


def _store_with_records(home, n: int):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=home)
    tbl = store.db.open_table("records")
    records = [_make_record(i) for i in range(n)]
    tbl.add([store._to_row(r) for r in records])
    return store, tbl, records


def _live_labels(db) -> list[int]:
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT vec_label FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()
    return [int(r["vec_label"]) for r in rows]


def test_concurrent_rebuilds_serialize_and_converge(iai_home_rebuild):
    store, _tbl, _records = _store_with_records(iai_home_rebuild, 40)
    db = store.db
    errors: list[str] = []

    def rebuild():
        try:
            db._rebuild_index_from_sqlite()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=rebuild) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"concurrent rebuilds raised: {errors}"

    labels = _live_labels(db)
    assert len(db._label_map) == len(labels)
    # get_items raises for any label missing from the active buffer.
    db._hnsw.get_items(labels)
    assert db._ann_journal is None
    store.close()


def test_big_k_query_restores_shared_ef(iai_home_rebuild):
    """ef is a property of the SHARED index: a big-k caller (a large teach
    weave, an exact-escalation probe) must not leave every later awake
    recall running at its inflated ef until the next rebuild."""
    store, _tbl, _records = _store_with_records(iai_home_rebuild, 30)
    db = store.db

    db._hnsw.set_ef(5)
    assert int(db._hnsw.ef) == 5

    cue = [0.0] * len(_records[0].embedding)
    cue[0] = 1.0
    got = store.query_similar(cue, k=20)
    assert got, "query must return neighbors"

    assert int(db._hnsw.ef) == 5, (
        f"shared ef left inflated at {db._hnsw.ef} after a k=20 query"
    )
    store.close()


def test_reuse_path_never_falls_back_under_churn(iai_home_rebuild):
    """The zero-per-cycle-allocation invariant, measured: repeated rebuilds
    over a churning corpus must never trip the fresh-allocation fallback.
    The counter observes the replace_deleted label-drop (hnswlib re-homing
    erases live lookup entries) that the in-place refill eliminates."""
    store, tbl, records = _store_with_records(iai_home_rebuild, 60)
    db = store.db
    db._rebuild_index_from_sqlite()

    for cycle in range(5):
        victims = records[cycle * 8:(cycle + 1) * 8]
        for v in victims:
            tbl.delete(f"id = '{v.id}'")
        newbies = [_make_record(1000 + cycle * 10 + i) for i in range(10)]
        tbl.add([store._to_row(r) for r in newbies])
        records.extend(newbies)

        result = db._rebuild_index_from_sqlite()
        labels = _live_labels(db)
        db._hnsw.get_items(labels)
        assert len(db._label_map) == len(labels)
        assert "reuse_collisions_total" in result

    assert db._reuse_collision_count == 0, (
        f"fresh-allocation fallback fired {db._reuse_collision_count} time(s) —"
        f" the in-place refill must hold the zero-alloc invariant under churn"
    )
    store.close()


def _pause_rebuild_before_swap(monkeypatch):
    """Patch the post-build verification hook to pause the rebuilder between
    standby build and swap — the exact window a concurrent write can vanish in."""
    from iai_mcp.hippo import HippoDB

    build_done = threading.Event()
    resume = threading.Event()
    real_check = HippoDB._standby_has_all_labels

    def paused_check(index, labels):
        build_done.set()
        assert resume.wait(timeout=15), "rebuild pause never released"
        return real_check(index, labels)

    monkeypatch.setattr(
        HippoDB, "_standby_has_all_labels", staticmethod(paused_check)
    )
    return build_done, resume


def test_add_during_rebuild_survives_swap(iai_home_rebuild, monkeypatch):
    store, tbl, _records = _store_with_records(iai_home_rebuild, 20)
    db = store.db
    build_done, resume = _pause_rebuild_before_swap(monkeypatch)

    rebuild_err: list[str] = []

    def rebuild():
        try:
            db._rebuild_index_from_sqlite()
        except Exception as exc:  # noqa: BLE001
            rebuild_err.append(f"{type(exc).__name__}: {exc}")

    t = threading.Thread(target=rebuild)
    t.start()
    assert build_done.wait(timeout=15)

    new_rec = _make_record(9999)
    tbl.add([store._to_row(new_rec)])
    new_label = db._label_map[str(new_rec.id)]

    resume.set()
    t.join(timeout=30)
    assert not rebuild_err, rebuild_err

    # The record committed mid-rebuild must be resolvable in the ACTIVE buffer
    # after the swap — a miss here is the silent recall record-loss.
    db._hnsw.get_items([new_label])
    assert str(new_rec.id) in db._label_map
    assert db._ann_journal is None
    store.close()


def test_delete_during_rebuild_not_resurrected(iai_home_rebuild, monkeypatch):
    import numpy as np

    store, tbl, records = _store_with_records(iai_home_rebuild, 20)
    db = store.db
    victim = records[7]
    victim_label = db._label_map[str(victim.id)]
    victim_vec = np.array(victim.embedding, dtype=np.float32).reshape(1, -1)

    build_done, resume = _pause_rebuild_before_swap(monkeypatch)

    t = threading.Thread(target=db._rebuild_index_from_sqlite)
    t.start()
    assert build_done.wait(timeout=15)

    tbl.delete(f"id = '{victim.id}'")

    resume.set()
    t.join(timeout=30)

    # The delete committed mid-rebuild must hold after the swap: the label is
    # gone from the map and the ANN search never returns it.
    assert str(victim.id) not in db._label_map
    k = min(10, db._hnsw.get_current_count())
    got_labels, _dists = db._hnsw.knn_query(victim_vec, k=k)
    assert victim_label not in got_labels[0].tolist()
    assert db._ann_journal is None
    store.close()
