from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from iai_mcp import hippea_cascade
from iai_mcp.community import CommunityAssignment
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


def _make_record(
    *,
    literal: str,
    community_id: UUID | None = None,
    centrality: float = 0.5,
    dim: int = 1024,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=literal,
        aaak_index="",
        embedding=[0.0] * dim,
        community_id=community_id,
        centrality=centrality,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


@pytest.fixture
def reset_warm_lru() -> None:
    hippea_cascade._warm_lru.clear()
    yield
    hippea_cascade._warm_lru.clear()


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password",
        lambda s, u, p: fake_store.__setitem__((s, u), p),
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None),
    )
    yield fake_store


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def test_compute_salient_communities_empty_history(
    store: MemoryStore, reset_warm_lru: None
) -> None:
    c1, c2, c3 = uuid4(), uuid4(), uuid4()
    assignment = CommunityAssignment(
        top_communities=[c1, c2, c3],
        community_centroids={c1: [0.0] * 4, c2: [0.0] * 4, c3: [0.0] * 4},
    )
    result = hippea_cascade.compute_salient_communities(store, assignment, top_k=3)
    assert result == [c1, c2, c3]


def test_compute_salient_communities_ranks_by_pe(
    store: MemoryStore, reset_warm_lru: None
) -> None:
    c_dominant, c_rare = uuid4(), uuid4()
    assignment = CommunityAssignment(
        top_communities=[c_dominant, c_rare],
        community_centroids={
            c_dominant: [0.0] * 4,
            c_rare: [0.0] * 4,
        },
    )
    c_mid = uuid4()
    assignment = CommunityAssignment(
        top_communities=[c_dominant, c_mid, c_rare],
        community_centroids={
            c_dominant: [0.0] * 4,
            c_mid: [0.0] * 4,
            c_rare: [0.0] * 4,
        },
    )
    for i in range(15):
        sid = f"s{i}"
        write_event(
            store, "session_started", {"session_id": sid, "idx": i},
            severity="info", session_id=sid,
        )
        if i < 9:
            cid = c_dominant
        elif i < 12:
            cid = c_mid
        else:
            cid = c_rare
        for _ in range(3):
            write_event(
                store, "retrieval_used",
                {"session_id": sid, "community_id": str(cid)},
                severity="info", session_id=sid,
            )
    top = hippea_cascade.compute_salient_communities(store, assignment, top_k=1)
    top3 = hippea_cascade.compute_salient_communities(store, assignment, top_k=3)
    assert c_dominant in top3, (
        f"dominant must be in top-3 salience set; got {top3}"
    )


def test_compute_salient_communities_variance_weighting(
    store: MemoryStore, reset_warm_lru: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c_stable, c_bursty = uuid4(), uuid4()
    assignment = CommunityAssignment(
        top_communities=[c_stable, c_bursty],
        community_centroids={
            c_stable: [0.0] * 4,
            c_bursty: [0.0] * 4,
        },
    )
    now = datetime.now(timezone.utc)
    sessions_mock = []
    retrievals_mock = []
    for day in range(4):
        sid = f"stable-{day}"
        ts = now - timedelta(days=day)
        sessions_mock.append(
            {"session_id": sid, "ts": ts, "data": {"session_id": sid}}
        )
        retrievals_mock.append(
            {"session_id": sid, "ts": ts,
             "data": {"session_id": sid, "community_id": str(c_stable)}}
        )
    for i in range(2):
        sid = f"bursty-{i}"
        ts = now
        sessions_mock.append(
            {"session_id": sid, "ts": ts, "data": {"session_id": sid}}
        )
        retrievals_mock.append(
            {"session_id": sid, "ts": ts,
             "data": {"session_id": sid, "community_id": str(c_bursty)}}
        )

    def _fake_query_events(_store, kind=None, since=None, limit=None):
        if kind == "session_started":
            return sessions_mock
        if kind == "retrieval_used":
            return retrievals_mock
        return []

    import iai_mcp.events as ev_mod
    monkeypatch.setattr(ev_mod, "query_events", _fake_query_events)

    top = hippea_cascade.compute_salient_communities(store, assignment, top_k=2)
    assert top[0] == c_stable, (
        f"stable must rank first: got {top}; "
        f"expected stable={c_stable} at position 0, bursty={c_bursty} at 1"
    )
    assert top[1] == c_bursty


def test_simplified_formula_at_low_data(
    store: MemoryStore, reset_warm_lru: None
) -> None:
    c1, c2 = uuid4(), uuid4()
    assignment = CommunityAssignment(
        top_communities=[c1, c2],
        community_centroids={c1: [0.0] * 4, c2: [0.0] * 4},
    )
    for i in range(2):
        write_event(
            store, "session_started", {"idx": i},
            severity="info", session_id=f"s{i}",
        )
    top = hippea_cascade.compute_salient_communities(store, assignment, top_k=2)
    assert top == [c1, c2]


def test_warm_records_populates_lru(
    store: MemoryStore, reset_warm_lru: None
) -> None:
    recs = [
        _make_record(literal=f"rec-{i}", dim=store.embed_dim) for i in range(3)
    ]
    for r in recs:
        store.insert(r)
    ids = [r.id for r in recs]
    inserted = asyncio.run(hippea_cascade.warm_records(ids, store))
    assert inserted == 3
    snap = hippea_cascade.snapshot_warm_ids()
    assert set(snap) == set(ids)


def test_lru_evicts_at_maxsize(reset_warm_lru: None) -> None:
    lru = hippea_cascade._warm_lru
    for _ in range(201):
        lru[uuid4()] = {"fake": True}
    assert len(lru) == 200


def test_lru_ttl_expires(monkeypatch: pytest.MonkeyPatch, reset_warm_lru: None) -> None:
    from cachetools import TTLCache

    fake_now = [1000.0]

    def _fake_timer() -> float:
        return fake_now[0]

    local_lru = TTLCache(maxsize=200, ttl=1800, timer=_fake_timer)
    rid = uuid4()
    local_lru[rid] = {"fake": True}
    assert rid in local_lru
    fake_now[0] += 1801
    assert rid not in local_lru


def test_cascade_is_read_only(
    store: MemoryStore, reset_warm_lru: None
) -> None:
    cid = uuid4()
    recs = [
        _make_record(literal=f"r{i}", community_id=cid, centrality=0.5,
                     dim=store.embed_dim)
        for i in range(3)
    ]
    for r in recs:
        store.insert(r)
    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * store.embed_dim},
        node_to_community={r.id: cid for r in recs},
        mid_regions={cid: [r.id for r in recs]},
    )
    for i in range(5):
        sid = f"sess-{i}"
        write_event(store, "session_started", {"idx": i},
                    severity="info", session_id=sid)
        write_event(store, "retrieval_used",
                    {"community_id": str(cid), "session_id": sid},
                    severity="info", session_id=sid)

    prov_before = {r.id: len(store.get(r.id).provenance or []) for r in recs}
    stats = asyncio.run(hippea_cascade.run_cascade(store, assignment, top_k=1))
    prov_after = {r.id: len(store.get(r.id).provenance or []) for r in recs}
    assert prov_before == prov_after, (
        f"C6 violation: provenance mutated by cascade. "
        f"before={prov_before} after={prov_after}"
    )
    assert stats["communities_selected"] >= 1


def test_cascade_no_api_key_in_source() -> None:
    src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "hippea_cascade.py"
    text = src.read_text()
    low = text.lower()
    assert "import anthropic" not in text
    assert "from anthropic" not in text
    assert "ANTHROPIC_API_KEY" not in text


def test_cascade_no_store_mutation_imports() -> None:
    src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "hippea_cascade.py"
    text = src.read_text()
    assert "store.insert(" not in text
    assert "store.append_provenance(" not in text
    assert "store.append_provenance_batch(" not in text
    assert "store.update(" not in text
    assert "store.boost_edges(" not in text
    assert "store.add_contradicts_edge(" not in text


def test_cascade_loop_yields_on_shutdown(tmp_path: Path) -> None:
    from iai_mcp import daemon
    from iai_mcp import daemon_state

    state_file = tmp_path / ".daemon-state.json"
    orig_path = daemon_state.STATE_PATH
    daemon_state.STATE_PATH = state_file
    try:
        async def _drive() -> float:
            state_file.write_text("{}")
            shutdown = asyncio.Event()
            fake_store = MagicMock()
            task = asyncio.create_task(
                daemon._hippea_cascade_loop(fake_store, shutdown)
            )
            await asyncio.sleep(0.1)
            t0 = time.monotonic()
            shutdown.set()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                raise
            return time.monotonic() - t0

        elapsed = asyncio.run(_drive())
        assert elapsed < 5.0, f"cascade loop did not yield within 5s: {elapsed}s"
    finally:
        daemon_state.STATE_PATH = orig_path


def _drive_one_cascade_iteration(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> int:
    from iai_mcp import daemon
    from iai_mcp import daemon_state
    from iai_mcp import retrieve

    state_file = tmp_path / ".daemon-state.json"
    orig_path = daemon_state.STATE_PATH
    daemon_state.STATE_PATH = state_file

    call_count = {"n": 0}

    def _fake_build_runtime_graph(_store):
        call_count["n"] += 1
        return None, None, 0

    monkeypatch.setattr(retrieve, "build_runtime_graph", _fake_build_runtime_graph)
    monkeypatch.setattr(daemon, "HIPPEA_CASCADE_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon, "_last_cascade_completed_at", 0.0)

    try:
        async def _drive() -> None:
            state_file.write_text("{}")
            state = daemon_state.load_state()
            state["hippea_cascade_request"] = {"pending": True, "session_id": "s1"}
            daemon_state.save_state(state)

            shutdown = asyncio.Event()
            task = asyncio.create_task(
                daemon._hippea_cascade_loop(store, shutdown)
            )
            await asyncio.sleep(0.2)
            shutdown.set()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                raise

        asyncio.run(_drive())
    finally:
        daemon_state.STATE_PATH = orig_path

    return call_count["n"]


def test_cascade_defers_read_when_consolidation_intent_pending(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_flag = store.root / "hippo" / ".consolidation-pending"
    intent_flag.parent.mkdir(parents=True, exist_ok=True)
    intent_flag.write_text("")

    call_count = _drive_one_cascade_iteration(store, tmp_path, monkeypatch)

    assert call_count == 0, (
        f"cascade must defer (skip build_runtime_graph) while the "
        f"consolidation intent flag is present; got {call_count} calls"
    )


def test_cascade_reads_when_no_consolidation_intent(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_flag = store.root / "hippo" / ".consolidation-pending"
    assert not intent_flag.exists()

    call_count = _drive_one_cascade_iteration(store, tmp_path, monkeypatch)

    assert call_count >= 1, (
        f"cascade must run build_runtime_graph when no consolidation intent "
        f"is signaled; got {call_count} calls"
    )
