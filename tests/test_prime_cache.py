"""Non-vacuous coverage for the priming cache: build/save/load/invalidate
over the encrypted `_hippo_meta` blob, and the PROC_MINE nightly tail that
writes it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import prime_cache
from iai_mcp.lilli.cycle.chunk import decay_proc_chunks, persist_proc_chunk
from iai_mcp.lilli.cycle.proc_mine import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _plant_transition(
    store: MemoryStore,
    *,
    src: str,
    dst: str,
    chunk_id: str | None,
    source: str = "cofired_mine",
    count: int = 5,
    session_count: int = 3,
) -> None:
    store.db.open_table("proc_transitions").merge_insert(
        ["src", "dst", "source"]
    ).execute(
        [
            {
                "src": src,
                "dst": dst,
                "source": source,
                "count": count,
                "session_count": session_count,
                "first_ts": "2026-01-01T00:00:00+00:00",
                "last_ts": "2026-01-01T00:00:01+00:00",
                "chunk_id": chunk_id,
                "updated_at": "2026-01-01T00:00:01+00:00",
            }
        ]
    )


# ---------------------------------------------------------------------------
# build / save / load / invalidate
# ---------------------------------------------------------------------------


_MINT_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Clears both chunk_decay's age and staleness windows with a wide margin --
# eligibility does not depend on the exact configured values.
_DECAY_NOW = _MINT_TS + timedelta(days=500)


def _candidate(pair: tuple[str, str]) -> CofirePairCandidate:
    return CofirePairCandidate(
        pair=pair,
        source="retrieval_cofired",
        count=PAIR_COUNT_FLOOR,
        session_count=MIN_DISTINCT_SESSIONS,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=_MINT_TS,
        last_ts=_MINT_TS + timedelta(minutes=5),
    )


def _backdate(store: MemoryStore, record_id, *, created_at, last_reviewed) -> None:
    store.db.open_table(RECORDS_TABLE).update(
        where=f"id = '{record_id}'",
        values={
            "created_at": created_at.isoformat(),
            "last_reviewed": last_reviewed.isoformat() if last_reviewed else None,
        },
    )


def test_build_excludes_tombstoned_chunk(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    dead_id = persist_proc_chunk(store, _candidate(("alice", "bob")))
    live_id = persist_proc_chunk(store, _candidate(("carol", "dave")))
    assert dead_id is not None and live_id is not None

    # Only the first chunk clears both decay windows (old + stale). The
    # second is old but freshly reviewed, so the staleness clause stays
    # false and it must remain live.
    _backdate(store, dead_id, created_at=_MINT_TS, last_reviewed=_MINT_TS)
    _backdate(
        store, live_id, created_at=_MINT_TS, last_reviewed=_DECAY_NOW - timedelta(days=5),
    )

    result = decay_proc_chunks(store, now=_DECAY_NOW)
    assert result["retired"] == 1

    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT tombstoned_at FROM records WHERE id = ?", (str(dead_id),),
        ).fetchone()
    assert row is not None and row["tombstoned_at"] is not None

    blob = prime_cache.build(store)

    dead_s, live_s = str(dead_id), str(live_id)
    assert dead_s not in blob["chunk_members"]
    assert live_s in blob["chunk_members"]
    assert dead_s not in blob["seed_to_chunks"].get("alice", [])
    assert live_s in blob["seed_to_chunks"]["carol"]


def test_build_lists_multiple_chunks_per_src(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    _plant_transition(store, src="A", dst="B", chunk_id="c1")
    _plant_transition(store, src="A", dst="C", chunk_id="c2")

    blob = prime_cache.build(store)

    assert blob["seed_to_chunks"]["A"] == ["c1", "c2"]
    assert blob["chunk_members"]["c1"] == ["A", "B"]
    assert blob["chunk_members"]["c2"] == ["A", "C"]


def test_build_excludes_rows_with_null_chunk_id(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    _plant_transition(store, src="A", dst="B", chunk_id="c1")
    _plant_transition(store, src="A", dst="D", chunk_id=None)

    blob = prime_cache.build(store)

    assert blob["seed_to_chunks"]["A"] == ["c1"]
    assert list(blob["chunk_members"].values()) == [["A", "B"]]


def test_save_load_round_trips_through_hippo_meta(tmp_path: Path) -> None:
    store, home = _fresh_store(tmp_path)
    blob = {
        "seed_to_chunks": {"A": ["c1", "c2"]},
        "chunk_members": {"c1": ["A", "B"], "c2": ["A", "C"]},
    }
    assert prime_cache.save(store, blob) is True

    store2 = MemoryStore(path=home / ".iai-mcp")
    loaded = prime_cache.load(store2)

    assert loaded["seed_to_chunks"] == {"A": ["c1", "c2"]}
    assert loaded["chunk_members"] == {"c1": ["A", "B"], "c2": ["A", "C"]}


def test_ciphertext_at_rest(tmp_path: Path) -> None:
    from iai_mcp.crypto import is_encrypted

    # A long, low-alphabet marker -- unlike a short id, it cannot collide
    # with base64 ciphertext bytes by chance.
    marker = "unmistakably-plaintext-marker-not-base64-safe-zzzzzzzzzzzzzz"
    store, _ = _fresh_store(tmp_path)
    prime_cache.save(
        store,
        {"seed_to_chunks": {"A": [marker]}, "chunk_members": {marker: ["A", "B"]}},
    )

    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (prime_cache.PRIME_CACHE_META_KEY,),
        ).fetchone()
    assert row is not None
    assert is_encrypted(row["value"])
    assert marker not in row["value"]


def test_load_on_fresh_store_returns_empty_and_never_raises(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    assert prime_cache.load(store) == {}


def test_load_raw_read_failure_degrades_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The raw `_hippo_meta` fetch, not just decrypt/parse, must be inside
    # the never-raise guard -- boot warmup calls load() unconditionally.
    # sqlite3.Connection.execute is a read-only C attribute on both drivers,
    # so the failure is injected via a stand-in db object instead.
    import threading

    class _RaisingConn:
        def execute(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated raw-read failure")

    class _RaisingDB:
        def __init__(self) -> None:
            self._conn = _RaisingConn()
            self._conn_lock = threading.RLock()

    store, _ = _fresh_store(tmp_path)
    monkeypatch.setattr(prime_cache, "_hippo_db", lambda _store: _RaisingDB())

    assert prime_cache.load(store) == {}


def test_load_corrupt_row_degrades_to_empty(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    with store.db._conn_lock:
        store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?",
            (prime_cache.PRIME_CACHE_META_KEY,),
        )
        store.db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (prime_cache.PRIME_CACHE_META_KEY, "not-encrypted-plaintext"),
        )
        store.db._conn.commit()

    assert prime_cache.load(store) == {}


def test_invalidate_forces_reread_after_save(tmp_path: Path) -> None:
    store, _ = _fresh_store(tmp_path)
    prime_cache.save(
        store, {"seed_to_chunks": {"A": ["c1"]}, "chunk_members": {"c1": ["A", "B"]}},
    )
    first = prime_cache.load(store)
    assert first["seed_to_chunks"] == {"A": ["c1"]}

    prime_cache.save(
        store,
        {
            "seed_to_chunks": {"A": ["c1", "c2"]},
            "chunk_members": {"c1": ["A", "B"], "c2": ["A", "C"]},
        },
    )
    # No invalidate() yet -- the process memo from the first load() must
    # still be served stale.
    still_stale = prime_cache.load(store)
    assert still_stale["seed_to_chunks"] == {"A": ["c1"]}

    prime_cache.invalidate(store)
    fresh = prime_cache.load(store)
    assert fresh["seed_to_chunks"] == {"A": ["c1", "c2"]}


# ---------------------------------------------------------------------------
# PROC_MINE nightly tail -- live-mint through to the persisted cache
# ---------------------------------------------------------------------------


def _run_proc_mine_writes_prime_cache(tmp_path: Path) -> None:
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.proc_mine import MIN_DISTINCT_SESSIONS, PAIR_COUNT_FLOOR
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from tests.test_proc_mine import _emit_cofired, _repeated_pair_ids
    from tests.test_proc_mine import _fresh_store as _proc_mine_fresh_store

    real_store, home = _proc_mine_fresh_store(tmp_path)
    pipeline = SleepPipeline(
        store=real_store,
        lifecycle_state_path=home / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=home / "logs"),
        quarantine_ttl_hours=24.0,
    )

    a, b = str(uuid4()), str(uuid4())
    for i in range(MIN_DISTINCT_SESSIONS):
        ids = _repeated_pair_ids((a, b), PAIR_COUNT_FLOOR, f"f{i}")
        _emit_cofired(real_store, f"sess-{i}", ids, ids)

    done, payload = pipeline._step_proc_mine(None)

    assert done is True
    assert set(payload.keys()) == {"candidates_gated", "chunks_persisted"}
    assert payload["chunks_persisted"] >= 1

    with real_store.db._conn_lock:
        row = real_store.db._conn.execute(
            "SELECT chunk_id FROM proc_transitions WHERE src = ? AND dst = ?",
            (a, b),
        ).fetchone()
    assert row is not None
    chunk_id = row["chunk_id"]
    assert chunk_id

    # Read the PERSISTED blob on a fresh store instance -- not a stale
    # in-process memo -- to prove the tail actually wrote to _hippo_meta.
    store2 = MemoryStore(path=home / ".iai-mcp")
    blob = prime_cache.load(store2)

    assert chunk_id in blob["seed_to_chunks"][a]
    assert blob["chunk_members"][chunk_id] == [a, b]


def test_proc_mine_writes_prime_cache(tmp_path: Path) -> None:
    _run_proc_mine_writes_prime_cache(tmp_path)


def test_proc_mine_writes_prime_cache_no_autoflush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Proves the cache-built assertion is not manufactured by the autouse
    # per-test buffer flush.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    _run_proc_mine_writes_prime_cache(tmp_path)
