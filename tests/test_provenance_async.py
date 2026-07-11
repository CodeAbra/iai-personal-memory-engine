from __future__ import annotations

import time

import pytest

from iai_mcp.store import MemoryStore
from tests.test_store import _make

def test_enqueue_fast(tmp_path):
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    real_batch = store.append_provenance_batch

    def slow_batch(pairs, records_cache=None):
        time.sleep(0.2)
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        t0 = time.perf_counter()
        q.enqueue([(r.id, {"ts": "x", "cue": "c", "session_id": "s"})])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms <= 20.0, f"enqueue took {elapsed_ms:.1f}ms (target <=2ms, headroom <=20ms)"
    finally:
        q.stop()

def test_flush_drains(tmp_path):
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        for i in range(10):
            q.enqueue([(r.id, {"ts": f"t{i}", "cue": f"c{i}", "session_id": "s"})])
        t0 = time.perf_counter()
        q.flush(timeout=2.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms <= 500.0, f"flush took {elapsed_ms:.1f}ms (target <=500ms)"
    finally:
        q.stop()

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 10

def test_atexit_flush(tmp_path, monkeypatch):
    import atexit as _atexit
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    captured: list = []

    def _fake_register(fn, *a, **kw):
        captured.append(fn)
        return fn

    monkeypatch.setattr(_atexit, "register", _fake_register)

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    q.enqueue([(r.id, {"ts": "t", "cue": "c", "session_id": "s"})])

    assert captured, "ProvenanceWriteQueue.start() must register atexit flush"
    captured[0]()

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 1
    q.stop()

@pytest.mark.perf
def test_pipeline_recall_does_not_block_on_merge_insert(tmp_path, monkeypatch):
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    dispatch(
        store, "memory_recall",
        {"cue": "warmup", "session_id": "s0", "cue_embedding": r.embedding},
    )

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    store.enable_provenance_queue(coalesce_ms=50)
    try:
        real_batch = store.append_provenance_batch

        def slow_batch(pairs, records_cache=None):
            time.sleep(0.5)
            return real_batch(pairs, records_cache=records_cache)

        store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

        t0 = time.perf_counter()
        dispatch(
            store,
            "memory_recall",
            {"cue": "q", "session_id": "s1", "cue_embedding": r.embedding},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 400.0, (
            f"pipeline_recall blocked on merge_insert: {elapsed_ms:.1f}ms "
            f"(queue should hand off; target <400ms given 500ms slow write)"
        )
    finally:
        store.disable_provenance_queue()

def test_mem05_preserved_after_drain(tmp_path):
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    store.enable_provenance_queue(coalesce_ms=50)
    try:
        dispatch(store, "memory_recall",
                 {"cue": "first", "session_id": "s1", "cue_embedding": r.embedding})
        dispatch(store, "memory_recall",
                 {"cue": "second", "session_id": "s2", "cue_embedding": r.embedding})
        dispatch(store, "memory_recall",
                 {"cue": "third", "session_id": "s3", "cue_embedding": r.embedding})
        store._provenance_queue.flush(timeout=2.0)  # type: ignore[attr-defined]
    finally:
        store.disable_provenance_queue()

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 3
    cues = [p["cue"] for p in got.provenance]
    assert cues == ["first", "second", "third"], f"order violated: {cues}"

def test_overflow_spill_round_trip(tmp_path, monkeypatch):
    import threading
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path / "store")
    r = _make()
    store.insert(r)

    monkeypatch.setenv("HOME", str(tmp_path))

    flushed_pairs: list = []
    flush_release = threading.Event()
    flush_release.clear()
    real_batch = store.append_provenance_batch

    def slow_batch(pairs, records_cache=None):
        flush_release.wait(timeout=10.0)
        flushed_pairs.extend(pairs)
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

    q = ProvenanceWriteQueue(store, coalesce_ms=10, max_queue_size=2,
                             max_batch_pairs=1)
    q.start()
    try:
        for i in range(5):
            q.enqueue([(r.id, {"ts": f"t{i}", "cue": f"c{i}",
                               "session_id": "sov"})])
        time.sleep(0.1)
        overflow_dir = store.root / ".provenance-overflow"
        spilled_before_release = list(overflow_dir.glob("*.jsonl"))
        assert len(spilled_before_release) >= 1, (
            f"expected at least 1 spilled file, got {len(spilled_before_release)} "
            f"(overflow dir contents: {list(overflow_dir.iterdir()) if overflow_dir.exists() else 'absent'})"
        )
        flush_release.set()
        deadline = time.time() + 12.0
        while time.time() < deadline:
            if not list(overflow_dir.glob("*.jsonl")):
                break
            time.sleep(0.2)
        q.flush(timeout=2.0)
    finally:
        q.stop()

    flushed_cues = [p[1]["cue"] for p in flushed_pairs]
    assert sorted(flushed_cues) == [f"c{i}" for i in range(5)], (
        f"expected all 5 cues flushed exactly once; got {sorted(flushed_cues)}"
    )
    overflow_dir = store.root / ".provenance-overflow"
    assert list(overflow_dir.glob("*.jsonl")) == [], (
        f"spill dir should be empty after drain; got {list(overflow_dir.iterdir())}"
    )

def test_overflow_dir_lazy_create(tmp_path, monkeypatch):
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path / "store")
    r = _make()
    store.insert(r)

    monkeypatch.setenv("HOME", str(tmp_path))

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        q.enqueue([(r.id, {"ts": "t", "cue": "c", "session_id": "s"})])
        q.flush(timeout=2.0)
    finally:
        q.stop()

    overflow_dir = store.root / ".provenance-overflow"
    assert not overflow_dir.exists(), (
        "overflow dir must not be created when no spill happens"
    )

def test_overflow_malformed_spill_file_quarantined(tmp_path, monkeypatch):
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path / "store")

    monkeypatch.setenv("HOME", str(tmp_path))
    overflow_dir = store.root / ".provenance-overflow"
    overflow_dir.mkdir(parents=True)
    bad_file = overflow_dir / "bad.jsonl"
    bad_file.write_text("this is not valid json at all\n")

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        time.sleep(6.5)
    finally:
        q.stop()

    assert not bad_file.exists()
    failed_files = list(overflow_dir.glob("*.failed-*.jsonl"))
    assert len(failed_files) == 1, (
        f"expected 1 failed-quarantined file; got {len(failed_files)} "
        f"(overflow dir contents: {list(overflow_dir.iterdir())})"
    )

def test_queue_disabled_falls_back_to_sync(tmp_path):
    import threading
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    assert getattr(store, "_provenance_queue", None) is None

    call_threads: list[int] = []
    real_batch = store.append_provenance_batch

    def tracking_batch(pairs, records_cache=None):
        call_threads.append(threading.get_ident())
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = tracking_batch  # type: ignore[method-assign]

    main_ident = threading.get_ident()
    dispatch(store, "memory_recall",
             {"cue": "q", "session_id": "s1", "cue_embedding": r.embedding})

    assert call_threads, "append_provenance_batch not called in sync fallback"
    assert call_threads[0] == main_ident, (
        f"sync fallback ran on thread {call_threads[0]!r}, expected main {main_ident!r}"
    )

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 1
