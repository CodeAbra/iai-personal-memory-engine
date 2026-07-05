from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.daemon import WATCHDOG_PROBE_TIMEOUT_SEC, _probe_status_roundtrip
from iai_mcp.hippea_cascade import (
    _install_warm,
    compute_and_fetch_warm,
    fetch_warm_records,
)
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


_N_SEED = 80
_PROBE_READ_TIMEOUT = 1.0
_HOLD_SEC = 8.0
_SERVED_RTT_CEIL = 1.0
_SERVED_FRACTION_MIN = 0.7


def _short_socket_path() -> Path:
    d = Path(tempfile.mkdtemp(prefix="iai-nodb-"))
    return d / "d.sock"


def _make_representative_record(vec, community_id, centrality: float) -> MemoryRecord:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    literal = ("verbatim recall content " * 40)[:960]
    provenance = [
        {"ts": now.isoformat(), "cue": "recall cue text", "session_id": f"s{i}"}
        for i in range(3)
    ]
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=literal,
        aaak_index="",
        embedding=vec.tolist(),
        community_id=community_id,
        centrality=centrality,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=provenance,
        created_at=now,
        updated_at=now,
        tags=["topic:alpha", "kind:note"],
        language="en",
        profile_modulation_gain={"empathy_gain": 0.5},
    )


def _seed_store(store: MemoryStore, n: int):
    dim = store._embed_dim
    cid = uuid4()
    ids = []
    rng = np.random.default_rng(42)
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        rec = _make_representative_record(vec, cid, float(i % 31))
        store.insert(rec)
        ids.append(rec.id)
    assignment = CommunityAssignment(
        top_communities=[cid],
        community_centroids={cid: [0.0] * dim},
        mid_regions={cid: ids},
    )
    return ids, assignment


def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )


def _build_state() -> dict:
    return {
        "fsm_state": "WAKE",
        "daemon_started_at": None,
        "last_tick_at": None,
        "quiet_window": None,
        "pending_digest": None,
        "scheduler_paused": False,
    }


class _ThreadedProbe:

    def __init__(self, sock_path: str, read_timeout: float):
        self._sock_path = sock_path
        self._read_timeout = read_timeout
        self._stop = threading.Event()
        self._worst = 0.0
        self._samples: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                ok = asyncio.run(
                    _probe_status_roundtrip(self._sock_path, self._read_timeout)
                )
            except Exception:  # noqa: BLE001
                ok = False
            rtt = (time.monotonic() - t0) if ok else float("inf")
            self._samples.append(rtt)
            if rtt == float("inf"):
                self._worst = float("inf")
            elif self._worst != float("inf"):
                self._worst = max(self._worst, rtt)
            time.sleep(0.05)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        self._worst = 0.0
        self._samples = []

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=12.0)

    def report(self):
        return self._worst, list(self._samples)


def _hold_conn_lock(
    store: MemoryStore, hold_sec: float,
    started: threading.Event, done: threading.Event,
) -> None:
    with store.db._conn_lock:
        started.set()
        time.sleep(hold_sec)
    done.set()


def _served_fraction(samples: list[float], ceil: float) -> float:
    if not samples:
        return 0.0
    served = sum(1 for s in samples if s != float("inf") and s <= ceil)
    return served / len(samples)


async def _serve(store: MemoryStore, sock_path: Path):
    server = SocketServer(store, state=_build_state())
    serve_task = asyncio.create_task(server.serve(socket_path=sock_path))
    for _ in range(100):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    return server, serve_task


async def _teardown_server(server: SocketServer, serve_task) -> None:
    server.shutdown_event.set()
    try:
        await asyncio.wait_for(serve_task, timeout=5.0)
    except Exception:  # noqa: BLE001
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def test_fetch_warm_records_is_sync_callable(tmp_path):
    import inspect
    assert callable(fetch_warm_records), "fetch_warm_records must be callable"
    assert not inspect.iscoroutinefunction(fetch_warm_records), (
        "fetch_warm_records must be sync (not async)"
    )


def test_install_warm_is_async_callable():
    import inspect
    assert callable(_install_warm), "_install_warm must be callable"
    assert inspect.iscoroutinefunction(_install_warm), (
        "_install_warm must be async"
    )


def test_compute_and_fetch_warm_is_sync_callable():
    import inspect
    assert callable(compute_and_fetch_warm), "compute_and_fetch_warm must be callable"
    assert not inspect.iscoroutinefunction(compute_and_fetch_warm), (
        "compute_and_fetch_warm must be sync (not async)"
    )


def test_fetch_warm_records_lock_free_no_lru_mutation(tmp_path):
    from iai_mcp import hippea_cascade
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    try:
        ids, _ = _seed_store(store, 5)
        hippea_cascade._warm_lru.clear()
        recs = fetch_warm_records(store, ids)
        assert len(recs) == 5, "fetch_warm_records should return all 5 records"
        assert len(hippea_cascade._warm_lru) == 0, (
            "fetch_warm_records must not insert into _warm_lru"
        )
    finally:
        store.close()


def test_install_warm_inserts_no_store_access(tmp_path):
    from iai_mcp import hippea_cascade
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    try:
        ids, _ = _seed_store(store, 3)
        recs = fetch_warm_records(store, ids)
        hippea_cascade._warm_lru.clear()

        get_called = []
        original_get = store.get

        def tracking_get(rid):
            get_called.append(threading.current_thread().ident)
            return original_get(rid)

        store.get = tracking_get
        inserted = asyncio.run(_install_warm(recs))
        assert inserted == 3
        assert len(get_called) == 0, (
            "_install_warm must not call store.get; called on threads: "
            f"{get_called}"
        )
    finally:
        store.close()


def test_compute_and_fetch_warm_matches_run_cascade_warm_set(tmp_path):
    from iai_mcp import hippea_cascade
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    try:
        ids, assignment = _seed_store(store, 20)
        from iai_mcp.events import write_event
        for i in range(5):
            sid = f"sess-{i}"
            write_event(store, "session_started", {}, severity="info", session_id=sid)

        hippea_cascade._warm_lru.clear()
        recs_tuple = compute_and_fetch_warm(store, assignment)
        if isinstance(recs_tuple, tuple):
            recs, top = recs_tuple
        else:
            recs = recs_tuple
            top = None

        hippea_cascade._warm_lru.clear()
        stats = asyncio.run(hippea_cascade.run_cascade(store, assignment))

        caf_ids = set(getattr(r, "id", None) for r in recs)
        assert stats["records_warmed"] == len(caf_ids), (
            f"compute_and_fetch_warm returned {len(caf_ids)} records but "
            f"run_cascade warmed {stats['records_warmed']}"
        )
    finally:
        store.close()


def test_probe_served_under_held_lock_and_cascade(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, assignment = _seed_store(store, _N_SEED)
    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cascade-test"
        )
        try:
            probe.start()
            await asyncio.sleep(0.4)
            probe.reset()

            started = threading.Event()
            done = threading.Event()
            worker = threading.Thread(
                target=_hold_conn_lock,
                args=(store, _HOLD_SEC, started, done),
                daemon=True,
            )
            worker.start()
            ok = await asyncio.to_thread(started.wait, 10.0)
            assert ok, "worker never acquired _conn_lock"

            loop = asyncio.get_event_loop()
            recs_result = await loop.run_in_executor(
                executor, compute_and_fetch_warm, store, assignment
            )
            if isinstance(recs_result, tuple):
                recs, _top = recs_result
            else:
                recs = recs_result
            inserted = await _install_warm(recs)

            await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.3)

            worst, samples = probe.report()
            assert samples, "probe produced no samples during test"
            served = _served_fraction(samples, _SERVED_RTT_CEIL)
            assert served >= _SERVED_FRACTION_MIN, (
                f"Loop was blocked while cascade ran on held-lock. "
                f"served_fraction={served:.2f}, worst={worst:.3f}s, "
                f"samples={samples[:10]}..., inserted={inserted}"
            )
        finally:
            executor.shutdown(wait=False)
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def test_no_store_get_on_loop_thread_during_cascade(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, assignment = _seed_store(store, 20)

    async def _body():
        loop = asyncio.get_event_loop()
        loop_thread_id = threading.current_thread().ident
        get_threads: list[int] = []
        original_get = store.get

        def tracking_get(rid):
            get_threads.append(threading.current_thread().ident)
            return original_get(rid)

        store.get = tracking_get

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cascade-discriminator"
        )
        try:
            recs_result = await loop.run_in_executor(
                executor, compute_and_fetch_warm, store, assignment
            )
            if isinstance(recs_result, tuple):
                recs, _top = recs_result
            else:
                recs = recs_result
            await _install_warm(recs)

            assert loop_thread_id not in get_threads, (
                f"store.get was called on the event-loop thread during cascade. "
                f"loop_thread_id={loop_thread_id}, get_threads={get_threads}"
            )
            if ids:
                assert len(get_threads) > 0, (
                    "store.get was never called — cascade did no work"
                )
        finally:
            executor.shutdown(wait=False)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def test_cascade_runs_on_dedicated_executor(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    ids, assignment = _seed_store(store, 10)

    async def _body():
        loop = asyncio.get_event_loop()
        seen_threads: list[threading.Thread] = []
        original_caf = compute_and_fetch_warm

        def sentinel_caf(store, assignment, **kwargs):
            seen_threads.append(threading.current_thread())
            return original_caf(store, assignment, **kwargs)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cascade-dedic"
        )
        try:
            from unittest.mock import patch
            with patch("iai_mcp.hippea_cascade.compute_and_fetch_warm", sentinel_caf):
                from iai_mcp import hippea_cascade as hc
                recs_result = await loop.run_in_executor(
                    executor, hc.compute_and_fetch_warm, store, assignment
                )
            if isinstance(recs_result, tuple):
                recs, _top = recs_result
            else:
                recs = recs_result
            await _install_warm(recs)

            assert seen_threads, "sentinel never ran — executor not used"
            for t in seen_threads:
                assert t.name.startswith("cascade-dedic"), (
                    f"cascade ran on wrong thread: {t.name!r} "
                    f"(expected 'cascade-dedic...')"
                )
        finally:
            executor.shutdown(wait=False)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def test_no_double_submit_under_burst(tmp_path, monkeypatch):
    import iai_mcp.daemon as daemon_mod

    call_times: list[float] = []
    original_caf = compute_and_fetch_warm

    def counting_caf(store, assignment, **kwargs):
        call_times.append(time.monotonic())
        time.sleep(0.1)
        return original_caf(store, assignment, **kwargs)

    state_holder = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "burst-test"},
    }

    def load_state_stub():
        return dict(state_holder)

    def save_state_stub(s):
        state_holder.clear()
        state_holder.update(s)

    def write_event_stub(*args, **kwargs):
        return None

    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)

    shutdown = asyncio.Event()

    async def _body():
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cascade-burst"
        )
        try:
            from unittest.mock import patch
            with (
                patch("iai_mcp.hippea_cascade.compute_and_fetch_warm", counting_caf),
                patch("iai_mcp.daemon_state.load_state", load_state_stub),
                patch("iai_mcp.daemon_state.save_state", save_state_stub),
                patch("iai_mcp.daemon.write_event", write_event_stub),
                patch.object(daemon_mod, "_cascade_executor", executor),
            ):
                cascade_task = asyncio.create_task(
                    daemon_mod._hippea_cascade_loop(store=None, shutdown=shutdown)
                )
                await asyncio.sleep(0.4)
                shutdown.set()
                try:
                    await asyncio.wait_for(cascade_task, timeout=6.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    cascade_task.cancel()
                    try:
                        await cascade_task
                    except (asyncio.CancelledError, Exception):
                        pass
        finally:
            executor.shutdown(wait=False)

        assert len(call_times) <= 2, (
            f"Double-submit detected: {len(call_times)} calls in 0.4s "
            f"(times={call_times}). The serial await must prevent concurrent dispatch."
        )

    asyncio.run(_body())


def test_lifecycle_tick_no_onloop_store_block(tmp_path):
    from iai_mcp.daemon import _store_is_empty
    import inspect

    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    _seed_store(store, 5)
    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        try:
            probe.start()
            await asyncio.sleep(0.3)
            probe.reset()

            started = threading.Event()
            done = threading.Event()
            worker = threading.Thread(
                target=_hold_conn_lock,
                args=(store, 4.0, started, done),
                daemon=True,
            )
            worker.start()
            ok = await asyncio.to_thread(started.wait, 10.0)
            assert ok, "worker never acquired lock"

            result = await asyncio.to_thread(_store_is_empty, store)

            await asyncio.to_thread(done.wait, 6.0)
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.2)

            worst, samples = probe.report()
            assert samples, "probe produced no samples"
            served = _served_fraction(samples, _SERVED_RTT_CEIL)
            assert served >= _SERVED_FRACTION_MIN, (
                f"Lifecycle tick store-empty check blocked the loop. "
                f"served_fraction={served:.2f}, worst={worst:.3f}s"
            )
        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()


def test_cascade_loop_uses_dedicated_executor_end_to_end(tmp_path, monkeypatch):
    import iai_mcp.daemon as daemon_mod

    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, assignment = _seed_store(store, _N_SEED)
    sock_path = _short_socket_path()

    caf_threads: list[str] = []
    from iai_mcp import hippea_cascade as hc_mod
    original_caf = hc_mod.compute_and_fetch_warm

    def tracking_caf(store, assignment, **kwargs):
        caf_threads.append(threading.current_thread().name)
        return original_caf(store, assignment, **kwargs)

    state_holder: dict = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "e2e-test"},
    }

    def load_state_stub():
        return dict(state_holder)

    def save_state_stub(s):
        state_holder.clear()
        state_holder.update(s)

    def write_event_stub(*args, **kwargs):
        return None

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="iai-cascade-e2e"
    )

    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)
    monkeypatch.setattr(daemon_mod, "_cascade_executor", executor)
    monkeypatch.setattr(hc_mod, "compute_and_fetch_warm", tracking_caf)

    shutdown = asyncio.Event()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        try:
            probe.start()
            await asyncio.sleep(0.3)
            probe.reset()

            started = threading.Event()
            done = threading.Event()
            worker = threading.Thread(
                target=_hold_conn_lock,
                args=(store, _HOLD_SEC, started, done),
                daemon=True,
            )
            worker.start()
            ok = await asyncio.to_thread(started.wait, 10.0)
            assert ok, "worker never acquired _conn_lock"

            from unittest.mock import patch
            with (
                patch(
                    "iai_mcp.runtime_graph_cache.load_recall_structural",
                    return_value=(assignment, [], 0, "normal"),
                ),
                patch("iai_mcp.daemon_state.load_state", load_state_stub),
                patch("iai_mcp.daemon_state.save_state", save_state_stub),
                patch("iai_mcp.daemon.write_event", write_event_stub),
            ):
                cascade_task = asyncio.create_task(
                    daemon_mod._hippea_cascade_loop(store=store, shutdown=shutdown)
                )
                await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
                await asyncio.to_thread(worker.join, 5.0)
                await asyncio.sleep(0.3)
                shutdown.set()
                try:
                    await asyncio.wait_for(cascade_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    cascade_task.cancel()
                    try:
                        await cascade_task
                    except (asyncio.CancelledError, Exception):
                        pass

            worst, samples = probe.report()
            assert samples, "probe produced no samples during e2e test"

            served = _served_fraction(samples, _SERVED_RTT_CEIL)
            assert served >= _SERVED_FRACTION_MIN, (
                f"Loop was blocked during _hippea_cascade_loop dispatch. "
                f"served_fraction={served:.2f}, worst={worst:.3f}s "
                f"(caf_threads={caf_threads})"
            )

            assert caf_threads, (
                "compute_and_fetch_warm was never called — cascade did not dispatch"
            )
            for t_name in caf_threads:
                assert t_name.startswith("iai-cascade-e2e"), (
                    f"compute_and_fetch_warm ran on wrong thread: {t_name!r}. "
                    f"Expected 'iai-cascade-e2e...' (dedicated executor). "
                    f"If this is 'MainThread' or an asyncio worker, the daemon "
                    f"regressed to on-loop dispatch."
                )

        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        executor.shutdown(wait=False)
        store.close()
