from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from tests._helpers import short_socket_path as _short_socket_path_base

from iai_mcp.community import CommunityAssignment
from iai_mcp.daemon import _probe_status_roundtrip
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_N_SEED = 80
_PROBE_READ_TIMEOUT = 1.0
_HOLD_SEC = 3.0
_SERVED_RTT_CEIL = 1.0

def _make_representative_record(vec, community_id, centrality: float) -> MemoryRecord:
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    literal = ("verbatim recall content " * 40)[:960]
    provenance = [
        {"ts": now.isoformat(), "cue": "recall cue text", "session_id": f"sess-{i}"}
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
        tags=["topic:alpha", "topic:beta", "kind:note"],
        language="en",
        profile_modulation_gain={"empathy_gain": 0.5, "detail_gain": 0.7},
    )

def _seed_store(store: MemoryStore, n: int):
    dim = store._embed_dim
    cid = uuid4()
    ids = []
    rng = np.random.default_rng(7)
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
            except Exception:  # noqa: BLE001 -- probe failure == unservable
                ok = False
            rtt = (time.monotonic() - t0) if ok else float("inf")
            self._samples.append(rtt)
            if rtt == float("inf"):
                self._worst = float("inf")
            elif self._worst != float("inf"):
                self._worst = max(self._worst, rtt)
            time.sleep(0.02)

    def start(self) -> None:
        self._thread.start()

    def reset(self) -> None:
        self._worst = 0.0
        self._samples = []

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=8.0)

    def report(self):
        return self._worst, list(self._samples)

def _hold_conn_lock(store: MemoryStore, hold_sec: float,
                    started: threading.Event, done: threading.Event) -> None:
    with store.db._conn_lock:
        started.set()
        time.sleep(hold_sec)
    done.set()

def _measured_get(store: MemoryStore, rid) -> float:
    t0 = time.monotonic()
    try:
        store.get(rid)
    except Exception:  # noqa: BLE001 -- timing the block, not asserting the value
        pass
    return time.monotonic() - t0

def _short_socket_path() -> Path:
    return _short_socket_path_base(prefix="iai-wedge-")

def _build_state() -> dict:
    return {
        "fsm_state": "WAKE",
        "daemon_started_at": None,
        "last_tick_at": None,
        "quiet_window": None,
        "pending_digest": None,
        "scheduler_paused": False,
    }

def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )

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

def test_fixture_smoke(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        try:
            probe.start()
            await asyncio.sleep(0.4)

            base_get = _measured_get(store, rid)
            assert base_get < 0.1, f"uncontended get too slow: {base_get:.4f}s"

            worst, samples = probe.report()
            assert samples, "probe produced no samples"
            assert worst < _SERVED_RTT_CEIL, (
                f"idle-loop probe should be served fast, worst={worst}"
            )
        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()
        shutil.rmtree(sock_path.parent, ignore_errors=True)

def _served_fraction(samples: list[float], ceil: float) -> float:
    if not samples:
        return 0.0
    served = sum(1 for s in samples if s != float("inf") and s <= ceil)
    return served / len(samples)

def test_on_loop_store_read_under_held_lock_stays_served(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
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
            block_sec = _measured_get(store, rid)
            await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.2)

            arm1_worst, arm1_samples = probe.report()
            # an on-loop store read no longer waits on _conn_lock at all: it
            # streams from a dedicated RO snapshot connection, so it stays
            # fast even while a writer holds the lock for the full window
            assert block_sec < _PROBE_READ_TIMEOUT, (
                "on-loop get blocked behind the held _conn_lock "
                f"(block={block_sec:.3f}s, hold={_HOLD_SEC}s) — reads must be "
                "served from the RO snapshot, never wait on the writer lock"
            )
            assert arm1_samples, "ARM-1 probe produced no samples"
            served = _served_fraction(arm1_samples, _SERVED_RTT_CEIL)
            assert served >= 0.8, (
                "probe should have stayed served (on-loop get is fast, never "
                f"blocks the loop) while a worker held _conn_lock; "
                f"served_fraction={served:.2f} samples={arm1_samples}"
            )
        finally:
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()
        shutil.rmtree(sock_path.parent, ignore_errors=True)

def test_off_loop_store_read_under_held_lock_keeps_probe_served(tmp_path):
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    ids, _assignment = _seed_store(store, _N_SEED)
    rid = ids[len(ids) // 2]

    sock_path = _short_socket_path()

    async def _body():
        server, serve_task = await _serve(store, sock_path)
        probe = _ThreadedProbe(str(sock_path), _PROBE_READ_TIMEOUT)
        ticks = {"n": 0}
        stop_ticker = asyncio.Event()

        async def _ticker():
            while not stop_ticker.is_set():
                await asyncio.sleep(0.05)
                ticks["n"] += 1

        ticker_task = asyncio.create_task(_ticker())
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
            ticks_before = ticks["n"]
            worker_get_sec = await asyncio.to_thread(_measured_get, store, rid)
            # the RO-snapshot read completes fast (see the assert below), so
            # ticks across the read alone would no longer be a meaningful
            # sample -- span the full held-lock window instead, which still
            # proves the loop never stalls for the writer's hold duration
            await asyncio.to_thread(done.wait, _HOLD_SEC + 10.0)
            ticks_after = ticks["n"]
            await asyncio.to_thread(worker.join, 5.0)
            await asyncio.sleep(0.2)

            arm2_worst, arm2_samples = probe.report()
            # the off-loop get streams from the RO snapshot too: it no longer
            # waits on _conn_lock, so it stays fast even under a held writer
            # lock (this is in addition to, not instead of, loop responsiveness)
            assert worker_get_sec < _PROBE_READ_TIMEOUT, (
                "to_thread get blocked behind the held _conn_lock "
                f"(get={worker_get_sec:.3f}s, hold={_HOLD_SEC}s) — reads must "
                "be served from the RO snapshot, never wait on the writer lock"
            )
            assert ticks_after - ticks_before >= 5, (
                "the event loop stalled while a writer held _conn_lock "
                f"(ticks advanced {ticks_after - ticks_before} over ~{_HOLD_SEC}s)"
            )
            assert arm2_samples, "ARM-2 probe produced no samples"
            served = _served_fraction(arm2_samples, _SERVED_RTT_CEIL)
            assert served >= 0.8, (
                "ARM-2 probe should have stayed served (loop free) even while a "
                f"worker held _conn_lock; served_fraction={served:.2f} "
                f"samples={arm2_samples}"
            )
        finally:
            stop_ticker.set()
            ticker_task.cancel()
            try:
                await ticker_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            probe.stop()
            await _teardown_server(server, serve_task)

    try:
        asyncio.run(_body())
    finally:
        store.close()
        shutil.rmtree(sock_path.parent, ignore_errors=True)
