"""Coverage for FIX D (async reconsolidation half): the daemon boot must call
``store.enable_async_writes()`` so that ``queue_reinforce`` (the per-hit
reconsolidation-on-retrieval write) is deferred onto the background
write-behind queue instead of running synchronously on the recall response
thread, and must call ``store.disable_async_writes()`` on shutdown so the
queue drains cleanly.

Two groups, mirroring the 177-05 / boot-backpressure-gate test idiom for a
large async closure (``daemon.main()``) with no direct construction seam:

1. behavioral -- construct a MemoryStore directly, call the exact
   ``enable_async_writes()`` path the daemon calls, and prove
   ``queue_reinforce`` enqueues (spy on ``reinforce_record`` to prove it is
   NOT called synchronously on the calling thread once the queue is live),
   plus the kill-switch and fail-safe arms.
2. wiring pins -- source-level assertions that ``daemon/__init__.py`` contains
   the boot enable call (guarded by the kill-switch) and the shutdown disable
   call, placed at the correct seams.

Every store here lives under ``tmp_path`` -- never a real or prod store.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"


def _make_record(text: str = "hello world") -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


# ---------------------------------------------------------------------------
# Group 1: behavioral -- the exact enable_async_writes() path the daemon calls
# ---------------------------------------------------------------------------


class TestEnableAsyncWritesBehavioral:
    def test_enable_async_writes_makes_queue_live(self, tmp_path: Path) -> None:
        store = MemoryStore(path=tmp_path)

        async def run() -> None:
            await store.enable_async_writes()

        asyncio.run(run())
        try:
            assert store._write_queue is not None
            assert store._reinforce_queue is not None
        finally:
            asyncio.run(store.disable_async_writes())

    def test_queue_reinforce_enqueues_not_sync_when_queue_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = MemoryStore(path=tmp_path)
        rec = _make_record()
        store.insert(rec)

        sync_calls: list = []
        orig_reinforce = store.reinforce_record

        def _spy_reinforce(*args, **kwargs):
            sync_calls.append((args, kwargs))
            return orig_reinforce(*args, **kwargs)

        monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

        async def run() -> None:
            await store.enable_async_writes()

        asyncio.run(run())
        try:
            # queue_reinforce must return immediately (enqueue), never call
            # reinforce_record synchronously on the calling thread.
            store.queue_reinforce([rec.id])
            assert sync_calls == [], (
                "queue_reinforce called reinforce_record synchronously on "
                "the calling thread while the reinforce queue was live -- "
                "reconsolidation must be deferred to the background thread"
            )
            # Give the background worker a bounded window to drain, then
            # confirm the reinforcement genuinely happened (not dropped).
            store._reinforce_queue.flush(timeout=2.0)
            assert len(sync_calls) == 1, (
                "the background reinforce worker never called "
                "reinforce_record -- the enqueued id was lost, not deferred"
            )
        finally:
            asyncio.run(store.disable_async_writes())

    def test_queue_reinforce_falls_back_to_sync_when_queue_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = MemoryStore(path=tmp_path)
        rec = _make_record()
        store.insert(rec)

        sync_calls: list = []
        orig_reinforce = store.reinforce_record

        def _spy_reinforce(*args, **kwargs):
            sync_calls.append((args, kwargs))
            return orig_reinforce(*args, **kwargs)

        monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

        # No enable_async_writes() called -- queue is absent.
        assert store._reinforce_queue is None
        store.queue_reinforce([rec.id])
        assert len(sync_calls) == 1, (
            "queue_reinforce must fall back to a synchronous reinforce_record "
            "call when the queue is not live -- never a silent no-op"
        )

    def test_disable_async_writes_drains_and_stops_cleanly(
        self, tmp_path: Path,
    ) -> None:
        store = MemoryStore(path=tmp_path)
        rec = _make_record()
        store.insert(rec)

        async def run() -> None:
            await store.enable_async_writes()

        asyncio.run(run())
        store.queue_reinforce([rec.id])

        asyncio.run(store.disable_async_writes())

        assert store._write_queue is None
        assert store._reinforce_queue is None
        assert store._async_loop is None
        assert store._async_thread is None

    def test_disable_async_writes_idempotent_when_never_enabled(
        self, tmp_path: Path,
    ) -> None:
        store = MemoryStore(path=tmp_path)
        # Must not raise when the queue was never enabled.
        asyncio.run(store.disable_async_writes())
        assert store._write_queue is None

    def test_enable_async_writes_idempotent(self, tmp_path: Path) -> None:
        store = MemoryStore(path=tmp_path)

        async def run() -> None:
            await store.enable_async_writes()
            first_queue = store._write_queue
            await store.enable_async_writes()
            assert store._write_queue is first_queue, (
                "a second enable_async_writes() call must be a no-op, not "
                "spin a second background thread/loop"
            )

        asyncio.run(run())
        asyncio.run(store.disable_async_writes())


# ---------------------------------------------------------------------------
# Group 2: wiring pins -- source-level assertions on daemon/__init__.py
# ---------------------------------------------------------------------------


class TestDaemonBootWiring:
    def test_boot_calls_enable_async_writes(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        assert "await store.enable_async_writes()" in src, (
            "daemon boot must call store.enable_async_writes() so "
            "reconsolidation-on-retrieval is queued off the recall path"
        )

    def test_boot_enable_is_guarded_by_kill_switch(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        assert "IAI_MCP_ASYNC_WRITES_OFF" in src, (
            "the boot enable call must be guarded by a kill-switch env var "
            "for A/B + rollback safety"
        )
        kill_switch_idx = src.index("IAI_MCP_ASYNC_WRITES_OFF")
        enable_idx = src.index("await store.enable_async_writes()")
        assert kill_switch_idx < enable_idx, (
            "the kill-switch check must guard the enable call, appearing "
            "before it in source"
        )

    def test_boot_enable_is_fail_safe(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        enable_idx = src.index("await store.enable_async_writes()")
        # The enable call must be wrapped in a try/except so a failure to
        # start the queue never fails boot -- find the try: immediately
        # preceding it and the except: immediately following it.
        segment = src[max(0, enable_idx - 200):enable_idx + 300]
        assert "try:" in segment
        assert "except Exception" in segment
        try_idx = segment.index("try:")
        except_idx = segment.index("except Exception")
        local_enable_idx = segment.index("await store.enable_async_writes()")
        assert try_idx < local_enable_idx < except_idx, (
            "enable_async_writes() must be inside a try/except Exception "
            "block so boot continues on failure (sync fallback)"
        )

    def test_boot_enable_placed_after_boot_health_before_socket_serve(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        health_idx = src.index('"hippo_boot_health"')
        enable_idx = src.index("await store.enable_async_writes()")
        socket_idx = src.index("mcp_socket = SocketServer(store, state=state)")
        assert health_idx < enable_idx < socket_idx, (
            "enable_async_writes must be wired in after the boot-health "
            "check and before the socket server starts serving recall"
        )

    def test_shutdown_calls_disable_async_writes(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        assert "await store.disable_async_writes()" in src, (
            "daemon shutdown must call store.disable_async_writes() so the "
            "queue drains and the background thread stops cleanly"
        )

    def test_shutdown_disable_is_after_shutdown_wait_and_task_cancellation(
        self,
    ) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        wait_idx = src.index("await shutdown.wait()")
        gather_idx = src.index("await asyncio.gather(*_cancel_targets, return_exceptions=True)")
        disable_idx = src.index("await store.disable_async_writes()")
        assert wait_idx < gather_idx < disable_idx, (
            "disable_async_writes must run in the shutdown teardown scope, "
            "after shutdown.wait() unblocks and background tasks are "
            "cancelled, so it always runs on a clean shutdown"
        )

    def test_shutdown_disable_never_fails_teardown(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        disable_idx = src.index("await store.disable_async_writes()")
        segment = src[max(0, disable_idx - 100):disable_idx + 250]
        assert "try:" in segment
        assert "except Exception" in segment
        try_idx = segment.index("try:")
        except_idx = segment.index("except Exception")
        local_disable_idx = segment.index("await store.disable_async_writes()")
        assert try_idx < local_disable_idx < except_idx, (
            "disable_async_writes() must be inside a try/except Exception "
            "block so a drain failure never blocks the rest of shutdown "
            "teardown (events/records/edges buffer flush, state save)"
        )

    def test_queue_reinforce_and_boot_preload_unchanged(self) -> None:
        # This plan must not touch queue_reinforce's routing logic or the
        # boot preload / L3 gate -- both already exist and route correctly
        # once the queue is live.
        from iai_mcp import daemon as _daemon_module
        from iai_mcp.store import _store as _store_module

        daemon_src = inspect.getsource(_daemon_module)
        assert "background_store_work(" in daemon_src, (
            "the boot preload L3 gate must remain untouched"
        )

        store_src = inspect.getsource(_store_module)
        assert "def queue_reinforce(self, record_ids" in store_src
        assert "Synchronous fallback: reinforce each id directly." in store_src
