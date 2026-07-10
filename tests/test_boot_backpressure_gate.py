"""Coverage for the boot-fleet low-priority single-flight gate.

`iai_mcp.retrieve.background_store_work` bounds worst-case concurrent
lock-holders in the daemon's boot background fleet (graph rebuild, sigma,
foraging, deferred-capture drain) to 1, without ever gating recall itself
(reader isolation already keeps recall off the writer path).

Five groups:
1. serialization -- observed max concurrency == 1, fair ordering
2. completion under contention -- a long holder + waiters all complete
3. fail-open -- a gate that cannot acquire still runs the body, with a warning
4. recall ungated -- source-level wiring pin (dispatch never touches the gate)
5. wiring pins -- each of the four boot launch sites routes its heavy body
   through the gate (source-level, mirroring the lifecycle-tick pause-gate
   test idiom for async closures with no construction seam)
"""
from __future__ import annotations

import inspect
import logging
import threading
import time

import pytest

from iai_mcp import retrieve


# ---------------------------------------------------------------------------
# Group 1: serialization -- observed max concurrency == 1, fair ordering
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_four_threads_observe_max_concurrency_one(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()
        order: list[str] = []

        def worker(n: int) -> None:
            nonlocal active, max_active
            with retrieve.background_store_work(
                f"worker-{n}", max_wait_s=1.0, max_retries=5, backoff_base_s=0.02,
            ):
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    order.append(f"start-{n}")
                time.sleep(0.05)
                with lock:
                    active -= 1
                    order.append(f"end-{n}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "a worker thread failed to complete"

        assert max_active == 1, (
            f"observed concurrency {max_active} > 1 -- the gate must bound "
            f"concurrent lock-holders to exactly 1"
        )
        # Every start must be immediately followed by its own end (no
        # interleaving) -- proves total ordering, not just a max-count bound.
        assert len(order) == 8
        for i in range(0, 8, 2):
            start_n = order[i].split("-")[1]
            end_n = order[i + 1].split("-")[1]
            assert start_n == end_n, (
                f"interleaved execution detected: {order[i]} not immediately "
                f"followed by end-{start_n} (got {order[i + 1]})"
            )

    def test_no_task_starved_past_max_wait_budget(self) -> None:
        # 4 contending workers, each retrying up to 5 times with a 0.05s
        # bounded wait -- total budget per worker is well above the 4-way
        # contention's worst case (3 * ~0.05s body time), so every worker
        # must finish inside its own retry budget without falling back to
        # fail-open (i.e. every worker actually acquired the gate).
        acquired_via_gate = [False] * 4
        lock = threading.Lock()

        def worker(n: int) -> None:
            with retrieve.background_store_work(
                f"fair-{n}", max_wait_s=0.3, max_retries=5, backoff_base_s=0.02,
            ):
                with lock:
                    acquired_via_gate[n] = True
                time.sleep(0.03)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert all(acquired_via_gate), (
            "every worker must acquire the gate within its retry budget under "
            "4-way contention -- a False here means a worker was starved past "
            "its bounded-wait budget"
        )


# ---------------------------------------------------------------------------
# Group 2: completion under contention -- a long holder + waiters all complete
# ---------------------------------------------------------------------------


class TestCompletionUnderContention:
    def test_long_holder_plus_three_waiters_all_complete(self) -> None:
        completed: list[str] = []
        lock = threading.Lock()

        def long_holder() -> None:
            with retrieve.background_store_work(
                "long-holder", max_wait_s=1.0, max_retries=10, backoff_base_s=0.02,
            ):
                time.sleep(0.15)
            with lock:
                completed.append("long-holder")

        def waiter(n: int) -> None:
            with retrieve.background_store_work(
                f"waiter-{n}", max_wait_s=0.3, max_retries=10, backoff_base_s=0.02,
            ):
                time.sleep(0.01)
            with lock:
                completed.append(f"waiter-{n}")

        threads = [threading.Thread(target=long_holder)]
        threads += [threading.Thread(target=waiter, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "a contending thread never completed"

        assert set(completed) == {"long-holder", "waiter-0", "waiter-1", "waiter-2"}, (
            f"not all 4 tasks completed under contention: {completed!r}"
        )


# ---------------------------------------------------------------------------
# Group 3: fail-open -- a gate that cannot acquire still runs the body
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_acquire_always_timing_out_still_runs_body(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            retrieve._BACKGROUND_STORE_WORK_GATE, "acquire", lambda timeout=None: False,
        )
        ran = []
        with caplog.at_level(logging.WARNING, logger="iai_mcp.retrieve"):
            with retrieve.background_store_work(
                "always-times-out", max_wait_s=0.01, max_retries=2, backoff_base_s=0.01,
            ):
                ran.append(True)

        assert ran == [True], "the body must still run when the gate cannot be acquired"
        assert any(
            "not acquired" in rec.message and "fail-open" in rec.message
            for rec in caplog.records
        ), "a warning must be logged when the gate falls back to fail-open"

    def test_acquire_raising_still_runs_body(self, monkeypatch, caplog) -> None:
        def _boom(timeout=None):
            raise RuntimeError("gate implementation bug")

        monkeypatch.setattr(retrieve._BACKGROUND_STORE_WORK_GATE, "acquire", _boom)
        ran = []
        with caplog.at_level(logging.WARNING, logger="iai_mcp.retrieve"):
            with retrieve.background_store_work(
                "acquire-raises", max_wait_s=0.01, max_retries=1, backoff_base_s=0.01,
            ):
                ran.append(True)

        assert ran == [True], (
            "a broken gate (acquire raising) must never prevent the body from "
            "running -- fail-open over contention purity"
        )
        assert any(
            "acquire raised" in rec.message for rec in caplog.records
        ), "a warning must be logged when acquire itself raises"

    def test_no_release_on_never_acquired(self, monkeypatch) -> None:
        # If acquire never returns True, release() must never be called --
        # otherwise a real semaphore would raise ValueError on an unbalanced
        # release once the monkeypatch is removed.
        release_calls = []
        monkeypatch.setattr(
            retrieve._BACKGROUND_STORE_WORK_GATE, "acquire", lambda timeout=None: False,
        )
        monkeypatch.setattr(
            retrieve._BACKGROUND_STORE_WORK_GATE, "release",
            lambda: release_calls.append(1),
        )
        with retrieve.background_store_work(
            "never-acquired", max_wait_s=0.01, max_retries=1, backoff_base_s=0.01,
        ):
            pass
        assert release_calls == [], (
            "release() must not be called when the gate was never acquired"
        )

    def test_real_semaphore_not_corrupted_after_fail_open(self) -> None:
        # End-to-end sanity: after a fail-open episode (forced via a very
        # short max_wait against a currently-held gate), the real semaphore
        # must still be usable by a subsequent caller -- no leaked permit,
        # no corrupted internal count.
        outer_entered = threading.Event()
        release_outer = threading.Event()

        def hold_gate() -> None:
            with retrieve.background_store_work(
                "outer-holder", max_wait_s=1.0, max_retries=5, backoff_base_s=0.02,
            ):
                outer_entered.set()
                release_outer.wait(timeout=5.0)

        t = threading.Thread(target=hold_gate)
        t.start()
        assert outer_entered.wait(timeout=5.0), "outer holder never entered the gate"

        # A short-budget caller contends against the held gate and falls
        # back to fail-open (never blocks the test).
        ran = []
        with retrieve.background_store_work(
            "fails-open-while-held", max_wait_s=0.05, max_retries=1, backoff_base_s=0.01,
        ):
            ran.append(True)
        assert ran == [True]

        release_outer.set()
        t.join(timeout=5.0)
        assert not t.is_alive()

        # The gate must still be acquirable now that the outer holder released.
        acquired = retrieve._BACKGROUND_STORE_WORK_GATE.acquire(timeout=1.0)
        assert acquired, "the real semaphore must not be left corrupted/exhausted"
        retrieve._BACKGROUND_STORE_WORK_GATE.release()


# ---------------------------------------------------------------------------
# Group 4: recall ungated -- source-level wiring pin (dispatch never touches
# the gate; recall preempts by construction, not by priority)
# ---------------------------------------------------------------------------


class TestRecallUngated:
    def test_core_dispatch_never_references_the_gate(self) -> None:
        from iai_mcp import core as _core_module

        src = inspect.getsource(_core_module)
        assert "background_store_work" not in src, (
            "core.dispatch (the recall/dispatch entry point) must never "
            "acquire the boot-fleet gate -- recall is reader-isolated and "
            "must always preempt the background fleet"
        )
        assert "_BACKGROUND_STORE_WORK_GATE" not in src


# ---------------------------------------------------------------------------
# Group 5: wiring pins -- each of the four launch sites routes its heavy body
# through the gate (source-level assertion, mirroring the lifecycle-tick
# pause-gate test idiom for async closures with no construction seam)
# ---------------------------------------------------------------------------


class TestLaunchSiteWiring:
    def test_boot_preload_routes_through_gate(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        body_start = src.index("def _boot_preload_body")
        body_end = src.index("async def _boot_preload(")
        segment = src[body_start:body_end]
        assert "background_store_work(" in segment
        assert "build_runtime_graph(store)" in segment
        gate_idx = segment.index("background_store_work(")
        call_idx = segment.index("build_runtime_graph(store)")
        assert gate_idx < call_idx, (
            "the gate must wrap build_runtime_graph, not merely appear "
            "somewhere in the same function"
        )

    def test_s4_background_scan_routes_through_gate(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        body_start = src.index("def _s4_body")
        body_end = src.index("await asyncio.to_thread(_s4_body)")
        segment = src[body_start:body_end]
        assert "background_store_work(" in segment
        assert "s4_background_scan(store, 50)" in segment
        gate_idx = segment.index("background_store_work(")
        call_idx = segment.index("s4_background_scan(store, 50)")
        assert gate_idx < call_idx

    def test_forage_for_connections_routes_through_gate(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        body_start = src.index("def _forage_body")
        body_end = src.index("_foraged = await asyncio.to_thread(_forage_body)")
        segment = src[body_start:body_end]
        assert "background_store_work(" in segment
        assert "forage_for_connections(store, 3)" in segment
        gate_idx = segment.index("background_store_work(")
        call_idx = segment.index("forage_for_connections(store, 3)")
        assert gate_idx < call_idx

    def test_drain_deferred_captures_routes_through_gate(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        body_start = src.index("def _drain_body")
        body_end = src.index("async def _drain_and_report(")
        segment = src[body_start:body_end]
        assert "background_store_work(" in segment
        assert "_drain(store)" in segment
        gate_idx = segment.index("background_store_work(")
        call_idx = segment.index("_drain(store)")
        assert gate_idx < call_idx

    def test_preload_ready_set_still_in_finally(self) -> None:
        # The gate wiring must not disturb the existing preload_ready.set()
        # finally-block contract -- a gate failure or exception inside the
        # gated body must still signal preload readiness.
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        boot_preload_start = src.index("async def _boot_preload(")
        boot_preload_end = src.index("asyncio.create_task(_boot_preload())")
        segment = src[boot_preload_start:boot_preload_end]
        assert "finally:" in segment
        assert "preload_ready.set()" in segment
        finally_idx = segment.index("finally:")
        set_idx = segment.index("preload_ready.set()")
        assert finally_idx < set_idx

    def test_all_four_sites_use_distinct_names(self) -> None:
        # Each site should identify itself distinctly in the gate's debug/
        # warning logs (observability for a prod-scale regression) -- not a
        # behavioral requirement, but a documented naming contract.
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        names = {
            "boot_preload",
            "s4_background_scan",
            "forage_for_connections",
            "drain_deferred_captures",
        }
        for name in names:
            assert f'background_store_work("{name}")' in src, (
                f"expected a background_store_work(\"{name}\") call site"
            )


# ---------------------------------------------------------------------------
# Cross-cutting: the gate module-level object itself
# ---------------------------------------------------------------------------


class TestGatePrimitive:
    def test_gate_is_bounded_semaphore_of_one(self) -> None:
        assert isinstance(retrieve._BACKGROUND_STORE_WORK_GATE, type(threading.BoundedSemaphore(1)))

    def test_default_constants_are_sane(self) -> None:
        assert retrieve._BACKGROUND_GATE_MAX_WAIT_SEC_DEFAULT > 0
        assert retrieve._BACKGROUND_GATE_MAX_RETRIES_DEFAULT >= 1
        assert retrieve._BACKGROUND_GATE_BACKOFF_BASE_SEC_DEFAULT > 0

    def test_exception_inside_body_still_releases_gate(self) -> None:
        with pytest.raises(ValueError):
            with retrieve.background_store_work(
                "raises-inside", max_wait_s=0.5, max_retries=2, backoff_base_s=0.01,
            ):
                raise ValueError("boom inside the gated body")

        # The gate must be released despite the body raising -- prove by
        # immediately re-acquiring it directly.
        acquired = retrieve._BACKGROUND_STORE_WORK_GATE.acquire(timeout=1.0)
        assert acquired, "gate was not released after an exception inside the body"
        retrieve._BACKGROUND_STORE_WORK_GATE.release()
