from __future__ import annotations

from iai_mcp.daemon import (
    HIBERNATION_DEMAND_WINDOW_SEC,
    _hibernation_demand_reason,
)
from iai_mcp.lifecycle import LifecycleEvent, compute_transition
from iai_mcp.lifecycle_state import LifecycleState


class TestHibernationDemandReason:
    def test_no_demand_returns_none(self) -> None:
        assert _hibernation_demand_reason(False, 100.0, 100.0, 200.0) is None

    def test_fresh_socket_request_after_boot_is_demand(self) -> None:
        reason = _hibernation_demand_reason(False, 150.0, 100.0, 160.0)
        assert reason == "wake_on_socket_request_in_hibernation"

    def test_socket_activity_at_boot_baseline_is_not_demand(self) -> None:
        # last_activity_ts initializes to construction time; equality with
        # the boot baseline must never read as demand or every boot wakes.
        assert _hibernation_demand_reason(False, 100.0, 100.0, 110.0) is None

    def test_socket_activity_before_boot_is_not_demand(self) -> None:
        assert _hibernation_demand_reason(False, 90.0, 100.0, 110.0) is None

    def test_stale_socket_request_is_not_demand(self) -> None:
        now = 150.0 + HIBERNATION_DEMAND_WINDOW_SEC + 1.0
        assert _hibernation_demand_reason(False, 150.0, 100.0, now) is None

    def test_live_wrapper_heartbeat_is_demand(self) -> None:
        reason = _hibernation_demand_reason(True, 100.0, 100.0, 200.0)
        assert reason == "wake_on_live_wrapper_heartbeat"

    def test_socket_evidence_wins_over_heartbeat(self) -> None:
        reason = _hibernation_demand_reason(True, 150.0, 100.0, 160.0)
        assert reason == "wake_on_socket_request_in_hibernation"

    def test_stale_socket_falls_back_to_heartbeat(self) -> None:
        now = 150.0 + HIBERNATION_DEMAND_WINDOW_SEC + 1.0
        reason = _hibernation_demand_reason(True, 150.0, 100.0, now)
        assert reason == "wake_on_live_wrapper_heartbeat"

    def test_window_is_exclusive_at_boundary(self) -> None:
        now = 150.0 + HIBERNATION_DEMAND_WINDOW_SEC
        assert _hibernation_demand_reason(False, 150.0, 100.0, now) is None


class TestRequestArrivedDispatchGuard:
    """REQUEST_ARRIVED returns WAKE from ANY state, including SLEEP — an
    unguarded dispatch would abort a running nightly consolidation cycle.
    Every dispatch site must sit inside a block that first proved the state
    is HIBERNATION."""

    _GUARD_LOOKBACK_LINES = 40

    def _dispatch_sites(self, text: str) -> list[int]:
        lines = text.splitlines()
        return [
            i for i, line in enumerate(lines)
            if "REQUEST_ARRIVED" in line
            and "LifecycleEvent" in line
            and "import" not in line
        ]

    def test_every_dispatch_site_is_hibernation_guarded(self) -> None:
        import inspect

        import iai_mcp.daemon as daemon_mod

        text = inspect.getsource(daemon_mod)
        lines = text.splitlines()
        sites = self._dispatch_sites(text)
        assert len(sites) == 2, (
            f"expected exactly 2 REQUEST_ARRIVED dispatch sites (boot + "
            f"lifecycle tick), found {len(sites)} — a new site must prove "
            f"the state is HIBERNATION before dispatching, then update this "
            f"count"
        )
        for i in sites:
            lookback = "\n".join(
                lines[max(0, i - self._GUARD_LOOKBACK_LINES):i]
            )
            assert "HIBERNATION" in lookback, (
                f"REQUEST_ARRIVED dispatch at daemon/__init__.py source "
                f"line {i + 1} has no HIBERNATION guard within "
                f"{self._GUARD_LOOKBACK_LINES} lines above it — it could "
                f"fire during SLEEP and abort consolidation"
            )

    def test_no_dispatch_sites_outside_the_daemon_module(self) -> None:
        import subprocess
        from pathlib import Path

        src_root = Path(__file__).resolve().parents[1] / "src" / "iai_mcp"
        out = subprocess.run(
            ["grep", "-rl", "REQUEST_ARRIVED", str(src_root)],
            capture_output=True, text=True, check=False,
        )
        offenders = [
            p for p in out.stdout.splitlines()
            if p
            and not p.endswith("lifecycle.py")
            and not p.endswith("daemon/__init__.py")
            and "__pycache__" not in p
        ]
        assert offenders == [], (
            f"REQUEST_ARRIVED referenced outside lifecycle.py / "
            f"daemon/__init__.py: {offenders} — new dispatch paths need a "
            f"HIBERNATION guard and coverage here"
        )


class TestHibernationWakeTransitions:
    def test_request_arrived_leaves_hibernation(self) -> None:
        assert (
            compute_transition(
                LifecycleState.HIBERNATION, LifecycleEvent.REQUEST_ARRIVED
            )
            is LifecycleState.WAKE
        )

    def test_wake_signal_leaves_hibernation(self) -> None:
        assert (
            compute_transition(
                LifecycleState.HIBERNATION, LifecycleEvent.WAKE_SIGNAL
            )
            is LifecycleState.WAKE
        )
