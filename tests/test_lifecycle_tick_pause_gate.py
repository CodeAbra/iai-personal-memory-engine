from __future__ import annotations

import inspect

from iai_mcp.daemon import _sleep_pipeline_gate


class TestSleepPipelineGate:
    def test_paused_true_blocks_pipeline(self) -> None:
        assert _sleep_pipeline_gate({"scheduler_paused": True}) is False

    def test_paused_false_allows_pipeline(self) -> None:
        assert _sleep_pipeline_gate({"scheduler_paused": False}) is True

    def test_paused_key_absent_allows_pipeline(self) -> None:
        assert _sleep_pipeline_gate({}) is True

    def test_none_state_allows_pipeline(self) -> None:
        assert _sleep_pipeline_gate(None) is True

    def test_empty_dict_allows_pipeline(self) -> None:
        assert _sleep_pipeline_gate({}) is True

    def test_truthy_string_paused_blocks_pipeline(self) -> None:
        # bool("1") is True in Python -- the gate uses plain bool() semantics
        # on whatever value is stored under scheduler_paused, so any truthy
        # value (not just the literal True) blocks the pipeline.
        assert _sleep_pipeline_gate({"scheduler_paused": "1"}) is False

    def test_falsy_string_paused_allows_pipeline(self) -> None:
        # bool("") is False -- documented so a future refactor does not
        # silently invert this on an empty-string sentinel.
        assert _sleep_pipeline_gate({"scheduler_paused": ""}) is True

    def test_gate_given_empty_dict_never_nameerrors(self) -> None:
        # Covers the pre-binding requirement: if _load_ds ever raised inside
        # lifecycle_tick's try-block, _ds must still be a valid {} the gate
        # can consult without a NameError.
        _ds: dict = {}
        assert _sleep_pipeline_gate(_ds) is True

    def test_pending_force_rem_overrides_pause(self) -> None:
        # An explicit force-rem is an intentional operator request and must
        # run the pipeline even while paused -- never silently swallowed.
        assert _sleep_pipeline_gate({
            "scheduler_paused": True,
            "force_rem_request": {"pending": True},
        }) is True

    def test_non_pending_force_rem_does_not_override_pause(self) -> None:
        # A consumed / absent force-rem does not lift the pause: idle-driven
        # auto-sleep (which sets no pending force_rem_request) still obeys it.
        assert _sleep_pipeline_gate({
            "scheduler_paused": True,
            "force_rem_request": {"pending": False},
        }) is False
        assert _sleep_pipeline_gate({
            "scheduler_paused": True,
            "force_rem_request": {},
        }) is False

    def test_pending_force_rem_when_not_paused_still_runs(self) -> None:
        assert _sleep_pipeline_gate({
            "force_rem_request": {"pending": True},
        }) is True


class TestLifecycleTickWiresGate:
    def test_sleep_branch_source_references_gate(self) -> None:
        # Behavioral harnessing of lifecycle_tick (an async closure nested
        # inside the daemon assembly function, with no seam to construct it
        # in isolation) would require standing up the full daemon loop --
        # disproportionate for this wiring check. Fallback per plan: a
        # source-level pin confirming the SLEEP branch actually consults
        # _sleep_pipeline_gate before running the sleep pipeline. Brittle but
        # honest -- documented here as a wiring pin, not a behavioral proof.
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        sleep_branch_start = src.index("if current is _LifecycleState.SLEEP")
        pipeline_run_call = src.index("_sleep_pipeline.run", sleep_branch_start)
        gate_call = src.index("_sleep_pipeline_gate(", sleep_branch_start)
        assert gate_call < pipeline_run_call, (
            "_sleep_pipeline_gate( must be consulted before _sleep_pipeline.run "
            "inside the SLEEP branch"
        )

    def test_ds_prebound_before_try_block(self) -> None:
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        prebind_idx = src.index("_ds: dict = {}")
        load_ds_idx = src.index("from iai_mcp.daemon_state import load_state as _load_ds")
        assert prebind_idx < load_ds_idx, (
            "_ds must be pre-bound to {} before the try-block that loads it, "
            "so a raised _load_ds can never leave _ds unbound at the gate"
        )


class TestSleepInterruptPredicate:
    def test_quiet_beacon_never_defers(self, monkeypatch) -> None:
        # Ambient socket traffic (capture hooks, refresh polls) must not
        # register as foreground; with the beacon quiet the predicate says
        # "do not defer" regardless of any socket-level activity. The beacon
        # is process-global, so pin it quiet explicitly.
        from iai_mcp import concurrency
        from iai_mcp.daemon import _sleep_interrupt_predicate

        monkeypatch.setattr(concurrency, "_foreground_last_touch", 0.0)
        monkeypatch.setattr(concurrency, "_foreground_inflight", 0)
        assert _sleep_interrupt_predicate() is False

    def test_inflight_foreground_defers(self) -> None:
        from iai_mcp import concurrency
        from iai_mcp.daemon import _sleep_interrupt_predicate

        concurrency.foreground_begin()
        try:
            assert _sleep_interrupt_predicate() is True
        finally:
            concurrency.foreground_end()

    def test_recent_foreground_stamp_defers_within_window(self, monkeypatch) -> None:
        import time as _time

        from iai_mcp import concurrency, daemon as _daemon_module
        from iai_mcp.daemon import _sleep_interrupt_predicate

        monkeypatch.setattr(
            _daemon_module, "SLEEP_FOREGROUND_RECENT_WINDOW_SEC", 0.15
        )
        monkeypatch.setattr(concurrency, "_foreground_inflight", 0)
        concurrency.foreground_touch()
        assert _sleep_interrupt_predicate() is True
        _time.sleep(0.25)
        assert _sleep_interrupt_predicate() is False

    def test_sleep_branch_source_uses_beacon_not_socket_activity(self) -> None:
        # Wiring pin (same style as the gate pin above): the SLEEP branch's
        # interrupt closure must delegate to the beacon predicate; a raw
        # last_activity_ts check there re-creates the ambient-traffic
        # starvation where a cycle defers at its first chunk forever.
        from iai_mcp import daemon as _daemon_module

        src = inspect.getsource(_daemon_module)
        sleep_branch_start = src.index("elif current is _LifecycleState.SLEEP:")
        run_call = src.index("_sleep_pipeline.run", sleep_branch_start)
        predicate_call = src.index("_sleep_interrupt_predicate()", sleep_branch_start)
        assert predicate_call < run_call
        assert "mcp_socket.last_activity_ts" not in src[sleep_branch_start:run_call]


class TestSleepCycleCooldown:
    """A clean completed cycle opens a refractory window; the pipeline must
    not loop back-to-back while the user is active (hibernation exit needs a
    30-min idle that an active user never provides)."""

    def test_no_clean_cycle_yet_never_gates(self):
        from iai_mcp.daemon import _sleep_cooldown_active

        assert _sleep_cooldown_active({}, 0.0, 1000.0) is False

    def test_within_cooldown_gates(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "600")
        assert _sleep_cooldown_active({}, 1000.0, 1000.0 + 599.0) is True

    def test_past_cooldown_releases(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "600")
        assert _sleep_cooldown_active({}, 1000.0, 1000.0 + 601.0) is False

    def test_force_rem_always_wins(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "600")
        ds = {"force_rem_request": {"pending": True}}
        assert _sleep_cooldown_active(ds, 1000.0, 1000.0 + 1.0) is False

    def test_bad_env_value_falls_back_to_default(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active, _sleep_cycle_cooldown_sec

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "not-a-number")
        assert _sleep_cycle_cooldown_sec() == 14400.0
        assert _sleep_cooldown_active({}, 1000.0, 1000.0 + 14399.0) is True
        assert _sleep_cooldown_active({}, 1000.0, 1000.0 + 14401.0) is False

    def test_tick_chain_wires_the_gate(self):
        # The gate must sit in the SLEEP dispatch chain of lifecycle_tick,
        # between the failure backoff and the pipeline run, and the clean-cycle
        # branch must stamp the refractory origin.
        import inspect

        from iai_mcp import daemon as daemon_mod

        src = inspect.getsource(daemon_mod)
        assert "_sleep_cooldown_active(" in src
        assert "_last_clean_cycle_mono[0] = time.monotonic()" in src


class TestSleepCooldownForceEscape:
    """The tick consumes a force's pending flag BEFORE the gate chain runs,
    so the cooldown must honor a recently-honored-but-not-yet-served force
    (both force-rem and the user sleep verb) — and close again once a clean
    cycle newer than the force completes."""

    def _honored(self, seconds_ago: float) -> str:
        from datetime import datetime, timedelta, timezone

        return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()

    def test_honored_unserved_force_opens_the_gate(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "14400")
        ds = {"force_rem_request": {"pending": False, "honored_at": self._honored(60)}}
        assert _sleep_cooldown_active(ds, 1000.0, 1001.0, 0.0) is False

    def test_honored_force_served_by_newer_clean_cycle_closes_the_gate(self, monkeypatch):
        import time as _time

        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "14400")
        ds = {"force_rem_request": {"pending": False, "honored_at": self._honored(600)}}
        served_wall = _time.time() - 60
        assert _sleep_cooldown_active(ds, 1000.0, 1001.0, served_wall) is True

    def test_user_sleep_request_escapes_like_force_rem(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "14400")
        ds = {"user_sleep_request": {"pending": True}}
        assert _sleep_cooldown_active(ds, 1000.0, 1001.0, 0.0) is False
        ds = {"user_sleep_request": {"pending": False, "honored_at": self._honored(30)}}
        assert _sleep_cooldown_active(ds, 1000.0, 1001.0, 0.0) is False

    def test_stale_honored_at_does_not_hold_the_gate_open(self, monkeypatch):
        from iai_mcp.daemon import _sleep_cooldown_active

        monkeypatch.setenv("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "14400")
        ds = {"force_rem_request": {"pending": False, "honored_at": self._honored(3 * 3600)}}
        assert _sleep_cooldown_active(ds, 1000.0, 1001.0, 0.0) is True
