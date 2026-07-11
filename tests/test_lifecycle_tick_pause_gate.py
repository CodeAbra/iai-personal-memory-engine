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
