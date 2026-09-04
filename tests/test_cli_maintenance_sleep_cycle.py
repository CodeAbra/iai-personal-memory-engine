from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iai_mcp.lifecycle_state import (
    default_state,
    load_state,
    save_state,
)
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        force=False,
        reset_quarantine=False,
        store_path=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def iai_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase")
    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir()
    import importlib
    from iai_mcp import lifecycle_state as _ls
    from iai_mcp import cli as _cli
    importlib.reload(_ls)
    importlib.reload(_cli)
    yield iai_dir
    importlib.reload(_ls)
    importlib.reload(_cli)


def _patch_store_open(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_store = MagicMock()
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    return fake_store


def _patch_pipeline_steps_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    _NOOP_STEPS = [
        (SleepStep.SCHEMA_MINE, "_step_schema_mine"),
        (SleepStep.KNOB_TUNE, "_step_knob_tune"),
        (SleepStep.DREAM_DECAY, "_step_dream_decay"),
        (SleepStep.ERASURE_AGENT, "_step_erasure_agent"),
        (SleepStep.OPTIMIZE_HIPPO, "_step_optimize_hippo"),
        (SleepStep.HIPPO_CLEANUP, "_step_hippo_cleanup"),
        (SleepStep.CLUSTER_REPLAY, "_step_cluster_replay"),
        (SleepStep.CRISIS_RECLUSTER, "_step_crisis_recluster"),
        (SleepStep.RECONSOLIDATION, "_step_reconsolidation"),
        (SleepStep.USER_MODEL_UPDATE, "_step_user_model_update"),
        (SleepStep.DMN_REFLECTION, "_step_dmn_reflection"),
        (SleepStep.CLUSTER_SUMMARY, "_step_cluster_summary"),
        (SleepStep.RECALL_INDEX_REBUILD, "_step_recall_index_rebuild"),
        (SleepStep.ENTITY_LINK, "_step_entity_link"),
        (SleepStep.CURIOSITY_MINE, "_step_curiosity_mine"),
        (SleepStep.EMBEDDING_INTEGRITY, "_step_embedding_integrity"),
        (SleepStep.COMMUNITY_NAMING, "_step_community_naming"),
        (SleepStep.RECONSOLIDATION_VALENCE, "_step_reconsolidation_valence"),
        (SleepStep.PROC_MINE, "_step_proc_mine"),
        (SleepStep.TRANSCRIPT_SWEEP_BACKSTOP, "_step_transcript_sweep_backstop"),
    ]

    for step, method_name in _NOOP_STEPS:
        def _make_noop(s=step):
            def _impl(self, _interrupt_check):
                return True, {}
            return _impl

        monkeypatch.setattr(
            SleepPipeline, method_name, _make_noop(),
        )

    assert {s for s, _ in _NOOP_STEPS} == set(SleepPipeline._STEP_ORDER)

    monkeypatch.setattr(
        SleepPipeline,
        "_run_essential_variable_tracker_hook",
        lambda self: None,
    )


def test_happy_path_runs_pipeline_and_prints_progress(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sleep cycle started." in out
    assert "[1/20] schema_mine" in out
    assert "[2/20] knob_tune" in out
    assert "[3/20] optimize_hippo" in out
    assert "[4/20] hippo_cleanup" in out
    assert "[5/20] dream_decay" in out
    assert "[6/20] erasure_agent" in out
    assert "[7/20] cluster_replay" in out
    assert "[8/20] reconsolidation" in out
    assert "[9/20] user_model_update" in out
    assert "[10/20] dmn_reflection" in out
    assert "[11/20] crisis_recluster" in out
    assert "[12/20] cluster_summary" in out
    assert "[13/20] recall_index_rebuild" in out
    assert "[14/20] entity_link" in out
    assert "[15/20] curiosity_mine" in out
    assert "[16/20] embedding_integrity" in out
    assert "[17/20] community_naming" in out
    assert "[18/20] reconsolidation_valence" in out
    assert "[19/20] proc_mine" in out
    assert "[20/20] transcript_sweep_backstop" in out
    assert "Sleep cycle complete" in out


def test_quarantined_without_force_returns_nonzero_with_message(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "test stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    captured = capsys.readouterr()
    assert "quarantined" in captured.err.lower()
    assert "test stuck" in captured.err
    assert "--force" in captured.err
    assert "--reset-quarantine" in captured.err


def test_force_runs_pipeline_when_quarantined(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "test stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(force=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "[13/20] recall_index_rebuild" in out
    assert "[14/20] entity_link" in out
    assert "Sleep cycle complete" in out

    record_after = load_state(LIFECYCLE_STATE_PATH)
    assert record_after["quarantine"] is not None


def test_reset_quarantine_clears_then_runs(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(reset_quarantine=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Quarantine cleared." in out
    assert "Sleep cycle complete" in out

    record_after = load_state(LIFECYCLE_STATE_PATH)
    assert record_after["quarantine"] is None


def test_reset_quarantine_when_not_quarantined_no_op(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(reset_quarantine=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Quarantine not active" in out
    assert "Sleep cycle complete" in out


def test_failure_returns_nonzero_with_error_in_stderr(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    def _raiser(self, _interrupt_check):
        raise RuntimeError("synthetic optimize failure")

    monkeypatch.setattr(
        SleepPipeline, "_step_optimize_hippo", _raiser,
    )

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    captured = capsys.readouterr()
    assert "[1/20] schema_mine" in captured.out
    assert "[2/20] knob_tune" in captured.out
    assert "[3/20] optimize_hippo ... FAILED" in captured.err
    assert "synthetic optimize failure" in captured.err


def test_failure_after_3rd_strike_prints_quarantine_hint(
    iai_root, monkeypatch, capsys,
):
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    def _raiser(self, _interrupt_check):
        raise RuntimeError("boom")

    monkeypatch.setattr(SleepPipeline, "_step_dream_decay", _raiser)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    cmd_maintenance_sleep_cycle(_make_args())
    cmd_maintenance_sleep_cycle(_make_args())
    capsys.readouterr()

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "quarantined for 24h" in captured.err
    assert "--reset-quarantine" in captured.err


def test_subparser_exposes_sleep_cycle_with_flags():
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "maintenance", "sleep-cycle",
        "--force", "--reset-quarantine",
    ])
    assert args.force is True
    assert args.reset_quarantine is True
    assert args.store_path is None
    assert args.maintenance_cmd == "sleep-cycle"


def test_subparser_defaults_force_false_reset_false():
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["maintenance", "sleep-cycle"])
    assert args.force is False
    assert args.reset_quarantine is False


def test_store_open_failure_returns_2(
    iai_root, monkeypatch, capsys,
):

    def _broken_store(path=None, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", _broken_store,
    )

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 2
    err = capsys.readouterr().err
    assert "could not open MemoryStore" in err
    assert "disk full" in err


def test_lock_held_returns_1_with_actionable_message(
    iai_root, monkeypatch, capsys,
):
    from iai_mcp.hippo import HippoLockHeldError

    def _raise_lock_held(path=None, **kw):
        raise HippoLockHeldError(str(Path(path or "") / "hippo" / ".lock"), "12345")

    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", _raise_lock_held,
    )

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    err = capsys.readouterr().err
    assert "daemon holds the store lock" in err.lower()
    assert "iai-mcp daemon status" in err


def _spawn_hold_exclusive_lock(store_path_str, ready_evt, release_evt, error_q):
    """Module-level so multiprocessing spawn can pickle-reference it by
    qualified name; a closure or nested function is not picklable."""
    from pathlib import Path as _Path

    try:
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        store = MemoryStore(
            path=_Path(store_path_str), access_mode=AccessMode.EXCLUSIVE,
        )
    except Exception as exc:  # noqa: BLE001 -- reported to parent, not raised here
        error_q.put(f"{type(exc).__name__}: {exc}")
        ready_evt.set()
        return
    try:
        ready_evt.set()
        release_evt.wait(timeout=30)
    finally:
        store.close()


def test_second_process_refused_while_lock_held_spawn_context(
    iai_root, monkeypatch, capsys,
):
    """Genuine cross-process refusal: a `spawn`-context child (not `fork`,
    which false-shares the fd; not a same-process open, which
    `_PROCESS_LOCKS` refcounts as the same holder) opens the store
    exclusively and blocks. The CLI's store-open in this process must be
    refused with exit 1 while the child holds it, and must succeed once
    the child releases it."""
    import multiprocessing as mp

    _patch_pipeline_steps_to_noop(monkeypatch)

    store_path = iai_root
    ctx = mp.get_context("spawn")
    ready_evt = ctx.Event()
    release_evt = ctx.Event()
    error_q = ctx.Queue()

    child = ctx.Process(
        target=_spawn_hold_exclusive_lock,
        args=(str(store_path), ready_evt, release_evt, error_q),
    )
    child.start()
    try:
        got_ready = ready_evt.wait(timeout=30)
        assert got_ready, "spawn-context child never signaled readiness"
        if not error_q.empty():
            pytest.fail(f"spawn-context child failed to open the store: {error_q.get()}")

        from iai_mcp.cli import cmd_maintenance_sleep_cycle

        rc = cmd_maintenance_sleep_cycle(_make_args())
        assert rc == 1
        err = capsys.readouterr().err
        assert "daemon holds the store lock" in err.lower()
    finally:
        release_evt.set()
        child.join(timeout=15)

    assert child.exitcode == 0

    rc_after_release = cmd_maintenance_sleep_cycle(_make_args())
    assert rc_after_release == 0
    out = capsys.readouterr().out
    assert "Sleep cycle complete" in out
