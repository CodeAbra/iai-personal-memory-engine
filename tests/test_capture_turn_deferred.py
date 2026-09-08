from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename semantics",
)


def _make_transcript_line(role: str, text: str) -> str:
    return json.dumps({"type": role, "message": {"role": role, "content": text}}) + "\n"


def _build_args(session_id: str, transcript_path: Path, max_turns: int = 200) -> argparse.Namespace:
    return argparse.Namespace(
        session_id=session_id,
        transcript_path=str(transcript_path),
        max_turns_per_call=max_turns,
    )


def test_first_call_writes_header_and_one_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "hello world enough chars"))

    rc = cmd_capture_turn_deferred(_build_args("S1", transcript))
    assert rc == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S1.live.jsonl"
    assert live.exists(), "live file must be created"
    lines = live.read_text().splitlines()
    assert len(lines) == 2, f"expected header + 1 event, got {lines}"
    header = json.loads(lines[0])
    assert header["version"] == 1
    assert header["session_id"] == "S1"
    assert "cwd" in header
    assert "deferred_at" in header
    event = json.loads(lines[1])
    assert event["role"] == "user"
    assert event["text"] == "hello world enough chars"


def test_second_call_appends_only_new_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "first user turn here please"))

    cmd_capture_turn_deferred(_build_args("S2", transcript))

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S2.live.jsonl"
    first_header = live.read_text().splitlines()[0]

    with transcript.open("a") as fh:
        fh.write(_make_transcript_line("assistant", "second assistant reply text"))

    cmd_capture_turn_deferred(_build_args("S2", transcript))

    lines = live.read_text().splitlines()
    assert len(lines) == 3, f"header + 2 events expected, got {lines}"
    assert lines[0] == first_header, "header line must not be rewritten"
    second = json.loads(lines[2])
    assert second["role"] == "assistant"
    assert second["text"] == "second assistant reply text"


def test_offset_persisted_atomically_as_line_count(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "a-turn with enough characters to qualify")
        + _make_transcript_line("assistant", "b-turn with enough characters here")
    )

    cmd_capture_turn_deferred(_build_args("S3", transcript))

    offset_file = tmp_path / ".iai-mcp" / ".capture-state" / "S3.offset"
    assert offset_file.exists()
    raw = offset_file.read_text().strip()
    parsed = int(raw)
    assert parsed == 2, f"expected line count 2 after 2 turns, got {parsed}"
    assert raw == str(parsed), "offset file must contain exactly the integer string"


def test_offset_resets_on_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "S4.offset").write_text("50")

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "".join(_make_transcript_line("user", f"turn number {i} long enough text") for i in range(3))
    )

    cmd_capture_turn_deferred(_build_args("S4", transcript))

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S4.live.jsonl"
    lines = live.read_text().splitlines()
    assert len(lines) == 4, f"header + 3 reprocessed events expected, got {lines}"
    new_offset = int((state_dir / "S4.offset").read_text().strip())
    assert new_offset == 3


def test_missing_transcript_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    rc = cmd_capture_turn_deferred(_build_args("S5", tmp_path / "does-not-exist.jsonl"))
    assert rc == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S5.live.jsonl"
    offset_file = tmp_path / ".iai-mcp" / ".capture-state" / "S5.offset"
    assert not live.exists()
    assert not offset_file.exists()


def test_invalid_role_lines_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "first valid user turn long enough")
        + _make_transcript_line("system", "system turn must be skipped now")
        + _make_transcript_line("tool_use", "tool use turn must also be skipped")
        + _make_transcript_line("assistant", "second valid assistant turn here")
    )

    cmd_capture_turn_deferred(_build_args("S6", transcript))

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S6.live.jsonl"
    lines = live.read_text().splitlines()
    assert len(lines) == 3, f"header + 2 valid events expected, got {lines}"
    roles = [json.loads(ln)["role"] for ln in lines[1:]]
    assert roles == ["user", "assistant"]


def test_max_turns_per_call_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "".join(_make_transcript_line("user", f"turn {i} text here long enough") for i in range(10))
    )

    args = _build_args("S7", transcript, max_turns=3)
    cmd_capture_turn_deferred(args)

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S7.live.jsonl"
    lines = live.read_text().splitlines()
    assert len(lines) == 4, f"header + 3 events expected with cap=3, got {lines}"

    offset = int((tmp_path / ".iai-mcp" / ".capture-state" / "S7.offset").read_text().strip())
    assert offset == 3


def test_invalid_session_id_never_touches_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "long enough text for capture here")
    )
    rc = cmd_capture_turn_deferred(_build_args("../evil", transcript))
    assert rc == 0
    assert not (tmp_path / ".iai-mcp" / ".deferred-captures").exists()


def test_user_turn_keyed_by_prompt_id_via_cli_mirror(tmp_path, monkeypatch):
    """cmd_capture_turn_deferred shares _parse_transcript_obj with the daemon
    walk, so the promptId re-key for role:user must propagate here with no
    separate edit — the join key the CLI surface stamps must match the one
    the immediate stdin capture would have stamped for the same prompt."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "cli mirror key scheme parity test text here"},
        "uuid": "cli-mirror-per-line-uuid",
        "promptId": "cli-mirror-per-prompt-id",
    }) + "\n")

    rc = cmd_capture_turn_deferred(_build_args("S10", transcript))
    assert rc == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "S10.live.jsonl"
    lines = live.read_text().splitlines()
    event = json.loads(lines[1])
    assert event["role"] == "user"
    assert event["source_uuid"] == "cli-mirror-per-prompt-id", (
        f"expected the promptId key to propagate to the CLI mirror; got {event!r}"
    )


def test_contended_lock_gives_up_without_consuming(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp import _flock
    from iai_mcp.cli import _capture as capmod
    from iai_mcp.cli import cmd_capture_turn_deferred

    monkeypatch.setattr(capmod, "_LOCK_POLL_TRIES", 2)
    monkeypatch.setattr(capmod, "_LOCK_POLL_INTERVAL", 0.01)

    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)
    holder = os.open(
        str(state_dir / "S9.capture.lock"), os.O_WRONLY | os.O_CREAT, 0o600
    )
    _flock.flock(holder, _flock.LOCK_EX | _flock.LOCK_NB)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "long enough text for capture here")
    )
    try:
        rc = cmd_capture_turn_deferred(_build_args("S9", transcript))
        assert rc == 0
        assert not (
            tmp_path / ".iai-mcp" / ".deferred-captures" / "S9.live.jsonl"
        ).exists()
        assert not (state_dir / "S9.offset").exists()
    finally:
        _flock.flock(holder, _flock.LOCK_UN)
        os.close(holder)

    rc = cmd_capture_turn_deferred(_build_args("S9", transcript))
    assert rc == 0
    assert (
        tmp_path / ".iai-mcp" / ".deferred-captures" / "S9.live.jsonl"
    ).exists()


def test_reconcile_happy_path_ok_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.capture import drain_active_live_captures, reconcile_session_capture
    from tests.conftest_shared import make_tmp_store

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "first user turn long enough text")
        + _make_transcript_line("assistant", "first assistant reply long enough")
        + _make_transcript_line("user", "second user turn long enough text")
    )

    cmd_capture_turn_deferred(_build_args("R1", transcript))

    store = make_tmp_store(tmp_path)
    try:
        drain_active_live_captures(store, exclude_session_id="other-session")
        recon = reconcile_session_capture(
            "R1", transcript_path=transcript, store=store, include_durable=True,
        )
    finally:
        store.close()

    assert recon["status"] == "ok"
    assert recon["missing_turns"] == 0
    assert recon["parse_failed"] == 0
    assert recon["expected_turns"] == 3
    assert recon["captured_turns"] == 3
    assert recon["durable_turns"] == 3


def test_reconcile_drop_one_spool_event_reports_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.capture import deferred_captures_dir, reconcile_session_capture

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "a-turn with enough characters to qualify")
        + _make_transcript_line("assistant", "b-turn with enough characters here")
    )
    cmd_capture_turn_deferred(_build_args("R2", transcript))

    live = deferred_captures_dir() / "R2.live.jsonl"
    lines = live.read_text().splitlines()
    assert len(lines) == 3, f"expected header + 2 events, got {lines}"
    # Drop the second event line: a spool write that never landed.
    live.write_text("\n".join(lines[:2]) + "\n")

    recon = reconcile_session_capture(
        "R2", transcript_path=transcript, include_durable=False,
    )
    assert recon["status"] == "partial"
    assert recon["missing_turns"] == 1
    assert recon["parse_failed"] == 0
    assert recon["expected_turns"] == 2
    assert recon["captured_turns"] == 1
    assert recon["durable_turns"] is None


def test_reconcile_malformed_transcript_line_non_vacuous(tmp_path, monkeypatch):
    """A malformed-but-complete transcript line must land in parse_failed --
    proving expected_turns is independent of the live parse that silently
    drops the same line, distinguishing this case from a plain spool drop."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.capture import reconcile_session_capture

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "first valid user turn long enough")
        + "{this is not valid json at all\n"
        + _make_transcript_line("assistant", "second valid assistant turn here")
    )
    cmd_capture_turn_deferred(_build_args("R3", transcript))

    recon = reconcile_session_capture(
        "R3", transcript_path=transcript, include_durable=False,
    )
    assert recon["status"] == "partial"
    assert recon["parse_failed"] >= 1
    assert recon["expected_turns"] == 3
    assert recon["captured_turns"] == 2
    assert recon["missing_turns"] == 1


def test_reconcile_rotated_only_spool_counts_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.capture import deferred_captures_dir, reconcile_session_capture

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "a rotated-only spool test turn text")
    )
    cmd_capture_turn_deferred(_build_args("R4", transcript))

    live = deferred_captures_dir() / "R4.live.jsonl"
    rotated = deferred_captures_dir() / "R4.live-1699999999-12345.jsonl"
    live.rename(rotated)

    recon = reconcile_session_capture(
        "R4", transcript_path=transcript, include_durable=False,
    )
    assert recon["captured_turns"] > 0
    assert recon["status"] == "ok"
    assert recon["expected_turns"] == recon["captured_turns"] == 1


def test_reconcile_include_durable_false_leaves_durable_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.capture import reconcile_session_capture

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "durable-none leg test turn text here")
    )
    cmd_capture_turn_deferred(_build_args("R5", transcript))

    recon = reconcile_session_capture(
        "R5", transcript_path=transcript, include_durable=False,
    )
    assert recon["durable_turns"] is None
    assert recon["status"] == "ok"
    assert recon["missing_turns"] == 0


def test_cheap_leg_shortfall_logs_structured_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "first valid user turn long enough")
        + "{this is not valid json at all\n"
    )

    with caplog.at_level("WARNING", logger="iai_mcp.cli._capture"):
        rc = cmd_capture_turn_deferred(_build_args("T1", transcript))
    assert rc == 0

    assert "capture-turn-deferred shortfall" in caplog.text
    assert "session=T1" in caplog.text
    assert "expected=2" in caplog.text
    assert "captured=1" in caplog.text
    assert "missing=1" in caplog.text
    assert "parse_failed=1" in caplog.text


def test_cheap_leg_reconcile_exception_does_not_break_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.cli import cmd_capture_turn_deferred
    import iai_mcp.capture as capture_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("reconcile exploded")

    monkeypatch.setattr(capture_mod, "reconcile_session_capture", _boom)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "capture must survive a broken reconcile")
    )

    rc = cmd_capture_turn_deferred(_build_args("T2", transcript))
    assert rc == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "T2.live.jsonl"
    assert live.exists()
    lines = live.read_text().splitlines()
    assert len(lines) == 2, f"expected header + 1 event despite reconcile failure, got {lines}"


def test_durable_leg_reports_captured_equals_durable_after_full_drain(
    tmp_path, monkeypatch, caplog,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import deferred_captures_dir, drain_active_live_captures
    from tests.conftest_shared import make_tmp_store

    deferred_dir = deferred_captures_dir()
    deferred_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "deferred_at": "2026-09-07T00:00:00+00:00",
        "session_id": "T3",
        "cwd": "/tmp",
    }
    event = {
        "text": "a fully drained durable-leg event long enough",
        "cue": "session T3 turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-09-07T00:00:01+00:00",
    }
    (deferred_dir / "T3.live.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(event) + "\n"
    )

    store = make_tmp_store(tmp_path)
    try:
        with caplog.at_level("WARNING", logger="iai_mcp.capture"):
            counts = drain_active_live_captures(store, exclude_session_id="other-session")
        assert counts["events_inserted"] == 1, counts
        assert "drain_active_durable_shortfall" not in caplog.text
    finally:
        store.close()


def test_durable_leg_flags_shortfall_when_insert_never_landed(
    tmp_path, monkeypatch, caplog,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import deferred_captures_dir, drain_active_live_captures
    from tests.conftest_shared import make_tmp_store

    deferred_dir = deferred_captures_dir()
    deferred_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "deferred_at": "2026-09-07T00:00:00+00:00",
        "session_id": "T4",
        "cwd": "/tmp",
    }
    # Too-short text: written to spool (captured) but skipped at drain
    # (never durable) -- exactly the gap the durable leg exists to catch.
    event = {
        "text": "ok",
        "cue": "session T4 turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-09-07T00:00:01+00:00",
    }
    (deferred_dir / "T4.live.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(event) + "\n"
    )

    store = make_tmp_store(tmp_path)
    try:
        with caplog.at_level("WARNING", logger="iai_mcp.capture"):
            counts = drain_active_live_captures(store, exclude_session_id="other-session")
        assert counts["events_inserted"] == 0, counts
        assert "drain_active_durable_shortfall" in caplog.text
        assert "session=T4" in caplog.text
        assert "captured=1" in caplog.text
        assert "durable=0" in caplog.text
    finally:
        store.close()


def test_insert_failed_first_line_skips_durable_leg_reconciliation(
    tmp_path, monkeypatch, caplog,
):
    """A pass that makes zero progress (the only new line hits
    insert-failed and breaks immediately, before the offset ever advances)
    must not enter the durable-leg reconciliation -- entering it here would
    compare captured_turns against durable_turns for a session with
    genuinely nothing new to report this pass, producing a spurious
    shortfall log line every subsequent pass with no new information."""
    monkeypatch.setenv("HOME", str(tmp_path))
    import iai_mcp.capture as capture_mod
    from iai_mcp.capture import deferred_captures_dir, drain_active_live_captures
    from tests.conftest_shared import make_tmp_store

    deferred_dir = deferred_captures_dir()
    deferred_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "deferred_at": "2026-09-07T00:00:00+00:00",
        "session_id": "T5",
        "cwd": "/tmp",
    }
    event = {
        "text": "some captured text, long enough to not be filtered",
        "cue": "session T5 turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-09-07T00:00:01+00:00",
    }
    (deferred_dir / "T5.live.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(event) + "\n"
    )

    def fake_capture_turn(*args, **kwargs):
        return {
            "status": "skipped",
            "record_id": None,
            "reason": "insert-failed: RuntimeError",
        }

    monkeypatch.setattr(capture_mod, "capture_turn", fake_capture_turn)

    store = make_tmp_store(tmp_path)
    try:
        with caplog.at_level("WARNING", logger="iai_mcp.capture"):
            counts = drain_active_live_captures(store, exclude_session_id="other-session")
        assert counts["events_inserted"] == 0, counts
        assert counts.get("events_skipped_insert_failed") == 1, counts
        assert "drain_active_durable_shortfall" not in caplog.text
    finally:
        store.close()
