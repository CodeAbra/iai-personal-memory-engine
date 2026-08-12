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
