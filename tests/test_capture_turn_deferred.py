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


def _race_worker(home: str, session_id: str, transcript_path: str, q) -> None:
    """Module-level (picklable under macOS's spawn start method) worker for
    the concurrent-capture regression test below."""
    import os as _os
    _os.environ["HOME"] = home
    from iai_mcp.cli import cmd_capture_turn_deferred
    try:
        rc = cmd_capture_turn_deferred(_build_args(session_id, Path(transcript_path)))
        q.put(("ok", rc))
    except Exception as exc:  # the pre-fix bug surfaces here
        q.put(("error", f"{type(exc).__name__}: {exc}"))


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


def test_concurrent_calls_do_not_race_on_shared_tmp_path(tmp_path, monkeypatch):
    """UserPromptSubmit's turn-capture and Stop's session-capture (which also
    calls this command) can fire close together for one exchange, both
    touching the same {session_id}.offset file. Before the fix, both writers
    shared one {session_id}.offset.tmp path, so whichever called os.replace()
    second raised FileNotFoundError once the first had already consumed it.
    Regression test: run genuinely concurrent OS processes (not threads —
    the fix keys uniqueness off os.getpid(), which threads would share) and
    assert none of them ever hit that error."""
    import multiprocessing as mp

    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _make_transcript_line("user", "concurrent race regression turn one")
        + _make_transcript_line("assistant", "concurrent race regression turn two")
    )
    session_id = "RACE1"

    ctx = mp.get_context("spawn" if platform.system() != "Linux" else "fork")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_race_worker, args=(str(tmp_path), session_id, str(transcript), q))
        for _ in range(10)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    results = [q.get(timeout=5) for _ in procs]
    errors = [r for r in results if r[0] == "error"]
    assert not errors, f"concurrent capture-turn-deferred calls raised: {errors}"

    offset_path = tmp_path / ".iai-mcp" / ".capture-state" / f"{session_id}.offset"
    assert offset_path.exists()
    assert int(offset_path.read_text().strip()) == 2, "offset must land on the real turn count"

    leftover_tmp = list((tmp_path / ".iai-mcp" / ".capture-state").glob(f"{session_id}.offset*.tmp"))
    assert not leftover_tmp, f"orphaned tmp files left behind: {leftover_tmp}"
