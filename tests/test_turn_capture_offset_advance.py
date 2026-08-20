from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOOK_FILE = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"

def _extract_py_script() -> str:
    text = HOOK_FILE.read_text()
    m = re.search(r"PY_SCRIPT='(.*?)'\s*\n", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find PY_SCRIPT heredoc in {HOOK_FILE}")
    return m.group(1)

def _run_py_script(
    py_script: str,
    session_id: str,
    transcript_path: Path,
    home_dir: Path,
) -> tuple[int, float]:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", py_script, session_id, str(transcript_path)],
        env=env,
        capture_output=True,
        timeout=15,
    )
    elapsed = time.monotonic() - t0
    return result.returncode, elapsed

def _make_transcript(path: Path, n_lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_lines):
            role = "user" if i % 2 == 0 else "assistant"
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": f"Turn {i}"},
            }) + "\n")

def _make_transcript_with_nonce(path: Path, n_lines: int, nonce: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_lines):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Turn {i} {nonce}" if (i == 0 and role == "user") else f"Turn {i}"
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": content},
            }) + "\n")

def _read_offset(state_dir: Path, session_id: str) -> int:
    offset_file = state_dir / f"{session_id}.offset"
    if not offset_file.exists():
        return -1
    return int(offset_file.read_text().strip() or "0")

def _count_live_turns(deferred_dir: Path, session_id: str) -> int:
    live_file = deferred_dir / f"{session_id}.live.jsonl"
    if not live_file.exists():
        return 0
    count = 0
    with live_file.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "role" in obj:
                    count += 1
            except Exception:
                pass
    return count

def _run_py_script_with_prompt(
    py_script: str,
    session_id: str,
    transcript_path: Path,
    home_dir: Path,
    prompt: str,
    prompt_id: str,
) -> tuple[int, float]:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    payload_tmp = home_dir / f"{session_id}-payload.json"
    payload_tmp.write_text(
        json.dumps({"prompt": prompt, "prompt_id": prompt_id}), encoding="utf-8"
    )
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", py_script, session_id, str(transcript_path),
         str(payload_tmp), prompt_id],
        env=env,
        capture_output=True,
        timeout=15,
    )
    elapsed = time.monotonic() - t0
    return result.returncode, elapsed


def _live_contains_text(deferred_dir: Path, session_id: str, text: str) -> bool:
    live_file = deferred_dir / f"{session_id}.live.jsonl"
    if not live_file.exists():
        return False
    with live_file.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "role" in obj and text in obj.get("text", ""):
                    return True
            except Exception:
                pass
    return False

def test_ha_refuted_large_transcript_advances_offset():
    py_script = _extract_py_script()
    sid = "test-ha-refutation"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 1520)

        (state_dir / f"{sid}.offset").write_text("1324")

        rc, elapsed = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        new_offset = _read_offset(state_dir, sid)
        assert new_offset == 1520, f"expected 1520, got {new_offset}"
        assert elapsed < 4.0, f"took {elapsed:.2f}s — unexpected timeout risk"

def test_hd_shorter_transcript_restarts_at_the_new_stream():
    py_script = _extract_py_script()
    sid = "test-hd-short-transcript"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 50)
        (state_dir / f"{sid}.offset").write_text("1324")

        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)

        # Unified rotation policy: a stream shorter than the stored offset
        # is a new stream — restart at zero and capture it, never hold the
        # dead offset until the new stream outgrows it (that is silent
        # capture loss); replays land on the same idempotency keys.
        assert final_offset == 50, (
            f"expected a full re-walk of the new 50-line stream, "
            f"final offset {final_offset}"
        )
        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns > 0, (
            "the new stream must be captured immediately after rotation"
        )

def test_normal_growing_transcript_advances_and_writes_turns():
    py_script = _extract_py_script()
    sid = "test-normal-grow"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 20)
        (state_dir / f"{sid}.offset").write_text("10")

        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 20, f"expected 20, got {final_offset}"

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns > 0, "expected at least one turn written for new lines"

def test_fresh_session_no_offset_captures_all_turns():
    py_script = _extract_py_script()
    sid = "test-fresh-session"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 10)

        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 10, f"expected offset=10, got {final_offset}"

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 10, (
            f"expected 10 turns captured, got {live_turns}"
        )

def test_stale_path_scan_fallback_captures_turns():
    py_script = _extract_py_script()
    sid = "test-scan-fallback"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        project_dir = home / ".claude" / "projects" / "-Users-areg-Desktop-Claude"
        project_dir.mkdir(parents=True, exist_ok=True)
        real_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript(real_transcript, 12)

        stale_path = home / "nonexistent" / f"{sid}.jsonl"

        rc, _ = _run_py_script(py_script, sid, stale_path, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 12, (
            f"canonical-first did not activate: offset={final_offset}, "
            f"expected 12 (all lines of real transcript)"
        )
        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 12, (
            f"expected 12 turns captured via canonical-first, got {live_turns}"
        )

def test_missing_transcript_everywhere_exits_cleanly():
    """No transcript resolvable anywhere (no prompt_id supplied either): the
    walk has nothing to do, equivalent to a run with a transcript but zero
    new lines — offset is still published (0, a no-op value), but no turn
    is written to the live spool."""
    py_script = _extract_py_script()
    sid = "test-missing-everywhere"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        (home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

        stale_path = home / "no-such-file.jsonl"
        rc, _ = _run_py_script(py_script, sid, stale_path, home)

        assert rc == 0
        assert _read_offset(state_dir, sid) == 0, "offset publishes as a no-op 0"
        assert _count_live_turns(deferred_dir, sid) == 0, "live file must not be created"

def test_present_but_empty_stdin_uses_canonical_and_writes_nonce():
    py_script = _extract_py_script()
    sid = "test-empty-stdin-canonical"
    nonce = "e7k9p"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        project_dir = home / ".claude" / "projects" / "-Users-areg-Desktop-Claude"
        project_dir.mkdir(parents=True, exist_ok=True)
        canonical_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript_with_nonce(canonical_transcript, 35, nonce)

        empty_stdin = home / "empty-transcript.jsonl"
        empty_stdin.write_text("")

        rc, _ = _run_py_script(py_script, sid, empty_stdin, home)

        assert rc == 0, f"hook exited {rc}"

        assert _live_contains_text(deferred_dir, sid, nonce), (
            f"nonce '{nonce}' not found in live file — canonical-first fallback did not fire. "
            f"This is the 7173b585 regression."
        )

        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 35, f"expected offset=35, got {final_offset}"

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns > 0, "no turns written despite 35-line canonical transcript"

def test_present_but_wrong_session_stdin_uses_canonical_not_stdin():
    py_script = _extract_py_script()
    sid = "test-wrong-session-stdin"
    nonce = "e7k9p"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        project_dir = home / ".claude" / "projects" / "-Users-areg-Desktop-Claude"
        project_dir.mkdir(parents=True, exist_ok=True)
        canonical_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript_with_nonce(canonical_transcript, 35, nonce)

        other_sid = "other-session-xyz"
        wrong_stdin = home / "wrong-session.jsonl"
        _make_transcript(wrong_stdin, 50)

        rc, _ = _run_py_script(py_script, sid, wrong_stdin, home)

        assert rc == 0

        assert _live_contains_text(deferred_dir, sid, nonce), (
            f"nonce '{nonce}' not found — canonical-first did not override longer wrong-session stdin. "
            f"A max-lines strategy would fail this test."
        )

        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 35, (
            f"offset should be 35 (canonical line count), got {final_offset}"
        )


def test_codex_rollout_captured_with_tool_trailer(tmp_path):
    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    transcript = tmp_path / "rollout.jsonl"
    rows = [
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "tighten the rollout parser please"}]}},
        {"type": "response_item", "payload": {"type": "function_call",
         "call_id": "c1", "name": "shell", "arguments": "{}"}},
        {"type": "response_item", "payload": {"type": "function_call_output",
         "call_id": "c1", "output": "ok"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Parser tightened and checks pass."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rc, _ = _run_py_script(py_script, "s-codex-hook", transcript, home)
    assert rc == 0, f"hook exited {rc}"

    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    state_dir = home / ".iai-mcp" / ".capture-state"
    assert _read_offset(state_dir, "s-codex-hook") == 4
    assert _count_live_turns(deferred_dir, "s-codex-hook") == 2, (
        "codex rollout dialogue must be captured, not silently consumed"
    )
    assert _live_contains_text(deferred_dir, "s-codex-hook", "[tools: shell]")


def test_trailer_survives_two_hook_fires(tmp_path):
    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    transcript = tmp_path / "t.jsonl"
    first = [
        {"type": "user", "message": {"role": "user", "content": "run the suite please"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash", "input": {}, "id": "t1"}]}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in first) + "\n")
    rc, _ = _run_py_script(py_script, "s-straddle-hook", transcript, home)
    assert rc == 0

    with transcript.open("a") as f:
        f.write(json.dumps({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "The suite is green after the fix."}]}}) + "\n")
    rc, _ = _run_py_script(py_script, "s-straddle-hook", transcript, home)
    assert rc == 0

    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _live_contains_text(deferred_dir, "s-straddle-hook", "[tools: Bash]"), (
        "pending tool names must persist across hook fires beside the offset"
    )


def test_codex_event_msg_never_double_captures(tmp_path):
    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    transcript = tmp_path / "rollout.jsonl"
    rows = [
        {"type": "event_msg", "payload": {"type": "user_message",
         "message": "duplicate of the prompt zebra91"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "the real prompt zebra91"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Answer to the zebra91 prompt."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rc, _ = _run_py_script(py_script, "s-codex-dup", transcript, home)
    assert rc == 0
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _count_live_turns(deferred_dir, "s-codex-dup") == 2, (
        "event_msg mirrors of user turns must not double-capture"
    )


def test_codex_filtered_user_turn_clears_pending_in_hook(tmp_path):
    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    transcript = tmp_path / "rollout.jsonl"
    rows = [
        {"type": "response_item", "payload": {"type": "function_call",
         "call_id": "c1", "name": "shell", "arguments": "{}"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text",
                      "text": "<environment_context>x</environment_context>"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text",
                      "text": "Totally unrelated later answer."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rc, _ = _run_py_script(py_script, "s-codex-clear", transcript, home)
    assert rc == 0
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _live_contains_text(deferred_dir, "s-codex-clear", "unrelated later answer")
    assert not _live_contains_text(deferred_dir, "s-codex-clear", "[tools:"), (
        "a filtered user turn must clear pending tool names in the hook too"
    )


def test_matching_fingerprint_never_resets_a_growing_stream(tmp_path):
    """The false-positive direction: a snapshot carrying the CORRECT fp of
    a stream that merely grew must continue from the stored offset with
    zero re-emission of already-captured lines."""
    import hashlib

    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)

    rows = [
        {"type": "user", "message": {"role": "user",
         "content": f"same stream captured line {i}"}}
        for i in range(3)
    ]
    transcript = tmp_path / "t.jsonl"
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    transcript.write_text(body)
    first_raw = body.splitlines(keepends=True)[0].encode("utf-8")
    (state_dir / "s-grow.turnstate.json").write_text(json.dumps({
        "offset": 3,
        "pending": [],
        "fp": hashlib.sha256(first_raw).hexdigest()[:16],
    }))
    (state_dir / "s-grow.offset").write_text("3")

    transcript.write_text(body + json.dumps(
        {"type": "user", "message": {"role": "user",
         "content": "fresh growth line only this must emit"}}
    ) + "\n")

    rc, _ = _run_py_script(py_script, "s-grow", transcript, home)
    assert rc == 0
    assert _read_offset(state_dir, "s-grow") == 4

    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _live_contains_text(deferred_dir, "s-grow", "fresh growth line")
    assert not _live_contains_text(deferred_dir, "s-grow", "captured line 0"), (
        "a matching fingerprint must never trigger a re-walk"
    )


def test_hook_then_cli_share_one_fingerprint(tmp_path, monkeypatch):
    """Cross-carrier pin: the hook publishes the fp, the deferred CLI reads
    it back on the SAME stream — the offset must advance monotonically with
    zero re-emission (a decode-skewed fingerprint would reset every
    alternation and replay the whole transcript each time)."""
    import argparse

    from iai_mcp.cli import cmd_capture_turn_deferred

    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    rows = [
        {"type": "user", "message": {"role": "user",
         "content": f"один общий поток строка {i} с не-ASCII"}}  # non-English fixture data: ru round-trip under test
        for i in range(2)
    ]
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    rc, _ = _run_py_script(py_script, "s-xc", transcript, home)
    assert rc == 0
    state_dir = home / ".iai-mcp" / ".capture-state"
    assert _read_offset(state_dir, "s-xc") == 2

    transcript.write_text(
        transcript.read_text(encoding="utf-8") + json.dumps(
            {"type": "user", "message": {"role": "user",
             "content": "хвост потока для второго носителя"}},  # non-English fixture data: ru round-trip under test
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        session_id="s-xc", transcript_path=str(transcript),
        max_turns_per_call=200,
    )
    assert cmd_capture_turn_deferred(args) == 0
    assert _read_offset(state_dir, "s-xc") == 3, (
        "the CLI must continue from the hook offset, never reset on its "
        "own fingerprint of the same stream"
    )
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    live = deferred_dir / "s-xc.live.jsonl"
    body = live.read_text(encoding="utf-8") if live.exists() else ""
    assert body.count("строка 0") <= 1, "re-emission across carriers"  # non-English fixture data: ru round-trip under test


def test_py_script_heredoc_contains_no_apostrophes():
    """The PY_SCRIPT block lives in a single-quoted shell heredoc: one
    apostrophe anywhere inside terminates the quoting and bash executes the
    remainder as shell. This has broken the hook once; the extraction finds
    the same terminator the shell does."""
    py_script = _extract_py_script()
    assert py_script.strip(), "PY_SCRIPT block must extract non-empty"
    # The extraction regex is non-greedy: an end-of-line apostrophe inside
    # the script would truncate the match and the scan below would pass on
    # the clean prefix. Anchoring the real final lines makes truncation
    # fail loudly first.
    assert py_script.rstrip().endswith("except Exception:\n    pass"), (
        "PY_SCRIPT extraction truncated - an apostrophe ended a line early"
    )
    offenders = [
        (i + 1, line)
        for i, line in enumerate(py_script.splitlines())
        if chr(39) in line
    ]
    assert not offenders, f"apostrophes inside PY_SCRIPT: {offenders[:5]}"


def test_rotate_and_regrow_past_old_offset_is_caught_by_fingerprint(tmp_path):
    """The length fence is blind when the replacement stream already grew
    past the dead offset: only the first-line fingerprint distinguishes the
    streams, and without it the first lines of the new stream are silently
    lost while stale pending survives."""
    import hashlib

    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)

    old_first = json.dumps(
        {"type": "user", "message": {"role": "user", "content": "old stream head"}}
    ) + "\n"
    (state_dir / "s-fp.turnstate.json").write_text(json.dumps({
        "offset": 3,
        "pending": ["StaleToolFromDeadStream"],
        "fp": hashlib.sha256(old_first.encode("utf-8")).hexdigest()[:16],
    }))
    (state_dir / "s-fp.offset").write_text("3")

    rows = [
        {"type": "user", "message": {"role": "user",
         "content": f"replacement stream line {i} well past the old offset"}}
        for i in range(5)
    ] + [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "reply on the replacement stream."}]}},
    ]
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rc, _ = _run_py_script(py_script, "s-fp", transcript, home)
    assert rc == 0
    assert _read_offset(state_dir, "s-fp") == 6

    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _live_contains_text(deferred_dir, "s-fp", "replacement stream line 0"), (
        "the head of the replacement stream must be captured, not skipped "
        "by the dead offset"
    )
    assert not _live_contains_text(deferred_dir, "s-fp", "[tools:"), (
        "pending from the dead stream must not trail a replacement-stream turn"
    )


def test_immediate_prompt_capture_writes_one_line_keyed_by_prompt_id(tmp_path):
    py_script = _extract_py_script()
    sid = "test-immediate-prompt"
    prompt_id = "e2b1a111-2222-3333-4444-555566667777"
    prompt = "please tighten the immediate capture path for this test case"

    home = tmp_path / "home"
    home.mkdir()
    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True)

    transcript = home / "transcript.jsonl"
    _make_transcript(transcript, 4)

    rc, _ = _run_py_script_with_prompt(py_script, sid, transcript, home, prompt, prompt_id)
    assert rc == 0

    live_file = deferred_dir / f"{sid}.live.jsonl"
    events = [json.loads(ln) for ln in live_file.read_text().splitlines() if ln.strip()]
    matches = [e for e in events if e.get("source_uuid") == prompt_id]
    assert len(matches) == 1, (
        f"expected exactly one immediate line keyed by prompt_id, got {matches}"
    )
    assert matches[0]["text"] == prompt
    assert matches[0]["role"] == "user"


def test_immediate_prompt_capture_skipped_when_prompt_id_absent(tmp_path):
    py_script = _extract_py_script()
    sid = "test-immediate-no-prompt-id"
    prompt = "this prompt arrives with no prompt_id at all, zero regression"

    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / ".capture-state").mkdir(parents=True)
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True)

    transcript = home / "transcript.jsonl"
    _make_transcript(transcript, 4)

    rc, _ = _run_py_script_with_prompt(py_script, sid, transcript, home, prompt, "")
    assert rc == 0

    live_file = deferred_dir / f"{sid}.live.jsonl"
    events = (
        [json.loads(ln) for ln in live_file.read_text().splitlines() if ln.strip()]
        if live_file.exists() else []
    )
    assert not any(prompt == e.get("text") for e in events), (
        "prompt_id-absent must skip the immediate capture entirely (current behavior)"
    )


def _build_prompt_submit_payload(sid: str, transcript: Path, prompt: str, prompt_id: str, cwd: Path) -> dict:
    return {
        "cwd": str(cwd),
        "hook_event_name": "UserPromptSubmit",
        "permission_mode": "default",
        "prompt": prompt,
        "prompt_id": prompt_id,
        "session_id": sid,
        "session_title": "test",
        "transcript_path": str(transcript),
    }


def _run_full_hook_shell(payload: dict, home: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_MCP_TURN_INJECT_DISABLED"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK_FILE)],
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_full_hook_shell_immediate_capture_multiline_prompt(tmp_path):
    """Bash-level: a multi-line prompt with an embedded tab must survive the
    shell's stdin extraction verbatim — a naive tab-joined single-line read
    of .prompt would truncate or shift fields on this input."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / ".capture-state").mkdir(parents=True)
    (home / ".iai-mcp" / ".deferred-captures").mkdir(parents=True)
    project_dir = home / ".claude" / "projects" / "-proj"
    project_dir.mkdir(parents=True)
    sid = "full-shell-multiline"
    transcript = project_dir / f"{sid}.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "filler assistant reply text"}]},
    }) + "\n")

    prompt_id = "11111111-2222-3333-4444-555566667777"
    prompt_text = "first line of a multi-line prompt\nsecond line has a literal\ttab too"
    payload = _build_prompt_submit_payload(sid, transcript, prompt_text, prompt_id, tmp_path)

    proc = _run_full_hook_shell(payload, home)
    assert proc.returncode == 0, proc.stderr

    live = home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl"
    assert live.exists()
    events = [json.loads(ln) for ln in live.read_text().splitlines() if ln.strip()]
    matches = [e for e in events if e.get("source_uuid") == prompt_id]
    assert len(matches) == 1, matches
    assert matches[0]["text"] == prompt_text, (
        "multi-line/tab prompt must survive the shell extraction byte-identical"
    )


def test_full_hook_shell_immediate_capture_without_jq(tmp_path):
    """Same multi-line/tab prompt through the python-fallback extraction
    path (PATH stripped of jq)."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / ".capture-state").mkdir(parents=True)
    (home / ".iai-mcp" / ".deferred-captures").mkdir(parents=True)
    project_dir = home / ".claude" / "projects" / "-proj"
    project_dir.mkdir(parents=True)
    sid = "full-shell-no-jq"
    transcript = project_dir / f"{sid}.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "filler assistant reply text"}]},
    }) + "\n")

    prompt_id = "22222222-3333-4444-5555-666677778888"
    prompt_text = "no-jq fallback line one\nno-jq fallback line two with a\ttab"
    payload = _build_prompt_submit_payload(sid, transcript, prompt_text, prompt_id, tmp_path)

    proc = _run_full_hook_shell(payload, home, extra_env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr

    live = home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl"
    assert live.exists()
    events = [json.loads(ln) for ln in live.read_text().splitlines() if ln.strip()]
    matches = [e for e in events if e.get("source_uuid") == prompt_id]
    assert len(matches) == 1, matches
    assert matches[0]["text"] == prompt_text


def test_full_hook_shell_immediate_capture_first_fire_no_transcript(tmp_path):
    """First fire of a brand new session: no canonical transcript under
    ~/.claude/projects and a nonexistent transcript_path. The immediate
    stdin-prompt capture needs no transcript and must still write the
    prompt_id-keyed line the same turn it lands."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp" / ".capture-state").mkdir(parents=True)
    (home / ".iai-mcp" / ".deferred-captures").mkdir(parents=True)
    sid = "first-fire-no-transcript"
    transcript = home / "does-not-exist" / f"{sid}.jsonl"

    prompt_id = "33333333-4444-5555-6666-777788889999"
    prompt_text = "the very first prompt of a brand new session with no transcript yet"
    payload = _build_prompt_submit_payload(sid, transcript, prompt_text, prompt_id, tmp_path)

    proc = _run_full_hook_shell(payload, home)
    assert proc.returncode == 0, proc.stderr

    live = home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl"
    assert live.exists(), (
        "first-fire immediate capture must still write to the live spool "
        "even though no transcript is resolvable yet"
    )
    events = [json.loads(ln) for ln in live.read_text().splitlines() if ln.strip()]
    matches = [e for e in events if e.get("source_uuid") == prompt_id]
    assert len(matches) == 1, matches
    assert matches[0]["text"] == prompt_text
    assert matches[0]["role"] == "user"


def test_rotation_clears_stale_pending_from_dead_stream(tmp_path):
    py_script = _extract_py_script()
    home = tmp_path / "home"
    home.mkdir()
    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True)
    (state_dir / "s-rot.turnstate.json").write_text(
        json.dumps({"offset": 5, "pending": ["OldBashCall"]})
    )
    (state_dir / "s-rot.offset").write_text("5")

    transcript = tmp_path / "t.jsonl"
    rows = [
        {"type": "user", "message": {"role": "user", "content": f"short line {i}"}}
        for i in range(2)
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rc, _ = _run_py_script(py_script, "s-rot", transcript, home)
    assert rc == 0
    # Unified rotation policy: the shrunken stream is a NEW stream — capture
    # restarts at zero and walks it, never waits for it to outgrow the dead
    # offset (silent loss); replays land on the same idempotency keys.
    assert _read_offset(state_dir, "s-rot") == 2

    rows = [
        {"type": "user", "message": {"role": "user",
         "content": f"regrown line {i} of the new stream"}}
        for i in range(5)
    ] + [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "assistant reply on the new stream."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rc, _ = _run_py_script(py_script, "s-rot", transcript, home)
    assert rc == 0

    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    assert _live_contains_text(deferred_dir, "s-rot", "new stream")
    assert not _live_contains_text(deferred_dir, "s-rot", "[tools:"), (
        "pending from a dead transcript must never trail a new-stream turn"
    )
