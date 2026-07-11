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

def test_hd_shorter_transcript_must_not_clobber_offset():
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

        assert final_offset >= 1324, (
            f"offset was clobbered: stored 1324, final {final_offset}. "
            f"Shorter transcript reset prev=0 and rewrote old turns."
        )

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 0, (
            f"re-emitted {live_turns} old turns as new events (clobber bug)"
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
        assert _read_offset(state_dir, sid) == -1, "offset must not be created"
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
