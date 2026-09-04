from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-transcript-sweep-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _user_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }


def _assistant_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _decode_spool_events(spool_path: Path) -> list[dict]:
    from iai_mcp.capture import _decode_spool_line

    events = []
    lines = spool_path.read_text(encoding="utf-8").splitlines()
    for raw in lines[1:]:  # first line is the header, never an event
        if not raw.strip():
            continue
        events.append(json.loads(_decode_spool_line(raw)))
    return events


def test_one_transcript_swept_drained_and_receipted(iai_home):
    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.transcript_sweep import sweep_once

    session_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    claude_root = iai_home / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-work" / f"{session_id}.jsonl"
    nonce = "transcript sweep tracer unique nonce nine nine nine"
    _write_transcript(
        transcript_path,
        [
            _user_line(
                nonce,
                uuid="u-1",
                ts="2026-09-02T20:00:00.000000+00:00",
            ),
            _assistant_line(
                "acknowledged the tracer request in full",
                uuid="a-1",
                ts="2026-09-02T20:00:01.000000+00:00",
            ),
        ],
    )

    summary = sweep_once(roots=[claude_root])

    assert summary["files_seen"] == 1
    assert summary["sessions_staged"] == 1
    assert summary["lines_staged"] == 2

    store = _open_store()
    try:
        drain_capture_backlog(store)
        turns = store.recent_user_turns(50, session_id=session_id)
        texts = [t.literal_surface for t in turns]
        assert any(nonce in (t or "") for t in texts), (
            f"nonce not found in recent_user_turns; got: {texts!r}"
        )
    finally:
        store.close()

    receipt_dir = iai_home / ".iai-mcp" / "logs"
    today = datetime.now(timezone.utc)
    receipt_path = receipt_dir / f"transcript-sweep-{today:%Y-%m-%d}.log"
    assert receipt_path.exists()
    receipt_bytes = receipt_path.read_bytes()
    receipt_text = receipt_bytes.decode("utf-8")
    receipt_lines = [ln for ln in receipt_text.splitlines() if ln.strip()]
    assert len(receipt_lines) == 1
    assert f"session={session_id}" in receipt_lines[0]
    assert "channel=transcript-sweep" in receipt_lines[0]
    assert nonce.encode("utf-8") not in receipt_bytes


def test_sweep_once_opens_no_store_and_holds_no_lock():
    import inspect
    import iai_mcp.transcript_sweep as mod

    source = inspect.getsource(mod)
    assert "MemoryStore" not in source
    assert "iai_mcp.store" not in source
    assert ".acquire(" not in source
    assert "Lock(" not in source


def _build_agent_home_tree(
    cowork_session_root: Path,
    *,
    account: str = "11111111-2222-4333-8444-555555555555",
    device: str = "66666666-7777-4888-9999-aaaaaaaaaaaa",
    ditto: str = "local_ditto_alice",
) -> Path:
    claude_dir = cowork_session_root / account / device / "agent" / ditto / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    return claude_dir


def test_transcript_roots_finds_shared_and_agent_home(tmp_path, monkeypatch):
    import iai_mcp.cli._cowork as cowork_mod
    from iai_mcp.transcript_sweep import _transcript_roots

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    shared_claude = tmp_path / ".claude"
    (shared_claude / "projects").mkdir(parents=True)

    cowork_session_root = tmp_path / "cowork-sessions"
    agent_claude = _build_agent_home_tree(cowork_session_root)
    monkeypatch.setattr(cowork_mod, "_cowork_session_roots", lambda: [cowork_session_root])

    roots = _transcript_roots()
    resolved = {r.resolve() for r in roots}
    assert shared_claude.resolve() in resolved
    assert agent_claude.resolve() in resolved


def test_transcript_roots_env_override_validated(tmp_path, monkeypatch):
    import iai_mcp.cli._cowork as cowork_mod
    from iai_mcp.transcript_sweep import _transcript_roots

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cowork_mod, "_cowork_session_roots", lambda: [])

    valid_dir = tmp_path / "custom-claude-home"
    valid_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(valid_dir))
    roots = _transcript_roots()
    assert valid_dir.resolve() in {r.resolve() for r in roots}

    for invalid_value in (
        "relative/not-absolute",
        str(tmp_path / "does-not-exist"),
        str(tmp_path / "a" / ".." / ".." / "etc"),
    ):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", invalid_value)
        roots = _transcript_roots()  # must not raise
        assert valid_dir.resolve() not in {r.resolve() for r in roots}

    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    roots = _transcript_roots()  # unset must not raise
    assert isinstance(roots, list)


def test_transcript_roots_dedupes_symlinked_root(tmp_path, monkeypatch):
    import iai_mcp.cli._cowork as cowork_mod
    from iai_mcp.transcript_sweep import _transcript_roots

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cowork_mod, "_cowork_session_roots", lambda: [])

    real_dir = tmp_path / "real-claude-home"
    real_dir.mkdir()
    shared_claude = tmp_path / ".claude"
    shared_claude.symlink_to(real_dir, target_is_directory=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(real_dir))

    roots = _transcript_roots()
    resolved = [r.resolve() for r in roots]
    assert resolved.count(real_dir.resolve()) == 1


def test_iter_transcript_files_excludes_side_files(tmp_path):
    from iai_mcp.transcript_sweep import _iter_transcript_files

    claude_root = tmp_path / ".claude"
    projects_dir = claude_root / "projects" / "-Users-alice-work"
    projects_dir.mkdir(parents=True)
    session_id = "cccccccc-dddd-4eee-8fff-000000000000"
    top_level = projects_dir / f"{session_id}.jsonl"
    top_level.write_text('{"type": "user"}\n', encoding="utf-8")

    side_dir = projects_dir / session_id
    side_dir.mkdir()
    (side_dir / "subagents").mkdir()
    (side_dir / "subagents" / "agent-1.jsonl").write_text("{}\n", encoding="utf-8")
    (side_dir / "tool-results").mkdir()
    (side_dir / "tool-results" / "hook-1-stdout.txt").write_text("output", encoding="utf-8")
    (side_dir / "audit.jsonl").write_text("{}\n", encoding="utf-8")

    found = list(_iter_transcript_files(claude_root))
    assert found == [top_level]


def test_iter_transcript_files_missing_projects_dir_is_skipped(tmp_path):
    from iai_mcp.transcript_sweep import _iter_transcript_files

    claude_root = tmp_path / "nonexistent-claude"
    assert list(_iter_transcript_files(claude_root)) == []


def test_resolver_reuses_platform_helper_no_duplicate_table():
    import inspect
    import iai_mcp.transcript_sweep as mod

    source = inspect.getsource(mod)
    assert "sys.platform" not in source
    assert "_cowork_session_roots" in source


def test_transcript_sweep_run_refuses_against_home_matching_the_account(
    iai_home, monkeypatch, capsys,
):
    """Pins: HOME resolving to the same path pwd.getpwuid reports refuses;
    --allow-live-home and the scheduled-run env marker both bypass it."""
    import pwd

    import iai_mcp.transcript_sweep as mod
    from iai_mcp.cli import main

    fake_pwent = type("_Pw", (), {"pw_dir": str(iai_home)})()
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: fake_pwent)
    monkeypatch.delenv(mod.SWEEP_SCHEDULED_ENV_VAR, raising=False)
    assert mod._targets_live_home() is True

    rc = main(["transcript-sweep", "run"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "--allow-live-home" in captured.err

    rc_allowed = main(["transcript-sweep", "run", "--allow-live-home"])
    assert rc_allowed == 0

    monkeypatch.setenv(mod.SWEEP_SCHEDULED_ENV_VAR, "1")
    rc_scheduled = main(["transcript-sweep", "run"])
    assert rc_scheduled == 0


def test_transcript_sweep_run_never_refuses_a_tmp_home(iai_home, monkeypatch):
    """Pins: an unfaked pwd lookup (real account home) never matches a
    tmp-dir HOME, so a normal test run is never refused."""
    import iai_mcp.transcript_sweep as mod
    from iai_mcp.cli import main

    monkeypatch.delenv(mod.SWEEP_SCHEDULED_ENV_VAR, raising=False)
    assert mod._targets_live_home() is False

    rc = main(["transcript-sweep", "run"])
    assert rc == 0


def test_transcript_sweep_run_help_names_no_internal_concepts():
    from iai_mcp.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as ei:
            main(["transcript-sweep", "run", "--help"])
    assert ei.value.code == 0
    out = buf.getvalue()
    lowered = out.lower()
    for forbidden in ("daemon", "courier", "phase", "experiment"):
        assert forbidden not in lowered, f"forbidden word {forbidden!r} in help text: {out!r}"


def test_since_line_unset_matches_original_behavior(iai_home):
    from iai_mcp.capture import write_deferred_captures

    transcript_path = iai_home / "plain-transcript.jsonl"
    _write_transcript(
        transcript_path,
        [_user_line("a plain unsliced turn with enough length", uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00")],
    )
    out_path = write_deferred_captures(
        session_id="plain-session", transcript_path=transcript_path, cwd=str(iai_home),
    )
    events = _decode_spool_events(out_path)
    assert len(events) == 1
    assert events[0]["cue"] == "session plain-session turn 1"


def test_since_line_preserves_tool_trailer_boundary_across_resume(iai_home):
    from iai_mcp.capture import _write_deferred_captures_impl

    session_id = "trailer-boundary-session"
    transcript_path = iai_home / "trailer-transcript.jsonl"
    tool_only_line = {
        "type": "assistant",
        "uuid": "a-tool-1",
        "timestamp": "2026-09-02T20:00:01.000000+00:00",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Bash", "id": "tu-1", "input": {}}],
        },
    }
    _write_transcript(
        transcript_path,
        [
            _user_line(
                "hello there this is the first user turn",
                uuid="u-1",
                ts="2026-09-02T20:00:00.000000+00:00",
            ),
            tool_only_line,
            _assistant_line(
                "the tool ran successfully and returned useful data",
                uuid="a-text-1",
                ts="2026-09-02T20:00:02.000000+00:00",
            ),
        ],
    )

    one_pass_path, _pending, _seen = _write_deferred_captures_impl(
        session_id, transcript_path, cwd=str(iai_home),
    )
    one_pass_events = _decode_spool_events(one_pass_path)
    assert any("[tools: Bash]" in (e.get("text") or "") for e in one_pass_events)

    pass1_path, pass1_pending, pass1_seen = _write_deferred_captures_impl(
        session_id, transcript_path, cwd=str(iai_home), max_turns=2,
    )
    # Decode immediately: the auto-generated spool filename is keyed on
    # session id + wall-clock second + pid, so a second same-session call
    # within the same second would silently overwrite this exact path.
    pass1_events = _decode_spool_events(pass1_path)

    pass2_path, _pass2_pending, _pass2_seen = _write_deferred_captures_impl(
        session_id, transcript_path, cwd=str(iai_home),
        since_line=pass1_seen, pending_tools=pass1_pending,
    )
    pass2_events = _decode_spool_events(pass2_path)
    resumed_events = pass1_events + pass2_events

    assert resumed_events == one_pass_events


def test_should_sweep_always_true_for_unseen_file(tmp_path):
    from iai_mcp.transcript_sweep import _should_sweep

    p = tmp_path / "new.jsonl"
    p.write_text("data", encoding="utf-8")
    assert _should_sweep(p, None, now=time.time()) is True


def test_should_sweep_skips_unchanged_file(tmp_path):
    from iai_mcp.transcript_sweep import _SweepState, _should_sweep

    p = tmp_path / "unchanged.jsonl"
    p.write_text("data", encoding="utf-8")
    st = p.stat()
    prior = _SweepState(mtime_ns=st.st_mtime_ns, size=st.st_size, lines_swept=5)
    assert _should_sweep(p, prior, now=time.time()) is False


def test_should_sweep_waits_for_quiet_window_on_active_file(tmp_path):
    from iai_mcp.transcript_sweep import _SweepState, _should_sweep

    p = tmp_path / "active.jsonl"
    p.write_text("data", encoding="utf-8")
    prior = _SweepState(mtime_ns=0, size=1, lines_swept=1)  # differs -> "changed"
    assert _should_sweep(p, prior, now=time.time()) is False


def test_first_sweep_bound_caps_unseen_large_transcript(iai_home, monkeypatch):
    import iai_mcp.transcript_sweep as mod

    monkeypatch.setattr(mod, "FIRST_SWEEP_LINE_BOUND", 10)

    session_id = "huge-pre-existing-session"
    claude_root = iai_home / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-huge" / f"{session_id}.jsonl"
    lines = [
        _user_line(f"user turn number {i:03d} padded to a safe length", uuid=f"u-{i}", ts="2026-09-02T20:00:00.000000+00:00")
        for i in range(30)
    ]
    _write_transcript(transcript_path, lines)

    summary = mod.sweep_once(roots=[claude_root])

    assert summary["files_seen"] == 1
    assert 0 < summary["lines_staged"] <= 10

    state = mod._read_sweep_state(session_id)
    assert state is not None
    assert state.lines_swept <= 10


def test_torn_final_line_not_lost_on_first_sweep(iai_home, monkeypatch):
    """A first-ever sweep (no quiet-window delay) that reads a transcript
    while its final line is still mid-write must not count that line
    toward the persisted high-water mark -- the writer finishes it and a
    later pass must still capture it, exactly once."""
    import iai_mcp.transcript_sweep as mod

    monkeypatch.setattr(mod, "QUIET_WINDOW_SEC", 0)

    session_id = "torn-line-session"
    claude_root = iai_home / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-torn" / f"{session_id}.jsonl"
    transcript_path.parent.mkdir(parents=True)

    first_line = json.dumps(
        _user_line(
            "the first turn, fully written before the torn tail",
            uuid="u-1",
            ts="2026-09-02T20:00:00.000000+00:00",
        ),
        ensure_ascii=False,
    )
    second_obj = _user_line(
        "the second turn, still being written when the first sweep runs",
        uuid="u-2",
        ts="2026-09-02T20:00:01.000000+00:00",
    )
    second_line = json.dumps(second_obj, ensure_ascii=False)
    torn_fragment = second_line[: len(second_line) // 2]

    transcript_path.write_text(first_line + "\n" + torn_fragment, encoding="utf-8")

    summary = mod.sweep_once(roots=[claude_root])
    assert summary["files_seen"] == 1
    assert summary["lines_staged"] == 1  # the torn tail must not be counted

    state = mod._read_sweep_state(session_id)
    assert state is not None
    assert state.lines_swept == 1  # high-water mark withheld past the torn line

    # The writer finishes the line.
    transcript_path.write_text(first_line + "\n" + second_line + "\n", encoding="utf-8")

    summary2 = mod.sweep_once(roots=[claude_root])
    assert summary2["lines_staged"] == 1  # only the newly-completed turn

    state2 = mod._read_sweep_state(session_id)
    assert state2 is not None
    assert state2.lines_swept == 2

    # An unchanged file is a no-op: the completed turn was staged exactly
    # once, never re-staged.
    summary3 = mod.sweep_once(roots=[claude_root])
    assert summary3["sessions_staged"] == 0


def test_malformed_midfile_line_logged_and_skipped(iai_home, caplog):
    """A genuinely malformed (not torn) line -- complete, newline-
    terminated, but not valid JSON -- must be skipped with a logged
    warning and never block later lines or get retried forever."""
    import logging

    from iai_mcp.capture import _write_deferred_captures_impl

    session_id = "malformed-midfile-session"
    transcript_path = iai_home / "malformed-transcript.jsonl"
    good_line = json.dumps(
        _user_line("a healthy turn before the corrupt one", uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00"),
        ensure_ascii=False,
    )
    trailing_line = json.dumps(
        _user_line("a healthy turn after the corrupt one", uuid="u-2", ts="2026-09-02T20:00:02.000000+00:00"),
        ensure_ascii=False,
    )
    transcript_path.write_text(
        good_line + "\nnot valid json at all\n" + trailing_line + "\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        out_path, _pending, seen = _write_deferred_captures_impl(
            session_id, transcript_path, cwd=str(iai_home),
        )

    assert seen == 3  # the malformed line still advances the count -- never re-attempted
    assert any("malformed" in r.message.lower() for r in caplog.records)
    events = _decode_spool_events(out_path)
    assert len(events) == 2


def test_sweep_isolates_one_failing_transcript_others_still_swept(iai_home, monkeypatch):
    """One transcript raising inside the per-file body must not abort the
    pass -- every other transcript is still discovered and staged."""
    import iai_mcp.transcript_sweep as mod

    claude_root = iai_home / ".claude"
    good_id = "good-session-aaaaaaaaaaaaaaaa"
    bad_id = "bad-session-bbbbbbbbbbbbbbbbbb"
    good_path = claude_root / "projects" / "-Users-alice-good" / f"{good_id}.jsonl"
    bad_path = claude_root / "projects" / "-Users-alice-bad" / f"{bad_id}.jsonl"
    _write_transcript(
        good_path,
        [_user_line("a healthy turn long enough to capture", uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00")],
    )
    _write_transcript(
        bad_path,
        [_user_line("a turn behind a write failure", uuid="u-2", ts="2026-09-02T20:00:00.000000+00:00")],
    )

    real_impl = mod._write_deferred_captures_impl

    def _flaky(session_id, *args, **kwargs):
        if session_id == bad_id:
            raise OSError("simulated write failure")
        return real_impl(session_id, *args, **kwargs)

    monkeypatch.setattr(mod, "_write_deferred_captures_impl", _flaky)

    summary = mod.sweep_once(roots=[claude_root])

    assert summary["files_seen"] == 2
    assert summary["files_failed"] == 1
    assert summary["sessions_staged"] == 1

    good_state = mod._read_sweep_state(good_id)
    assert good_state is not None
    bad_state = mod._read_sweep_state(bad_id)
    assert bad_state is None  # the failing file never reached the state write


def test_orphan_sweep_state_reclaimed_when_transcript_removed(iai_home):
    import iai_mcp.transcript_sweep as mod

    session_id = "will-be-removed"
    claude_root = iai_home / ".claude"
    transcript_path = claude_root / "projects" / "-Users-alice-x" / f"{session_id}.jsonl"
    _write_transcript(
        transcript_path,
        [_user_line("first sweep content long enough to capture", uuid="u-1", ts="2026-09-02T20:00:00.000000+00:00")],
    )

    mod.sweep_once(roots=[claude_root])
    state_path = mod._sweep_state_path(session_id)
    assert state_path.exists()

    transcript_path.unlink()
    mod.sweep_once(roots=[claude_root])
    assert not state_path.exists()


def test_sweep_state_suffix_excluded_from_time_based_gc():
    import inspect
    from iai_mcp import capture

    source = inspect.getsource(capture)
    idx = source.index("_CAPTURE_STATE_SUFFIXES = (")
    end_idx = source.index(")", idx)
    tuple_block = source[idx:end_idx]
    assert ".transcript-sweep" not in tuple_block
