"""End-to-end coverage for `iai import`: cold-start seeding from existing
Claude Code (and, in later tasks, Codex CLI) transcripts through the
already-tested O(N) `capture_transcript` spine.

Hermetic: fixture transcripts + the autouse tmp-home fixtures in
conftest.py only. Never reads the real `~/.claude` / `~/.codex` /
`~/.iai-mcp`.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


ALICE_CLAUDE_TURNS = [
    ("user", "alice asks how the sleep pipeline orders its consolidation steps"),
    ("assistant", "alice: the pipeline runs SCHEMA_MINE then KNOB_TUNE overnight"),
    ("user", "alice wants to know how recall ranks candidate memories"),
    ("assistant", "alice: recall ranks by cosine plus centrality minus age"),
]


def _make_claude_transcript(
    directory: Path, session_id: str, turns: list[tuple[str, str]],
) -> Path:
    """Write a Claude-Code-shaped JSONL transcript (one line per turn),
    mirroring the fixture shape in test_capture_transcript_flush.py."""
    transcript_path = directory / f"{session_id}.jsonl"
    lines = []
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, (role, text) in enumerate(turns, start=1):
        ts = base_ts.replace(second=i % 60)
        turn = {
            "type": role,
            "uuid": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "sessionId": session_id,
            "message": {"role": role, "content": text},
        }
        lines.append(json.dumps(turn))
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript_path


def _make_args(path, *, source: str = "claude", **extra) -> argparse.Namespace:
    ns = argparse.Namespace(path=str(path), source=source, include_subagents=False)
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


def _open_store_for_read():
    """Open a MemoryStore over the autouse hermetic root (mirrors the
    pattern in test_iai_cli_upload.py -- HOME/DEFAULT_STORAGE_PATH are
    patched to a tmp root by conftest's autouse fixture)."""
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _count_episodic_records(store) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL AND tier = 'episodic'"
        ).fetchone()
    return int(row[0]) if row else 0


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_import_claude_seeds_store(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-1", ALICE_CLAUDE_TURNS)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(claude_dir))
    assert rc == 0, buf.getvalue()

    store = _open_store_for_read()
    try:
        assert _count_episodic_records(store) == len(ALICE_CLAUDE_TURNS), (
            f"[{driver}] expected {len(ALICE_CLAUDE_TURNS)} episodic records"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reimport_claude_is_idempotent(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-1", ALICE_CLAUDE_TURNS)

    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = cmd_import(_make_args(claude_dir))
    assert rc1 == 0, buf1.getvalue()

    store = _open_store_for_read()
    try:
        first_count = _count_episodic_records(store)
    finally:
        store.close()

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = cmd_import(_make_args(claude_dir))
    assert rc2 == 0, buf2.getvalue()
    out2 = buf2.getvalue()
    assert "reinforced=4" in out2 or "reinforced=" in out2, out2

    store2 = _open_store_for_read()
    try:
        second_count = _count_episodic_records(store2)
    finally:
        store2.close()

    assert second_count == first_count, (
        f"[{driver}] re-import must not insert duplicates: "
        f"{first_count} -> {second_count}"
    )


ALICE_CODEX_TURNS = [
    ("user", "alice asks the codex agent to explain the resume watermark"),
    ("assistant", "alice: the watermark is written only after flush_record_buffer"),
]


def _make_codex_rollout(
    directory: Path, filename: str, turns: list[tuple[str, str]],
) -> Path:
    """Write a Codex-rollout-shaped JSONL transcript (one `response_item`
    event per turn), matching `capture_transcript`'s existing codex branch."""
    rollout_path = directory / filename
    lines = []
    for i, (role, text) in enumerate(turns, start=1):
        event = {
            "timestamp": f"2026-01-01T00:00:{i:02d}Z",
            "ordinal": i,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "text", "text": text}],
                "id": str(uuid.uuid4()),
            },
        }
        lines.append(json.dumps(event))
    rollout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rollout_path


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_import_codex_seeds_store(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    codex_root = tmp_path / "codex-src"
    sessions_dir = codex_root / "sessions" / "2026" / "01" / "01"
    sessions_dir.mkdir(parents=True)
    _make_codex_rollout(sessions_dir, "rollout-20260101-abc.jsonl", ALICE_CODEX_TURNS)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(codex_root, source="codex"))
    assert rc == 0, buf.getvalue()

    store = _open_store_for_read()
    try:
        assert _count_episodic_records(store) == len(ALICE_CODEX_TURNS), (
            f"[{driver}] expected {len(ALICE_CODEX_TURNS)} episodic records"
        )
    finally:
        store.close()


def _make_shared_dir(tmp_path: Path, label: str) -> Path:
    """A directory holding both a Claude-shaped file and a nested Codex
    rollout file -- used to prove `--source` isolates by shape, not by
    accidentally importing whichever file the walk happens to hit first."""
    shared_root = tmp_path / f"shared-{label}"
    shared_root.mkdir()
    _make_claude_transcript(shared_root, f"sess-claude-{label}", ALICE_CLAUDE_TURNS)
    sessions_dir = shared_root / "sessions" / "2026" / "01" / "01"
    sessions_dir.mkdir(parents=True)
    _make_codex_rollout(
        sessions_dir, f"rollout-20260101-{label}.jsonl", ALICE_CODEX_TURNS,
    )
    return shared_root


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_source_codex_isolates_from_claude_in_shared_directory(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    shared_root = _make_shared_dir(tmp_path, "codex-case")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(shared_root, source="codex"))
    assert rc == 0, buf.getvalue()

    store = _open_store_for_read()
    try:
        count = _count_episodic_records(store)
    finally:
        store.close()
    assert count == len(ALICE_CODEX_TURNS), (
        f"[{driver}] --source codex must import only the codex file, got "
        f"{count} records (expected {len(ALICE_CODEX_TURNS)})"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_source_claude_isolates_from_codex_in_shared_directory(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    shared_root = _make_shared_dir(tmp_path, "claude-case")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(shared_root, source="claude"))
    assert rc == 0, buf.getvalue()

    store = _open_store_for_read()
    try:
        count = _count_episodic_records(store)
    finally:
        store.close()
    assert count == len(ALICE_CLAUDE_TURNS), (
        f"[{driver}] --source claude must import only the claude file, got "
        f"{count} records (expected {len(ALICE_CLAUDE_TURNS)})"
    )


def test_dry_run_writes_nothing(tmp_path):
    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-dry", ALICE_CLAUDE_TURNS)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(claude_dir, dry_run=True))
    assert rc == 0, buf.getvalue()
    out = buf.getvalue()
    assert f"would_import={len(ALICE_CLAUDE_TURNS)}" in out, out

    store = _open_store_for_read()
    try:
        assert _count_episodic_records(store) == 0, (
            "dry-run must write nothing to the store"
        )
    finally:
        store.close()


def test_import_warns_content_stored_unredacted(tmp_path):
    """`iai import` must warn, on stderr, that imported content is stored
    as-is and is not scanned for secrets -- non-blocking, no prompt."""
    from contextlib import redirect_stderr

    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-secrets-warning", ALICE_CLAUDE_TURNS)

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cmd_import(_make_args(claude_dir))
    assert rc == 0, out.getvalue()
    err_text = err.getvalue()
    assert "not scanned for secrets" in err_text, err_text
    assert "API keys" in err_text, err_text
    # stdout carries only the run summary -- the notice never lands there.
    assert "not scanned for secrets" not in out.getvalue()


def test_import_warns_content_stored_unredacted_on_dry_run(tmp_path):
    """The notice also fires on `--dry-run`, which reads sources but writes
    nothing -- the warning is about the source review, not the write."""
    from contextlib import redirect_stderr

    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-secrets-warning-dry", ALICE_CLAUDE_TURNS)

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cmd_import(_make_args(claude_dir, dry_run=True))
    assert rc == 0, out.getvalue()
    assert "not scanned for secrets" in err.getvalue(), err.getvalue()


def test_empty_rollout_file_counts_as_empty_files_not_skipped(tmp_path):
    """A rollout file with zero parseable turns (format drift / non-message
    events only) increments `empty_files`, is NOT folded into `skipped`,
    and the command still exits 0."""
    from iai_mcp.iai_cli import cmd_import

    codex_root = tmp_path / "codex-empty"
    sessions_dir = codex_root / "sessions" / "2026" / "01" / "01"
    sessions_dir.mkdir(parents=True)
    empty_rollout = sessions_dir / "rollout-20260101-empty.jsonl"
    # Valid JSON, valid response_item envelope, but not a "message" payload
    # -- e.g. a function_call event, which _parse_transcript_obj skips.
    empty_rollout.write_text(
        json.dumps({
            "timestamp": "2026-01-01T00:00:01Z",
            "ordinal": 1,
            "type": "response_item",
            "payload": {"type": "function_call", "name": "some_tool"},
        }) + "\n",
        encoding="utf-8",
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(codex_root, source="codex"))
    assert rc == 0, buf.getvalue()
    out = buf.getvalue()
    assert "empty_files=1" in out, out
    assert "skipped=0" in out, out


def test_busy_hint_on_consolidation_window_collision(monkeypatch, tmp_path):
    """When the SHARED open hits a consolidation-window lock collision,
    `_open_store_shared_or_fail("import")` returns None and prints the busy
    hint; `cmd_import` must surface that degrade cleanly (exit 1, no raw
    traceback) instead of leaving the branch unverified."""
    from contextlib import redirect_stderr

    from iai_mcp import iai_cli

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-busy", ALICE_CLAUDE_TURNS)

    def _boom():
        raise RuntimeError("store is locked by another process")

    monkeypatch.setattr(iai_cli, "_open_store_shared", _boom)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_import(_make_args(claude_dir))
    assert rc == 1
    assert iai_cli._STORE_BUSY_HINT in err.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Resume state: skip-unchanged, durability-precedes-watermark, and the
# never-mint-a-directive guard. Exercised on the lilli driver -- the resume
# state file is plain JSON and not driver-visible, but the durability +
# idempotency seam it depends on (flush_record_buffer, capture_turn) is the
# prod-truth driver's contract.
# ---------------------------------------------------------------------------

def _spy_capture_transcript(monkeypatch):
    """Wrap import_sessions.capture_transcript, recording every filename it
    is actually invoked for, so a resume test can assert a file was NOT
    re-parsed rather than merely re-inserted-as-zero."""
    import iai_mcp.import_sessions as _mod

    real_capture = _mod.capture_transcript
    calls: list[str] = []

    def _spy(store, path, *, session_id):
        calls.append(Path(path).name)
        return real_capture(store, path, session_id=session_id)

    monkeypatch.setattr(_mod, "capture_transcript", _spy)
    return calls


def test_resume_skips_unchanged_files_on_rerun(tmp_path, monkeypatch):
    _select_driver("lilli", monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    _make_claude_transcript(claude_dir, "sess-1", ALICE_CLAUDE_TURNS)

    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = cmd_import(_make_args(claude_dir))
    assert rc1 == 0, buf1.getvalue()

    store = _open_store_for_read()
    try:
        first_count = _count_episodic_records(store)
    finally:
        store.close()

    calls = _spy_capture_transcript(monkeypatch)

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = cmd_import(_make_args(claude_dir))
    assert rc2 == 0, buf2.getvalue()

    assert calls == [], (
        f"unchanged file must be skipped via resume state without being "
        f"re-parsed, but capture_transcript was called for {calls!r}"
    )

    store2 = _open_store_for_read()
    try:
        second_count = _count_episodic_records(store2)
    finally:
        store2.close()
    assert second_count == first_count, (
        f"resumed run must stay idempotent: {first_count} -> {second_count}"
    )


def test_file_missing_from_state_is_reimported_after_simulated_interruption(
    tmp_path, monkeypatch,
):
    """A partial resume-state file (as a crash mid-corpus would leave
    behind) must NOT block the file it never got to: absence from state
    always means re-import, present-with-matching-mtime always means skip."""
    _select_driver("lilli", monkeypatch)
    from iai_mcp.iai_cli import cmd_import
    from iai_mcp.store import MemoryStore

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    file_a = _make_claude_transcript(claude_dir, "sess-a", ALICE_CLAUDE_TURNS)
    file_b = _make_claude_transcript(claude_dir, "sess-b", ALICE_CODEX_TURNS)

    # Fabricate a partial state file, as if a prior run flushed + watermarked
    # A durably, then crashed before ever touching B.
    store = MemoryStore()
    state_dir = Path(store.db._store_root) / "import-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "claude.json").write_text(
        json.dumps({
            str(file_a.resolve()): {"mtime": file_a.stat().st_mtime, "counts": {}},
        }),
        encoding="utf-8",
    )
    store.close()

    calls = _spy_capture_transcript(monkeypatch)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(claude_dir))
    assert rc == 0, buf.getvalue()

    assert file_a.name not in calls, "file present in state (matching mtime) must be skipped"
    assert file_b.name in calls, "file absent from state must be re-imported"

    store2 = _open_store_for_read()
    try:
        count = _count_episodic_records(store2)
    finally:
        store2.close()
    assert count == len(ALICE_CODEX_TURNS), (
        f"only file B's turns should land this run (A was never actually "
        f"captured, only fabricated as 'seen'), got {count}"
    )


def test_directive_marker_line_never_mints_directive(tmp_path, monkeypatch):
    """A transcript line whose text carries the anchored directive-marker
    prefix is stored as an ordinary episodic record -- capture_transcript
    is never called with directive_marker_allowed=True, so no directive
    is ever minted from bulk-imported history."""
    _select_driver("lilli", monkeypatch)
    from iai_mcp.iai_cli import cmd_import

    claude_dir = tmp_path / "claude-src"
    claude_dir.mkdir()
    directive_text = "standing directive: alice always prefers concise answers"
    _make_claude_transcript(claude_dir, "sess-directive", [
        ("user", directive_text),
        ("assistant", "alice: noted, concise answers from now on"),
    ])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_import(_make_args(claude_dir))
    assert rc == 0, buf.getvalue()

    store = _open_store_for_read()
    try:
        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT COUNT(*) FROM records WHERE directive = 1"
            ).fetchone()
        directive_count = int(row[0]) if row else 0
        assert directive_count == 0, (
            "the import path must never mint a directive from bulk history"
        )
        assert _count_episodic_records(store) == 2, (
            "the directive-marker-prefixed line must still land as an "
            "ordinary episodic record, not be dropped"
        )
    finally:
        store.close()
