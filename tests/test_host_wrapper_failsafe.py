"""The per-host wrapper scripts inherit the core skew contract: any failure
degrades to empty stdout + exit 0, and a successful recall is wrapped in the
host's envelope with nothing else on stdout."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX shell hooks"
)

HOOKS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "iai_mcp" / "_deploy" / "hooks"
)

WRAPPERS = (
    "iai-mcp-cursor-session-recall.sh",
    "iai-mcp-cursor-turn-capture.sh",
    "iai-mcp-antigravity-recall.sh",
    "iai-mcp-antigravity-capture.sh",
    "iai-mcp-hermes-recall.sh",
    "iai-mcp-hermes-capture.sh",
)


def _stage(tmp_path: Path, core_script: str | None, core_name: str) -> Path:
    """Copy one wrapper + a stub core into an isolated dir."""
    staged = tmp_path / "hooks"
    staged.mkdir()
    for name in WRAPPERS:
        (staged / name).write_bytes((HOOKS_DIR / name).read_bytes())
        (staged / name).chmod(0o755)
    if core_script is not None:
        core = staged / core_name
        core.write_text(core_script)
        core.chmod(core.stat().st_mode | stat.S_IXUSR)
    return staged


def _run(staged: Path, wrapper: str, payload: dict, home: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IAI_")}
    env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(staged / wrapper)],
        input=json.dumps(payload),
        env=env, capture_output=True, text=True, timeout=30.0,
    )


class TestCursorWrapper:
    def test_wraps_markdown_in_additional_context(self, tmp_path: Path) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf '## Identity\\nhello'\n",
            "iai-mcp-session-recall.sh",
        )
        proc = _run(staged, "iai-mcp-cursor-session-recall.sh",
                    {"session_id": "s"}, tmp_path)
        assert proc.returncode == 0
        out = json.loads(proc.stdout)
        assert out == {"additional_context": "## Identity\nhello"}

    def test_failing_core_degrades_silently(self, tmp_path: Path) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\necho boom >&2\nexit 1\n",
            "iai-mcp-session-recall.sh",
        )
        proc = _run(staged, "iai-mcp-cursor-session-recall.sh",
                    {"session_id": "s"}, tmp_path)
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    def test_missing_core_degrades_silently(self, tmp_path: Path) -> None:
        staged = _stage(tmp_path, None, "unused")
        (staged / "iai-mcp-session-recall.sh").unlink(missing_ok=True)
        proc = _run(staged, "iai-mcp-cursor-session-recall.sh",
                    {"session_id": "s"}, tmp_path)
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    def test_turn_capture_wrapper_swallows_core_envelope(
        self, tmp_path: Path,
    ) -> None:
        # beforeSubmitPrompt cannot inject: even a core that prints a
        # Claude-shaped envelope must reach Cursor as silence.
        staged = _stage(
            tmp_path,
            "#!/usr/bin/env bash\n"
            "printf '{\"hookSpecificOutput\": {\"x\": 1}}'\n",
            "iai-mcp-turn-capture.sh",
        )
        proc = _run(staged, "iai-mcp-cursor-turn-capture.sh",
                    {"session_id": "s", "transcript_path": "/nonexistent"},
                    tmp_path)
        assert proc.returncode == 0
        assert proc.stdout == ""

    def test_turn_capture_wrapper_disables_inject_gate(
        self, tmp_path: Path,
    ) -> None:
        seen = tmp_path / "env.txt"
        staged = _stage(
            tmp_path,
            f"#!/usr/bin/env bash\nprintf '%s' \"${{IAI_MCP_TURN_INJECT_DISABLED:-}}\" > {seen}\n",
            "iai-mcp-turn-capture.sh",
        )
        proc = _run(staged, "iai-mcp-cursor-turn-capture.sh",
                    {"session_id": "s"}, tmp_path)
        assert proc.returncode == 0
        assert seen.read_text() == "1"


class TestAntigravityRecallWrapper:
    def test_first_invocation_camel_case_emits_user_message(
        self, tmp_path: Path,
    ) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'recall payload'\n",
            "iai-mcp-session-recall.sh",
        )
        proc = _run(staged, "iai-mcp-antigravity-recall.sh",
                    {"conversationId": "c1", "invocationNum": 1}, tmp_path)
        assert proc.returncode == 0
        out = json.loads(proc.stdout)
        assert out == {"injectSteps": [{"userMessage": "recall payload"}]}

    def test_later_invocation_snake_case_emits_ephemeral(
        self, tmp_path: Path,
    ) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'tier slice'\n",
            "iai-mcp-per-turn-recall.sh",
        )
        proc = _run(staged, "iai-mcp-antigravity-recall.sh",
                    {"conversation_id": "c1", "invocation_num": 4}, tmp_path)
        assert proc.returncode == 0
        out = json.loads(proc.stdout)
        assert out == {"injectSteps": [{"ephemeralMessage": "tier slice"}]}

    def test_garbage_stdin_degrades_silently(self, tmp_path: Path) -> None:
        staged = _stage(tmp_path, None, "unused")
        proc = subprocess.run(
            ["bash", str(staged / "iai-mcp-antigravity-recall.sh")],
            input="not json", env={**os.environ, "HOME": str(tmp_path)},
            capture_output=True, text=True, timeout=30.0,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    def test_unknown_counter_uses_marker_not_durable_reinjection(
        self, tmp_path: Path,
    ) -> None:
        # No parseable invocation counter: the first run for a conversation
        # may emit the durable brief, every later run must degrade to the
        # ephemeral slice — never a durable userMessage per turn.
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'brief'\n",
            "iai-mcp-session-recall.sh",
        )
        per_turn = staged / "iai-mcp-per-turn-recall.sh"
        per_turn.write_text("#!/usr/bin/env bash\nprintf 'slice'\n")
        per_turn.chmod(0o755)
        first = _run(staged, "iai-mcp-antigravity-recall.sh",
                     {"conversationId": "c-marker"}, tmp_path)
        assert json.loads(first.stdout) == {
            "injectSteps": [{"userMessage": "brief"}]
        }
        second = _run(staged, "iai-mcp-antigravity-recall.sh",
                      {"conversationId": "c-marker"}, tmp_path)
        assert json.loads(second.stdout) == {
            "injectSteps": [{"ephemeralMessage": "slice"}]
        }

    def test_multi_megabyte_payload_survives(self, tmp_path: Path) -> None:
        # The payload rides stdin, not an env var — env strings hit ARG_MAX
        # (128 KiB per string on Linux) and would silently drop the recall.
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'brief'\n",
            "iai-mcp-session-recall.sh",
        )
        big = {
            "conversationId": "c-big",
            "invocationNum": 1,
            "junk": "x" * (2 * 1024 * 1024),
        }
        proc = _run(staged, "iai-mcp-antigravity-recall.sh", big, tmp_path)
        assert json.loads(proc.stdout) == {
            "injectSteps": [{"userMessage": "brief"}]
        }

    def test_unknown_counter_and_unknown_conversation_stays_ephemeral(
        self, tmp_path: Path,
    ) -> None:
        # No conversation id at all: no marker is possible, so the durable
        # path must never be chosen.
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'brief'\n",
            "iai-mcp-session-recall.sh",
        )
        per_turn = staged / "iai-mcp-per-turn-recall.sh"
        per_turn.write_text("#!/usr/bin/env bash\nprintf 'slice'\n")
        per_turn.chmod(0o755)
        proc = _run(staged, "iai-mcp-antigravity-recall.sh", {}, tmp_path)
        assert json.loads(proc.stdout) == {
            "injectSteps": [{"ephemeralMessage": "slice"}]
        }


class TestAntigravityCaptureWrapper:
    def test_swaps_in_full_transcript_and_translates_keys(
        self, tmp_path: Path,
    ) -> None:
        seen = tmp_path / "seen.json"
        staged = _stage(
            tmp_path,
            f"#!/usr/bin/env bash\ncat > {seen}\n",
            "iai-mcp-session-capture.sh",
        )
        tdir = tmp_path / "logs"
        tdir.mkdir()
        (tdir / "transcript.jsonl").write_text("{}\n")
        (tdir / "transcript_full.jsonl").write_text("{}\n")
        proc = _run(staged, "iai-mcp-antigravity-capture.sh", {
            "conversationId": "c9",
            "transcriptPath": str(tdir / "transcript.jsonl"),
        }, tmp_path)
        assert proc.returncode == 0
        forwarded = json.loads(seen.read_text())
        assert forwarded["session_id"] == "c9"
        assert forwarded["transcript_path"].endswith("transcript_full.jsonl")

    def test_keeps_short_transcript_when_full_absent(self, tmp_path: Path) -> None:
        seen = tmp_path / "seen.json"
        staged = _stage(
            tmp_path,
            f"#!/usr/bin/env bash\ncat > {seen}\n",
            "iai-mcp-session-capture.sh",
        )
        tdir = tmp_path / "logs"
        tdir.mkdir()
        (tdir / "transcript.jsonl").write_text("{}\n")
        proc = _run(staged, "iai-mcp-antigravity-capture.sh", {
            "conversationId": "c9",
            "transcriptPath": str(tdir / "transcript.jsonl"),
        }, tmp_path)
        assert proc.returncode == 0
        forwarded = json.loads(seen.read_text())
        assert forwarded["transcript_path"].endswith("transcript.jsonl")


class TestHermesRecallWrapper:
    def test_first_turn_wraps_in_context_envelope(self, tmp_path: Path) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nprintf 'identity block'\n",
            "iai-mcp-session-recall.sh",
        )
        proc = _run(staged, "iai-mcp-hermes-recall.sh",
                    {"session_id": "h1", "is_first_turn": True}, tmp_path)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"context": "identity block"}

    def test_failing_core_degrades_silently(self, tmp_path: Path) -> None:
        staged = _stage(
            tmp_path, "#!/usr/bin/env bash\nexit 1\n",
            "iai-mcp-session-recall.sh",
        )
        proc = _run(staged, "iai-mcp-hermes-recall.sh",
                    {"session_id": "h1", "is_first_turn": True}, tmp_path)
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""


class TestHermesCaptureWrapper:
    def _seed_db(self, home: Path, session: str) -> Path:
        import sqlite3

        hermes = home / ".hermes"
        hermes.mkdir(parents=True)
        db = hermes / "state.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
            " role TEXT, content TEXT, timestamp TEXT)"
        )
        rows = [
            (1, session, "user", "question one", "t1"),
            (2, session, "assistant", "answer one", "t2"),
            (3, session, "tool", "ignored", "t3"),
            (4, "other-session", "user", "not ours", "t4"),
        ]
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        return db

    def test_reads_db_and_writes_deferred_spool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        self._seed_db(home, "h-sess")
        staged = _stage(tmp_path, None, "unused")
        env = os.environ.copy()
        env["HOME"] = str(home)
        env.pop("IAI_MCP_HERMES_HOME", None)
        proc = subprocess.run(
            ["bash", str(staged / "iai-mcp-hermes-capture.sh")],
            input=json.dumps({"session_id": "h-sess"}),
            env=env, capture_output=True, text=True, timeout=30.0,
        )
        assert proc.returncode == 0
        deferred = home / ".iai-mcp" / ".deferred-captures"
        spools = list(deferred.glob("h-sess.live-*.jsonl"))
        assert len(spools) == 1
        # The spool must appear only complete: no .tmp sibling survives.
        assert not list(deferred.glob("*.tmp"))
        lines = [json.loads(x) for x in spools[0].read_text().splitlines()]
        assert lines[0]["version"] == 1
        events = lines[1:]
        assert [(e["role"], e["text"]) for e in events] == [
            ("user", "question one"), ("assistant", "answer one"),
        ]
        assert all(e["tier"] == "episodic" for e in events)
        # Watermark makes the second run a no-op.
        proc2 = subprocess.run(
            ["bash", str(staged / "iai-mcp-hermes-capture.sh")],
            input=json.dumps({"session_id": "h-sess"}),
            env=env, capture_output=True, text=True, timeout=30.0,
        )
        assert proc2.returncode == 0
        spools_after = list((home / ".iai-mcp" / ".deferred-captures").glob(
            "h-sess.live-*.jsonl"
        ))
        assert len(spools_after) == 1

    def test_missing_db_degrades_silently(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        staged = _stage(tmp_path, None, "unused")
        env = os.environ.copy()
        env["HOME"] = str(home)
        env.pop("IAI_MCP_HERMES_HOME", None)
        proc = subprocess.run(
            ["bash", str(staged / "iai-mcp-hermes-capture.sh")],
            input=json.dumps({"session_id": "x"}),
            env=env, capture_output=True, text=True, timeout=30.0,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""
