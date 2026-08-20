"""Version-skew failsafe: plugin hooks may be NEWER than the installed
package (the plugin half auto-refreshes; the pip half does not). Every hook
must degrade to empty stdout + exit 0 against an older CLI that rejects a
subcommand or flag it has never heard of — and the two heredoc-python hooks
must stay importless so no package version can break them at all."""

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

#: An old argparse CLI facing an unknown subcommand: usage on stderr,
#: NOTHING useful on stdout, exit 2.
OLD_CLI_STUB = (
    "#!/usr/bin/env bash\n"
    "echo 'usage: iai-mcp [-h] {capture,doctor} ...' >&2\n"
    "echo \"iai-mcp: error: invalid choice: '$1'\" >&2\n"
    "exit 2\n"
)

#: A hostile variant: garbage on stdout too. The hooks' rc-gate must still
#: keep the session context clean.
OLD_CLI_STDOUT_NOISE_STUB = (
    "#!/usr/bin/env bash\n"
    "echo 'DeprecationWarning: flag renamed'\n"
    "echo 'usage: ...' >&2\n"
    "exit 2\n"
)


def _stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_hook(
    hook: str,
    home: Path,
    stdin_payload: dict,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_MCP_RECALL_HOOK_TIMEOUT"] = "5"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOKS_DIR / hook)],
        input=json.dumps(stdin_payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=30.0,
    )


def _home_with_stub(tmp_path: Path, stub_script: str) -> Path:
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    stub = _stub_cli(stub_dir, stub_script)
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub))
    return home


class TestSessionRecallAgainstOlderCli:
    def test_unknown_subcommand_degrades_silently(self, tmp_path: Path) -> None:
        home = _home_with_stub(tmp_path, OLD_CLI_STUB)
        proc = _run_hook(
            "iai-mcp-session-recall.sh", home,
            {"session_id": "x", "source": "startup",
             "cwd": "/tmp", "transcript_path": ""},
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "", proc.stdout

    def test_stdout_noise_from_old_cli_never_reaches_the_session(
        self, tmp_path: Path,
    ) -> None:
        home = _home_with_stub(tmp_path, OLD_CLI_STDOUT_NOISE_STUB)
        proc = _run_hook(
            "iai-mcp-session-recall.sh", home,
            {"session_id": "x", "source": "startup",
             "cwd": "/tmp", "transcript_path": ""},
        )
        assert proc.returncode == 0, proc.stderr
        assert "DeprecationWarning" not in proc.stdout
        assert proc.stdout.strip() == "", proc.stdout


class TestSessionCaptureAgainstOlderCli:
    def _payload(self, tmp_path: Path) -> dict:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "hello"}}) + "\n"
        )
        return {
            "session_id": "skew-test", "cwd": "/tmp",
            "transcript_path": str(transcript),
        }

    def test_unknown_subcommand_degrades_silently(self, tmp_path: Path) -> None:
        home = _home_with_stub(tmp_path, OLD_CLI_STUB)
        proc = _run_hook(
            "iai-mcp-session-capture.sh", home, self._payload(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "", proc.stdout

    def test_missing_cli_binary_degrades_silently(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".iai-mcp").mkdir(parents=True)
        (home / ".iai-mcp" / ".cli-path").write_text(str(tmp_path / "gone"))
        proc = _run_hook(
            "iai-mcp-session-capture.sh", home, self._payload(tmp_path),
            # PATH without any iai-mcp so resolution falls through every rung.
            extra_env={"PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "", proc.stdout


class TestHeredocHooksAreImportless:
    """The per-turn hooks embed python that reads sidecar files directly.
    Their skew immunity IS the absence of iai_mcp imports — an import would
    couple every turn to whatever package version happens to be installed."""

    @pytest.mark.parametrize("hook", [
        "iai-mcp-per-turn-recall.sh",
        "iai-mcp-turn-capture.sh",
    ])
    def test_no_package_import(self, hook: str) -> None:
        text = (HOOKS_DIR / hook).read_text()
        assert "import iai_mcp" not in text
        assert "from iai_mcp" not in text

    def test_all_four_hooks_exist(self) -> None:
        names = {p.name for p in HOOKS_DIR.glob("*.sh")}
        assert {
            "iai-mcp-session-recall.sh", "iai-mcp-session-capture.sh",
            "iai-mcp-per-turn-recall.sh", "iai-mcp-turn-capture.sh",
        } <= names


class TestOldDaemonToleratesImmediateEvent:
    """The immediate-capture block re-keys role:user to promptId and stamps
    source_uuid on the spool event. A daemon predating this change already
    reads source_uuid via ev.get (never a required key), so draining an
    event carrying it must not crash — the true skew is the key-scheme
    change, shipped in lockstep with the hook, not this field."""

    def test_drain_reads_source_uuid_field_without_crashing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-hook-skew-passphrase")
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
        import keyring.core
        keyring.core._keyring_backend = None

        from iai_mcp.capture import drain_active_live_captures
        from iai_mcp.store import MemoryStore

        session_id = "old-daemon-tolerance-session"
        text = "an immediate stdin-captured prompt for the old-daemon tolerance probe"
        deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True)
        # The immediate-capture block writes to {session_id}.live.jsonl, the
        # same active-spool shape the per-turn hook already uses — drained by
        # drain_active_live_captures, not the rotated-file deferred drain.
        live = deferred_dir / f"{session_id}.live.jsonl"
        header = {
            "version": 1, "deferred_at": "2026-08-18T09:00:00.000Z",
            "session_id": session_id, "cwd": "/tmp",
        }
        event = {
            "text": text,
            "cue": f"session {session_id} turn",
            "tier": "episodic",
            "role": "user",
            "ts": "2026-08-18T09:00:00.000Z",
            "source_uuid": "immediate-prompt-id-old-daemon-probe",
        }
        with live.open("w") as fh:
            fh.write(json.dumps(header) + "\n")
            fh.write(json.dumps(event) + "\n")

        store = MemoryStore()
        try:
            counts = drain_active_live_captures(store, exclude_session_id="drainer-session")
        except Exception as exc:  # noqa: BLE001 -- the assertion below is the real check
            pytest.fail(f"drain must tolerate the source_uuid-bearing immediate event: {exc!r}")

        assert counts.get("events_inserted", 0) == 1, counts

        turns = store.recent_user_turns(n=10, session_id=session_id)
        matches = [t for t in turns if text in (t.literal_surface or "")]
        assert len(matches) == 1, (
            f"expected exactly 1 stored row for the immediate event; found {len(matches)}"
        )
