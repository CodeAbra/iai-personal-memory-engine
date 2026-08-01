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
