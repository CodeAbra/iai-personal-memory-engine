"""Both halves of the session hook pair resolve the CLI the same way.

A stock `pip install` puts the binary in `~/.local/bin`; the capture half
found it there while the recall half knew only two paths — so capture worked
and recall silently skipped. The recall hook now carries the capture hook's
resolution order: env override, cache, PATH lookup, then the shared
candidate list.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_HOOKS = Path(__file__).resolve().parents[1] / "src/iai_mcp/_deploy/hooks"
_RECALL = _HOOKS / "iai-mcp-session-recall.sh"
_CAPTURE = _HOOKS / "iai-mcp-session-capture.sh"


def _candidates(script: Path) -> list[str]:
    body = script.read_text(encoding="utf-8")
    m = re.search(r"for candidate in \\\n(.*?)\n\s*do", body, flags=re.DOTALL)
    assert m is not None, f"no candidate list in {script.name}"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_hook_pair_shares_one_candidate_list() -> None:
    assert _candidates(_RECALL) == _candidates(_CAPTURE), (
        "the two hook halves drifted apart — a stock install would capture "
        "but never recall (or vice versa)"
    )


def test_recall_hook_resolves_cli_from_path(tmp_path) -> None:
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "iai-mcp"
    fake.write_text(
        "#!/bin/sh\necho 'RESOLVED_VIA_PATH_MARKER'\n", encoding="utf-8"
    )
    fake.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = str(home)
    env["PATH"] = f"{bindir}:/usr/bin:/bin"
    env.pop("IAI_MCP_SESSION_RECALL_CLI", None)

    proc = subprocess.run(
        [str(_RECALL)], input='{"session_id": "s-lookup-test"}',
        capture_output=True, text=True, env=env, timeout=20,
    )
    assert proc.returncode == 0
    assert "RESOLVED_VIA_PATH_MARKER" in proc.stdout
    # First successful resolution is cached for later session-ends.
    assert (home / ".iai-mcp" / ".cli-path").read_text(
        encoding="utf-8"
    ) == str(fake)


def test_recall_hook_finds_user_site_install(tmp_path) -> None:
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    fake = local_bin / "iai-mcp"
    fake.write_text(
        "#!/bin/sh\necho 'RESOLVED_VIA_CANDIDATE_MARKER'\n", encoding="utf-8"
    )
    fake.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = str(home)
    env["PATH"] = "/usr/bin:/bin"  # no iai-mcp on PATH
    env.pop("IAI_MCP_SESSION_RECALL_CLI", None)

    proc = subprocess.run(
        [str(_RECALL)], input='{"session_id": "s-candidate-test"}',
        capture_output=True, text=True, env=env, timeout=20,
    )
    assert proc.returncode == 0
    assert "RESOLVED_VIA_CANDIDATE_MARKER" in proc.stdout
