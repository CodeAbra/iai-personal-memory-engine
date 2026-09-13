"""The narrow /clear-continuation override must never leak a prior session's
private focal goal to a session that lacks its own fresh marker -- absence,
staleness, and a wrong-sid marker must all still block (established privacy
invariant preserved). A companion test proves the producer (SessionStart
hook) and the consumer (per-turn hook) resolve the marker's store root
identically.
"""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

_CONSUMER_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)
_PRODUCER_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-session-recall.sh"
)

_PRIOR_GOAL = "session one private focal goal"


def _write_continuity(root: Path, recorded_sid: str) -> None:
    (root / ".session-continuity.cached.md").write_text(
        f"<iai-mcp-live-state>\ngoal: {_PRIOR_GOAL}\n"
        "</iai-mcp-live-state>\n<iai-mcp-agent-registry>\n</iai-mcp-agent-registry>\n",
        encoding="utf-8",
    )
    (root / ".session-continuity.state.json").write_text(
        f'{{"session_id": "{recorded_sid}"}}', encoding="utf-8",
    )


def _run_consumer_hook(root: Path, stdin_json: str) -> str:
    env = dict(os.environ)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    env["IAI_MCP_STORE"] = str(root)
    env["IAI_MCP_ROOT"] = str(root)
    proc = subprocess.run(
        [str(_CONSUMER_HOOK)],
        input=stdin_json,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


def test_privacy_a_no_marker_stays_blocked(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    _write_continuity(root, "s1")

    out = _run_consumer_hook(root, '{"prompt": "hi", "session_id": "s3"}')

    assert _PRIOR_GOAL not in out


def test_privacy_b_stale_marker_no_override(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    _write_continuity(root, "s1")

    marker = root / ".session-clear-continuation.s3"
    marker.write_text("", encoding="utf-8")
    stale = time.time() - 30000  # past the 21600s (6h) TTL
    os.utime(marker, (stale, stale))

    out = _run_consumer_hook(root, '{"prompt": "hi", "session_id": "s3"}')

    assert _PRIOR_GOAL not in out


def test_privacy_d_future_mtime_negative_age_no_override(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    _write_continuity(root, "s1")

    marker = root / ".session-clear-continuation.s3"
    marker.write_text("", encoding="utf-8")
    future = time.time() + 30000  # mtime ahead of now -> negative file_age
    os.utime(marker, (future, future))

    out = _run_consumer_hook(root, '{"prompt": "hi", "session_id": "s3"}')

    assert _PRIOR_GOAL not in out


def test_privacy_c_wrong_sid_marker_no_override(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    _write_continuity(root, "s1")

    (root / ".session-clear-continuation.s2").write_text("", encoding="utf-8")

    out = _run_consumer_hook(root, '{"prompt": "hi", "session_id": "s3"}')

    assert _PRIOR_GOAL not in out


def _make_stub_cli(dir_: Path) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text("#!/usr/bin/env bash\nexit 1\n")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def test_producer_path_marker_lands_at_iai_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    stub_cli = _make_stub_cli(stub_dir)

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_MCP_ROOT"] = str(root)
    env.pop("IAI_MCP_STORE", None)
    env["IAI_MCP_SESSION_RECALL_CLI"] = str(stub_cli)

    proc = subprocess.run(
        ["sh", str(_PRODUCER_HOOK)],
        input='{"session_id": "s2", "source": "clear"}',
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    marker = root / ".session-clear-continuation.s2"
    assert marker.is_file(), (
        "producer must stamp the marker under the consumer's IAI_ROOT "
        "three-level fallback -- proving path agreement under the "
        "IAI_MCP_ROOT-only legacy env shape"
    )
