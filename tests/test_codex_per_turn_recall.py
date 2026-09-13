"""Synthetic-Codex host acceptance for the installed per-turn-recall hook.

Runs the shared hook through the real Codex installer path (install_codex_hooks
writes the script + renderer companion under a throwaway codex home, hooks.json
wires it under UserPromptSubmit) rather than invoking a copy of the script
directly, so a deploy-path regression (missing companion, wrong hooks_dir,
broken reinstall/uninstall) surfaces here even when the emitter unit coverage
in test_per_turn_recall_safety.py stays green.

Hermetic: throwaway HOME/IAI_MCP_STORE/IAI_MCP_CODEX_HOME and a local AF_UNIX
listener under the test's own temp root. Never the owner's live store, real
socket, or real ~/.codex.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pytest

from iai_mcp.cli._codex_hooks import (
    _RECALL_RENDER_HELPER,
    install_codex_hooks,
    status_codex_hooks,
    uninstall_codex_hooks,
)

_HOOK_NAME = "iai-mcp-per-turn-recall.sh"
_LEAKABLE_ENV = (
    "HOME", "IAI_MCP_STORE", "IAI_MCP_ROOT", "IAI_MCP_CODEX_HOME",
    "IAI_MCP_PER_TURN_SOCKET_ACCEL", "IAI_MCP_WORKING_TIER_FRESH_SEC",
    "IAI_MCP_FORESIGHT_FRESH_SEC", "IAI_MCP_RUNNING_AGENT_TTL_SEC",
    "IAI_MCP_RECALL_SOCKET_TIMEOUT", "IAI_DAEMON_SOCKET_PATH",
    "IAI_MCP_DIRECTIVES_OFF", "IAI_MCP_WORKING_TIER_CACHE",
    "IAI_MCP_FORESIGHT_PACK", "CLAUDE_PLUGIN_ROOT",
)


@dataclass(frozen=True)
class _InstalledCodex:
    root: Path
    home: Path
    codex_home: Path
    store: Path
    socket: Path


@pytest.fixture
def installed_codex(monkeypatch: pytest.MonkeyPatch):
    # Short root: the fake recall socket's AF_UNIX path has a ~104 byte limit,
    # well below pytest's nested default tmp_path.
    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="iai-cxpt-"))
    installed = _InstalledCodex(
        root=root, home=root / "home", codex_home=root / "codex-home",
        store=root / "store", socket=root / "recall.sock",
    )
    for path in (installed.home, installed.store):
        path.mkdir(mode=0o700)
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(installed.codex_home))
    assert install_codex_hooks() == 0
    try:
        yield installed
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _recall_script_path(installed: _InstalledCodex) -> Path:
    data = json.loads((installed.codex_home / "hooks.json").read_text())
    for entry in data["hooks"].get("UserPromptSubmit", []):
        for handler in entry.get("hooks") or []:
            command = handler.get("command") or ""
            if _HOOK_NAME in command:
                argv = shlex.split(command)
                assert argv[0] == "bash"
                return Path(argv[1])
    raise AssertionError(f"{_HOOK_NAME} not wired under UserPromptSubmit")


def _hook_env(installed: _InstalledCodex, **overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _LEAKABLE_ENV}
    env.update(
        HOME=str(installed.home),
        IAI_MCP_STORE=str(installed.store),
        IAI_MCP_CODEX_HOME=str(installed.codex_home),
        IAI_DAEMON_SOCKET_PATH=str(installed.socket),
        IAI_MCP_PER_TURN_SOCKET_ACCEL="0",
        TMPDIR=str(installed.root),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.update(overrides)
    return env


def _run_installed_hook(
    installed: _InstalledCodex, payload: dict, **env_overrides: str
) -> subprocess.CompletedProcess[bytes]:
    script = _recall_script_path(installed)
    result = subprocess.run(
        ["/bin/bash", str(script)],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(installed.root), env=_hook_env(installed, **env_overrides),
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result


class _RecallPeer:
    """Minimal fake daemon: accept one connection, read to newline, reply once."""

    def __init__(self, path: Path, reply: bytes) -> None:
        self.path, self.reply = path, reply
        self.requests: "list[dict]" = []

    def __enter__(self) -> "_RecallPeer":
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(1)
        self.server.settimeout(5)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.thread.join(timeout=5)
        self.server.close()

    def _serve(self) -> None:
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            buf = bytearray()
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
            if buf:
                self.requests.append(json.loads(buf))
            try:
                conn.sendall(self.reply)
            except OSError:
                pass


def test_install_wires_recall_hook_to_installed_script_matching_template(
    installed_codex: _InstalledCodex,
) -> None:
    script = _recall_script_path(installed_codex)
    assert script.is_absolute()
    assert script.parent == installed_codex.codex_home / "hooks"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    template = resources.files("iai_mcp") / "_deploy" / "hooks" / _HOOK_NAME
    assert script.read_bytes() == template.read_bytes()

    renderer = script.with_name(_RECALL_RENDER_HELPER)
    assert renderer.is_file()
    renderer_template = (
        resources.files("iai_mcp") / "_deploy" / "hooks" / _RECALL_RENDER_HELPER
    )
    assert renderer.read_bytes() == renderer_template.read_bytes()


def test_installed_hook_emits_working_tier_and_foresight_together(
    installed_codex: _InstalledCodex,
) -> None:
    sid = "codex-sess-coexist"
    (installed_codex.store / f".working-tier.{sid}.cached.md").write_text(
        "goal: ship the recall hardening\nnext action: land the adapted tests\n",
        encoding="utf-8",
    )
    (installed_codex.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: the agent will re-run the release gate next\n",
        encoding="utf-8",
    )
    result = _run_installed_hook(
        installed_codex, {"session_id": sid, "prompt": "status?"}
    )
    out = result.stdout
    assert out.count(b"<iai-mcp-working-tier>") == 1
    assert b"land the adapted tests" in out
    assert out.count(b"<iai-mcp-foresight>") == 1
    assert b"release gate" in out
    assert out.count(b"<iai-mcp-live-state>") == 0, (
        "emit_live_state must stay suppressed once emit_working_tier already "
        "carried this turn's goal/next-action through the installed script"
    )


def test_installed_hook_emits_live_state_fallback_and_agent_registry(
    installed_codex: _InstalledCodex,
) -> None:
    # No working-tier snapshot for this session -> emit_live_state and
    # emit_working_tier stay silent; the eager continuity cache carries both
    # the live-state and agent-registry blocks instead.
    (installed_codex.store / ".session-continuity.cached.md").write_text(
        "<iai-mcp-live-state>\n"
        "goal: recover the pending agent\n"
        "next action: resume the running task\n"
        "</iai-mcp-live-state>\n"
        "<iai-mcp-agent-registry>\n"
        "- executor: running deploy checklist\n"
        "</iai-mcp-agent-registry>\n",
        encoding="utf-8",
    )
    result = _run_installed_hook(
        installed_codex, {"session_id": "codex-sess-fresh", "prompt": "status?"}
    )
    out = result.stdout
    assert b"<iai-mcp-working-tier>" not in out
    assert out.count(b"<iai-mcp-live-state>") == 1
    assert b"resume the running task" in out
    assert out.count(b"<iai-mcp-agent-registry>") == 1
    assert b"executor: running deploy checklist" in out


def test_installed_hook_emits_directives_without_session_id(
    installed_codex: _InstalledCodex,
) -> None:
    (installed_codex.store / ".directives.cached.md").write_text(
        "always confirm before a destructive git operation\n", encoding="utf-8",
    )
    result = _run_installed_hook(installed_codex, {"prompt": "status?"})
    out = result.stdout
    assert out.count(b"<iai-mcp-directives>") == 1
    assert b"destructive git operation" in out
    assert b"<iai-mcp-working-tier>" not in out


def test_installed_socket_recall_uses_deployed_renderer_companion(
    installed_codex: _InstalledCodex,
) -> None:
    reply = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "result": {"hits": [{"literal_surface": "alice waters the garden"}]},
    }).encode("utf-8") + b"\n"
    with _RecallPeer(installed_codex.socket, reply) as peer:
        result = _run_installed_hook(
            installed_codex, {"prompt": "garden status"},
            IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
        )
        assert peer.requests == [{
            "jsonrpc": "2.0", "id": 1, "method": "memory_recall",
            "params": {"cue": "garden status", "limit": 3},
        }]
    assert result.stdout == (
        b"<iai-mcp-recall>\n- alice waters the garden\n</iai-mcp-recall>\n"
    )


def test_installed_socket_recall_degrades_when_renderer_missing(
    installed_codex: _InstalledCodex,
) -> None:
    # A prior installer regression dropped the renderer companion on deploy;
    # deleting it here reproduces that failure mode and proves the socket
    # path still degrades to a silent no-op instead of a crash.
    script = _recall_script_path(installed_codex)
    script.with_name(_RECALL_RENDER_HELPER).unlink()
    reply = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "result": {"hits": [{"literal_surface": "alice waters the garden"}]},
    }).encode("utf-8") + b"\n"
    with _RecallPeer(installed_codex.socket, reply) as peer:
        result = _run_installed_hook(
            installed_codex, {"prompt": "garden status"},
            IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
        )
        assert peer.requests, "the connection must still be attempted"
    assert result.stdout == b""
    assert result.stderr == b""


def test_installed_socket_recall_respects_explicit_disable(
    installed_codex: _InstalledCodex,
) -> None:
    reply = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "result": {"hits": [{"literal_surface": "should never be requested"}]},
    }).encode("utf-8") + b"\n"
    with _RecallPeer(installed_codex.socket, reply) as peer:
        result = _run_installed_hook(
            installed_codex, {"prompt": "garden status"},
            IAI_MCP_PER_TURN_SOCKET_ACCEL="0",
        )
        assert peer.requests == []
    assert result.stdout == b""


def test_reinstall_repairs_installed_script_and_hook_still_runs(
    installed_codex: _InstalledCodex,
) -> None:
    hooks_json = installed_codex.codex_home / "hooks.json"
    data = json.loads(hooks_json.read_text())
    data["hooks"].setdefault("Stop", []).append(
        {"hooks": [{"type": "command", "command": "echo foreign-owner-hook"}]}
    )
    hooks_json.write_text(json.dumps(data))

    script = _recall_script_path(installed_codex)
    script.unlink()
    assert not script.exists()

    assert install_codex_hooks() == 0
    assert script.exists()

    data2 = json.loads(hooks_json.read_text())
    stop_cmds = [
        h["command"] for e in data2["hooks"]["Stop"] for h in e["hooks"]
    ]
    assert "echo foreign-owner-hook" in stop_cmds

    (installed_codex.store / ".directives.cached.md").write_text(
        "repaired install still injects\n", encoding="utf-8",
    )
    result = _run_installed_hook(installed_codex, {"prompt": "status?"})
    assert b"repaired install still injects" in result.stdout


def test_uninstall_removes_installed_script_and_status_reports_not_registered(
    installed_codex: _InstalledCodex, capsys: pytest.CaptureFixture[str],
) -> None:
    script = _recall_script_path(installed_codex)
    renderer = script.with_name(_RECALL_RENDER_HELPER)
    assert script.exists() and renderer.exists()

    assert uninstall_codex_hooks() == 0
    assert not script.exists()
    assert not renderer.exists()

    capsys.readouterr()
    assert status_codex_hooks() == 1
    out = capsys.readouterr().out
    assert "NOT REGISTERED" in out
