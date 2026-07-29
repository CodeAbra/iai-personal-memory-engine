"""Codex host wiring: the same four hook scripts registered in
~/.codex/hooks.json, and the rollout transcript format consumed by the
shared parser. Codex's hook stdin carries the same field names the
scripts already read, so nothing here adapts payloads — only
registration and transcript format differ from the Claude host.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.capture import _parse_transcript_line
from iai_mcp.cli._codex_hooks import (
    _HOOK_SCRIPTS,
    install_codex_hooks,
    status_codex_hooks,
    uninstall_codex_hooks,
)


@pytest.fixture()
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex-home"
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(home))
    return home


class TestInstaller:
    def test_install_writes_scripts_and_wires_events(self, codex_home, capsys):
        assert install_codex_hooks() == 0
        for name in _HOOK_SCRIPTS:
            dst = codex_home / "hooks" / name
            assert dst.exists(), name
            assert dst.stat().st_mode & 0o100, f"{name} must be executable"

        data = json.loads((codex_home / "hooks.json").read_text())
        events = data["hooks"]
        assert any(
            "iai-mcp-session-capture.sh" in h["command"]
            for e in events["Stop"] for h in e["hooks"]
        )
        submit_cmds = [
            h["command"] for e in events["UserPromptSubmit"] for h in e["hooks"]
        ]
        assert any("iai-mcp-turn-capture.sh" in c for c in submit_cmds)
        assert any("iai-mcp-per-turn-recall.sh" in c for c in submit_cmds)
        ss = events["SessionStart"]
        assert any(
            "iai-mcp-session-recall.sh" in h["command"]
            for e in ss for h in e["hooks"]
        )
        assert ss[0]["matcher"] == "startup|resume|clear|compact"

    def test_install_is_idempotent(self, codex_home):
        install_codex_hooks()
        first = json.loads((codex_home / "hooks.json").read_text())
        install_codex_hooks()
        second = json.loads((codex_home / "hooks.json").read_text())
        assert first == second

    def test_install_preserves_foreign_hooks(self, codex_home):
        (codex_home).mkdir(parents=True)
        foreign = {
            "hooks": {
                "Stop": [
                    {"hooks": [{"type": "command", "command": "echo mine"}]}
                ]
            }
        }
        (codex_home / "hooks.json").write_text(json.dumps(foreign))
        install_codex_hooks()
        data = json.loads((codex_home / "hooks.json").read_text())
        cmds = [h["command"] for e in data["hooks"]["Stop"] for h in e["hooks"]]
        assert "echo mine" in cmds
        assert any("iai-mcp-session-capture.sh" in c for c in cmds)

    def test_uninstall_removes_only_ours(self, codex_home):
        (codex_home).mkdir(parents=True)
        (codex_home / "hooks.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{"hooks": [{"type": "command", "command": "echo mine"}]}]
            }
        }))
        install_codex_hooks()
        assert uninstall_codex_hooks() == 0
        for name in _HOOK_SCRIPTS:
            assert not (codex_home / "hooks" / name).exists()
        data = json.loads((codex_home / "hooks.json").read_text())
        cmds = [h["command"] for e in data["hooks"]["Stop"] for h in e["hooks"]]
        assert cmds == ["echo mine"]
        assert "UserPromptSubmit" not in data["hooks"]

    def test_status_reflects_wiring(self, codex_home):
        assert status_codex_hooks() == 1
        install_codex_hooks()
        assert status_codex_hooks() == 0
        uninstall_codex_hooks()
        assert status_codex_hooks() == 1


def _rollout_line(role: str, text: str, kind: str = "input_text") -> str:
    return json.dumps({
        "timestamp": "2026-07-28T21:00:00.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": kind, "text": text}],
        },
    })


class TestRolloutParser:
    def test_user_and_assistant_messages_parse(self):
        role, text, _uuid, ts = _parse_transcript_line(
            _rollout_line("user", "please rename the deploy target")
        )
        assert (role, text) == ("user", "please rename the deploy target")
        assert ts == "2026-07-28T21:00:00.000Z"

        role, text, _u, _t = _parse_transcript_line(
            _rollout_line("assistant", "renamed and verified", kind="output_text")
        )
        assert (role, text) == ("assistant", "renamed and verified")

    def test_session_meta_and_event_msg_are_skipped(self):
        meta = json.dumps({
            "timestamp": "t", "type": "session_meta",
            "payload": {"id": "abc", "cwd": "/x"},
        })
        assert _parse_transcript_line(meta) is None
        # event_msg duplicates every user turn; consuming it would
        # double-capture.
        dup = json.dumps({
            "timestamp": "t", "type": "event_msg",
            "payload": {"type": "user_message", "message": "hello"},
        })
        assert _parse_transcript_line(dup) is None

    def test_environment_context_wrapper_is_filtered(self):
        line = _rollout_line(
            "user", "<environment_context>\n  <cwd>/x</cwd>\n</environment_context>"
        )
        assert _parse_transcript_line(line) is None

    def test_schema_drift_tolerated(self):
        # Unknown content kinds still parse as long as a text field exists;
        # a message with no text yields None instead of raising.
        odd = json.dumps({
            "type": "response_item",
            "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "future_kind", "text": "still readable"}],
            },
        })
        parsed = _parse_transcript_line(odd)
        assert parsed is not None and parsed[1] == "still readable"

        empty = json.dumps({
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": []},
        })
        assert _parse_transcript_line(empty) is None

    def test_claude_format_unchanged(self):
        claude = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "plain claude line"},
            "uuid": "u1", "timestamp": "t1",
        })
        assert _parse_transcript_line(claude) == (
            "user", "plain claude line", "u1", "t1",
        )
