"""Hook<->daemon lockstep: an edited hook reports stale, a freshly installed
hook reports clean."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _per_turn_status_line(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("Per-turn installed:"):
            return line
    raise AssertionError(f"no 'Per-turn installed:' line in status output:\n{out}")


def test_hook_lockstep_detects_stale_install(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0
    capsys.readouterr()

    pt_dst = fake_home / ".claude" / "hooks" / "iai-mcp-per-turn-recall.sh"
    assert pt_dst.exists(), "install did not deploy the per-turn hook"
    pt_text = pt_dst.read_text(encoding="utf-8")
    assert "emit_live_state" in pt_text, (
        "the packaged per-turn hook template must carry the gate-free "
        "live-state emitter"
    )
    assert "emit_agent_registry" in pt_text, (
        "the packaged per-turn hook template must carry the session-agnostic "
        "running-agent registry emitter"
    )
    assert "emit_live_state_fallback" in pt_text, (
        "the packaged per-turn hook template must carry the phase/step "
        "fallback emitter that fires when the session-scoped snapshot is "
        "absent"
    )

    # A freshly installed hook matches the packaged template byte-for-byte.
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    clean_out = capsys.readouterr().out
    assert "STALE" not in _per_turn_status_line(clean_out), clean_out

    # Simulate an installed hook that predates a template edit -- the
    # hash-diff detector must flag it, proving an old hook can never
    # silently keep reading a cache shape the daemon no longer writes.
    original = pt_dst.read_bytes()
    pt_dst.write_bytes(original + b"\n# stale installed copy\n")

    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    stale_out = capsys.readouterr().out
    assert "STALE" in _per_turn_status_line(stale_out), stale_out

    # Reinstall restores lockstep.
    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0
    capsys.readouterr()

    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    fixed_out = capsys.readouterr().out
    assert "STALE" not in _per_turn_status_line(fixed_out), fixed_out
    assert pt_dst.read_bytes() == original, (
        "reinstall must restore the exact packaged-template bytes"
    )
