"""Unit tests for the Windows auth-token self-heal (iai_mcp._ipc).

Regression coverage for the incident where .daemon.token vanished out-of-band
while the daemon kept running: the TCP port stayed up but every NEW client
failed the auth handshake (looked like "daemon not running") until a manual
restart. reassert_token_if_missing() lets the daemon restore the file from its
in-memory token on the next tick.
"""
from __future__ import annotations

import pytest

from iai_mcp import _ipc


@pytest.fixture
def token_env(tmp_path, monkeypatch):
    """Point the token file at a hermetic tmp path and neutralize the Windows
    ACL call (icacls) so the logic is testable on any platform."""
    endpoint = tmp_path / "daemon"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(endpoint))
    monkeypatch.setattr(_ipc, "_restrict_token_file", lambda _p: None)
    return _ipc._token_file_path()


def test_reassert_recreates_missing_token(token_env, monkeypatch):
    monkeypatch.setattr(_ipc, "IS_WINDOWS", True)
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", "deadbeef" * 8)
    assert not token_env.exists()

    assert _ipc.reassert_token_if_missing() is True
    assert token_env.exists()
    assert token_env.read_text(encoding="utf-8").strip() == "deadbeef" * 8


def test_reassert_is_noop_when_token_present(token_env, monkeypatch):
    monkeypatch.setattr(_ipc, "IS_WINDOWS", True)
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", "abc123" * 8)
    token_env.write_text("existing-token", encoding="utf-8")

    assert _ipc.reassert_token_if_missing() is False
    # Must not clobber a token already on disk.
    assert token_env.read_text(encoding="utf-8") == "existing-token"


def test_reassert_noop_without_in_memory_token(token_env, monkeypatch):
    # A client process never generated a token: it must NOT create one.
    monkeypatch.setattr(_ipc, "IS_WINDOWS", True)
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", None)

    assert _ipc.reassert_token_if_missing() is False
    assert not token_env.exists()


def test_reassert_noop_on_posix(token_env, monkeypatch):
    monkeypatch.setattr(_ipc, "IS_WINDOWS", False)
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", "ffff" * 16)

    assert _ipc.reassert_token_if_missing() is False
    assert not token_env.exists()


def test_generate_token_populates_in_memory_copy(token_env, monkeypatch):
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", None)
    token = _ipc._generate_token()
    assert _ipc._CURRENT_TOKEN == token
    assert token_env.read_text(encoding="utf-8").strip() == token


def test_remove_token_clears_in_memory_copy(token_env, monkeypatch):
    monkeypatch.setattr(_ipc, "IS_WINDOWS", True)
    monkeypatch.setattr(_ipc, "_CURRENT_TOKEN", "cafe" * 16)
    token_env.write_text("cafe" * 16, encoding="utf-8")

    _ipc._remove_token_file()

    # In-memory copy cleared so a later reassert cannot resurrect a token the
    # daemon deliberately tore down.
    assert _ipc._CURRENT_TOKEN is None
    assert _ipc.reassert_token_if_missing() is False
    assert not token_env.exists()
