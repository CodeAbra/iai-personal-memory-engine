from __future__ import annotations

import fcntl

import pytest

from iai_mcp.doctor import check_c_lock_healthy


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo_dir = tmp_path / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    return hippo_dir


def test_absent_lock_is_healthy(tmp_store):
    lock_path = tmp_store / ".lock"
    assert not lock_path.exists()

    result = check_c_lock_healthy()

    assert result.passed is True
    assert result.name == "(c) lock file healthy"
    assert "not yet initialized" in result.detail


def test_idle_acquirable_lock_is_healthy(tmp_store):
    lock_path = tmp_store / ".lock"
    lock_path.write_bytes(b"")
    assert lock_path.exists()

    result = check_c_lock_healthy()

    assert result.passed is True
    assert result.name == "(c) lock file healthy"
    assert "acquirable" in result.detail


def test_held_lock_is_healthy(tmp_store):
    lock_path = tmp_store / ".lock"
    lock_path.write_bytes(b"")

    held_fd = None
    try:
        with open(lock_path, "r") as held:
            held_fd = held.fileno()
            fcntl.flock(held_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            result = check_c_lock_healthy()

            assert result.passed is True
            assert result.name == "(c) lock file healthy"
            assert "held" in result.detail
            fcntl.flock(held_fd, fcntl.LOCK_UN)
    finally:
        pass


def test_probe_does_not_create_lock_file(tmp_store):
    lock_path = tmp_store / ".lock"
    assert not lock_path.exists()

    check_c_lock_healthy()

    assert not lock_path.exists(), "probe must not fabricate a lock file"
