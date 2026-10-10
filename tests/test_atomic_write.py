"""Shared atomic text-file write used by the host config installers."""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.cli._atomic_write import _atomic_write_text


def test_write_creates_file_with_content(tmp_path):
    target = tmp_path / "hooks.json"
    _atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'


def test_no_tmp_sibling_left_behind_on_success(tmp_path):
    target = tmp_path / "hooks.json"
    _atomic_write_text(target, "content")
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"hooks.json"}


def test_overwrite_replaces_content_atomically(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("old content", encoding="utf-8")
    _atomic_write_text(target, "new content")
    assert target.read_text(encoding="utf-8") == "new content"
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"config.toml"}


def test_preserves_existing_file_permissions(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    _atomic_write_text(target, "new")
    assert (target.stat().st_mode & 0o777) == 0o640


def test_tmp_file_removed_on_write_failure(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"

    def _boom(*args, **kwargs):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr("os.fsync", _boom)
    with pytest.raises(OSError):
        _atomic_write_text(target, "content")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
