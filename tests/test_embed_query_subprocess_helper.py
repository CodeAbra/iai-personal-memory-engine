"""Unit coverage for `_embed_query_via_subprocess`'s error paths: the
exact crashes it was consolidated to fix (unhandled IndexError on empty
output, unhandled TimeoutExpired, and an unparseable last line) must now
all raise the single well-defined `EmbedWorkerError` instead.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from iai_mcp.tools import EmbedWorkerError, _embed_query_via_subprocess


def test_empty_stdout_raises_embed_worker_error_not_index_error(monkeypatch) -> None:
    completed = types.SimpleNamespace(returncode=0, stdout="\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: completed)

    with pytest.raises(EmbedWorkerError):
        _embed_query_via_subprocess("some query")


def test_timeout_expired_raises_embed_worker_error(monkeypatch) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="embed_worker", timeout=120)

    monkeypatch.setattr("subprocess.run", _raise_timeout)

    with pytest.raises(EmbedWorkerError, match="timed out"):
        _embed_query_via_subprocess("some query")


def test_unparseable_output_raises_embed_worker_error_not_json_decode_error(monkeypatch) -> None:
    completed = types.SimpleNamespace(returncode=0, stdout="not json\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: completed)

    with pytest.raises(EmbedWorkerError):
        _embed_query_via_subprocess("some query")


def test_nonzero_exit_raises_embed_worker_error_with_stderr(monkeypatch) -> None:
    completed = types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: completed)

    with pytest.raises(EmbedWorkerError, match="boom"):
        _embed_query_via_subprocess("some query")


def test_well_formed_output_still_parses(monkeypatch) -> None:
    completed = types.SimpleNamespace(returncode=0, stdout="[0.1, 0.2, 0.3]\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: completed)

    assert _embed_query_via_subprocess("some query") == [0.1, 0.2, 0.3]
