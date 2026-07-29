"""The perf-tripwire retry contract: one assertion miss is absorbed on a
fresh store, two consecutive misses stay red, and non-assertion errors
propagate immediately — the helper must never convert a crash into a
retry.
"""
from __future__ import annotations

import pytest

from tests.conftest_shared import retry_once_on_assertion


def test_single_miss_passes_on_retry(tmp_path):
    calls = []

    @retry_once_on_assertion
    def probe(base):
        calls.append(base.name)
        if len(calls) == 1:
            raise AssertionError("jitter")

    probe(tmp_path)
    assert calls == ["attempt1", "attempt2"], (
        "each attempt must run in its own fresh subdirectory"
    )


def test_double_miss_stays_red(tmp_path):
    @retry_once_on_assertion
    def probe(base):
        raise AssertionError("real regression")

    with pytest.raises(AssertionError, match="real regression"):
        probe(tmp_path)


def test_non_assertion_errors_propagate_without_retry(tmp_path):
    calls = []

    @retry_once_on_assertion
    def probe(base):
        calls.append(1)
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        probe(tmp_path)
    assert calls == [1], "a crash must not be retried"


def test_extra_fixture_args_pass_through(tmp_path):
    @retry_once_on_assertion
    def probe(base, marker, kw=None):
        assert marker == "x" and kw == "y"
        return "ok"

    assert probe(tmp_path, "x", kw="y") == "ok"
