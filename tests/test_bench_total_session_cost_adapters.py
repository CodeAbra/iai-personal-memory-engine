from __future__ import annotations

import json
import subprocess
from unittest import mock


from bench.total_session_cost import (
    _SCRIPT,
    _run_claude_mem_adapter,
    _run_mempalace_adapter,
    main,
    run_total_session_cost,
)


def _fixed_counter(text: str) -> int:
    return max(1, len(text.split()))


def test_mempalace_adapter_signature():
    result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    assert result is None or isinstance(result, int)


def test_mempalace_adapter_absent_cli_returns_none(capsys):
    with mock.patch("bench.total_session_cost.shutil.which", return_value=None):
        result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    assert result is None
    err = capsys.readouterr().err
    assert "bench_adapter_unavailable" in err
    assert "mempalace" in err


def test_mempalace_adapter_live_run_sums_stdout_tokens():

    def fake_which(name):
        return "/fake/bin/mempalace" if name == "mempalace" else None

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="one two three",
            stderr="",
        )

    with mock.patch("bench.total_session_cost.shutil.which", side_effect=fake_which), \
         mock.patch("bench.total_session_cost.subprocess.run", side_effect=fake_run):
        result = _run_mempalace_adapter(_SCRIPT, _fixed_counter)
    assert result == 3 * len(_SCRIPT)


def test_measure_mempalace_flag_populates_refs(monkeypatch, capsys):

    def fake_which(name):
        return "/fake/bin/mempalace" if name == "mempalace" else None

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="hello world",
            stderr="",
        )

    with mock.patch("bench.total_session_cost.shutil.which", side_effect=fake_which), \
         mock.patch("bench.total_session_cost.subprocess.run", side_effect=fake_run):
        rc = main(["--wake-depth", "minimal", "--measure-mempalace"])

    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert "mempalace_measured" in result["refs"]
    assert isinstance(result["refs"]["mempalace_measured"], int)
    assert result["refs"]["mempalace_measured"] > 0


def test_claude_mem_adapter_mirrors_mempalace_shape(capsys):
    with mock.patch("bench.total_session_cost.shutil.which", return_value=None):
        result = _run_claude_mem_adapter(_SCRIPT, _fixed_counter)
    assert result is None
    err = capsys.readouterr().err
    assert "bench_adapter_unavailable" in err
    assert "claude-mem" in err


def test_live_measurement_wins_over_manual_ref():

    with mock.patch("bench.total_session_cost.shutil.which",
                    side_effect=lambda n: "/fake/bin/mempalace" if n == "mempalace" else None), \
         mock.patch("bench.total_session_cost.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        args=[], returncode=0,
                        stdout="token " * 5000,
                        stderr="",
                    )):
        result = run_total_session_cost(
            wake_depth="minimal",
            mempalace_ref=10,
            measure_mempalace=True,
            count_tokens_fn=_fixed_counter,
        )
    assert "mempalace_measured" in result["refs"]
    assert "mempalace_manual" in result["refs"]
    assert result["refs"]["mempalace_manual"] == 10
    assert result["passed"] is True
