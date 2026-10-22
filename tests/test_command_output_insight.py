"""Command-run insight aggregate: actionable, measurement-scoped, never hollow.

Hermetic isolation, binding for every test here: IAI_TOOLS_ROOT points at a
tmp dir (monkeypatch, read at call time); the executor daemon and socket
never touch the user's real ~/.iai-tools store. Corpus runs go through the
real exec_cli.run_and_render path -- never a hand-authored meta.json dict,
which can silently diverge from the daemon's own serialization (field
names, started_at units).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_MAX_CONNECT_RETRIES = 3


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


@pytest.fixture
def exec_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so the
    # socket lives in its own short-path directory directly under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", "1.0")
    try:
        yield {"root": root}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _run_with_retry(run_and_render, *args, **kwargs):
    """Bounded retry around a known intermittent executor-daemon connect
    wobble (tracked separately) -- not a defect of this test's own
    aggregate wiring.
    """
    last_error: "Exception | None" = None
    for attempt in range(_MAX_CONNECT_RETRIES):
        try:
            return run_and_render(*args, **kwargs)
        except ConnectionRefusedError as e:
            last_error = e
            time.sleep(0.2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def _seed_mixed_corpus(exec_env: dict) -> dict:
    """Runs a small, mixed-exit-code corpus through the real executor path
    and returns the results keyed by a short label, for tests to pick out
    the distinctive failing run by its own returned run_id.
    """
    import sys

    from iai_mcp.tools.exec_cli import run_and_render

    results = {}
    results["ok_echo"] = _run_with_retry(run_and_render, "echo", ["insight-marker"])
    results["plain_false"] = _run_with_retry(run_and_render, "false", [])
    results["distinctive_failure"] = _run_with_retry(
        run_and_render, sys.executable, ["-c", "import sys; sys.exit(3)"]
    )
    results["ok_true"] = _run_with_retry(run_and_render, "true", [])
    return results


def test_insight_surfaces_failing_command(exec_env: dict) -> None:
    from iai_mcp.tools.stats import gather_command_insight

    results = _seed_mixed_corpus(exec_env)
    artifacts_root = exec_env["root"] / "artifacts"

    report = gather_command_insight(artifacts_root=artifacts_root)

    failing_run_ids = {r["run_id"] for r in report["failing_runs"]}
    assert results["distinctive_failure"]["run_id"] in failing_run_ids
    assert results["plain_false"]["run_id"] in failing_run_ids
    assert results["ok_echo"]["run_id"] not in failing_run_ids
    assert results["ok_true"]["run_id"] not in failing_run_ids

    distinctive_entry = next(
        r for r in report["failing_runs"] if r["run_id"] == results["distinctive_failure"]["run_id"]
    )
    assert distinctive_entry["exit_code"] == 3
    assert "sys.exit(3)" in " ".join(distinctive_entry["args"])


def test_insight_actionable_fields(exec_env: dict) -> None:
    from iai_mcp.tools.stats import gather_command_insight

    results = _seed_mixed_corpus(exec_env)
    artifacts_root = exec_env["root"] / "artifacts"

    report = gather_command_insight(artifacts_root=artifacts_root)

    # Exactly the measured-actionable subset ships -- no bare count/frequency
    # field (measured and dropped; see the evidence file).
    assert set(report.keys()) == {"failing_runs", "most_recent_failure"}

    for entry in report["failing_runs"]:
        assert set(entry.keys()) == {"run_id", "cmd", "args", "exit_code", "started_at"}
        assert entry["exit_code"] not in (0, None)

    most_recent = report["most_recent_failure"]
    assert most_recent is not None
    assert most_recent["run_id"] == results["distinctive_failure"]["run_id"] or most_recent["run_id"] == results["plain_false"]["run_id"]
    assert most_recent == max(report["failing_runs"], key=lambda r: r["started_at"])


def test_insight_empty_history_is_safe(tmp_path: Path) -> None:
    from iai_mcp.tools.stats import gather_command_insight

    report = gather_command_insight(artifacts_root=tmp_path / "no-such-artifacts-dir")
    assert report == {"failing_runs": [], "most_recent_failure": None}


def test_insight_artifacts_root_override(exec_env: dict, tmp_path: Path) -> None:
    from iai_mcp.tools.stats import gather_command_insight

    _seed_mixed_corpus(exec_env)

    other_root = tmp_path / "unrelated-empty-artifacts"
    report = gather_command_insight(artifacts_root=other_root)
    assert report == {"failing_runs": [], "most_recent_failure": None}

    real_root = exec_env["root"] / "artifacts"
    report = gather_command_insight(artifacts_root=real_root)
    assert report["failing_runs"], "the explicit override must read the real corpus, not a default derivation"
