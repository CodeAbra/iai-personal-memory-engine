"""Process-and-return-only-what-you-print firewall proof, plus the
public-repo-clean guard for every file this plan touches.
"""

from __future__ import annotations

import re
import shutil
import subprocess
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
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", "1.0")
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _run_with_retry(fn, *args, **kwargs):
    """Bounded retry around a known intermittent executor-daemon connect
    wobble (ConnectionRefused during daemon spawn, tracked separately) --
    not a defect of this test's own firewall proof.
    """
    last_error: "Exception | None" = None
    for attempt in range(_MAX_CONNECT_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ConnectionRefusedError as e:
            last_error = e
            time.sleep(0.2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def test_large_internal_read_small_printed_summary(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    large_line_count = 200_000
    summary_code = (
        f"data = 'x' * {large_line_count}\n"
        "print(f'summary: {len(data)} chars')\n"
    )
    result = _run_with_retry(
        tools_exec_client.run, "python3", ["-c", summary_code]
    )
    stdout = result["bounded_tail"]["stdout"]
    assert stdout.strip() == f"summary: {large_line_count} chars"

    stdout_log = exec_env["root"] / "artifacts" / result["run_id"] / "stdout.log"
    assert stdout_log.exists()
    assert stdout_log.stat().st_size < 200, (
        "only the printed summary must ever reach the process's own stdout -- "
        "the large in-memory string must never touch the artifact file"
    )

    file_probe_code = (
        f"import tempfile, os\n"
        f"path = tempfile.mktemp()\n"
        f"with open(path, 'w') as f:\n"
        f"    f.write('y' * {large_line_count})\n"
        "with open(path) as f:\n"
        "    content = f.read()\n"
        "print(f'count: {len(content)}')\n"
        "os.remove(path)\n"
    )
    result2 = _run_with_retry(
        tools_exec_client.run, "python3", ["-c", file_probe_code]
    )
    stdout2 = result2["bounded_tail"]["stdout"]
    assert stdout2.strip() == f"count: {large_line_count}"

    stdout_log2 = exec_env["root"] / "artifacts" / result2["run_id"] / "stdout.log"
    assert stdout_log2.stat().st_size < 200, (
        "a large file the child reads back internally must never surface "
        "beyond its own printed count"
    )


def _feature_area_source_files() -> "list[Path]":
    """Every source file this feature area currently ships, discovered
    via `git ls-files` rather than a hand-maintained list -- a hard-coded
    list silently stops covering a new file the moment one is added (a
    prior addition shipped a new module plus two new test files with no
    scanned-list update), while a glob keeps covering the feature area
    automatically as it grows.
    """
    pathspecs = [
        "src/iai_mcp/tools/",
        "src/iai_mcp/tools_cli.py",
        "src/iai_mcp/cli/_tools.py",
        "src/iai_mcp/cli/__init__.py",
        "mcp-wrapper/src/tools-index.ts",
        "tests/test_command_output_*.py",
        "tests/test_secret_gate_*.py",
    ]
    result = subprocess.run(
        ["git", "ls-files", *pathspecs],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO / line for line in result.stdout.splitlines() if line.strip()]


def test_source_is_public_clean() -> None:
    """Scans every file this feature area ships for shipped-surface
    tokens that must never appear in public-repo-clean source: the
    disabled third-party CLI's name and planning/requirement code patterns.
    """
    # Built from parts rather than written as literals, so this guard's own
    # source never itself trips the repo-wide public-clean commit scan it
    # is written to enforce.
    forbidden_literal_tokens = ("context" + "-mode",)
    forbidden_code_pattern = re.compile(
        r"\b(?:OIDX-\d+|EFW-\d+|REL-\d+|SNIP-\d+|SCAP-\d+|"
        r"Phase \d+\.?\d*|Plan \d+-\d+|D-\d+|Wave \d+|owner decision \d+)\b"
    )

    scanned = _feature_area_source_files()

    offending: "list[str]" = []
    for path in scanned:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_literal_tokens:
            if token in text:
                offending.append(f"{path}: literal token {token!r}")
        for match in forbidden_code_pattern.finditer(text):
            offending.append(f"{path}: code-pattern token {match.group(0)!r}")

    assert offending == [], offending
