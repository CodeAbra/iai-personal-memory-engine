"""Console-script coverage for `iai-tools session-search` and `iai-tools stats`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from iai_mcp import tools_cli
from iai_mcp.tools.session import capture_event

REPO = Path(__file__).resolve().parent.parent


def test_session_search_prints_filtered_events(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "tools-root"))

    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/alpha.py",
            "session_id": "sess-cli",
            "source_hook": "PostToolUse",
        }
    )
    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/beta.py",
            "session_id": "sess-cli",
            "source_hook": "PostToolUse",
        }
    )

    rc = tools_cli.main(["session-search", "sess-cli", "--substring", "alpha"])
    assert rc == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["data"] == "src/alpha.py"
    assert event["session_id"] == "sess-cli"


def test_session_search_empty_result_prints_no_events(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "tools-root"))

    rc = tools_cli.main(["session-search", "sess-missing"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "no events"


def test_session_search_category_filter_scopes_to_one_category(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "tools-root"))

    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/gamma.py",
            "session_id": "sess-cat",
            "source_hook": "PostToolUse",
        }
    )
    capture_event(
        {
            "tool_name": "Bash",
            "command": "false",
            "is_error": True,
            "session_id": "sess-cat",
            "source_hook": "PostToolUse",
        }
    )

    rc = tools_cli.main(["session-search", "sess-cat", "--category", "error"])
    assert rc == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["category"] == "error"


def _run_tools_cli(args: list[str], *, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_stats_reports_real_counts_and_stays_isolated_under_home_override(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def alpha():\n    return 1\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)
    env["IAI_TOOLS_ROOT"] = str(tmp_path / "tools-root")

    index_proc = _run_tools_cli(["index", str(corpus)], env=env)
    assert index_proc.returncode == 0, index_proc.stderr

    capture_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from iai_mcp.tools.session import capture_event;"
            "capture_event({'tool_name': 'Edit', 'file_path': 'src/x.py', "
            "'session_id': 'sess-stats', 'source_hook': 'PostToolUse'})",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert capture_proc.returncode == 0, capture_proc.stderr

    stats_proc = _run_tools_cli(["stats"], env=env)
    assert stats_proc.returncode == 0, stats_proc.stderr
    payload = json.loads(stats_proc.stdout.strip())

    assert payload["code_index"]["whole_file"]["segment_count"] == 1
    assert payload["code_index"]["symbol"]["segment_count"] == 1
    assert payload["code_index"]["xref"]["segment_count"] == 1
    assert payload["sessions"]["session_count"] == 1
    assert payload["sessions"]["total_events"] == 1
    assert payload["executor"]["run_count"] == 0

    # Isolation post-condition: everything landed under the overridden HOME
    # and IAI_TOOLS_ROOT, never under the real invoking user's directories.
    assert (tmp_path / ".iai-mcp" / "tools").is_dir()
    assert (tmp_path / "tools-root" / "sessions" / "sess-stats.jsonl").exists()
