"""Coverage for the reconciled `iai-mcp tools` operator CLI subcommands:
search-symbol, search-xref, session-search, and stats.

Each test drives the `cmd_tools_*` handler directly against isolated
temporary stores -- either by monkeypatching the underlying default-root
helper the handler's callee resolves against (real I/O, no home-directory
dependence) or, for session-search, via the session store's own
`IAI_TOOLS_ROOT` env override.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iai_mcp.study as study_glue
import iai_mcp.tools as tools_glue
import iai_mcp.tools.session as session_glue
from iai_mcp.cli._tools import (
    cmd_tools_index,
    cmd_tools_search_symbol,
    cmd_tools_search_xref,
    cmd_tools_session_search,
    cmd_tools_stats,
)
from iai_mcp.study import TOOLS_INDEX_DOT_DIR_ALLOWLIST
from iai_mcp.tools.session import capture_event
from iai_mcp.tools_cli import _cmd_index


def test_search_symbol_dispatches_and_prints_ranked_hits(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    symbol_root = tmp_path / "symbols"
    monkeypatch.setattr(tools_glue, "default_symbol_store_root", lambda: symbol_root)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "alpha.py"
    target.write_text("def zebracorn_finder():\n    return 1\n")
    tools_glue.index_symbols([str(target)], store_root=symbol_root)

    rc = cmd_tools_search_symbol(argparse.Namespace(query="zebracorn_finder", k=10))
    assert rc == 0

    out = capsys.readouterr().out
    assert "alpha.py" in out


def test_search_symbol_reports_no_hits_cleanly(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(tools_glue, "default_symbol_store_root", lambda: tmp_path / "empty-symbols")

    rc = cmd_tools_search_symbol(argparse.Namespace(query="absent_marker", k=10))
    assert rc == 0
    assert capsys.readouterr().out.strip() == "no hits"


def test_search_xref_dispatches_and_prints_base_hits(tmp_path: Path, monkeypatch, capsys) -> None:
    store_root = tmp_path / "tools"
    xref_root = tmp_path / "xref"
    monkeypatch.setattr(tools_glue, "default_store_root", lambda: store_root)
    monkeypatch.setattr(tools_glue, "default_xref_store_root", lambda: xref_root)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "beta.py"
    target.write_text("def zebracorn_marker():\n    return 1\n")
    tools_glue.index_paths([str(target)], store_root=store_root)
    tools_glue.index_xref([str(target)], store_root=xref_root)

    rc = cmd_tools_search_xref(argparse.Namespace(query="zebracorn_marker", k=10))
    assert rc == 0

    out = capsys.readouterr().out
    assert "beta.py" in out


def test_session_search_dispatches_and_prints_events(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "tools-root"))

    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/gamma.py",
            "session_id": "sess-op",
            "source_hook": "PostToolUse",
        }
    )

    rc = cmd_tools_session_search(
        argparse.Namespace(
            session_id="sess-op", substring=None, category=None, since=None, until=None
        )
    )
    assert rc == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["data"] == "src/gamma.py"
    assert event["session_id"] == "sess-op"


def test_session_search_empty_result_prints_no_events(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "tools-root"))

    rc = cmd_tools_session_search(
        argparse.Namespace(
            session_id="sess-missing", substring=None, category=None, since=None, until=None
        )
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "no events"


def test_stats_dispatches_and_prints_real_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    store_root = tmp_path / "tools"
    symbol_root = tmp_path / "symbols"
    xref_root = tmp_path / "xref"
    sessions_root = tmp_path / "tools-root" / "sessions"

    monkeypatch.setattr(tools_glue, "default_store_root", lambda: store_root)
    monkeypatch.setattr(tools_glue, "default_symbol_store_root", lambda: symbol_root)
    monkeypatch.setattr(tools_glue, "default_xref_store_root", lambda: xref_root)
    monkeypatch.setattr(session_glue, "default_session_store_root", lambda: sessions_root)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "delta.py"
    target.write_text("def marker():\n    return 1\n")
    tools_glue.index_paths([str(target)], store_root=store_root)

    capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/delta.py",
            "session_id": "sess-stats-op",
            "source_hook": "PostToolUse",
        },
        store_root=sessions_root,
    )

    rc = cmd_tools_stats(argparse.Namespace())
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["code_index"]["whole_file"]["segment_count"] == 1
    assert payload["sessions"]["session_count"] == 1
    assert payload["sessions"]["total_events"] == 1
    assert payload["executor"]["run_count"] == 0


def test_both_index_call_sites_pass_the_dot_dir_allowlist_constant(
    tmp_path: Path, monkeypatch
) -> None:
    """The console script's _cmd_index and the operator CLI's cmd_tools_index
    are two duplicated implementations, patched together by design -- this
    proves both actually pass the same shared allowlist constant, not just
    one of them."""
    recorded_allowlists: list = []

    def _spy_iter_study_files(path, **kwargs):
        recorded_allowlists.append(kwargs.get("dot_dir_allowlist"))
        return []

    monkeypatch.setattr(study_glue, "iter_study_files", _spy_iter_study_files)
    monkeypatch.setattr(tools_glue, "index_all", lambda paths: 0)
    monkeypatch.setattr(tools_glue, "index_paths", lambda paths: 0)

    target = tmp_path / "target"
    target.mkdir()
    ns = argparse.Namespace(path=str(target), ecosystem=None, package=None, project_dir=None)

    rc_console = _cmd_index(ns)
    rc_operator = cmd_tools_index(ns)

    assert rc_console == 0
    assert rc_operator == 0
    assert recorded_allowlists == [TOOLS_INDEX_DOT_DIR_ALLOWLIST, TOOLS_INDEX_DOT_DIR_ALLOWLIST]
