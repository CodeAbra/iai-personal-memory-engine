"""Confirms the code-index CLI's discovery call sites pass an explicit
max_files=sys.maxsize into iter_study_files, never relying on its
bounded 500-file default -- a caller that omits this silently indexes
only a truncated prefix of the real corpus with no error raised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch


def test_cli_tools_index_passes_explicit_max_files(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp.cli._tools import cmd_tools_index

    seen: dict[str, object] = {}

    def fake_iter_study_files(root, *, max_files=None, **kwargs):
        seen["root"] = root
        seen["max_files"] = max_files
        return []

    monkeypatch.setattr("iai_mcp.study.iter_study_files", fake_iter_study_files)
    with patch("iai_mcp.tools.index_paths") as mock_index_paths:
        args = argparse.Namespace(path=str(tmp_path))
        cmd_tools_index(args)

    assert seen["max_files"] == sys.maxsize, (
        "cmd_tools_index must pass max_files=sys.maxsize -- the 500 default "
        "silently truncates the real corpus"
    )
    mock_index_paths.assert_called_once()


def test_tools_cli_index_passes_explicit_max_files(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp import tools_cli

    seen: dict[str, object] = {}

    def fake_iter_study_files(root, *, max_files=None, **kwargs):
        seen["root"] = root
        seen["max_files"] = max_files
        return []

    monkeypatch.setattr("iai_mcp.study.iter_study_files", fake_iter_study_files)
    with patch("iai_mcp.tools.index_all") as mock_index_all:
        args = argparse.Namespace(path=str(tmp_path))
        tools_cli._cmd_index(args)

    assert seen["max_files"] == sys.maxsize, (
        "iai-tools index must pass max_files=sys.maxsize -- the 500 default "
        "silently truncates the real corpus"
    )
    mock_index_all.assert_called_once()
