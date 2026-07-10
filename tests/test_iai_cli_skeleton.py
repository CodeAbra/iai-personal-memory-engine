from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import pytest


def test_iai_cli_module_importable():
    from iai_mcp import iai_cli

    assert hasattr(iai_cli, "main")
    assert callable(iai_cli.main)
    assert hasattr(iai_cli, "__version__")


def test_iai_no_args_prints_logo_and_help(monkeypatch):
    from iai_mcp.iai_cli import main

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([])
    output = buf.getvalue()

    assert rc == 0
    assert "iai" in output.lower() or "█" in output
    assert "recall" in output
    assert "capture" in output


def test_iai_no_color_env_strips_ansi(monkeypatch):
    from iai_mcp.iai_cli import main

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        main([])
    output = buf.getvalue()
    assert "\x1b[" not in output, f"NO_COLOR violated: ANSI escape in output: {output!r}"


def test_iai_non_tty_stdout_strips_ansi(monkeypatch):
    from iai_mcp.iai_cli import main

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        main([])
    output = buf.getvalue()
    assert "\x1b[" not in output


def test_iai_version_flag():
    from iai_mcp.iai_cli import main, __version__

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as ei:
            main(["--version"])
    assert ei.value.code == 0
    assert __version__ in buf.getvalue()


def test_iai_unknown_subcommand_errors(monkeypatch):
    from iai_mcp.iai_cli import main

    err = io.StringIO()
    with redirect_stderr(err):
        with pytest.raises(SystemExit) as ei:
            main(["nope-not-a-real-command"])
    assert ei.value.code != 0


def test_iai_recall_subcommand_registered():
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    actions = {a.dest for a in parser._actions}
    assert "cmd" in actions


def test_iai_capture_subcommand_registered():
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["capture", "hello world"])
    assert args.cmd == "capture"
    assert args.text == "hello world"
