from __future__ import annotations

import io
import sys
import tomllib
from contextlib import redirect_stdout
from pathlib import Path

import pytest

EXPECTED_VERSION = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
)["project"]["version"]


def test_canonical_version_is_expected():
    import iai_mcp

    assert iai_mcp.__version__ == EXPECTED_VERSION


def test_installed_distribution_matches_project_version():
    import iai_mcp
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        installed = distribution("iai-mcp")
    except PackageNotFoundError:
        pytest.skip("source-tree test without an installed distribution")
    install_root = Path(installed.locate_file("")).resolve()
    if not Path(iai_mcp.__file__).resolve().is_relative_to(install_root):
        pytest.skip("source tree shadows a different installed distribution")
    assert installed.version == EXPECTED_VERSION


def test_cli_version_is_single_sourced():
    """The CLI must read the canonical version, not carry its own literal."""
    import iai_mcp
    from iai_mcp import iai_cli

    assert iai_cli.__version__ == iai_mcp.__version__


def test_version_flag_prints_semver():
    from iai_mcp.iai_cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
    output = buf.getvalue()

    assert exit_info.value.code == 0
    assert output.strip() == f"iai {EXPECTED_VERSION}"


def test_banner_shows_version(monkeypatch):
    from iai_mcp.iai_cli import _print_logo

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_logo()
    output = buf.getvalue()

    assert f"v{EXPECTED_VERSION}" in output


def test_no_args_invocation_shows_version(monkeypatch):
    from iai_mcp.iai_cli import main

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([])
    output = buf.getvalue()

    assert rc == 0
    assert f"v{EXPECTED_VERSION}" in output
