from __future__ import annotations

import contextlib
import io



def test_logo_has_no_mcp_columns(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    from iai_mcp.iai_cli import _print_logo

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_logo()
    output = buf.getvalue()

    assert "███╗   ███╗" not in output, (
        "M-column signature found — the `MCP` wordmark was not replaced in the logo"
    )
    assert "█" in output, "No box-drawing chars found — logo appears empty after edit"


def test_tagline_reads_iai_cli(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    from iai_mcp.iai_cli import _print_logo

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_logo()
    output = buf.getvalue()

    assert "iai-cli · terminal memory for your agent" in output, (
        f"Tagline not found in logo output. Got:\n{output!r}"
    )
    assert "iai-mcp ·" not in output, (
        "Old `iai-mcp ·` tagline still present in logo output"
    )


def test_brand_internals_intact():
    from iai_mcp import iai_cli  # noqa: F401  module importable

    assert callable(iai_cli._color), "_color helper missing"
    assert callable(iai_cli._print_logo), "_print_logo missing"


def test_doctor_header_reads_iai(monkeypatch, tmp_path):
    monkeypatch.setenv("NO_COLOR", "1")
    from iai_mcp.doctor import print_checklist

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_checklist([])
    output = buf.getvalue()

    assert "iai doctor" in output, (
        f"Doctor header missing `iai doctor`. Got:\n{output!r}"
    )
    assert "IAI-MCP Doctor" not in output, (
        "Old `IAI-MCP Doctor` header still present in doctor output"
    )


def test_consent_banner_header_reads_iai():
    from iai_mcp.cli import CONSENT_BANNER

    assert "iai Sleep Daemon" in CONSENT_BANNER, (
        f"Consent banner header missing `iai Sleep Daemon`. Banner:\n{CONSENT_BANNER!r}"
    )
    assert "IAI-MCP Sleep Daemon" not in CONSENT_BANNER, (
        "Old `IAI-MCP Sleep Daemon` header still present in consent banner"
    )
