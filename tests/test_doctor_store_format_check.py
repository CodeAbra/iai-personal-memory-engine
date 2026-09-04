"""``doctor`` store-format check: (gg) store format.

Reads the backing file's 16-byte header only and never constructs a store --
the very operation the legacy format's read-only-open defect orphans the
write-ahead-log sidecars under. Covers the three file states (absent,
native, legacy), the zero-open proof, and the cross-surface consistency of
the advisory's quoted command sequence against the real argument parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_LILLI_MAGIC_PROBE_LEN = 16


def _write_header(path: Path, magic: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(magic.ljust(_LILLI_MAGIC_PROBE_LEN, b"\x00"))


def _lilli_magic() -> bytes:
    from iai_mcp.lillibrain.constants import DB_MAGIC

    return DB_MAGIC


def _sqlite_magic() -> bytes:
    return b"SQLite format 3\x00"


def test_absent_file_passes_with_no_store_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_gg_store_format

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    monkeypatch.setattr(
        "iai_mcp.doctor._resolve_hippo_db_path", lambda: db_path
    )

    result = check_gg_store_format()
    assert result.passed is True
    assert result.status == "PASS"
    assert "no store yet" in result.detail.lower()


def test_native_format_passes_and_names_the_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_gg_store_format

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _write_header(db_path, _lilli_magic())
    monkeypatch.setattr(
        "iai_mcp.doctor._resolve_hippo_db_path", lambda: db_path
    )
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    result = check_gg_store_format()
    assert result.passed is True
    assert result.status == "PASS"
    assert "native" in result.detail.lower()


def test_legacy_format_is_advisory_with_format_defect_and_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.doctor import check_gg_store_format

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _write_header(db_path, _sqlite_magic())
    monkeypatch.setattr(
        "iai_mcp.doctor._resolve_hippo_db_path", lambda: db_path
    )
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")

    result = check_gg_store_format()
    assert result.passed is True
    assert result.status == "WARN"
    detail_l = result.detail.lower()
    assert "legacy" in detail_l
    assert "orphan" in detail_l
    assert "read-only" in detail_l
    assert "daemon stop" in result.detail
    assert "daemon start" in result.detail
    assert "--swap" in result.detail
    assert "--apply" in result.detail
    assert "--yes" in result.detail
    assert str(db_path) in result.detail


def test_zero_open_across_all_three_file_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch both the store class constructor and the raw-open helper to
    raise, then run the check against absent, native and legacy files --
    it must return a result for every one, never propagate."""
    from iai_mcp.doctor import check_gg_store_format
    from iai_mcp.hippo import HippoDB

    def _raise_ctor(*a, **kw):
        raise AssertionError("check_gg_store_format must never construct a store")

    def _raise_open(*a, **kw):
        raise AssertionError("check_gg_store_format must never call the raw-open helper")

    monkeypatch.setattr(HippoDB, "__init__", _raise_ctor)
    monkeypatch.setattr(
        "iai_mcp.doctor._storage_checks.open_store_conn", _raise_open
    )

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    monkeypatch.setattr(
        "iai_mcp.doctor._resolve_hippo_db_path", lambda: db_path
    )

    # (1) absent file
    assert not db_path.exists()
    r_absent = check_gg_store_format()
    assert r_absent.status == "PASS"

    # (2) native file
    _write_header(db_path, _lilli_magic())
    r_native = check_gg_store_format()
    assert r_native.status == "PASS"

    # (3) legacy file
    _write_header(db_path, _sqlite_magic())
    r_legacy = check_gg_store_format()
    assert r_legacy.status == "WARN"


def test_check_function_body_has_no_store_or_connection_reference() -> None:
    """The function's own source contains no reference to the store class,
    the raw-open helper, or any connection object -- its only file access
    is the header read inside the driver resolver."""
    import inspect

    from iai_mcp.doctor._storage_checks import check_gg_store_format

    src = inspect.getsource(check_gg_store_format)
    assert "HippoDB" not in src
    assert "open_store_conn" not in src
    assert "_conn" not in src


def test_registry_places_store_format_after_daemon_code_current() -> None:
    import inspect

    from iai_mcp.doctor import run_diagnosis

    src = inspect.getsource(run_diagnosis)
    assert src.count("check_gg_store_format()") == 1
    ff_idx = src.index("check_ff_daemon_code_current()")
    gg_idx = src.index("check_gg_store_format()")
    z_idx = src.index("check_z_avx2_support()")
    assert ff_idx < gg_idx < z_idx


def test_no_repair_action_registered_for_store_format_check() -> None:
    import inspect

    from iai_mcp.doctor import _plan_repair_actions

    src = inspect.getsource(_plan_repair_actions)
    assert "check_gg_store_format" not in src
    assert "(gg) store format" not in src


def test_advisory_command_sequence_parses_through_the_real_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extract the quoted preview and apply invocations out of the advisory
    detail and parse both through the real argument parser -- proves the
    instructions match the command that actually exists."""
    from iai_mcp.doctor import check_gg_store_format
    from iai_mcp.cli import _build_parser

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    _write_header(db_path, _sqlite_magic())
    monkeypatch.setattr(
        "iai_mcp.doctor._resolve_hippo_db_path", lambda: db_path
    )
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")

    result = check_gg_store_format()
    lines = [
        line.strip()
        for line in result.detail.splitlines()
        if line.strip().startswith("iai-mcp migrate-to-lilli")
    ]
    assert len(lines) == 2, lines

    preview_line, apply_line = lines
    assert "--apply" not in preview_line
    assert "--apply" in apply_line
    assert "--yes" in apply_line

    def _argv(line: str) -> list[str]:
        # strip the leading "iai-mcp " program name
        return line.split()[1:]

    parser = _build_parser()
    ns_preview = parser.parse_args(_argv(preview_line))
    assert ns_preview.swap is True
    assert ns_preview.apply is False

    ns_apply = parser.parse_args(_argv(apply_line))
    assert ns_apply.swap is True
    assert ns_apply.apply is True
    assert ns_apply.yes is True
