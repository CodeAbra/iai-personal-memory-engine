"""CLI surface for the in-place hippo/ swap: ``iai-mcp migrate-to-lilli --swap``.

Drives the real argument parser and the real command functions against a
legacy-format store built by the swap module's own fixture builder (imported,
never duplicated). Covers the two mutation gates (``--apply`` and ``--yes``),
the dry-run preview, a completed apply, a real blocker, the already-native
no-op, and that the pre-existing non-swap invocation is untouched.

Hermetic: tmp HOME + tmp store, single process, no xdist. Fixtures use
``alice``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from test_migrate_to_lilli_swap import _build_legacy_source_with_file_key

from iai_mcp.cli import _build_parser
from iai_mcp.cli._analytics import cmd_migrate_to_lilli, cmd_migrate_to_lilli_swap


def _parse(argv: list[str]):
    parser = _build_parser()
    return parser.parse_args(argv)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def test_swap_parses_without_dst_but_dst_required_without_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """--swap with no --dst parses cleanly; without --swap, no --dst fails
    at the command body with a clear error and a non-zero exit."""
    ns = _parse(
        ["migrate-to-lilli", "--src", str(tmp_path / "hippo" / "brain.sqlite3"), "--swap"]
    )
    assert ns.swap is True
    assert ns.dst is None

    ns_no_swap = _parse(
        ["migrate-to-lilli", "--src", str(tmp_path / "hippo" / "brain.sqlite3")]
    )
    assert ns_no_swap.swap is False
    assert ns_no_swap.dst is None

    rc = cmd_migrate_to_lilli(ns_no_swap)
    assert rc != 0
    err = capsys.readouterr().err
    assert "--dst" in err
    assert "required" in err


def test_apply_without_confirmation_gate_never_calls_swap_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """--apply without --yes exits non-zero and never invokes the swap
    function in apply mode -- assert on the decision, not an absent side
    effect."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"
    hippo_mtime_before = (root / "hippo").stat().st_mtime_ns

    calls: list[dict] = []

    def _recording_swap(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"mode": "apply", "swapped": True, "blockers": []}

    import iai_mcp.migrate as _migrate_mod

    monkeypatch.setattr(_migrate_mod, "swap_migrated_store", _recording_swap)

    ns = _parse(
        [
            "migrate-to-lilli",
            "--src",
            str(src_db),
            "--swap",
            "--apply",
        ]
    )
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc != 0
    assert calls == [], "swap function must not be called when the confirmation gate is missing"
    err = capsys.readouterr().err
    assert "--yes" in err
    assert (root / "hippo").stat().st_mtime_ns == hippo_mtime_before


def test_swap_alone_prints_dry_run_report_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """--swap with neither --apply nor --yes previews and exits zero without
    any filesystem write."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=5)
    src_db = root / "hippo" / "brain.sqlite3"

    entries_before = sorted(p.name for p in root.iterdir())
    hippo_mtime_before = (root / "hippo").stat().st_mtime_ns

    ns = _parse(["migrate-to-lilli", "--src", str(src_db), "--swap"])
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc == 0

    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "stdlib" in out
    assert str(root) in out
    assert "hippo.sqlite-backup-" in out

    entries_after = sorted(p.name for p in root.iterdir())
    assert entries_after == entries_before
    assert (root / "hippo").stat().st_mtime_ns == hippo_mtime_before
    assert not (root / ".swap-in-progress").exists()


def test_swap_apply_with_both_gates_performs_the_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """--swap --apply --yes calls the swap function in apply mode and prints
    the resulting summary."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=6)
    src_db = root / "hippo" / "brain.sqlite3"

    ns = _parse(
        ["migrate-to-lilli", "--src", str(src_db), "--swap", "--apply", "--yes"]
    )
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc == 0

    out = capsys.readouterr().out
    assert "apply" in out
    assert "swapped" in out.lower()

    from iai_mcp.hippo._db import _resolve_effective_driver

    assert _resolve_effective_driver(str(src_db)) == "lilli"
    assert (root / f"hippo.sqlite-backup-{_today_utc()}").is_dir()


def test_blockers_exit_nonzero_with_each_blocker_on_its_own_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """A real blocker (a same-day backup dir already present) makes the
    apply-mode command exit non-zero and print each blocker on its own
    line."""
    root = tmp_path / "live"
    root.mkdir()
    _build_legacy_source_with_file_key(root, monkeypatch=monkeypatch, n=4)
    src_db = root / "hippo" / "brain.sqlite3"

    (root / f"hippo.sqlite-backup-{_today_utc()}").mkdir()

    ns = _parse(
        ["migrate-to-lilli", "--src", str(src_db), "--swap", "--apply", "--yes"]
    )
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc != 0

    out = capsys.readouterr().out
    assert "blockers" in out
    assert "already exists" in out

    from iai_mcp.hippo._db import _resolve_effective_driver

    assert _resolve_effective_driver(str(src_db)) == "stdlib"


def test_swap_apply_oserror_is_caught_and_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """An OSError raised by swap_migrated_store (the crash-safety path this
    phase's own test proves reachable on the second swap rename) is caught
    at the CLI surface and reported as a pointer to the marker/backup/
    staging paths, not a raw traceback."""
    root = tmp_path / "live"
    root.mkdir()
    src_db = root / "hippo" / "brain.sqlite3"

    def _raising_swap(*args, **kwargs):
        raise OSError("simulated failure on the second swap rename")

    import iai_mcp.migrate as _migrate_mod

    monkeypatch.setattr(_migrate_mod, "swap_migrated_store", _raising_swap)

    ns = _parse(
        ["migrate-to-lilli", "--src", str(src_db), "--swap", "--apply", "--yes"]
    )
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc != 0

    err = capsys.readouterr().err
    assert "simulated failure" in err
    assert str(root) in err
    assert ".swap-in-progress" in err
    assert "hippo.sqlite-backup" in err
    assert ".migrate-staging" in err


def test_already_native_store_prints_noop_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """A store already in the native format reports the no-op and exits
    zero, regardless of mode."""
    root = tmp_path / "live"
    root.mkdir()
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    from iai_mcp.store import MemoryStore

    store = MemoryStore(root)
    store.db.close()
    src_db = root / "hippo" / "brain.sqlite3"

    ns = _parse(["migrate-to-lilli", "--src", str(src_db), "--swap"])
    rc = cmd_migrate_to_lilli_swap(ns)
    assert rc == 0

    out = capsys.readouterr().out
    assert "already native" in out


def test_without_swap_flag_reaches_the_original_migration_path_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without --swap the command still delegates to the pre-existing
    migrate + verify path with the same arguments, and never touches the
    swap command function."""
    src = str(tmp_path / "hippo" / "brain.sqlite3")
    dst = str(tmp_path / "dst")

    calls: dict = {}

    class _FakeReport:
        rows_copied = {"records": 3}
        elapsed_sec = 0.1
        peak_rss_mb = 1.0
        max_vec_label = 2

    class _FakeDim:
        ok = True
        reason = "ok"

    class _FakeVerifyReport:
        ok = True
        dimensions = {"A_counts": _FakeDim()}

    def _fake_migrate(src_arg, dst_arg, *, batch, prune_telemetry_before):
        calls["migrate"] = (src_arg, dst_arg, batch, prune_telemetry_before)
        return _FakeReport()

    def _fake_verify(src_arg, dst_arg, key, *, src_root):
        calls["verify"] = (src_arg, dst_arg, key, src_root)
        return _FakeVerifyReport()

    import iai_mcp.migrate as _migrate_mod
    from iai_mcp.crypto import CryptoKey

    monkeypatch.setattr(_migrate_mod, "migrate_sqlite_to_lilli", _fake_migrate)
    monkeypatch.setattr(_migrate_mod, "verify_store_equality", _fake_verify)
    monkeypatch.setattr(CryptoKey, "get_or_create", lambda self: b"\x00" * 32)

    def _fail_swap(*args, **kwargs):
        raise AssertionError("the swap command must not run when --swap is absent")

    monkeypatch.setattr(
        "iai_mcp.cli._analytics.cmd_migrate_to_lilli_swap", _fail_swap
    )

    ns = _parse(
        ["migrate-to-lilli", "--src", src, "--dst", dst, "--batch", "10"]
    )
    rc = cmd_migrate_to_lilli(ns)
    assert rc == 0
    assert calls["migrate"] == (src, dst, 10, None)


def test_help_text_names_the_gates_and_the_dated_backup(capsys):
    """The migration command's help text states the swap, apply, and
    confirmation flags, and that the previous store is kept under a dated
    backup. There is no separate --dry-run flag: omitting --apply already
    previews without writing anything."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-to-lilli", "--help"])
    out = capsys.readouterr().out
    assert "--swap" in out
    assert "--apply" in out
    assert "--yes" in out
    assert "backup" in out.lower()
    assert "--dry-run" not in out
