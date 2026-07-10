from __future__ import annotations

import argparse
import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        dry_run=False,
        apply=False,
        yes=False,
        store_path=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _patch_psutil_alive(
    monkeypatch: pytest.MonkeyPatch, *, pid: int, cmdline: list[str],
) -> None:
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = cmdline
    fake_psutil = MagicMock()
    fake_psutil.Process.return_value = fake_proc
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)


def _make_optimize_report(
    *, rows_before: int = 0, rows_after: int = 0,
) -> dict:
    base = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "size_bytes_before": 0,
        "size_bytes_after": 0,
        "vacuum_elapsed_sec": 0.0,
        "hnswlib_rebuild_elapsed_sec": 0.0,
        "elapsed_sec": 0.0,
    }
    return {
        "records": dict(base),
        "edges": dict(base),
        "events": dict(base),
    }


def _make_fake_store(record_ids: list[str]) -> MagicMock:
    fake_store = MagicMock()
    fake_tbl = MagicMock()
    fake_tbl.count_rows.return_value = len(record_ids)
    fake_df = MagicMock()
    fake_df.__getitem__.return_value.tolist.return_value = list(record_ids)
    fake_tbl.to_pandas.return_value = fake_df
    fake_store.db.open_table.return_value = fake_tbl
    return fake_store


@pytest.fixture
def iai_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase")
    try:
        import keyring.core
        keyring.core._keyring_backend = None
    except ImportError:
        pass

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir()
    hippo_dir = iai_dir / "hippo"
    hippo_dir.mkdir(parents=True)
    (hippo_dir / "brain.sqlite3").write_bytes(b"SQLite format 3" + b"\x00" * 85)
    import importlib
    from iai_mcp import cli as _cli
    importlib.reload(_cli)
    yield iai_dir
    importlib.reload(_cli)


def test_dry_run_prints_metrics_no_optimize_call(iai_root, monkeypatch, capsys):
    fake_store = _make_fake_store([])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    from iai_mcp.cli import cmd_maintenance_compact_records
    with patch(
        "iai_mcp.maintenance.optimize_hippo_storage"
    ) as mock_opt:
        rc = cmd_maintenance_compact_records(_make_args(dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "dry-run"
    assert "db_size_mb" in payload["metrics"]["pre"]
    assert "records_count" in payload["metrics"]["pre"]
    assert payload["metrics"]["post"] is None
    mock_opt.assert_not_called()


def test_apply_with_yes_runs_optimize(iai_root, monkeypatch, capsys):
    from iai_mcp import cli as _cli

    fake_store = _make_fake_store(["id1", "id2", "id3", "id4", "id5"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    mock_opt = MagicMock(return_value=_make_optimize_report(
        rows_before=5, rows_after=5,
    ))
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_hippo_storage", mock_opt,
    )

    rc = _cli.cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0
    assert mock_opt.call_count == 1


def test_preflight_refuses_when_daemon_alive(iai_root, monkeypatch, capsys):
    state_path = iai_root / ".daemon-state.json"
    state_path.write_text(json.dumps({"daemon_pid": os.getpid()}))
    _patch_psutil_alive(
        monkeypatch, pid=os.getpid(),
        cmdline=["python", "-m", "iai_mcp.daemon"],
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    with patch(
        "iai_mcp.maintenance.optimize_hippo_storage"
    ) as mock_opt:
        rc = cmd_maintenance_compact_records(
            _make_args(apply=True, yes=True),
        )
    assert rc == 1
    err = capsys.readouterr().err
    assert "daemon running" in err
    mock_opt.assert_not_called()


def test_preflight_skips_when_daemon_state_missing(
    iai_root, monkeypatch, capsys,
):
    assert not (iai_root / ".daemon-state.json").exists()

    fake_store = _make_fake_store([])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    mock_opt = MagicMock(return_value=_make_optimize_report())
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_hippo_storage", mock_opt,
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0
    assert mock_opt.call_count == 1


def test_record_id_set_invariant_aborts_on_divergence(
    iai_root, monkeypatch, capsys,
):
    fake_store = _make_fake_store(["id1", "id2", "id3"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_hippo_storage",
        MagicMock(return_value=_make_optimize_report(
            rows_before=3, rows_after=2,
        )),
    )
    pre_set = {"id1", "id2", "id3"}
    post_set = {"id1", "id2"}
    metrics_seq = [
        {
            "db_size_mb": 0.0,
            "records_count": 3, "record_id_set": pre_set,
        },
        {
            "db_size_mb": 0.0,
            "records_count": 2, "record_id_set": post_set,
        },
    ]
    call_counter = {"n": 0}

    def _stub_metrics(*args, **kwargs):
        i = call_counter["n"]
        call_counter["n"] += 1
        return metrics_seq[min(i, 1)]

    monkeypatch.setattr(
        "iai_mcp.cli._maintenance_compact_metrics", _stub_metrics,
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "ABORT" in err
    assert "divergence" in err

    failed = list(iai_root.glob(".maintenance-compact-FAILED-*.json"))
    assert len(failed) == 1
    payload = json.loads(failed[0].read_text())
    assert payload["status"] == "aborted"
    assert payload["reason"] == "record_id_set divergence post-optimize"
    assert payload["missing_ids_count"] == 1


def test_audit_file_written_on_apply(iai_root, monkeypatch, capsys):
    fake_store = _make_fake_store(["id1", "id2"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_hippo_storage",
        MagicMock(return_value=_make_optimize_report(
            rows_before=2, rows_after=2,
        )),
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0

    audits = list(iai_root.glob(".maintenance-compact-*.json"))
    audits = [a for a in audits if "FAILED" not in a.name]
    assert len(audits) == 1, (
        f"expected exactly 1 audit file, got {audits}"
    )
    payload = json.loads(audits[0].read_text())
    assert payload["status"] == "ok"
    assert "metrics_pre" in payload
    assert "metrics_post" in payload
    assert "elapsed_sec" in payload


def test_dry_run_no_audit_file(iai_root, monkeypatch, capsys):
    fake_store = _make_fake_store([])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(_make_args(dry_run=True))
    assert rc == 0
    audits = list(iai_root.glob(".maintenance-compact-*.json"))
    assert audits == []


def test_yes_required_with_apply_in_non_tty(iai_root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=False),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires --yes" in err
