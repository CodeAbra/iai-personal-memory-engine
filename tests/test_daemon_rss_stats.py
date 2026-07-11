"""Unit coverage for the rss_stats read method and its operator CLI.

`core.dispatch(store, "rss_stats", {})` returns the process-local resident-set
snapshot; an unknown method still raises. The `iai-mcp daemon rss-stats` CLI
prints the live snapshot via JSON-RPC and a log-derived per-step delta table,
degrading gracefully to the table alone when the daemon socket is unreachable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from iai_mcp.core import dispatch
from iai_mcp.core import UnknownMethodError
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


_SNAPSHOT_KEYS = (
    "rss_kib",
    "vmmap_region_count",
    "vm_allocate_kib",
    "pa_pool_bytes",
    "pa_pool_max",
    "numba_nrt_alloc_count",
    "numba_nrt_free_count",
)


def test_rss_stats_dispatch_shape(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "rss_stats", {})
    assert isinstance(result, dict)
    for key in _SNAPSHOT_KEYS:
        assert key in result, f"rss_stats result missing {key!r}: {sorted(result)}"


def test_rss_stats_unknown_method_still_raises(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path)
    with pytest.raises(UnknownMethodError):
        dispatch(store, "definitely_not_a_method", {})


def _seed_events_log(store_root: Path, date_str: str, rows: list[dict]) -> Path:
    log_dir = store_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"lifecycle-events-{date_str}.jsonl"
    with log_file.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return log_file


def test_cli_rss_stats_daemon_down_fallback(tmp_path, monkeypatch, capsys) -> None:
    from iai_mcp import cli as _cli
    from iai_mcp.cli._daemon import cmd_daemon_rss_stats

    # Daemon down: the JSON-RPC helper returns None.
    monkeypatch.setattr(_cli, "_send_jsonrpc_request", lambda *a, **k: None)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).isoformat()
    _seed_events_log(
        tmp_path,
        today,
        [
            {
                "ts": ts,
                "event": "sleep_step_completed",
                "step": "RECONSOLIDATION",
                "step_num": 3,
                "duration_sec": 1.2,
                "rss_delta_kib": 4096,
            },
            {
                "ts": ts,
                "event": "sleep_step_completed",
                "step": "OPTIMIZE_HIPPO",
                "step_num": 12,
                "duration_sec": 0.8,
                "rss_delta_kib": 2048,
            },
        ],
    )

    rc = cmd_daemon_rss_stats(SimpleNamespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "unreachable" in out.lower()
    # The log-derived table still renders both completed steps.
    assert "RECONSOLIDATION" in out
    assert "OPTIMIZE_HIPPO" in out


def test_cli_rss_stats_live_snapshot(tmp_path, monkeypatch, capsys) -> None:
    from iai_mcp import cli as _cli
    from iai_mcp.cli._daemon import cmd_daemon_rss_stats

    fake = {
        "result": {
            "rss_kib": 204800,
            "vmmap_region_count": 289,
            "vm_allocate_kib": 16,
            "pa_pool_bytes": 0,
            "pa_pool_max": 1024,
            "numba_nrt_alloc_count": 10,
            "numba_nrt_free_count": 4,
            "t": 1.0,
        }
    }
    monkeypatch.setattr(_cli, "_send_jsonrpc_request", lambda *a, **k: fake)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    rc = cmd_daemon_rss_stats(SimpleNamespace())
    assert rc == 0
    out = capsys.readouterr().out
    # The live block surfaces the RSS / regions / pool figures.
    assert "204800" in out or "200" in out  # rss_kib or MiB rendering
    assert "289" in out  # region count
