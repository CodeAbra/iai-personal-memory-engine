from __future__ import annotations

import argparse
import json as _json
import os
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


READ_TIMEOUT_S = 1.5
FALLBACK_LOWER_S = READ_TIMEOUT_S
FALLBACK_CEILING_S = 3.5

WAKE_FAILFAST_LOWER_S = 1.8
WAKE_FAILFAST_CEILING_S = 3.5


N_FILLER = 5


def _make_hermetic_store(tmp_path: Path) -> Path:
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(42)
    for i in range(N_FILLER):
        v = rng.random(EMBED_DIM).astype(np.float32)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))
    flush_record_buffer(store)
    try:
        store.close()
    except Exception:
        pass
    return store_root


def _short_sock_path() -> str:
    fd, path = tempfile.mkstemp(prefix="iai_asleep_", suffix=".sock", dir=tempfile.gettempdir())
    os.close(fd)
    os.unlink(path)
    return path


def _start_stall_server(sock_path: str, stall_seconds: float = 60.0) -> threading.Event:
    ready = threading.Event()

    def _server():
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(sock_path)
        srv.listen(5)
        ready.set()
        srv.settimeout(120.0)
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(4096)
            except OSError:
                pass
            time.sleep(stall_seconds)
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    ready.wait(timeout=2.0)
    return ready


def _write_lifecycle_state(store_root: Path, state: str, seconds_ago: float) -> None:
    from iai_mcp.lifecycle_state import save_state, default_state

    lc_path = store_root / "lifecycle_state.json"
    record = dict(default_state())
    record["current_state"] = state
    since = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    record["since_ts"] = since
    record["last_activity_ts"] = since
    save_state(record, path=lc_path)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr

    def _raising_funnel(_store):
        raise RuntimeError("hermetic: no real embedder in asleep-skip tests")

    _sr._WARM_LOCAL_STORE = None
    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)
    yield
    _sr._WARM_LOCAL_STORE = None


def test_hung_daemon_falls_back_within_read_timeout(monkeypatch, tmp_path, capsys):
    """A genuinely stalled daemon (SLEEP lifecycle state, socket accepts but
    never replies) must still be tried first -- lifecycle state is never a
    pre-emptive skip (see test_iai_cli_recall.py::
    test_recall_contacts_daemon_while_lifecycle_state_is_sleep for the healthy
    SLEEP-daemon path). Latency is bounded by IAI_RECALL_READ_TIMEOUT, then
    cmd_recall falls back to direct-store recall -- never an instant skip,
    never an unbounded hang.
    """
    store_root = _make_hermetic_store(tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    _write_lifecycle_state(store_root, "SLEEP", seconds_ago=60.0)

    sock_path = _short_sock_path()
    ready = _start_stall_server(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_RECALL_READ_TIMEOUT", str(READ_TIMEOUT_S))

    import iai_mcp.iai_cli as _iai_cli

    args = argparse.Namespace(cue="test recall cue", limit=5, json=True)

    t0 = time.perf_counter()
    rc = _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    assert rc == 0
    assert elapsed >= FALLBACK_LOWER_S, (
        f"iai recall against a stalled daemon returned in {elapsed:.2f}s -- "
        f"too fast; expected >= {FALLBACK_LOWER_S}s (the daemon must still be "
        "tried, bounded by IAI_RECALL_READ_TIMEOUT, not pre-emptively skipped)."
    )
    assert elapsed < FALLBACK_CEILING_S, (
        f"iai recall against a stalled daemon took {elapsed:.2f}s -- expected "
        f"< {FALLBACK_CEILING_S}s (bounded read-timeout, then store fallback)."
    )

    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert payload["_source"] != "daemon", (
        f"expected the stalled daemon to be bypassed via fallback, got _source={payload['_source']!r}"
    )


def test_wake_state_still_pays_failfast_rpc(monkeypatch, tmp_path):
    store_root = _make_hermetic_store(tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    _write_lifecycle_state(store_root, "WAKE", seconds_ago=10.0)

    sock_path = _short_sock_path()
    ready = _start_stall_server(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_RECALL_READ_TIMEOUT", "2.0")

    import iai_mcp.iai_cli as _iai_cli

    args = argparse.Namespace(cue="wake regression check", limit=5, json=True)

    t0 = time.perf_counter()
    _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    assert elapsed >= WAKE_FAILFAST_LOWER_S, (
        f"WAKE state recall returned in {elapsed:.2f}s — too fast; "
        "the ~2s RPC should still be issued for a WAKE daemon. "
        f"Expected >= {WAKE_FAILFAST_LOWER_S}s (read_timeout = 2.0s)."
    )
    assert elapsed < WAKE_FAILFAST_CEILING_S, (
        f"WAKE state recall took {elapsed:.2f}s — too slow; expected < {WAKE_FAILFAST_CEILING_S}s."
    )
