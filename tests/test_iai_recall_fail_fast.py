from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


FAIL_FAST_CEILING_S = 3.5

FAST_CEILING_S = 2.0

N_FILLER = 5


def _make_hermetic_store(tmp_path: Path) -> Path:
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(12345)
    for i in range(N_FILLER):
        v = rng.random(EMBED_DIM).astype(np.float32)
        store.insert(_make(text=f"User record {i}", vec=v.tolist()))
    flush_record_buffer(store)
    try:
        store.close()
    except Exception:
        pass
    return store_root


def _unix_socket_server_stall(sock_path: str, stall_seconds: float = 60.0) -> threading.Event:
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


def _unix_socket_server_fast(sock_path: str, hits: list[dict]) -> threading.Event:
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
        srv.settimeout(10.0)
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(4096)
                resp = {
                    "jsonrpc": "2.0", "id": 1,
                    "result": {
                        "hits": hits,
                        "anti_hits": [],
                        "activation_trace": [],
                        "budget_used": 100,
                        "ann_path_used": True,
                    }
                }
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except OSError:
                pass
            finally:
                conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    ready.wait(timeout=2.0)
    return ready


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_and_stub_construct(monkeypatch):
    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr

    def _raising_funnel(_store):
        raise RuntimeError("hermetic: no real embedder construct in fail-fast tests")

    _sr._WARM_LOCAL_STORE = None
    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)

    yield

    _sr._WARM_LOCAL_STORE = None


def test_slow_daemon_degrades_in_under_3s(monkeypatch, tmp_path, short_socket):
    sock_path = str(short_socket)
    store_root = _make_hermetic_store(tmp_path)

    ready = _unix_socket_server_stall(sock_path, stall_seconds=60.0)
    assert ready.is_set(), "Stall server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    # Pin the read timeout: this test verifies the fail-fast MACHINERY (the
    # socket read gives up and the degrade path engages within the ceiling).
    # The shipped default is deliberately longer — it waits out a
    # consolidating daemon for a real answer; policy rides the env override.
    monkeypatch.setenv("IAI_RECALL_READ_TIMEOUT", "2.0")

    import iai_mcp.iai_cli as _iai_cli

    def _fast_degrade(store_root_arg, cue, n=5, *, session_id=None):
        return [{"literal_surface": "User degrade hit", "score": 0.0, "_source": "daemon-down-degrade"}]

    monkeypatch.setattr(_iai_cli, "recall_semantic_warm" if hasattr(_iai_cli, "recall_semantic_warm") else "_recall_warm", _fast_degrade, raising=False)

    import iai_mcp.semantic_recall as _sr
    monkeypatch.setattr(_sr, "recall_semantic_warm", _fast_degrade)

    import argparse
    args = argparse.Namespace(cue="test query", limit=5, json=False)

    t0 = time.perf_counter()
    returncode = _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    assert elapsed < FAIL_FAST_CEILING_S, (
        f"iai recall took {elapsed:.2f}s on a stalled daemon — expected <=3s. "
        f"The LAT-04 short read_timeout fix may not be active."
    )

    assert returncode == 0, f"cmd_recall returned non-zero: {returncode}"


def test_fast_daemon_uses_daemon_hits_no_degrade(monkeypatch, tmp_path, short_socket):
    sock_path = str(short_socket)
    store_root = _make_hermetic_store(tmp_path)

    daemon_hits = [
        {"record_id": "00000000-0000-0000-0000-000000000001", "score": 0.95,
         "reason": "cosine 0.95", "literal_surface": "User daemon memory hit 1",
         "adjacent_suggestions": []},
    ]
    ready = _unix_socket_server_fast(sock_path, hits=daemon_hits)
    assert ready.is_set(), "Fast server failed to bind"

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", sock_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    degrade_called = []

    import iai_mcp.semantic_recall as _sr
    _orig_warm = _sr.recall_semantic_warm

    def _spy_warm(*a, **kw):
        degrade_called.append(True)
        return _orig_warm(*a, **kw)

    monkeypatch.setattr(_sr, "recall_semantic_warm", _spy_warm)

    import iai_mcp.iai_cli as _iai_cli
    import argparse

    import io
    captured = io.StringIO()
    import sys
    orig_stdout = sys.stdout
    sys.stdout = captured

    t0 = time.perf_counter()
    try:
        args = argparse.Namespace(cue="test query", limit=5, json=True)
        returncode = _iai_cli.cmd_recall(args)
    finally:
        sys.stdout = orig_stdout

    elapsed = time.perf_counter() - t0

    assert elapsed < FAST_CEILING_S, f"Fast daemon recall took {elapsed:.2f}s"
    assert returncode == 0

    assert not degrade_called, (
        "Degrade path was invoked even though the daemon replied promptly — "
        "the LAT-04 fix must not degrade on a fast daemon."
    )

    output = captured.getvalue()
    assert "daemon memory hit" in output or '"_source": "daemon"' in output or '"source": "daemon"' in output or "daemon" in output, (
        f"Expected daemon result in output, got: {output!r}"
    )


def test_down_socket_degrades_fast(monkeypatch, tmp_path):
    absent_sock = str(tmp_path / "absent.sock")
    store_root = _make_hermetic_store(tmp_path)

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", absent_sock)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    import iai_mcp.iai_cli as _iai_cli
    import argparse

    t0 = time.perf_counter()
    args = argparse.Namespace(cue="test query", limit=5, json=False)
    returncode = _iai_cli.cmd_recall(args)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Down-socket degrade took {elapsed:.2f}s"
    assert returncode == 0
