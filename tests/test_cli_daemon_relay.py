"""iai teach/search/watch coexist with a live daemon: while the socket is
alive the CLI relays through it (the daemon is the single writer on the
production engine); a dead socket falls back to the direct store; a rejected
direct open prints one clean line, never a traceback."""
from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading

import pytest


class _FakeDaemonSocket:
    """Line-oriented JSON-RPC unix-socket server answering like the daemon."""

    def __init__(self, sock_path, handler):
        self.sock_path = str(sock_path)
        self.requests: list[dict] = []
        outer = self

        class _H(socketserver.StreamRequestHandler):
            def handle(self):
                line = self.rfile.readline()
                if not line:
                    return
                req = json.loads(line.decode("utf-8"))
                outer.requests.append(req)
                result = handler(req)
                resp = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))

        class _Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self._server = _Server(self.sock_path, _H)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    root = tmp_path / ".iai-mcp"
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    return root


@pytest.fixture
def short_sock(monkeypatch):
    # AF_UNIX paths are capped ~104 bytes on macOS; pytest tmp paths exceed
    # it. Bind at a short mkdtemp path and point the CLI's socket resolver
    # there.
    import shutil
    import tempfile
    from pathlib import Path

    d = tempfile.mkdtemp(prefix="iai-sock-", dir="/tmp")
    sock_path = Path(d) / "d.sock"
    from iai_mcp import iai_cli

    monkeypatch.setattr(iai_cli, "_daemon_socket_path", lambda: sock_path)
    yield sock_path
    shutil.rmtree(d, ignore_errors=True)


def test_search_relays_over_live_socket(store_root, short_sock, capsys):
    from iai_mcp.iai_cli import cmd_search

    canned = {"hits": [{"record_id": "x", "surface": "relayed hit",
                        "tier": "episodic", "lane": "lexical", "score": 1.0}]}

    def handler(req):
        assert req["method"] == "memory_search"
        assert req["params"]["query"] == "needle"
        return canned

    with _FakeDaemonSocket(short_sock, handler) as srv:
        rc = cmd_search(argparse.Namespace(query="needle", limit=5, json=True))
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out == canned
    assert len(srv.requests) == 1


def test_teach_relays_file_over_live_socket(store_root, short_sock, tmp_path, capsys):
    from iai_mcp.iai_cli import cmd_teach

    doc = tmp_path / "notes.md"
    doc.write_text("Sleep consolidation strengthens memory traces overnight.")

    report = {
        "source": "notes.md", "doc_tag": "doc:notes.md", "session_id": "study",
        "chunks_total": 1, "inserted": 1, "reinforced": 0, "skipped": 0,
        "edges_formed": 0, "contradictions_resolved": 0,
        "contradiction_candidates": 0, "schemas_induced": 0,
        "recall_tested": 1, "recall_verified": 1, "replay_queued": 1,
        "superseded_chunks": 0, "status": "studied",
    }

    def handler(req):
        assert req["method"] == "brain_view"
        assert req["params"]["verb"] == "teach"
        kwargs = req["params"]["kwargs"]
        assert kwargs["filename"] == "notes.md"
        assert kwargs["content_b64"]
        assert kwargs["session_id"] == "study"
        return report

    with _FakeDaemonSocket(short_sock, handler) as srv:
        rc = cmd_teach(argparse.Namespace(
            path=str(doc), session_id=None, reconcile=False, json=True,
        ))
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["inserted"] == 1
    assert len(srv.requests) == 1


def test_dead_socket_falls_back_to_direct(store_root, short_sock, tmp_path, capsys):
    # A socket FILE that is not a live listener must not be treated as alive.
    from iai_mcp.iai_cli import _daemon_alive

    assert _daemon_alive() is False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(short_sock))
    s.close()
    # Path exists as a socket inode but nothing listens: alive-check passes,
    # so the relay call itself must fail cleanly into the fallback.
    from iai_mcp.iai_cli import _is_relay_failure, _relay_rpc

    res = _relay_rpc("memory_search", {"query": "x", "k": 1}, timeout=2.0)
    assert _is_relay_failure(res)
    assert res["status"] == "daemon_down"


def test_locked_store_prints_clean_message(store_root, capsys, monkeypatch):
    import sqlite3

    from iai_mcp import iai_cli

    def _raise_locked(store_root=None):
        raise sqlite3.OperationalError(
            "engine connection: storage error: io error: store is locked:"
            " try_lock failed because the operation would block"
        )

    monkeypatch.setattr(iai_cli, "_open_store_shared", _raise_locked)
    store = iai_cli._open_store_shared_or_fail("search")
    assert store is None
    err = capsys.readouterr().err
    assert "memory is busy" in err
    assert "Traceback" not in err
