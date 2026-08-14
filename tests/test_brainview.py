"""Brain-view server: loopback dashboard over the awake hippocampus.

Daemon-independent by construction (store reads via ro_conn/get_batch,
writes through the shared capture spine). There is deliberately NO delete
surface — the only destructive-looking verb is a forget HINT that feeds the
existing decay/erasure funnel and leaves the record intact.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.brainview import make_server
from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def served(tmp_path):
    store = MemoryStore(path=tmp_path)
    server = make_server(store, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield store, port
    server.shutdown()


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, r.read()


def _post(port: int, path: str, payload: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _seed(store: MemoryStore, text: str, session: str = "s") -> str:
    result = capture_turn(
        store, cue="", text=text, tier="episodic", session_id=session, role="user",
    )
    assert result["status"] == "inserted", result
    flush_record_buffer(store)
    return result["record_id"]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_page_and_overview(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed(store, "The brain watches itself think.")
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, body = _get(port, "/")
        assert status == 200 and b"the brain will study it" in body
        status, body = _get(port, "/api/overview")
        overview = json.loads(body)
        assert status == 200
        assert overview["counts"]["episodic"] == 1
        assert "lifecycle" in overview
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_overview_reports_knowledge_coverage(driver, tmp_path, monkeypatch):
    """counts.coverage = share of live episodic records consolidated into a
    live semantic summary via consolidated_from edges — NOT the summary/
    moment count ratio."""
    from uuid import UUID

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    covered = _seed(store, "The hippocampus binds episodes together during sleep.")
    _seed(store, "A stray uncondensed moment about coral reefs.")

    summary = capture_turn(
        store, cue="", text="Knowledge: sleep binds episodes into durable schemas.",
        tier="semantic", session_id="system", role="assistant",
    )
    assert summary["status"] == "inserted", summary
    flush_record_buffer(store)
    store.boost_edges(
        [(UUID(summary["record_id"]), UUID(covered))],
        edge_type="consolidated_from",
        delta=1.0,
    )

    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _status, body = _get(port, "/api/overview")
        counts = json.loads(body)["counts"]
        assert counts["episodic"] == 2 and counts["semantic"] == 1, counts
        assert abs(counts["coverage"] - 0.5) < 1e-6, (
            f"one of two episodic records is consolidated: {counts}"
        )
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graph_serves_verbatim_decrypted_surfaces(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    text = "Ultrastability requires an essential-variable monitor."
    _seed(store, text)
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _status, body = _get(port, "/api/graph")
        graph = json.loads(body)
        assert graph["nodes"], "seeded record must appear in the graph"
        node = graph["nodes"][0]
        assert node["surface"] == text, "surface must be verbatim, decrypted"
        assert node["tier"] == "episodic"
        assert "pinned" in node and "centrality" in node
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_goes_through_dedup_gated_spine(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    # A dashboard capture is a NOTE, not a spoken turn: its identity is
    # content-bound (same text -> exact-key reinforce), matching uploads.
    store = MemoryStore(path=tmp_path)
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        first = _post(port, "/api/capture", {"text": "Variety absorbs variety."})
        assert first["status"] == "inserted", first
        again = _post(port, "/api/capture", {"text": "Variety absorbs variety."})
        assert again["status"] == "reinforced", (
            f"[{driver}] identical note must reinforce, not duplicate: {again}"
        )
        assert again["record_id"] == first["record_id"]
        with store.db._conn_lock:
            row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
        assert int(row[0]) == 1, f"[{driver}] duplicate row stored"
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_forget_hint_feeds_decay_never_deletes(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "An obsolete fact the user wants to fade out.")
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = _post(port, "/api/forget", {"id": rid})
        assert result["status"] == "queued_for_forgetting", result
        rec = store.get(UUID(rid))
        assert rec is not None, f"[{driver}] hint must NEVER delete the record"
        assert rec.literal_surface == "An obsolete fact the user wants to fade out."
        assert float(rec.centrality or 0.0) == 0.0
        assert rec.never_decay is False
        assert any(
            p.get("cue") == "user-forget-hint" for p in (rec.provenance or [])
        ), "the hint must be traceable in provenance"
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_forget_hint_refuses_pinned(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "A pinned foundational fact.")
    from iai_mcp.store import RECORDS_TABLE

    store.db.open_table(RECORDS_TABLE).update(
        where=f"id = '{rid}'", values={"pinned": True},
    )
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = _post(port, "/api/forget", {"id": rid})
        assert result["status"] == "refused", result
        rec = store.get(UUID(rid))
        assert rec is not None and rec.pinned
    finally:
        server.shutdown()


def test_no_delete_surface_exists(served):
    import urllib.error

    _store, port = served
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/delete",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read())
        assert payload["status"] == "error", "no delete endpoint may exist"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404, "no delete endpoint may exist"
    source = open("src/iai_mcp/brainview.py", encoding="utf-8").read()
    assert "DELETE FROM" not in source, "brainview must never issue a SQL DELETE"

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_endpoint_studies_uploaded_file(driver, tmp_path, monkeypatch):
    import base64

    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_PATSEP_LINK_THRESHOLD", "0.45")
    store = MemoryStore(path=tmp_path)
    _seed(store, "The hippocampus consolidates memories during sleep.")
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        content = base64.b64encode(
            b"Sleep consolidation strengthens hippocampal memory traces overnight. "
            b"The hippocampus replays recent experiences while the cortex extracts "
            b"patterns. Deep sleep stages are when the brain prunes weak synaptic "
            b"connections. Memory retrieval works best after consolidation completes."
        ).decode("ascii")
        report = _post(
            port, "/api/teach",
            {"filename": "notes.md", "content_b64": content},
        )
        assert report["status"] == "studied", report
        assert report["inserted"] >= 1
        assert report["edges_formed"] >= 1, (
            f"[{driver}] an uploaded file must weave into existing memory: {report}"
        )
        bad = _post(port, "/api/teach", {"filename": "x.exe", "content_b64": content})
        assert bad["status"] == "error"
    finally:
        server.shutdown()


def test_page_teaches_by_file_not_by_text(served):
    """Words belong to the chat; iai learns from FILES. The page must offer a
    file dropzone and no free-text capture input."""
    _store, port = served
    _status, body = _get(port, "/")
    assert b"the brain will study it" in body
    assert b'type="file"' in body
    assert b"captext" not in body, "free-text capture input must not exist"

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_search_finds_memory_semantically(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "Ashby's law of requisite variety governs control.")
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        res = _post(port, "/api/search", {"q": "variety and control"})
        assert res["hits"], f"[{driver}] semantic search found nothing"
        top = res["hits"][0]
        assert top["id"] == rid
        assert top["cos"] > 0.4
        assert "surface" in top and "tier" in top
        assert _post(port, "/api/search", {"q": ""}) == {"hits": []}
    finally:
        server.shutdown()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_rescue_cancels_fading_and_untombstones(driver, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "A memory the user first faded, then changed their mind about.")
    server = make_server(store, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert _post(port, "/api/forget", {"id": rid})["status"] == "queued_for_forgetting"
        # Simulate the sleep pipeline having tombstoned it since.
        from iai_mcp.store import RECORDS_TABLE

        store.db.open_table(RECORDS_TABLE).update(
            where=f"id = '{rid}'",
            values={"tombstoned_at": datetime.now(timezone.utc), "live": 0},
        )
        _status, body = _get(port, "/api/graph")
        ghosts = [n for n in json.loads(body)["nodes"] if n.get("ghost")]
        assert ghosts and ghosts[0]["id"] == rid, (
            f"[{driver}] a tombstoned-but-not-erased memory must be visible as a ghost"
        )

        assert _post(port, "/api/rescue", {"id": rid})["status"] == "rescued"
        rec = store.get(UUID(rid))
        assert rec is not None
        assert rec.last_reviewed is not None, "rescue must stamp a fresh review"
        assert float(rec.centrality) == 1.0
        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT tombstoned_at FROM records WHERE id = ?", [rid]
            ).fetchone()
        assert row[0] is None, f"[{driver}] rescue must un-tombstone"
        assert any(
            p.get("cue") == "user-rescue" for p in (rec.provenance or [])
        ), "rescue must be traceable in provenance"
    finally:
        server.shutdown()


def test_daemon_action_is_bypass_safe(served):
    """No daemon socket -> an honest daemon_down answer, never an error page."""
    _store, port = served
    res = _post(port, "/api/daemon", {"action": "sleep"})
    assert res["status"] == "daemon_down", res
    res = _post(port, "/api/daemon", {"action": "nonsense"})
    assert res["status"] == "error"


class _FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def _lifecycle_view(tmp_path):
    from iai_mcp.brainview import BrainView

    store = MemoryStore(path=tmp_path)
    return BrainView(store)


def test_daemon_lifecycle_start_uses_service_manager(tmp_path, monkeypatch):
    """start/restart reach the platform service manager, never the socket."""
    import subprocess

    view = _lifecycle_view(tmp_path)
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    for action in ("start", "restart"):
        res = view.daemon_action(action)
        assert res == {"status": "ok", "action": action}, res
    assert len(seen) == 2
    import platform

    if platform.system() == "Darwin":
        assert all(cmd[0] == "launchctl" and "kickstart" in cmd for cmd in seen)
    elif platform.system() == "Linux":
        assert all(cmd[:2] == ["systemctl", "--user"] for cmd in seen)


def test_daemon_lifecycle_stop_tolerates_already_stopped(tmp_path, monkeypatch):
    import subprocess

    view = _lifecycle_view(tmp_path)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: _FakeProc(3, stderr="Could not signal: No such process"),
    )
    res = view.daemon_action("stop")
    assert res["status"] == "ok" and res.get("note") == "already stopped", res


def test_daemon_lifecycle_reports_manager_failure(tmp_path, monkeypatch):
    import subprocess

    view = _lifecycle_view(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: _FakeProc(1, stderr="permission denied"),
    )
    res = view.daemon_action("restart")
    assert res["status"] == "error" and "permission denied" in res["reason"], res


def test_daemon_lifecycle_missing_manager_binary_is_soft_error(tmp_path, monkeypatch):
    import subprocess

    view = _lifecycle_view(tmp_path)

    def raise_missing(cmd, **kw):
        raise FileNotFoundError("launchctl not found")

    monkeypatch.setattr(subprocess, "run", raise_missing)
    res = view.daemon_action("start")
    assert res["status"] == "error" and "not found" in res["reason"], res


def test_dashboard_page_ships_daemon_power_controls(served):
    _store, port = served
    status, body = _get(port, "/")
    assert status == 200
    page = body.decode("utf-8")
    assert "btn-power" in page and "btn-stop" in page
    assert "wake the subconscious" in page


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_shared_stores_coexist_in_process(driver, tmp_path, monkeypatch):
    from iai_mcp.hippo import AccessMode

    _select_driver(driver, monkeypatch)
    first = MemoryStore(
        path=tmp_path, access_mode=AccessMode.SHARED, persist_index=False,
    )
    _seed(first, "Two shared holders read the same hippocampus.")
    second = MemoryStore(
        path=tmp_path, access_mode=AccessMode.SHARED, persist_index=False,
    )
    server = make_server(second, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, body = _get(port, "/api/overview")
        overview = json.loads(body)
        assert status == 200
        assert overview["counts"]["episodic"] == 1
    finally:
        server.shutdown()
        second.close()
        first.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_only_persisting_handle_writes_hnsw(driver, tmp_path, monkeypatch):
    from iai_mcp.hippo import AccessMode

    _select_driver(driver, monkeypatch)
    shared_root = tmp_path / "shared"
    store = MemoryStore(
        path=shared_root, access_mode=AccessMode.SHARED, persist_index=False,
    )
    _seed(store, "A non-persisting handle must never write the index.")
    hnsw_path = store.db._hnsw_path
    store.db._save_index_atomic()
    assert not hnsw_path.exists(), "gated handle wrote the hnsw file"
    store.close()
    assert not hnsw_path.exists(), "close() bypassed the persist gate"

    excl_root = tmp_path / "exclusive"
    persisting = MemoryStore(path=excl_root)
    _seed(persisting, "The persisting handle owns the on-disk index.")
    hnsw_path_excl = persisting.db._hnsw_path
    persisting.db._save_index_atomic()
    assert hnsw_path_excl.exists(), "persisting handle must write the hnsw file"
    persisting.close()
    mtime_after_owner = hnsw_path_excl.stat().st_mtime_ns

    # A distinct fresh path avoids reopening excl_root: the lilli engine's
    # OS-level advisory file lock is not guaranteed released synchronously by
    # close() within one process (a pre-existing engine constraint unrelated
    # to persist_index), so this rail sticks to open-once-per-path.
    reader_root = tmp_path / "reader"
    reader = MemoryStore(
        path=reader_root, access_mode=AccessMode.SHARED, persist_index=False,
    )
    _seed(reader, "A non-persisting reader on its own path must stay gated too.")
    reader_hnsw_path = reader.db._hnsw_path
    reader.db._save_index_atomic()
    assert not reader_hnsw_path.exists(), "gated handle wrote the hnsw file"
    reader.close()
    assert hnsw_path_excl.stat().st_mtime_ns == mtime_after_owner, (
        "an unrelated non-persisting handle rewrote the owner's hnsw file"
    )


_CHILD_SHARED_HOLDER = """
import sys
from pathlib import Path

from iai_mcp.hippo import AccessMode, HippoDB

db = HippoDB(Path(sys.argv[1]), access_mode=AccessMode.SHARED)
print("ready", flush=True)
sys.stdin.readline()
db.close()
"""


def test_cross_process_shared_coexists_exclusive_conflicts(tmp_path, monkeypatch):
    import os as _os
    import subprocess
    import sys as _sys

    from iai_mcp.hippo import AccessMode, HippoLockHeldError

    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    # Exercises HippoDB's own flock (access_mode), which is driver-independent
    # architecture — pinned to stdlib in BOTH processes regardless of the
    # ambient env: the lilli engine holds its own single-process OS lock below
    # this flock (no two processes can hold engine connections at once), so a
    # lilli-driver run would fail at the engine layer before ever reaching the
    # flock under test. Engine semantics are covered by test_hippo_multiprocess.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    child_env = dict(_os.environ)
    child = subprocess.Popen(  # noqa: S603 -- argv list, no shell
        [_sys.executable, "-c", _CHILD_SHARED_HOLDER, str(tmp_path)],
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = child.stdout.readline()
        assert line.strip() == "ready", f"child holder failed to open: {line!r}"

        store = MemoryStore(
            path=tmp_path, access_mode=AccessMode.SHARED, persist_index=False,
        )
        store.close()

        with pytest.raises(HippoLockHeldError):
            MemoryStore(path=tmp_path)
    finally:
        try:
            child.stdin.close()
        except OSError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=10)


def _fake_daemon_socket(sock_path, response_result):
    """Minimal JSON-RPC unix server: answers every request with the given
    result. Returns (thread, stop_fn)."""
    import socket as _socket
    import threading as _threading

    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(2)
    srv.settimeout(0.2)
    stop = _threading.Event()

    def loop():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf or b"{}")
                resp = {"jsonrpc": "2.0", "id": req.get("id"),
                        "result": response_result(req)}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            finally:
                conn.close()
        srv.close()

    t = _threading.Thread(target=loop, daemon=True)
    t.start()
    return t, stop.set


def test_view_relays_to_live_daemon_socket():
    """Socket present -> every data verb goes to the daemon (single writer);
    the HTTP process opens no store at all."""
    import shutil
    import tempfile

    from iai_mcp.brainview import BrainView

    # AF_UNIX sun_path is ~104 bytes on macOS; pytest tmp_path is too deep.
    root = Path(tempfile.mkdtemp(prefix="bv", dir="/tmp"))
    seen = []

    def answer(req):
        seen.append(req)
        return {"counts": {"episodic": 42}, "relayed": True}

    _t, stop = _fake_daemon_socket(root / ".daemon.sock", answer)
    try:
        view = BrainView(store_root=root)
        out = view.overview()
        assert out.get("relayed") is True and out["counts"]["episodic"] == 42
        assert view.store is None, "relay mode must not open a direct store"
        assert seen and seen[0]["method"] == "brain_view"
        assert seen[0]["params"]["verb"] == "overview"
    finally:
        stop()
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_view_reads_daemon_free_without_holding_the_writer(driver, tmp_path, monkeypatch):
    """No socket -> reads serve off the RO snapshot; the writable store is
    never opened for a read (a polling window must not block a daemon boot)."""
    from iai_mcp.brainview import BrainView

    _select_driver(driver, monkeypatch)
    seed_store = MemoryStore(path=tmp_path)
    _seed(seed_store, "Direct fallback still answers from the hippocampus.")
    seed_store.close()

    view = BrainView(store_root=tmp_path)
    out = view.overview()
    assert out["counts"]["episodic"] == 1
    assert view.store is None, "a read must not open the writable store"


def test_stale_socket_cooldown_then_direct(tmp_path):
    """A socket FILE with no listener must not wedge the view in relay mode:
    one failed relay buys a cooldown, the next call serves direct."""
    import shutil
    import socket as _socket
    import tempfile

    from iai_mcp.brainview import BrainView

    short = Path(tempfile.mkdtemp(prefix="bv", dir="/tmp"))
    tmp_path = short
    sock_path = tmp_path / ".daemon.sock"
    holder = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    holder.bind(str(sock_path))
    holder.close()  # path stays on disk as a dead socket file

    seed_store = MemoryStore(path=tmp_path)
    _seed(seed_store, "Cooldown flips a dead relay to direct service.")
    seed_store.close()

    view = BrainView(store_root=tmp_path)
    first = view.overview()
    assert first["counts"]["episodic"] == 1, (
        f"dead relay must fall through to the RO snapshot, got {first}"
    )
    assert view.store is None, "the RO fallback must not open a direct store"
    second = view.overview()
    assert second["counts"]["episodic"] == 1, (
        "post-cooldown reads keep serving off the RO snapshot"
    )
    assert view.store is None, "reads never hold the writer"
    shutil.rmtree(short, ignore_errors=True)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_core_brain_view_verb_serves_daemon_side(driver, tmp_path, monkeypatch):
    """The daemon-side half of the relay: core dispatch runs the view verbs
    against the live store, with a hard verb whitelist."""
    from iai_mcp.core import dispatch

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed(store, "Server-side view answers over the daemon's own store.")
    out = dispatch(store, "brain_view", {"verb": "overview", "kwargs": {}})
    assert out["counts"]["episodic"] == 1
    bad = dispatch(store, "brain_view", {"verb": "shutdown", "kwargs": {}})
    assert bad["status"] == "error" and "unknown brain verb" in bad["reason"]
    # the server-side view is pinned direct: it must never try to relay to
    # its own socket
    from iai_mcp.brainview import view_for_store

    assert view_for_store(store)._pinned_direct is True
    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ro_snapshot_fallback_serves_reads_when_writer_fenced(
    driver, tmp_path, monkeypatch
):
    """The brain stays visible through its own consolidation: when neither the
    relay nor a writable open is available, read verbs serve from a lock-free
    RO snapshot — counts, decrypted surfaces, ghosts, economy."""
    from iai_mcp.brainview import BrainView

    _select_driver(driver, monkeypatch)
    seed_store = MemoryStore(path=tmp_path)
    _seed(seed_store, "The observable brain keeps observing itself.")
    seed_store.close()

    view = BrainView(store_root=tmp_path)
    # Force the fenced-writer situation: writable open refused, no socket.
    def refuse_open():
        raise RuntimeError("writable open fenced for the test")

    monkeypatch.setattr(view, "_open_direct", refuse_open)

    out = view.overview()
    assert out["counts"]["episodic"] == 1, out
    graph = view.graph()
    assert graph["nodes"], "RO fallback must serve nodes"
    assert graph["nodes"][0]["surface"] == (
        "The observable brain keeps observing itself."
    ), "RO fallback must serve DECRYPTED verbatim surfaces"
    eco = view.economy()
    assert "packs_served" in eco and "agent_searches" in eco
    ev = view.events()
    assert isinstance(ev, list)
    # writes stay honest — no silent RO write path
    res = view.capture("must not be silently swallowed")
    assert res.get("status") == "memory_busy", res


def test_socket_observation_does_not_reset_activity(tmp_path):
    """brain_view READ verbs are observation: they must not bump the
    activity timestamp that defers the sleep pipeline."""
    import asyncio

    from iai_mcp.socket_server import SocketServer

    store = MemoryStore(path=tmp_path)
    server = SocketServer(store)

    class _R:
        def __init__(self, lines):
            self._lines = list(lines)
        def at_eof(self):
            return not self._lines
        async def readline(self):
            return self._lines.pop(0) if self._lines else b""

    class _W:
        def __init__(self):
            self.out = b""
        def write(self, b):
            self.out += b
        async def drain(self):
            return None
        def close(self):
            return None
        async def wait_closed(self):
            return None

    def run(req_obj):
        server.last_activity_ts = 1000.0
        r = _R([(json.dumps(req_obj) + "\n").encode()])
        w = _W()
        asyncio.run(server.handle(r, w))
        return server.last_activity_ts

    ts_after_read = run({
        "jsonrpc": "2.0", "id": 1, "method": "brain_view",
        "params": {"verb": "overview", "kwargs": {}},
    })
    assert ts_after_read == 1000.0, "observation must not reset activity"

    # browse is the folder-explorer poll (every 90s) — it MUST be observation,
    # or an open dashboard starves consolidation forever (the regression this
    # locks against).
    ts_after_browse = run({
        "jsonrpc": "2.0", "id": 3, "method": "brain_view",
        "params": {"verb": "browse", "kwargs": {}},
    })
    assert ts_after_browse == 1000.0, "browse poll must not reset activity"

    ts_after_write = run({
        "jsonrpc": "2.0", "id": 2, "method": "brain_view",
        "params": {"verb": "capture", "kwargs": {"text": "real user work"}},
    })
    assert ts_after_write != 1000.0, "a write IS user work and must reset it"
    store.close()


def test_idle_view_releases_the_writer(tmp_path, monkeypatch):
    """An unwatched dashboard must yield the store: the engine admits one
    writer per file, and a daemon rebooting overnight must not crash against
    a viewer nobody is looking at."""
    import time as _time

    from iai_mcp.brainview import BrainView

    monkeypatch.setattr(BrainView, "_IDLE_RELEASE_S", 0.3)
    seed = MemoryStore(path=tmp_path)
    _seed(seed, "Politeness is holding the lock only while serving.")
    seed.close()

    view = BrainView(store_root=tmp_path)
    out = view.overview()
    assert out["counts"]["episodic"] == 1
    assert view.store is None, (
        "reads must NEVER open the writable store — a polling dashboard "
        "would block a rebooting daemon forever"
    )
    res = view.capture("a write verb briefly holds the writer")
    assert res.get("status") in ("inserted", "reinforced"), res
    assert view.store is not None
    deadline = _time.time() + 30
    while view.store is not None and _time.time() < deadline:
        _time.sleep(0.5)
    assert view.store is None, "idle janitor must release the writer after writes stop"
    # reads keep answering off the RO snapshot the whole time
    again = view.overview()
    assert again["counts"]["episodic"] == 2


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_browse_folders_reflect_the_brains_own_groupings(driver, tmp_path, monkeypatch):
    """The explorer invents no taxonomy: time buckets, communities, studied
    documents, sessions and curation state — all read from what the brain
    already tracks."""
    from iai_mcp.brainview import BrainView
    from iai_mcp.store import RECORDS_TABLE

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "Ashby's law: only variety absorbs variety.", session="s-one")
    _seed(store, "Groceries: rye bread and tomatoes.", session="s-two")
    store.db.open_table(RECORDS_TABLE).update(
        where=f"id = '{rid}'", values={"pinned": True},
    )
    view = BrainView(store)

    tree = view.browse()["folders"]
    today = next(b for b in tree["time"] if b["key"] == "today")
    assert today["n"] == 2, tree["time"]
    assert next(sp for sp in tree["special"] if sp["key"] == "pinned")["n"] == 1
    assert {sf["key"] for sf in tree["sessions"]} == {"s-one", "s-two"}

    items = view.browse({"kind": "time", "key": "today"})["items"]
    assert len(items) == 2
    assert any("variety absorbs variety" in it["surface"] for it in items), (
        "folder contents must carry decrypted verbatim surfaces"
    )

    pinned_items = view.browse({"kind": "special", "key": "pinned"})["items"]
    assert [it["id"] for it in pinned_items] == [rid]

    sess = view.browse({"kind": "session", "key": "s-two"})["items"]
    assert len(sess) == 1 and "Groceries" in sess[0]["surface"]


@pytest.mark.parametrize("driver", ["stdlib"])
def test_browse_teaching_folder_lists_studied_document(driver, tmp_path, monkeypatch):
    from iai_mcp.brainview import BrainView
    from iai_mcp.study import study_document

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    doc = tmp_path / "cybernetics-notes.md"
    doc.write_text(
        "Feedback loops regulate essential variables.\n\n"
        "Homeostats adapt by changing their own structure.\n",
        encoding="utf-8",
    )
    report = study_document(store, path=doc, session_id="study")
    assert report.get("inserted", 0) + report.get("reinforced", 0) > 0, report
    from iai_mcp.store import flush_record_buffer

    flush_record_buffer(store)
    view = BrainView(store)
    tree = view.browse()["folders"]
    assert tree["teachings"], "a studied document must appear as a folder"
    key = tree["teachings"][0]["key"]
    assert key.startswith("doc:")
    items = view.browse({"kind": "teaching", "key": key})["items"]
    assert any("Feedback loops" in it["surface"] for it in items)


def test_browse_page_ships_explorer(served):
    _store, port = served
    _status, body = _get(port, "/")
    page = body.decode("utf-8")
    assert "foldtree" in page and "openFolder" in page
    _status, body = _get(port, "/api/browse")
    tree = json.loads(body)
    assert "folders" in tree


def test_foreign_origin_and_host_are_rejected(served):
    """Loopback is not browser-proof: a hostile web page can POST at
    127.0.0.1 (CSRF) and DNS-rebinding can alias a foreign Host here.
    Both must bounce with 403; local no-Origin clients keep working."""
    import urllib.error

    _store, port = served

    def req(path, method="POST", headers=None, data=b"{}"):
        r = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data if method == "POST" else None,
            headers={"Content-Type": "application/json", **(headers or {})},
            method=method,
        )
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    code, body = req("/api/forget", headers={"Origin": "https://evil.example"})
    assert code == 403 and body["reason"] == "forbidden origin"
    code, _ = req("/api/overview", method="GET",
                  headers={"Host": "attacker.example"})
    assert code == 403
    # leg

def test_http_hardening_content_length_and_methods(served):
    """DoS + error-shape hardening: bad Content-Length is a JSON 400 (not a
    hung thread), and unsupported methods are JSON 405 (not an HTML 501)."""
    import http.client

    _store, port = served
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    # negative content length -> 400 JSON, promptly
    conn.putrequest("POST", "/api/capture")
    conn.putheader("Content-Length", "-1")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    conn.send(b"")
    r = conn.getresponse()
    assert r.status == 400
    body = json.loads(r.read())
    assert body["status"] == "error"
    conn.close()
    # unsupported method -> 405 JSON
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("DELETE", "/api/forget", body=b"{}")
    r = conn.getresponse()
    assert r.status == 405
    assert "application/json" in r.getheader("Content-Type", "")
    conn.close()


def test_browse_rejects_non_dict_folder(served):
    _store, port = served
    res = _post(port, "/api/browse", {"folder": "not-a-dict"})
    assert res["status"] == "error" and "object" in res["reason"], res


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_accepts_code_suffix_and_wrapped_base64(driver, tmp_path, monkeypatch):
    """The dashboard teach must accept what the CLI accepts (code files) and
    tolerate line-wrapped base64 (the Unix `base64` default)."""
    import base64 as _b64

    from iai_mcp.brainview import BrainView

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    view = BrainView(store)
    body = b"def homeostat():\n    return 'regulates essential variables'\n"
    wrapped = _b64.encodebytes(body).decode("ascii")  # 76-col wrapped, has \n
    res = view.teach("regulator.py", wrapped)
    assert res.get("status") == "studied", res


def test_ro_read_fresh_install_returns_empty_shell(tmp_path):
    """A brand-new install (no store file) shows an empty brain, not busy."""
    from iai_mcp.brainview import BrainView

    fresh = tmp_path / "never-booted"
    view = BrainView(store_root=fresh)
    ov = view.overview()
    assert ov.get("counts", {}).get("episodic") == 0, ov
    assert view.graph()["nodes"] == []
    assert "folders" in view.browse()


def test_janitor_never_closes_store_mid_write(tmp_path, monkeypatch):
    """The idle janitor must not release a store a direct write is still
    using: a long teach (> idle window) must complete against a live store."""
    import threading
    import time as _time

    from iai_mcp.brainview import BrainView

    monkeypatch.setattr(BrainView, "_IDLE_RELEASE_S", 0.2)
    seed = MemoryStore(path=tmp_path)
    _seed(seed, "A store the janitor must not yank mid-write.")
    seed.close()

    view = BrainView(store_root=tmp_path)
    view.overview()  # opens a direct store

    observed_none = {"hit": False}

    def slow_capture_direct(text):
        # simulate a long write: sleep past the idle window while "active"
        _time.sleep(1.0)
        if view.store is None:
            observed_none["hit"] = True
        return {"status": "ok"}

    monkeypatch.setattr(view, "capture_direct", slow_capture_direct)
    monkeypatch.setattr(view, "_relay_alive", lambda ignore_cooldown=False: False)

    done = threading.Event()

    def run():
        view._call("capture", {"text": "x"})
        done.set()

    threading.Thread(target=run, daemon=True).start()
    # give the janitor multiple ticks to (wrongly) fire during the write
    assert done.wait(5.0), "capture did not finish"
    assert not observed_none["hit"], "janitor closed the store mid-write"


@pytest.mark.parametrize(
    ("action", "expected_type", "reply"),
    [
        ("sleep", "user_initiated_sleep", {"ok": True, "state": "TRANSITIONING"}),
        ("sleep", "user_initiated_sleep", {"ok": False, "reason": "already_sleeping"}),
        ("wake", "force_wake", {"ok": True, "reason": "wake_queued"}),
        ("consolidate", "force_rem", {"ok": True, "reason": "rem_queued"}),
    ],
)
def test_daemon_action_uses_control_protocol(action, expected_type, reply, tmp_path):
    """sleep/wake/consolidate are CONTROL messages (type-protocol). A fake
    daemon socket must receive `type`, not a JSON-RPC `method`."""
    import shutil
    import socket as _socket
    import tempfile
    import threading

    from iai_mcp.brainview import BrainView

    root = tempfile.mkdtemp(prefix="bv", dir="/tmp")
    sock_path = f"{root}/.daemon.sock"
    seen = {}
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1); srv.settimeout(5)

    def serve():
        try:
            conn, _ = srv.accept()
            buf = b""
            while not buf.endswith(b"\n"):
                c = conn.recv(4096)
                if not c:
                    break
                buf += c
            seen["req"] = json.loads(buf)
            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            conn.close()
        except OSError:
            pass

    t = threading.Thread(target=serve, daemon=True); t.start()
    try:
        view = BrainView(store_root=root)
        res = view.daemon_action(action)
        t.join(3)
        assert seen["req"]["type"] == expected_type, seen["req"]
        assert "method" not in seen["req"], "control msg must not be JSON-RPC"
        if action == "sleep":
            assert seen["req"]["reason"] == "BrainView dashboard control"
        assert res["status"] == "ok", res
    finally:
        srv.close()
        shutil.rmtree(root, ignore_errors=True)


def test_consolidate_button_in_page(served):
    _store, port = served
    _status, body = _get(port, "/")
    page = body.decode("utf-8")
    assert "btn-consolidate" in page and "sort memories now" in page


def test_lifecycle_label_reads_current_state(tmp_path):
    from iai_mcp.brainview import BrainView

    (tmp_path / "lifecycle_state.json").write_text(
        json.dumps({"current_state": "SLEEP"}), encoding="utf-8",
    )
    view = BrainView(store_root=tmp_path)
    assert view._lifecycle_label() == "SLEEP"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pin_toggle_protects_and_releases(driver, tmp_path, monkeypatch):
    """Pin protects (erasure gate + lossy merges refuse); unpin releases;
    both leave a provenance trace."""
    from iai_mcp.brainview import BrainView

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rid = _seed(store, "A fact worth protecting forever.")
    view = BrainView(store)

    res = view.pin(rid, True)
    assert res["status"] == "pinned", res
    rec = store.get(UUID(rid))
    assert rec.pinned and rec.never_merge
    assert any(p.get("cue") == "user-pin" for p in rec.provenance or [])
    refused = view.forget_hint(rid)
    assert refused["status"] == "refused", "pin must block the forget hint"

    res = view.pin(rid, False)
    assert res["status"] == "unpinned", res
    rec = store.get(UUID(rid))
    assert not rec.pinned and not rec.never_merge


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_surface_returns_full_decrypted_text(driver, tmp_path, monkeypatch):
    from iai_mcp.brainview import BrainView

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    long_text = "Verbatim recall is guaranteed. " * 30
    rid = _seed(store, long_text.strip())
    view = BrainView(store)
    res = view.surface(rid)
    assert res["surface"] == long_text.strip(), "full text, not the excerpt"


def test_pin_copy_buttons_in_page(served):
    _store, port = served
    _status, body = _get(port, "/")
    page = body.decode("utf-8")
    assert "btn-pin" in page and "btn-copy" in page
    assert "/api/pin" in page and "/api/surface" in page
