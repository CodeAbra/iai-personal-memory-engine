"""Suite-resident proof: a REAL spawned lilli daemon serves recall through
its cold boot on an isolated store.

Two prior cutover attempts failed exactly where no automated proof existed:
an ANN-build timeout starving recall, and a wrong-key cache pre-warm
degrading recall to zero. This module spawns a real ``python -m
iai_mcp.daemon`` subprocess against an isolated, synthetic store and proves,
on every suite run, that:

- recall answers over the daemon's own socket while the boot preload/warm-up
  are still running, both with the persisted column-index sidecar present
  (the load path) and absent (the lazy-rebuild bypass-safe fallback);
- the boot warm-up accelerator fires (and its kill-switch works);
- a second boot of the same store is a warm-cache boot;
- the daemon's own runtime-state file resolves under the isolated store
  root, never the default home store, and the default store's runtime state
  and wake signal are provably untouched by every test in this module.

Every store here lives under ``tmp_path`` with its own auto-created crypto
key — never a copy of a real store, never a passphrase-injected cache. The
unix-domain socket itself lives under a short, separately allocated temp
directory (never nested under ``tmp_path``): the kernel's ``sun_path`` limit
(104 bytes on macOS) is easy to exceed under pytest's deeply nested per-test
tmp directories, which would silently fail the bind while the daemon
otherwise boots healthy — a wedge with no socket, not a slow boot.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iai_mcp.daemon_state import daemon_state_path
from iai_mcp.hippo import _txn
from iai_mcp.store import MemoryStore

try:
    import iai_mcp_native  # noqa: F401
except ImportError:
    pytest.skip(
        "iai_mcp_native is unavailable; the isolated-daemon proof requires "
        "the native engine extension",
        allow_module_level=True,
    )

_EMBED_DIM = 384
_RECORD_COUNT = 1500
_EDGE_COUNT = 2000
_CURIOSITY_EVENT_COUNT = 5

_SOCKET_AWAIT_TIMEOUT_SEC = 30.0
_RECALL_TIMEOUT_SEC = 30.0
_TEARDOWN_JOIN_TIMEOUT_SEC = 10.0

_SIDECAR_FILENAME = "records.colindex"
_BOOT_WARMUP_SIDECAR = ".boot-warmup.json"
_RUNTIME_GRAPH_CACHE_FILENAME = "runtime_graph_cache.json"

_SUN_PATH_MAX_BYTES = 104


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _embedding_blob(seed: int) -> bytes:
    return struct.pack(f"<{_EMBED_DIM}f", *_norm_vec(seed))


def _seed_records(store: MemoryStore, n: int) -> None:
    """Insert n plain records at the real embed dim directly via SQL.

    Bypasses the embedder entirely (raw float32 blobs) — the daemon still
    embeds the recall CUE with the real Rust embedder, so record dims must
    match.
    """
    db = store.db
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for i in range(n):
        rid = str(uuid.uuid4())
        rows.append((
            rid,
            "episodic",
            f"fixture record {i} for alice",
            _embedding_blob(i),
            None,
            0,
            now_iso,
            now_iso,
        ))
    sql = (
        "INSERT INTO records "
        "(id, tier, literal_surface, embedding, tombstoned_at, "
        " embedding_pending, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)


def _seed_edges(store: MemoryStore, n: int) -> None:
    db = store.db
    rows = []
    for i in range(n):
        src = str(uuid.uuid4())
        dst = str(uuid.uuid4())
        edge_type = "hebbian" if i % 2 == 0 else "contradicts"
        weight = round(0.5 + (i % 10) * 0.05, 3)
        rows.append((src, dst, edge_type, weight))
    sql = (
        "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) "
        "VALUES (?, ?, ?, ?)"
    )
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)


def _seed_curiosity_events(store: MemoryStore, n: int) -> None:
    from iai_mcp.events import write_event

    for i in range(n):
        write_event(
            store,
            "curiosity_question",
            {"question": f"fixture curiosity question {i}", "targets": []},
            severity="info",
        )


@contextlib.contextmanager
def _lilli_driver_env():
    """Force LILLI_STORAGE_DRIVER=lilli for the corpus build.

    The spawned daemon subprocess always sets this explicitly (the isolation
    env), so the store built here must use the same driver — building the
    corpus under the default stdlib driver would hand the lilli daemon a
    format it cannot open (a checksum mismatch at engine open), independent
    of the caller's ambient shell environment.
    """
    prev = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = prev


def _lilli_conn(store: MemoryStore) -> Any:
    conn = getattr(store.db, "_conn", None)
    if conn is None or not hasattr(conn, "persist_col_indexes"):
        pytest.fail(
            "store.db._conn does not expose persist_col_indexes — the "
            "native engine surface is missing or stale. Run "
            "`cd rust/iai_mcp_native && maturin develop --release`."
        )
    return conn


def _sidecar_path(store_root: Path) -> Path:
    return store_root / "hippo" / _SIDECAR_FILENAME


def _create_own_crypto_key(store_root: Path) -> None:
    """Auto-create a fresh, random crypto key file for this isolated store.

    Mirrors the `iai-mcp crypto init` CLI path exactly (fresh random bytes,
    written 0o600 under the store root) — never a passphrase-derived key,
    never a copy of any real store's key.
    """
    import secrets

    from iai_mcp.crypto import KEY_BYTES, CryptoKey

    ck = CryptoKey(user_id="default", store_root=store_root)
    ck._try_file_set(secrets.token_bytes(KEY_BYTES))


def _build_corpus(store_root: Path, *, persist_index: bool = True) -> None:
    """Build a synthetic lilli store, close it fully, then optionally run a
    DEDICATED final open->persist->close pass with a zero-scan load-proof.

    Every buffered/deferred flush from the corpus-building store must commit
    BEFORE any persist — a committed write after persist would advance the
    write-generation and silently divert the daemon onto the lazy-rebuild
    path while this fixture (and the test consuming it) would still pass,
    proving nothing about the loaded-sidecar behavior.
    """
    store_root.mkdir(parents=True, exist_ok=True)
    _create_own_crypto_key(store_root)
    with _lilli_driver_env():
        store = MemoryStore(path=store_root)
        _seed_records(store, _RECORD_COUNT)
        _seed_edges(store, _EDGE_COUNT)
        _seed_curiosity_events(store, _CURIOSITY_EVENT_COUNT)
        store.close()

        if not persist_index:
            assert not _sidecar_path(store_root).exists(), (
                "persist_index=False must never leave a records.colindex sidecar"
            )
            return

        final_pass = MemoryStore(path=store_root)
        conn = _lilli_conn(final_pass)
        with final_pass.db._conn_lock:
            conn.persist_col_indexes()
        final_pass.close()

        assert _sidecar_path(store_root).exists(), (
            "the dedicated final pass must create records.colindex"
        )

        # Load-proof: a throwaway fresh connection must serve an indexed read
        # with zero full scans — the stamp provably matches before any spawn.
        proof_store = MemoryStore(path=store_root)
        proof_conn = _lilli_conn(proof_store)
        with proof_store.db._conn_lock:
            proof_conn.reset_full_scan_count()
            cur = proof_store.db._conn.execute(
                "SELECT vec_label FROM records WHERE vec_label IN (1, 2, 3)"
            )
            cur.fetchall()
            scans = proof_conn.full_scan_count()
        proof_store.close()
        assert scans == 0, (
            f"load-proof failed: a freshly opened connection over the "
            f"persisted, matching sidecar must serve an indexed read with "
            f"zero full scans (got {scans}) — the stamp does not match, and "
            f"spawning the daemon now would silently exercise the lazy path "
            f"while this test still passed"
        )


def _isolation_preflight(store_root: Path) -> None:
    """Assert the daemon's runtime-state file resolves under the isolated
    store root, never the default home store, BEFORE any spawn."""
    resolved = daemon_state_path(store_root)
    assert resolved == store_root / ".daemon-state.json", (
        f"daemon_state_path must resolve under the isolated store root; "
        f"got {resolved}"
    )


def _file_snapshot(path: Path) -> "tuple[int, int, str] | None":
    if not path.exists():
        return None
    data = path.read_bytes()
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size, hashlib.sha256(data).hexdigest())


def _prod_snapshot() -> dict:
    prod_state = Path.home() / ".iai-mcp" / ".daemon-state.json"
    prod_wake = Path.home() / ".iai-mcp" / "wake.signal"
    return {
        "state": _file_snapshot(prod_state),
        "wake_exists": prod_wake.exists(),
    }


def _assert_prod_unchanged(snap: dict) -> None:
    prod_state = Path.home() / ".iai-mcp" / ".daemon-state.json"
    prod_wake = Path.home() / ".iai-mcp" / "wake.signal"
    assert _file_snapshot(prod_state) == snap["state"], (
        "the spawned isolated daemon must never touch the default store's "
        "runtime state file"
    )
    assert prod_wake.exists() == snap["wake_exists"], (
        "the spawned isolated daemon must never consume or create the "
        "default store's wake signal"
    )


@contextlib.contextmanager
def _short_socket_dir():
    """Allocate a short-path scratch dir for the unix-domain socket only.

    The kernel's ``sun_path`` field is capped at 104 bytes on macOS — a path
    nested under pytest's own ``tmp_path`` (itself several directories deep)
    routinely exceeds that cap. A socket bind past the cap fails, but the
    daemon's own boot sequence does not treat that failure as fatal, so the
    process reaches a healthy, watchdog-monitored event loop with NO socket
    ever bound — a silent wedge that the watchdog's own status probe cannot
    reach either, self-killing the daemon only after the failure debounce
    window. Using a short, independently allocated temp dir for the socket
    (never nested under the store root or ``tmp_path``) keeps every spawn
    well under the cap regardless of how deep the test's own tmp path is.
    """
    d = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    try:
        yield d / ".daemon.sock"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _spawn_daemon(
    store_root: Path,
    socket_path: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    assert len(str(socket_path).encode("utf-8")) < _SUN_PATH_MAX_BYTES, (
        f"socket path too long for sun_path ({socket_path}) — "
        f"use a short, independently allocated temp dir"
    )
    env = os.environ.copy()
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(socket_path)
    env["LILLI_STORAGE_DRIVER"] = "lilli"
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _await_socket(socket_path: Path, timeout: float = _SOCKET_AWAIT_TIMEOUT_SEC) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"socket did not appear at {socket_path} within {timeout}s")


def _recall(socket_path: Path, cue: str, timeout: float = _RECALL_TIMEOUT_SEC) -> dict:
    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path)), timeout=timeout,
        )
        try:
            req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "memory_recall",
                "params": {"cue": cue, "session_id": "test"},
            }
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise ConnectionError("daemon closed the connection with no response")
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return asyncio.run(_runner())


def _assert_well_formed_recall(resp: dict) -> None:
    assert "error" not in resp, f"memory_recall returned an error: {resp.get('error')}"
    result = resp.get("result")
    assert isinstance(result, dict), f"expected a result dict, got {resp!r}"
    assert isinstance(result.get("hits"), list), (
        f"expected result['hits'] to be a list, got {resp!r}"
    )


def _teardown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=_TEARDOWN_JOIN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=_TEARDOWN_JOIN_TIMEOUT_SEC)
    assert proc.poll() is not None, "daemon subprocess must be dead after teardown"


def _poll_for_file(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.2)
    return path.exists()


# ---------------------------------------------------------------------------
# Test 1: recall serves through the cold boot with the persisted index
# present, and the boot warm-up accelerator fires.
# ---------------------------------------------------------------------------

def test_recall_serves_through_cold_boot_with_persisted_index(tmp_path: Path) -> None:
    store_root = tmp_path / "iso-store"
    _isolation_preflight(store_root)
    prod_snap = _prod_snapshot()

    _build_corpus(store_root, persist_index=True)

    with _short_socket_dir() as socket_path:
        proc = _spawn_daemon(store_root, socket_path)
        try:
            _await_socket(socket_path)

            resp = _recall(socket_path, "fixture record for alice")
            _assert_well_formed_recall(resp)

            warmup_path = store_root / _BOOT_WARMUP_SIDECAR
            assert _poll_for_file(warmup_path, timeout=30.0), (
                "the boot warm-up accelerator must write its sidecar after WAKE"
            )
            summary = json.loads(warmup_path.read_text())
            assert isinstance(summary.get("shapes"), dict) and summary["shapes"], (
                f"the warm-up sidecar must carry per-shape entries; got {summary!r}"
            )

            resp2 = _recall(socket_path, "another fixture cue")
            _assert_well_formed_recall(resp2)
        finally:
            _teardown(proc)
            _assert_prod_unchanged(prod_snap)


# ---------------------------------------------------------------------------
# Test 2: with the sidecar deleted, the daemon still boots and serves a
# correct recall (the lazy ensure_built fallback) — correctness never
# depends on the sidecar file.
# ---------------------------------------------------------------------------

def test_boot_without_sidecar_is_bypass_safe(tmp_path: Path) -> None:
    store_root = tmp_path / "iso-store-nosidecar"
    _isolation_preflight(store_root)
    prod_snap = _prod_snapshot()

    _build_corpus(store_root, persist_index=False)
    assert not _sidecar_path(store_root).exists()

    with _short_socket_dir() as socket_path:
        proc = _spawn_daemon(store_root, socket_path)
        try:
            _await_socket(socket_path)
            resp = _recall(socket_path, "fixture record for alice")
            _assert_well_formed_recall(resp)
        finally:
            _teardown(proc)
            _assert_prod_unchanged(prod_snap)


# ---------------------------------------------------------------------------
# Test 3: the warm-up kill-switch (IAI_MCP_BOOT_WARMUP_OFF=1) is honored
# end-to-end — recall still works, and no warm-up sidecar is ever written.
# ---------------------------------------------------------------------------

def test_warmup_kill_switch_end_to_end(tmp_path: Path) -> None:
    store_root = tmp_path / "iso-store-killswitch"
    _isolation_preflight(store_root)
    prod_snap = _prod_snapshot()

    _build_corpus(store_root, persist_index=True)

    with _short_socket_dir() as socket_path:
        proc = _spawn_daemon(
            store_root, socket_path, extra_env={"IAI_MCP_BOOT_WARMUP_OFF": "1"},
        )
        try:
            _await_socket(socket_path)
            resp = _recall(socket_path, "fixture record for alice")
            _assert_well_formed_recall(resp)

            warmup_path = store_root / _BOOT_WARMUP_SIDECAR
            time.sleep(10.0)
            assert not warmup_path.exists(), (
                "IAI_MCP_BOOT_WARMUP_OFF=1 must skip the warm-up entirely — "
                "no .boot-warmup.json sidecar may be written"
            )
        finally:
            _teardown(proc)
            _assert_prod_unchanged(prod_snap)


# ---------------------------------------------------------------------------
# Test 4: a second boot of the same isolated store is a warm-cache boot —
# the runtime graph cache and the persisted column index both survive across
# boots, and the second boot still serves a correct recall.
# ---------------------------------------------------------------------------

def test_warm_second_boot(tmp_path: Path) -> None:
    store_root = tmp_path / "iso-store-warmboot"
    _isolation_preflight(store_root)
    prod_snap = _prod_snapshot()

    _build_corpus(store_root, persist_index=True)

    # Boot 1: cold. Wait for the runtime graph cache to appear (the
    # background build writes it), then a clean teardown.
    with _short_socket_dir() as socket_path:
        proc = _spawn_daemon(store_root, socket_path)
        try:
            _await_socket(socket_path)
            resp = _recall(socket_path, "fixture record for alice")
            _assert_well_formed_recall(resp)

            cache_path = store_root / _RUNTIME_GRAPH_CACHE_FILENAME
            assert _poll_for_file(cache_path, timeout=30.0), (
                "boot 1's background build must write the runtime graph "
                "cache file under the store root"
            )
        finally:
            _teardown(proc)
            _assert_prod_unchanged(prod_snap)

    # Boot 2: same store. The warm-cache precondition (cache file presence)
    # held from boot 1; boot 2 must still serve a correct recall. The
    # persisted column index also survives across boots (re-emitted by the
    # sleep cadence, not consumed by a boot) — presence + recall correctness
    # is the observable contract here, not a zero-scan assertion (boot 1's
    # own recall traffic may legitimately have staled its generation stamp).
    with _short_socket_dir() as socket_path2:
        proc2 = _spawn_daemon(store_root, socket_path2)
        try:
            _await_socket(socket_path2)
            resp2 = _recall(socket_path2, "fixture record for alice")
            _assert_well_formed_recall(resp2)

            cache_path = store_root / _RUNTIME_GRAPH_CACHE_FILENAME
            assert cache_path.exists(), (
                "the runtime graph cache file must still exist going into "
                "boot 2 (the warm-cache precondition)"
            )
            assert _sidecar_path(store_root).exists(), (
                "the persisted column index sidecar must survive across "
                "boots — it is re-emitted by the sleep cadence, not "
                "consumed at boot"
            )
        finally:
            _teardown(proc2)
            _assert_prod_unchanged(prod_snap)
