"""Shared repro harness for the prod-scale isolated-daemon warm-recall gate.

Reused by the headline regression test and the phase's later verification
pass. Every helper here operates on a COPY of a preserved snapshot, never the
live store: the caller is responsible for pointing ``locate_repro_source`` at
a fixed, non-live snapshot path, and every spawn here uses a scratch HOME plus
a scratch store root, never the real ``~/.iai-mcp``.

The daemon resolves most of its runtime state under ``IAI_MCP_STORE`` (store
root), but the deferred-capture backlog directory and its log directory are
anchored to ``Path.home() / ".iai-mcp"`` unconditionally (not overridden by
``IAI_MCP_STORE``) -- so full isolation additionally requires a scratch
``HOME`` for the spawned daemon subprocess, not just a scratch store root.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SUN_PATH_MAX_BYTES = 104

# The pre-rollback live snapshot: a stdlib brain.sqlite3, the migrate SOURCE
# for a fresh copy. Never the live prod store, never opened in place.
_REPRO_SNAPSHOT_DIRNAME = "hippo.stdlib-prev-1783098188"

_PROD_HIPPO_DIRNAME = "hippo"
_PROD_SOCKET_NAME = ".daemon.sock"


class ProdPathGuardError(RuntimeError):
    """Raised when a helper is pointed at the live prod store or socket."""


def _real_home() -> Path:
    """The real (non-test) HOME, independent of any HOME override already in
    this process's environment (pytest's own hermeticity fixture redirects
    HOME for every test) -- resolved via the OS password database, mirroring
    ``iai_mcp.hippo._operator_home``, so it is immune to env overrides."""
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


def _assert_not_prod_path(path: Path) -> None:
    """Refuse any path that resolves under the real ``~/.iai-mcp/hippo`` or the
    real prod socket -- the one guard every helper in this module runs before
    touching a filesystem path."""
    resolved = path.resolve()
    real_prod_hippo = (_real_home() / ".iai-mcp" / _PROD_HIPPO_DIRNAME).resolve()
    real_prod_sock = (_real_home() / ".iai-mcp" / _PROD_SOCKET_NAME).resolve()
    if resolved == real_prod_hippo or real_prod_hippo in resolved.parents:
        raise ProdPathGuardError(
            f"refusing to operate on the live prod store: {resolved}"
        )
    if resolved == real_prod_sock:
        raise ProdPathGuardError(
            f"refusing to operate on the live prod socket: {resolved}"
        )


def locate_repro_source() -> "Path | None":
    """Return the pre-rollback stdlib snapshot's brain.sqlite3, or None.

    NEVER returns the live prod path (``~/.iai-mcp/hippo/brain.sqlite3``).
    If the fixed snapshot is absent (e.g. CI without the corpus), the caller
    must skip -- this function never falls back to prod.
    """
    candidate = _real_home() / ".iai-mcp" / _REPRO_SNAPSHOT_DIRNAME / "brain.sqlite3"
    _assert_not_prod_path(candidate)
    if not candidate.exists():
        return None
    return candidate


def _create_scratch_crypto_key(dst_root: Path) -> None:
    """Copy the REAL prod crypto key into the scratch store root.

    Never mints/overwrites a key -- the migrated copy's ciphertext (surface,
    provenance) is AEAD-bound to the real prod key via per-record AAD, so a
    freshly minted key would make every field undecryptable.
    """
    src_key = _real_home() / ".iai-mcp" / ".crypto.key"
    if not src_key.exists():
        raise ProdPathGuardError(
            f"prod crypto key not found at {src_key}; cannot proceed without "
            f"copying the real key (never mint a fresh one for a real-data copy)"
        )
    dst_key = dst_root / ".crypto.key"
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_key, dst_key)
    os.chmod(dst_key, 0o600)


def make_fresh_lilli_copy(
    src_db: Path, dst_root: Path, *, stale_cache: bool = True
) -> Any:
    """Copy src_db to scratch, migrate into a FRESH dst_root, optionally drop
    a deliberately drift-mismatched runtime_graph_cache.json onto dst_root
    AFTER migrate returns.

    Returns the MigrateReport from ``migrate_sqlite_to_lilli``.
    """
    _assert_not_prod_path(src_db)
    if dst_root.exists():
        raise ValueError(
            f"dst_root must be fresh (absent): {dst_root} already exists"
        )

    scratch_src_dir = dst_root.parent / f"{dst_root.name}-src-copy"
    scratch_src_dir.mkdir(parents=True, exist_ok=True)
    scratch_src_db = scratch_src_dir / "brain.sqlite3"
    shutil.copyfile(src_db, scratch_src_db)

    # The crypto key MUST exist at dst_root BEFORE migrate runs: every
    # migrated literal_surface/provenance_json ciphertext is AEAD-bound to
    # the ORIGINAL prod key (AAD=id), and migrate's own finalize step
    # (_prewarm_graph_cache) opens a MemoryStore(dst_root) and decrypts every
    # record while building the graph cache. Copying the key AFTER migrate
    # would let MemoryStore auto-mint a throwaway random key at migrate time,
    # making every record fail decrypt during the pre-warm (and later at
    # daemon boot) -- the exact "wrong cache-crypto-key -> empty graph ->
    # recall=0" failure mode from a prior cutover attempt.
    _create_scratch_crypto_key(dst_root)

    from iai_mcp.migrate import migrate_sqlite_to_lilli

    report = migrate_sqlite_to_lilli(scratch_src_db, dst_root)

    # migrate_sqlite_to_lilli already fingerprint-invalidates any stale cache
    # it finds at dst_root up front (there is none yet here -- dst_root was
    # fresh) and pre-warms a MATCHING cache as part of finalize. To reproduce
    # a cache whose corpus fingerprint drifts far past tolerance from the
    # migrated store, the mismatched cache must be written AFTER migrate
    # returns -- otherwise migrate's own up-front invalidation would just
    # delete it again before the daemon ever boots.
    if stale_cache:
        _write_drift_mismatched_cache(dst_root)

    return report


def _write_drift_mismatched_cache(dst_root: Path) -> None:
    """Build a real cache for dst_root, then mutate its payload_record_count
    far enough from the true corpus size to blow the drift-tolerance gate,
    re-encrypt, and overwrite runtime_graph_cache.json in place.

    Building a real cache first (rather than hand-rolling ciphertext) means
    every other field (assignment, rich_club, key vector, generation) stays
    structurally valid -- only the corpus fingerprint is deliberately wrong,
    reproducing a stale-cache-vs-fresh-corpus drift built for a much larger
    prior corpus than the store it now sits beside.
    """
    from iai_mcp.store import MemoryStore
    from iai_mcp import retrieve as _retrieve
    from iai_mcp import runtime_graph_cache as _rgc
    from iai_mcp.crypto import encrypt_field

    prev_store_env = os.environ.get("IAI_MCP_STORE")
    prev_driver_env = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    try:
        store = MemoryStore(path=dst_root)
        try:
            _retrieve.build_runtime_graph(store)
            cache_path = _rgc._cache_path(store)
            if not cache_path.exists():
                raise RuntimeError(
                    f"expected build_runtime_graph to write a cache at "
                    f"{cache_path}, but it does not exist"
                )
            data = _rgc._load_and_decrypt_cache(store)
            if data is None:
                raise RuntimeError(
                    f"failed to decrypt the freshly built cache at {cache_path}"
                )
            true_count = int(data.get("payload_record_count") or 0)
            # Mirror a real-world worst case: a cache built for a much larger
            # prior corpus sitting beside a much smaller fresh store, a
            # mismatch far beyond the drift tolerance in either direction.
            # Push the mismatch upward (larger) here so it cannot collide
            # with a corpus that happens to be tiny.
            data["payload_record_count"] = true_count + max(10_000, true_count * 2)
            serialised = json.dumps(data, ensure_ascii=False)
            key = _rgc._cache_encryption_key(store)
            ciphertext = encrypt_field(serialised, key, _rgc._CACHE_AAD)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="ascii") as f:
                f.write(ciphertext)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, cache_path)
        finally:
            store.close()
    finally:
        if prev_store_env is None:
            os.environ.pop("IAI_MCP_STORE", None)
        else:
            os.environ["IAI_MCP_STORE"] = prev_store_env
        if prev_driver_env is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = prev_driver_env


@dataclass
class DaemonHandle:
    socket_path: Path
    proc: subprocess.Popen
    _socket_dir: Path
    _terminated: bool = field(default=False, init=False)

    def terminate(self, timeout: float = 10.0) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
        shutil.rmtree(self._socket_dir, ignore_errors=True)


def spawn_isolated_daemon(
    base: Path,
    driver: str,
    *,
    scratch_home: "Path | None" = None,
    extra_env: "dict[str, str] | None" = None,
) -> DaemonHandle:
    """Spawn ``python -X faulthandler -m iai_mcp.daemon`` fully isolated.

    ``base`` is IAI_MCP_STORE (the store root). ``scratch_home`` (defaults to
    base.parent / "scratch-home") becomes HOME for the subprocess -- the
    deferred-capture backlog dir and its log dir are anchored to
    ``Path.home() / ".iai-mcp"`` unconditionally, not overridden by
    IAI_MCP_STORE, so a scratch HOME is required for full isolation.
    """
    _assert_not_prod_path(base)
    if scratch_home is None:
        scratch_home = base.parent / "scratch-home"
    scratch_home.mkdir(parents=True, exist_ok=True)
    (scratch_home / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    socket_dir = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    socket_path = socket_dir / ".daemon.sock"
    assert len(str(socket_path).encode("utf-8")) < _SUN_PATH_MAX_BYTES, (
        f"socket path too long for sun_path ({socket_path}) -- "
        f"use a short, independently allocated temp dir"
    )

    env = os.environ.copy()
    # Warm the model cache before HOME is overridden below: the real cache
    # holds the already-installed embedder (as prod does), so the spawned
    # daemon's embedder build is the warm fixed cost instead of a cold
    # re-download. HF_HOME (not HF_HUB_CACHE, which already has "hub"
    # appended) is what the resolver reads and falls back from -- pointing
    # it at HF_HUB_CACHE's value here would double-join into ".../hub/hub"
    # and force a cold miss. setdefault so an explicitly-set parent HF_HOME
    # always wins. Paired with a fail-loud offline flag: a wrong or missing
    # cache path must error, never silently re-download and launder a fast
    # green that is actually a cold miss.
    _real_hf_home = os.environ.get("HF_HOME") or str(_real_home() / ".cache" / "huggingface")
    env.setdefault("HF_HOME", _real_hf_home)
    env.setdefault("IAI_MCP_EMBED_OFFLINE", "1")
    env["HOME"] = str(scratch_home)
    env["IAI_MCP_STORE"] = str(base)
    env["IAI_DAEMON_SOCKET_PATH"] = str(socket_path)
    env["LILLI_STORAGE_DRIVER"] = driver
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [sys.executable, "-X", "faulthandler", "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return DaemonHandle(socket_path=socket_path, proc=proc, _socket_dir=socket_dir)


def await_socket(socket_path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"socket did not appear at {socket_path} within {timeout}s")


def raw_recall(socket_path: Path, cue: str, timeout_s: float = 30.0) -> dict:
    """Raw JSON-RPC memory_recall client. Raises on timeout."""

    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path)), timeout=timeout_s,
        )
        try:
            req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "memory_recall",
                "params": {"cue": cue, "session_id": "warm-recall-repro"},
            }
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
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


@dataclass
class DeferredBacklogState:
    backlog_path: Path
    seeded_texts: "list[str]"
    session_id: str


def seed_deferred_backlog(
    base: Path, *, scratch_home: Path, n: int = 50
) -> DeferredBacklogState:
    """Drop a deferred-capture backlog file in the EXACT path/format the
    daemon's boot drain (``drain_deferred_captures``) consumes.

    The header + event-line format mirrors ``write_deferred_captures``
    exactly (version=1, session_id, cwd, then one JSON object per line with
    text/cue/tier/role/ts). The backlog directory is anchored to
    ``scratch_home / ".iai-mcp" / ".deferred-captures"`` -- the caller must
    pass the SAME scratch_home used for ``spawn_isolated_daemon`` so the
    spawned daemon's ``Path.home()`` resolves to this same directory.

    Returns state (backlog path, seeded texts, session id) so the caller can
    assert the boot drain actually fired (the events must be discoverable in
    the store post-drain, not merely that a file was dropped).
    """
    _assert_not_prod_path(base)
    deferred_dir = scratch_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    session_id = f"warm-recall-repro-{uuid.uuid4().hex[:8]}"
    final_name = f"{session_id}-{int(time.time())}-{os.getpid()}.jsonl"
    out_path = deferred_dir / final_name
    tmp_path = deferred_dir / f"{final_name}.tmp"

    seeded_texts: list[str] = []
    with tmp_path.open("w") as fh:
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": str(base),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in range(n):
            text = (
                f"deferred backlog fixture turn {i} for the warm recall "
                f"repro session {session_id}"
            )
            seeded_texts.append(text)
            event = {
                "text": text,
                "cue": f"session {session_id} turn {i}",
                "tier": "episodic",
                "role": "user" if i % 2 == 0 else "assistant",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)

    return DeferredBacklogState(
        backlog_path=out_path, seeded_texts=seeded_texts, session_id=session_id
    )
