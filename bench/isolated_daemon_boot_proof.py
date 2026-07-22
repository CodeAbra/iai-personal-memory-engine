"""End-to-end isolated daemon boot proof: a real daemon subprocess on a lilli
copy of a production corpus, measured cold and warm, against a validated
stdlib reference.

This is the gate the earlier in-process models could not provide: a REAL
``python -m iai_mcp.daemon`` subprocess, spawned on an isolated copy that
carries a freshly persisted column-index sidecar (the state a post-migration
persist or a nightly sleep cycle leaves behind), must answer its first recall
correctly and in budget, must warm-reboot correctly, and must survive one
LLM-free sleep cycle that re-emits the sidecar -- all while a reference build
proves the corpus a live cutover would move is not silently corrupted or
degraded along the way.

Usage (source defaults to ~/.iai-mcp-isotest -- NEVER the real home store):

    source .venv/bin/activate
    .venv/bin/python bench/isolated_daemon_boot_proof.py \\
        --source ~/.iai-mcp-isotest --workdir /tmp/scratch --out /tmp/scratch/proof.json

Seven strictly sequential stages, each heavy step in its own subprocess:

  1. stage_copies            -- prod evidence pre; fresh lilli proof copy;
                                 cheap consistency check; persist-prep subprocess.
  2. stage_reference_build   -- reverse-copy the lilli proof copy into a fresh
                                 stdlib root (source .crypto.key travels FIRST);
                                 validate with the shipped migrator-verifier
                                 (or the counts+byte-sample fallback if blocked).
  3. stage_reference_recall  -- in-process stdlib recall over the fixed cues;
                                 the stdlib cold first-recall bar.
  4. stage_daemon_proof      -- spawn the lilli daemon on the proof copy
                                 (sidecar present); cold timeline + recall@10.
  5. stage_warm_reboot       -- re-spawn on the same copy; warm boot timeline.
  6. stage_sleep_smoke       -- LLM-free force_rem smoke, out-of-band
                                 completion detection, sidecar re-persist check.
  7. stage_verdict           -- compute every gate; prod evidence post; emit
                                 the aggregate JSON.

Prod refusal lives in every stage body (never only the orchestrator) so a
mistaken direct invocation of any stage can never touch the real store.
"""

from __future__ import annotations

import argparse
import asyncio
import copy as _copy
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


# ---------------------------------------------------------------------------
# The frozen cue set -- 8 varied cues, frozen before the first run, never
# re-picked. Distinct from the 6-cue set used by the earlier warm-up/index-
# persist validations (this proof exercises more cue variety end-to-end).
# ---------------------------------------------------------------------------
FIXED_CUES: tuple[str, ...] = (
    "what does alice prefer for her morning workflow",
    "the decision we made about the database migration rollback plan",
    "project codename for the new recommendation engine",
    "what did we discuss last week about the release schedule",
    "explain the hashing algorithm used for the projection matrix",
    "who is alice's main contact on the infrastructure team",
    "the bug we fixed in the reinforcement queue drain",
    "alice's opinion on the new onboarding flow",
)


def cue_set_sha256() -> str:
    joined = "\x1f".join(FIXED_CUES).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


_SIDECAR_FILENAME = "records.colindex"
_BOOT_WARMUP_SIDECAR = ".boot-warmup.json"
_SUN_PATH_MAX_BYTES = 104

_SOCKET_AWAIT_TIMEOUT_SEC = 30.0
# The first recall on a cold boot can legitimately queue behind the boot
# warm-up accelerator's own probes (which hold the shared connection lock
# while it runs in a background thread) -- a slow-but-honest first recall
# must be measured, not mistaken for a hang. 90s gives ample margin above
# the warm-up's own measured cost (order tens of seconds on a large corpus)
# plus the recall dispatch itself.
_RECALL_TIMEOUT_SEC = 90.0
_TEARDOWN_JOIN_TIMEOUT_SEC = 15.0

_STEADY_STATE_BUDGET_SEC = 3.0
_SOCKET_UP_BUDGET_SEC = 5.0
_INTERRUPT_QUIESCE_SEC = 35.0  # > daemon's own 30s INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC
_SLEEP_SMOKE_BOUNDED_WAIT_SEC = 600.0
_SEMANTIC_HIT_EPS = 0.98  # bench/recall_accuracy.py convention, reused for tie explanations


# ---------------------------------------------------------------------------
# Prod-refusal guard -- lives in every stage body, not only the orchestrator.
# ---------------------------------------------------------------------------


def _resolve_prod_root() -> Path:
    return Path.home() / ".iai-mcp"


def _refuse_if_prod(store_root: Path) -> None:
    prod_root = _resolve_prod_root()
    try:
        resolved = store_root.expanduser().resolve()
    except OSError:
        resolved = store_root.expanduser()
    if resolved == prod_root.resolve():
        print(
            f"REFUSED: store root resolves to the prod store ({prod_root}). "
            "This harness never opens prod.",
            file=sys.stderr,
        )
        sys.exit(2)


def _require_isolation_env() -> None:
    if not os.environ.get("IAI_MCP_STORE"):
        print(
            "REFUSED: IAI_MCP_STORE is not set in this process's environment.",
            file=sys.stderr,
        )
        sys.exit(2)


def _preflight_daemon_state_path(proof_root: Path) -> None:
    """Hard-fail if the state-isolation fix is missing, or resolves wrong."""
    try:
        from iai_mcp.daemon_state import daemon_state_path
    except ImportError as exc:
        print(
            f"REFUSED: `from iai_mcp.daemon_state import daemon_state_path` "
            f"does not import ({exc}) -- the store-isolation fix is not in "
            "HEAD. This proof must not run without it (a live daemon's "
            "force_rem ack could land in the PROD state file).",
            file=sys.stderr,
        )
        sys.exit(2)
    resolved = daemon_state_path(proof_root)
    expected = proof_root / ".daemon-state.json"
    if resolved != expected:
        print(
            f"REFUSED: daemon_state_path({proof_root}) resolved to {resolved}, "
            f"expected {expected} -- isolation preflight failed.",
            file=sys.stderr,
        )
        sys.exit(2)


def _sidecar_path(store_root: Path) -> Path:
    return store_root / "hippo" / _SIDECAR_FILENAME


def _stat_or_none(path: Path) -> dict | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _sha256_or_none(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _prod_daemon_pid() -> "int | None":
    """Read-only: the pid the real prod daemon last recorded for itself.

    Used only to attribute a prod brain.sqlite3/daemon-state.json diff to an
    INDEPENDENT, already-running prod daemon (the user's own long-lived
    launchd/systemd process, entirely outside this proof's control) rather
    than to this harness -- this harness never opens ~/.iai-mcp and never
    signals that pid. Never written to, never signaled.
    """
    prod_state = _resolve_prod_root() / ".daemon-state.json"
    try:
        raw = json.loads(prod_state.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pid = raw.get("daemon_pid")
    return int(pid) if isinstance(pid, int) else None


def _prod_daemon_pid_alive(pid: "int | None") -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _prod_evidence() -> dict:
    prod_root = _resolve_prod_root()
    prod_db = prod_root / "hippo" / "brain.sqlite3"
    prod_state = prod_root / ".daemon-state.json"
    prod_wake = prod_root / "wake.signal"
    pid = _prod_daemon_pid()
    return {
        "brain_sqlite3": {**(_stat_or_none(prod_db) or {}), "sha256": _sha256_or_none(prod_db)}
        if prod_db.exists()
        else None,
        "daemon_state_json": {
            **(_stat_or_none(prod_state) or {}),
            "sha256": _sha256_or_none(prod_state),
        }
        if prod_state.exists()
        else None,
        "wake_signal_exists": prod_wake.exists(),
        "independent_prod_daemon_pid": pid,
        "independent_prod_daemon_alive": _prod_daemon_pid_alive(pid),
    }


# ---------------------------------------------------------------------------
# Subprocess dispatch: every heavy stage body runs in its own subprocess with
# the full isolation env, invoked by this module re-executing itself with a
# --stage-worker flag.  Strictly sequential -- one process alive at a time.
# ---------------------------------------------------------------------------


def _iso_env(store_root: Path, *, driver: str, socket_path: Path | None = None) -> dict[str, str]:
    env = _copy.deepcopy(dict(os.environ))
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    if driver == "lilli":
        env["LILLI_STORAGE_DRIVER"] = "lilli"
    else:
        env.pop("LILLI_STORAGE_DRIVER", None)
    if socket_path is not None:
        env["IAI_DAEMON_SOCKET_PATH"] = str(socket_path)
    return env


def _run_worker_subprocess(worker_name: str, args: dict, *, store_root: Path, driver: str) -> dict:
    """Spawn this module as a fresh subprocess running one internal worker."""
    payload_path = Path(tempfile.mkstemp(prefix=f"iai-proof-{worker_name}-in-", suffix=".json")[1])
    out_path = Path(tempfile.mkstemp(prefix=f"iai-proof-{worker_name}-out-", suffix=".json")[1])
    try:
        payload_path.write_text(json.dumps(args))
        env = _iso_env(store_root, driver=driver)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage-worker",
            worker_name,
            "--worker-in",
            str(payload_path),
            "--worker-out",
            str(out_path),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"worker '{worker_name}' failed (exit {proc.returncode}):\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return json.loads(out_path.read_text())
    finally:
        payload_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Worker: persist-prep (stage_copies) -- persist_col_indexes on the proof copy.
# ---------------------------------------------------------------------------


def _worker_persist_prep(args: dict) -> dict:
    store_root = Path(args["store_root"])
    _refuse_if_prod(store_root)
    _require_isolation_env()

    from contextlib import nullcontext

    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_root, access_mode=AccessMode.EXCLUSIVE)
    conn = getattr(store.db, "_conn", None)
    if conn is None or not hasattr(conn, "persist_col_indexes"):
        print(
            "REFUSED: engine Connection missing persist_col_indexes -- stale "
            "native build. Run `cd rust/iai_mcp_native && maturin develop "
            "--release`.",
            file=sys.stderr,
        )
        sys.exit(2)

    lock = getattr(store.db, "_conn_lock", None)
    lock_ctx = lock if lock is not None else nullcontext()

    t0 = time.perf_counter()
    with lock_ctx:
        conn.persist_col_indexes()
    persist_ms = (time.perf_counter() - t0) * 1000.0

    try:
        store.close()
    except Exception:  # noqa: BLE001 -- teardown must not mask the measurement
        pass

    sidecar = _sidecar_path(store_root)
    result = {
        "persist_ms": persist_ms,
        "sidecar_exists": sidecar.exists(),
        "sidecar_bytes": sidecar.stat().st_size if sidecar.exists() else 0,
        "sidecar_mtime_ns": sidecar.stat().st_mtime_ns if sidecar.exists() else None,
    }
    if not result["sidecar_exists"]:
        print(
            "FAIL: persist_col_indexes ran but left no records.colindex "
            "sidecar -- the durable fix under proof did not take effect.",
            file=sys.stderr,
        )
        sys.exit(1)
    return result


def _worker_consistency_check(args: dict) -> dict:
    """Cheap boot-health mirror: hnsw active count == sqlite active count."""
    store_root = Path(args["store_root"])
    _refuse_if_prod(store_root)
    _require_isolation_env()

    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_root, access_mode=AccessMode.SHARED)
    try:
        db = store.db
        sqlite_count = int(
            db._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()[0]
        )
        active_label_count = int(len(db._label_map))
        hnsw_raw_count = int(db._hnsw.get_current_count())
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
    return {
        "sqlite_count": sqlite_count,
        "hnsw_active_count": active_label_count,
        "hnsw_raw_count": hnsw_raw_count,
        "consistent": sqlite_count == active_label_count == hnsw_raw_count,
    }


# ---------------------------------------------------------------------------
# Worker: reverse-copy the lilli proof copy into a fresh stdlib reference root.
# Modeled on migrate/_to_lilli.py:141-260, direction reversed.
# ---------------------------------------------------------------------------


def _worker_reverse_copy(args: dict) -> dict:
    src_root = Path(args["src_root"])  # lilli proof copy (source, read-only)
    dst_root = Path(args["dst_root"])  # fresh stdlib reference root (dest)
    batch = int(args.get("batch", 500))

    _refuse_if_prod(src_root)
    _refuse_if_prod(dst_root)

    # BEFORE opening the dest: the source's own crypto key travels with the
    # copy, never a minted key or passphrase (crypto.py:43 -- the key file
    # lives under store_root).
    src_key = src_root / ".crypto.key"
    dst_root.mkdir(parents=True, exist_ok=True)
    if src_key.exists():
        shutil.copy2(src_key, dst_root / ".crypto.key")
    else:
        print(
            f"REFUSED: source proof copy has no .crypto.key at {src_key} -- "
            "the reference must decrypt with the SAME key as the source.",
            file=sys.stderr,
        )
        sys.exit(2)

    # The runtime graph cache (community assignment + rich-club + degree map)
    # must travel with the copy too. Recall dispatch reads it to gate/rank
    # candidates by community structure; without it, dispatch degrades to an
    # empty community assignment ("cold_degrade") and ranks the k-boundary by
    # raw cosine alone. The daemon serves recall WITH this cache present, so a
    # faithful stdlib reference must dispatch with the SAME structural state --
    # otherwise the recall comparison pits a structure-informed path against a
    # structure-blind one and diverges at the k-boundary for reasons that have
    # nothing to do with the storage engine. The cache is a corpus-derived JSON
    # validated at load time against the destination's own corpus counts, so it
    # ports across the storage driver.
    src_graph_cache = src_root / "runtime_graph_cache.json"
    graph_cache_copied = False
    if src_graph_cache.exists():
        shutil.copy2(src_graph_cache, dst_root / "runtime_graph_cache.json")
        graph_cache_copied = True

    # Open the SOURCE (lilli-format) read-only via the engine, driver-aware.
    saved_store = os.environ.get("IAI_MCP_STORE")
    saved_driver = os.environ.get("LILLI_STORAGE_DRIVER")
    try:
        os.environ["IAI_MCP_STORE"] = str(src_root)
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
        from iai_mcp.hippo import HippoDB as SrcHippoDB

        src_db = SrcHippoDB(src_root, read_only=True)
    finally:
        if saved_store is None:
            os.environ.pop("IAI_MCP_STORE", None)
        else:
            os.environ["IAI_MCP_STORE"] = saved_store
        if saved_driver is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = saved_driver

    _ALL_TABLES = (
        "records",
        "edges",
        "events",
        "budget_ledger",
        "ratelimit_ledger",
        "_hippo_meta",
    )
    _EVENTS_TS_COLUMN = "ts"

    rows_copied: dict[str, int] = {}
    max_vec_label = 0
    t0 = time.monotonic()

    try:
        # The DEST env has LILLI_STORAGE_DRIVER removed (stdlib) -- forced
        # around the open per the migrator's env-dance pattern (IAI_MCP_STORE
        # overrides the path arg in _resolve_root).
        saved_store2 = os.environ.get("IAI_MCP_STORE")
        saved_driver2 = os.environ.get("LILLI_STORAGE_DRIVER")
        os.environ["IAI_MCP_STORE"] = str(dst_root)
        os.environ.pop("LILLI_STORAGE_DRIVER", None)
        try:
            from iai_mcp.hippo import HippoDB as DstHippoDB

            dst_db = DstHippoDB(dst_root)
            try:
                with src_db._conn_lock:
                    for table in _ALL_TABLES:
                        cols_rows = src_db._conn.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                        cols = [r[1] if not hasattr(r, "keys") else r["name"] for r in cols_rows]
                        if not cols:
                            rows_copied[table] = 0
                            continue
                        cl = ", ".join(cols)
                        ph = ", ".join("?" for _ in cols)
                        verb = "INSERT OR REPLACE" if table == "_hippo_meta" else "INSERT"
                        ins = f"{verb} INTO {table} ({cl}) VALUES ({ph})"

                        if (
                            table == "events"
                            and _EVENTS_TS_COLUMN in cols
                        ):
                            cur = src_db._conn.execute(
                                f"SELECT {cl} FROM {table} ORDER BY rowid"
                            )
                        else:
                            cur = src_db._conn.execute(f"SELECT {cl} FROM {table}")

                        copied = 0
                        with dst_db._conn_lock:
                            while True:
                                rows = cur.fetchmany(batch)
                                if not rows:
                                    break

                                def _row_get(r, c):
                                    return r[c] if hasattr(r, "keys") else r[cols.index(c)]

                                tuples = [tuple(_row_get(r, c) for c in cols) for r in rows]
                                dst_db._conn.executemany(ins, tuples)
                                dst_db._conn.commit()
                                copied += len(rows)
                                if table == "records" and "vec_label" in cols:
                                    for r in rows:
                                        v = _row_get(r, "vec_label")
                                        if v is not None:
                                            max_vec_label = max(max_vec_label, int(v))
                        rows_copied[table] = copied

                # stdlib AUTOINCREMENT tracks max(vec_label) natively -- no HWM
                # bump call needed (the forward migrator's Pitfall-1 fix is a
                # lilli-only concern; the dest here is stdlib).
                dst_db._rebuild_index_from_sqlite()
            finally:
                dst_db.close()
        finally:
            if saved_store2 is None:
                os.environ.pop("IAI_MCP_STORE", None)
            else:
                os.environ["IAI_MCP_STORE"] = saved_store2
            if saved_driver2 is None:
                os.environ.pop("LILLI_STORAGE_DRIVER", None)
            else:
                os.environ["LILLI_STORAGE_DRIVER"] = saved_driver2
    finally:
        try:
            src_db.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed_sec = time.monotonic() - t0
    return {
        "rows_copied": rows_copied,
        "max_vec_label": max_vec_label,
        "elapsed_sec": elapsed_sec,
        "graph_cache_copied": graph_cache_copied,
    }


# ---------------------------------------------------------------------------
# Worker: reference validation via the shipped migrator-verifier.
# ---------------------------------------------------------------------------


def _worker_reference_validate(args: dict) -> dict:
    ref_root = Path(args["ref_root"])  # stdlib reference (native SQLite direction: src)
    proof_root = Path(args["proof_root"])  # lilli proof copy (dst)
    _refuse_if_prod(ref_root)
    _refuse_if_prod(proof_root)

    from iai_mcp.crypto import CryptoKey
    from iai_mcp.migrate._to_lilli_verify import verify_store_equality

    key = CryptoKey(user_id="default", store_root=ref_root).get_or_create()

    src_db_path = ref_root / "hippo" / "brain.sqlite3"
    report = verify_store_equality(
        src_db_path,
        proof_root,
        key,
        src_root=ref_root,
        src_is_lilli=False,
    )
    return {
        "ok": report.ok,
        "sampled_n": report.sampled_n,
        "dimensions": {
            name: {"ok": d.ok, "reason": d.reason, "counts": d.counts}
            for name, d in report.dimensions.items()
        },
    }


def _worker_reference_validate_fallback(args: dict) -> dict:
    """Counts + >=500-row sampled byte-compare fallback, only if the shipped
    verifier is mechanically blocked."""
    ref_root = Path(args["ref_root"])
    proof_root = Path(args["proof_root"])
    _refuse_if_prod(ref_root)
    _refuse_if_prod(proof_root)

    import sqlite3

    from iai_mcp.hippo import HippoDB

    ref_conn = sqlite3.connect(f"file:{ref_root / 'hippo' / 'brain.sqlite3'}?mode=ro", uri=True)
    ref_conn.row_factory = sqlite3.Row

    saved_store = os.environ.get("IAI_MCP_STORE")
    saved_driver = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["IAI_MCP_STORE"] = str(proof_root)
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    try:
        proof_db = HippoDB(proof_root, read_only=True)
    finally:
        if saved_store is None:
            os.environ.pop("IAI_MCP_STORE", None)
        else:
            os.environ["IAI_MCP_STORE"] = saved_store
        if saved_driver is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = saved_driver

    tables = ("records", "edges", "events", "budget_ledger", "ratelimit_ledger", "_hippo_meta")
    counts_match: dict[str, dict] = {}
    byte_samples: dict[str, dict] = {}
    try:
        for table in tables:
            ref_n = ref_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            with proof_db._conn_lock:
                proof_n = proof_db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            counts_match[table] = {"ref": ref_n, "proof": proof_n, "equal": ref_n == proof_n}

            if table == "records":
                sample_cols = ("id", "literal_surface", "embedding")
            elif table == "edges":
                sample_cols = ("src", "dst", "edge_type")
            else:
                byte_samples[table] = {"sampled": 0, "mismatches": 0}
                continue

            cl = ", ".join(sample_cols)
            rows = ref_conn.execute(f"SELECT {cl} FROM {table} LIMIT 500").fetchall()
            mismatches = 0
            for r in rows:
                rid = r["id"] if "id" in sample_cols else None
                with proof_db._conn_lock:
                    if rid is not None:
                        proof_row = proof_db._conn.execute(
                            f"SELECT {cl} FROM {table} WHERE id = ?", (rid,)
                        ).fetchone()
                    else:
                        proof_row = None
                if proof_row is None:
                    mismatches += 1
                    continue
                for c in sample_cols:
                    ref_val = r[c]
                    proof_val = proof_row[c] if hasattr(proof_row, "keys") else proof_row[sample_cols.index(c)]
                    if ref_val != proof_val:
                        mismatches += 1
                        break
            byte_samples[table] = {"sampled": len(rows), "mismatches": mismatches}
    finally:
        ref_conn.close()
        try:
            proof_db.close()
        except Exception:  # noqa: BLE001
            pass

    ok = all(v["equal"] for v in counts_match.values()) and all(
        v["mismatches"] == 0 for v in byte_samples.values()
    )
    return {"ok": ok, "counts_match": counts_match, "byte_samples": byte_samples}


# ---------------------------------------------------------------------------
# Worker: reference recall run (in-process, stdlib env).
# ---------------------------------------------------------------------------


def _worker_reference_recall(args: dict) -> dict:
    ref_root = Path(args["ref_root"])
    _refuse_if_prod(ref_root)
    _require_isolation_env()

    from iai_mcp import core
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    t_cold0 = time.perf_counter()
    store = MemoryStore(path=ref_root, access_mode=AccessMode.EXCLUSIVE)
    store.enable_reinforce_queue()

    recalls = []
    for i, cue in enumerate(FIXED_CUES):
        t0 = time.perf_counter()
        resp = core.dispatch(
            store, "memory_recall", {"cue": cue, "session_id": "isolated_daemon_boot_proof"}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        hits = resp.get("hits", [])
        top10 = [h["record_id"] for h in hits[:10]]
        recalls.append(
            {
                "index": i,
                "cue": cue,
                "elapsed_ms": elapsed_ms,
                "hits": len(hits),
                "top10_ids": top10,
                # "cold-structural-degrade" here means the reference dispatched
                # with an empty community assignment -- a structure-blind path
                # that cannot be a faithful truth bar for the structure-informed
                # daemon. The graph cache must be present in the reference root
                # (copied by the reverse-copy stage) for the comparison to be
                # apples-to-apples.
                "source": resp.get("_source"),
            }
        )
        if i == 0:
            stdlib_cold_first_recall_ms = elapsed_ms

    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass

    reference_structure_informed = all(
        r.get("source") != "cold-structural-degrade" for r in recalls
    )
    graph_cache_present = (ref_root / "runtime_graph_cache.json").exists()

    return {
        "stdlib_cold_first_recall_ms": stdlib_cold_first_recall_ms,
        "recalls": recalls,
        "graph_cache_present": graph_cache_present,
        "reference_structure_informed": reference_structure_informed,
    }


# ---------------------------------------------------------------------------
# Daemon spawn/recall/teardown helpers (mirrors the 175-04 suite proof).
# ---------------------------------------------------------------------------


def _short_socket_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="iai-sock-"))


def _spawn_daemon(store_root: Path, socket_path: Path) -> subprocess.Popen:
    assert len(str(socket_path).encode("utf-8")) < _SUN_PATH_MAX_BYTES, (
        f"socket path too long for sun_path ({socket_path})"
    )
    env = _iso_env(store_root, driver="lilli", socket_path=socket_path)
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _await_socket(socket_path: Path, timeout: float = _SOCKET_AWAIT_TIMEOUT_SEC) -> float:
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return time.monotonic() - t0
        time.sleep(0.05)
    raise TimeoutError(f"socket did not appear at {socket_path} within {timeout}s")


_SOCKET_READ_LIMIT_BYTES = 16 * 1024 * 1024  # a rich memory_recall response on a
# large corpus (many hits, literal_surface text, activation traces) can exceed
# asyncio's 64KB StreamReader default -- 16MB gives ample margin.


def _recall_over_socket(socket_path: Path, cue: str, timeout: float = _RECALL_TIMEOUT_SEC) -> dict:
    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path), limit=_SOCKET_READ_LIMIT_BYTES),
            timeout=timeout,
        )
        try:
            req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "memory_recall",
                "params": {"cue": cue, "session_id": "isolated_daemon_boot_proof"},
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


def _send_control_message(socket_path: Path, msg: dict, timeout: float = 15.0) -> dict:
    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path), limit=_SOCKET_READ_LIMIT_BYTES),
            timeout=timeout,
        )
        try:
            writer.write((json.dumps(msg) + "\n").encode("utf-8"))
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


def _teardown_daemon(proc: subprocess.Popen) -> dict:
    stdout, stderr = "", ""
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=_TEARDOWN_JOIN_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_TEARDOWN_JOIN_TIMEOUT_SEC)
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=2.0)
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- best-effort log capture
        pass
    return {"returncode": proc.poll(), "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]}


def _extract_hit_ids(resp: dict) -> "list[str] | None":
    result = resp.get("result")
    if not isinstance(result, dict):
        return None
    hits = result.get("hits")
    if not isinstance(hits, list):
        return None
    return [h.get("record_id") for h in hits if isinstance(h, dict)]


# ---------------------------------------------------------------------------
# Worker: spawned-daemon proof (cold boot on the lilli proof copy).
# ---------------------------------------------------------------------------


def _worker_daemon_cold_proof(args: dict) -> dict:
    store_root = Path(args["store_root"])
    _refuse_if_prod(store_root)

    if not _sidecar_path(store_root).exists():
        print(
            "FAIL: records.colindex is absent before the cold-boot daemon "
            "spawn -- the proof would silently measure the lazy path.",
            file=sys.stderr,
        )
        sys.exit(1)

    socket_dir = _short_socket_dir()
    socket_path = socket_dir / ".daemon.sock"
    proc = _spawn_daemon(store_root, socket_path)
    try:
        socket_up_sec = _await_socket(socket_path)

        t0 = time.perf_counter()
        resp0 = _recall_over_socket(socket_path, FIXED_CUES[0])
        first_recall_ms = (time.perf_counter() - t0) * 1000.0
        first_recall_ids = _extract_hit_ids(resp0)

        warmup_landed_sec = None
        warmup_path = store_root / _BOOT_WARMUP_SIDECAR
        t_warmup0 = time.monotonic()
        while time.monotonic() - t_warmup0 < 30.0:
            if warmup_path.exists():
                warmup_landed_sec = time.monotonic() - t_warmup0
                break
            time.sleep(0.2)

        recalls = [
            {
                "index": 0,
                "cue": FIXED_CUES[0],
                "elapsed_ms": first_recall_ms,
                "top10_ids": (first_recall_ids or [])[:10],
                "well_formed": "error" not in resp0 and first_recall_ids is not None,
            }
        ]
        for i, cue in enumerate(FIXED_CUES[1:], start=1):
            t0 = time.perf_counter()
            resp = _recall_over_socket(socket_path, cue)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ids = _extract_hit_ids(resp)
            recalls.append(
                {
                    "index": i,
                    "cue": cue,
                    "elapsed_ms": elapsed_ms,
                    "top10_ids": (ids or [])[:10],
                    "well_formed": "error" not in resp and ids is not None,
                }
            )

        return {
            "socket_up_sec": socket_up_sec,
            "warmup_landed_sec": warmup_landed_sec,
            "recalls": recalls,
        }
    finally:
        teardown_info = _teardown_daemon(proc)
        shutil.rmtree(socket_dir, ignore_errors=True)
        _ = teardown_info  # captured for potential debugging, not required by the gate


def _worker_daemon_warm_reboot(args: dict) -> dict:
    store_root = Path(args["store_root"])
    _refuse_if_prod(store_root)

    socket_dir = _short_socket_dir()
    socket_path = socket_dir / ".daemon.sock"
    proc = _spawn_daemon(store_root, socket_path)
    try:
        socket_up_sec = _await_socket(socket_path)
        t0 = time.perf_counter()
        resp = _recall_over_socket(socket_path, FIXED_CUES[0])
        first_recall_ms = (time.perf_counter() - t0) * 1000.0
        ids = _extract_hit_ids(resp)
        return {
            "socket_up_sec": socket_up_sec,
            "first_recall_ms": first_recall_ms,
            "well_formed": "error" not in resp and ids is not None,
            "hit_count": len(ids) if ids is not None else 0,
        }
    finally:
        _teardown_daemon(proc)
        shutil.rmtree(socket_dir, ignore_errors=True)


def _worker_sleep_smoke(args: dict) -> dict:
    """LLM-free sleep smoke: force_rem, close the socket, stay off it >=30s,
    detect completion out-of-band, check the sidecar re-persist."""
    store_root = Path(args["store_root"])
    _refuse_if_prod(store_root)

    from iai_mcp.lifecycle_state import lifecycle_state_path

    sidecar = _sidecar_path(store_root)
    pre_smoke_mtime_ns = sidecar.stat().st_mtime_ns if sidecar.exists() else None

    socket_dir = _short_socket_dir()
    socket_path = socket_dir / ".daemon.sock"
    proc = _spawn_daemon(store_root, socket_path)
    try:
        _await_socket(socket_path)

        ack = _send_control_message(
            socket_path,
            {"type": "force_rem", "ts": datetime.now(timezone.utc).isoformat()},
        )

        # Close contact with the socket entirely for >= the daemon's own
        # 30s INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC -- polling the socket here
        # would perpetually interrupt the very pipeline under test.
        time.sleep(_INTERRUPT_QUIESCE_SEC)

        lstate_path = lifecycle_state_path(store_root)
        deadline = time.monotonic() + _SLEEP_SMOKE_BOUNDED_WAIT_SEC
        completion_marker = None
        last_lifecycle_state = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                # Process exited on its own -- check whether this was a clean
                # HIBERNATION exit after a completed pipeline (stderr tail).
                break
            if lstate_path.exists():
                try:
                    raw = json.loads(lstate_path.read_text())
                    last_lifecycle_state = raw.get("current_state")
                    progress = raw.get("sleep_cycle_progress")
                    if (
                        last_lifecycle_state in ("WAKE", "HIBERNATION")
                        and progress is None
                    ):
                        completion_marker = "lifecycle_state_cleared"
                        break
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(2.0)

        exited_on_own = proc.poll() is not None
        stderr_tail = ""
        if exited_on_own:
            try:
                _, stderr_bytes = proc.communicate(timeout=5.0)
                stderr_tail = (stderr_bytes or b"").decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

        end_state: str
        post_smoke_recall_well_formed = None
        if exited_on_own:
            hibernation_markers = ("HIBERNATION", "lifecycle_hibernation_exit", "SLEEP_CYCLE_DONE")
            completed_cleanly = any(m in stderr_tail for m in hibernation_markers)
            if completed_cleanly:
                end_state = "hibernation_exit_after_completed_pipeline"
                # Re-boot the same copy; post-smoke recall must be well-formed.
                socket_dir2 = _short_socket_dir()
                socket_path2 = socket_dir2 / ".daemon.sock"
                proc2 = _spawn_daemon(store_root, socket_path2)
                try:
                    _await_socket(socket_path2)
                    resp = _recall_over_socket(socket_path2, FIXED_CUES[0])
                    ids = _extract_hit_ids(resp)
                    post_smoke_recall_well_formed = (
                        "error" not in resp and ids is not None
                    )
                finally:
                    _teardown_daemon(proc2)
                    shutil.rmtree(socket_dir2, ignore_errors=True)
            else:
                end_state = "died_uncleanly"
        else:
            # Still alive -- a post-smoke recall over the same socket must be
            # well-formed and non-empty.
            resp = _recall_over_socket(socket_path, FIXED_CUES[0])
            ids = _extract_hit_ids(resp)
            post_smoke_recall_well_formed = "error" not in resp and ids is not None
            end_state = "alive_post_pipeline"

        post_smoke_mtime_ns = sidecar.stat().st_mtime_ns if sidecar.exists() else None
        sidecar_re_persisted = (
            pre_smoke_mtime_ns is not None
            and post_smoke_mtime_ns is not None
            and post_smoke_mtime_ns > pre_smoke_mtime_ns
        )

        return {
            "force_rem_ack": ack,
            "end_state": end_state,
            "completion_marker": completion_marker,
            "last_lifecycle_state": last_lifecycle_state,
            "post_smoke_recall_well_formed": post_smoke_recall_well_formed,
            "pre_smoke_sidecar_mtime_ns": pre_smoke_mtime_ns,
            "post_smoke_sidecar_mtime_ns": post_smoke_mtime_ns,
            "sidecar_re_persisted": sidecar_re_persisted,
            "llm_free": True,  # IAI_MCP_RECONSOLIDATION_TIER1=0 pinned in _iso_env
            "smoke_pass": end_state
            in ("alive_post_pipeline", "hibernation_exit_after_completed_pipeline")
            and bool(post_smoke_recall_well_formed),
        }
    finally:
        if proc.poll() is None:
            _teardown_daemon(proc)
        shutil.rmtree(socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker dispatch table.
# ---------------------------------------------------------------------------

_WORKERS = {
    "persist_prep": _worker_persist_prep,
    "consistency_check": _worker_consistency_check,
    "reverse_copy": _worker_reverse_copy,
    "reference_validate": _worker_reference_validate,
    "reference_validate_fallback": _worker_reference_validate_fallback,
    "reference_recall": _worker_reference_recall,
    "daemon_cold_proof": _worker_daemon_cold_proof,
    "daemon_warm_reboot": _worker_daemon_warm_reboot,
    "sleep_smoke": _worker_sleep_smoke,
}


def _stage_worker_main(name: str, in_path: Path, out_path: Path) -> None:
    fn = _WORKERS.get(name)
    if fn is None:
        print(f"REFUSED: unknown worker '{name}'", file=sys.stderr)
        sys.exit(2)
    args = json.loads(in_path.read_text())
    result = fn(args)
    out_path.write_text(json.dumps(result, default=str))


# ---------------------------------------------------------------------------
# Orchestrator: the 7 stages, invoking workers via fresh subprocesses.
# ---------------------------------------------------------------------------


def stage_copies(source: Path, workdir: Path) -> dict:
    prod_pre = _prod_evidence()

    ts = int(time.time() * 1000)
    proof_root = workdir / f"iai-isoproof-{ts}"
    print(f"[stage_copies] copying {source} -> {proof_root}")
    subprocess.run(["cp", "-R", str(source), str(proof_root)], check=True)

    consistency = _run_worker_subprocess(
        "consistency_check", {"store_root": str(proof_root)},
        store_root=proof_root, driver="lilli",
    )
    if not consistency.get("consistent"):
        print(
            f"FAIL: consistency check on the fresh proof copy diverged: "
            f"{consistency}",
            file=sys.stderr,
        )

    persist_prep = _run_worker_subprocess(
        "persist_prep", {"store_root": str(proof_root)},
        store_root=proof_root, driver="lilli",
    )

    return {
        "prod_pre": prod_pre,
        "proof_root": str(proof_root),
        "consistency": consistency,
        "persist_prep": persist_prep,
    }


def stage_reference_build(proof_root: Path, workdir: Path) -> dict:
    ts = int(time.time() * 1000)
    ref_root = workdir / f"iai-stdlib-ref-{ts}"
    reverse_copy = _run_worker_subprocess(
        "reverse_copy",
        {"src_root": str(proof_root), "dst_root": str(ref_root)},
        store_root=proof_root, driver="lilli",
    )

    try:
        validation = _run_worker_subprocess(
            "reference_validate",
            {"ref_root": str(ref_root), "proof_root": str(proof_root)},
            store_root=ref_root, driver="stdlib",
        )
        validation_mode = "shipped_verifier"
    except Exception as exc:  # noqa: BLE001 -- fall back only if mechanically blocked
        print(
            f"[stage_reference_build] shipped verifier blocked ({exc}); "
            "falling back to counts+byte-sample",
            file=sys.stderr,
        )
        validation = _run_worker_subprocess(
            "reference_validate_fallback",
            {"ref_root": str(ref_root), "proof_root": str(proof_root)},
            store_root=ref_root, driver="stdlib",
        )
        validation_mode = "fallback_counts_and_byte_sample"

    return {
        "ref_root": str(ref_root),
        "reverse_copy": reverse_copy,
        "validation_mode": validation_mode,
        "validation": validation,
    }


def stage_reference_recall(ref_root: Path) -> dict:
    return _run_worker_subprocess(
        "reference_recall", {"ref_root": str(ref_root)}, store_root=ref_root, driver="stdlib",
    )


def stage_daemon_proof(proof_root: Path) -> dict:
    return _run_worker_subprocess(
        "daemon_cold_proof", {"store_root": str(proof_root)}, store_root=proof_root, driver="lilli",
    )


def stage_warm_reboot(proof_root: Path) -> dict:
    return _run_worker_subprocess(
        "daemon_warm_reboot", {"store_root": str(proof_root)}, store_root=proof_root, driver="lilli",
    )


def stage_sleep_smoke(proof_root: Path) -> dict:
    return _run_worker_subprocess(
        "sleep_smoke", {"store_root": str(proof_root)}, store_root=proof_root, driver="lilli",
    )


def _recall_at_10(daemon_top10: "list[str]", ref_top10: "list[str]") -> float:
    if not ref_top10:
        return 1.0  # nothing to miss
    denom = min(10, len(ref_top10))
    overlap = len(set(daemon_top10[:10]) & set(ref_top10[:10]))
    return overlap / denom


def stage_verdict(
    prod_pre: dict,
    daemon_cold: dict,
    ref_recall: dict,
    warm_reboot: dict,
    smoke: dict,
    reference_validation: dict,
) -> dict:
    prod_post = _prod_evidence()

    daemon_recalls = daemon_cold["recalls"]
    ref_recalls = ref_recall["recalls"]
    per_cue_recall_at_10 = []
    for d, r in zip(daemon_recalls, ref_recalls):
        r10 = _recall_at_10(d["top10_ids"], r["top10_ids"])
        per_cue_recall_at_10.append(
            {
                "index": d["index"],
                "cue": d["cue"],
                "daemon_ms": d["elapsed_ms"],
                "daemon_hit_count": len(d["top10_ids"]),
                "recall_at_10": r10,
            }
        )
    mean_recall_at_10 = (
        sum(x["recall_at_10"] for x in per_cue_recall_at_10) / len(per_cue_recall_at_10)
        if per_cue_recall_at_10
        else 0.0
    )

    first_recall_correct = (
        daemon_recalls[0]["well_formed"]
        and len(daemon_recalls[0]["top10_ids"]) > 0
        and per_cue_recall_at_10[0]["recall_at_10"] >= 0.98
    )
    first_recall_in_steady_band = daemon_recalls[0]["elapsed_ms"] / 1000.0 <= _STEADY_STATE_BUDGET_SEC

    steady_state_by_recall3 = True
    for rc in per_cue_recall_at_10[:4]:  # recalls 0..3 inclusive (index 3 = "recall 3")
        if rc["index"] >= 1 and rc["daemon_ms"] / 1000.0 > _STEADY_STATE_BUDGET_SEC:
            steady_state_by_recall3 = False

    prod_bytes_unchanged = prod_pre == prod_post
    # Attribution: a byte-level prod diff is only a HARNESS violation if no
    # independently-alive prod daemon pid explains it. This proof never opens
    # or signals ~/.iai-mcp (structurally true by construction -- see
    # _resolve_prod_root's only call sites), so if the recorded daemon_pid
    # was alive throughout, any prod diff is attributable to that INDEPENDENT
    # process (the user's own long-lived launchd/systemd daemon) doing its
    # normal job concurrently with this proof, not to this harness.
    independent_daemon_pre = bool(prod_pre.get("independent_prod_daemon_alive"))
    independent_daemon_post = bool(prod_post.get("independent_prod_daemon_alive"))
    prod_diff_attributable_to_independent_daemon = (
        not prod_bytes_unchanged and independent_daemon_pre and independent_daemon_post
        and prod_pre.get("independent_prod_daemon_pid") == prod_post.get("independent_prod_daemon_pid")
    )
    prod_untouched_by_harness = prod_bytes_unchanged or prod_diff_attributable_to_independent_daemon

    gates = {
        "socket_up_cold_lt_5s": daemon_cold["socket_up_sec"] < _SOCKET_UP_BUDGET_SEC,
        "socket_up_warm_lt_5s": warm_reboot["socket_up_sec"] < _SOCKET_UP_BUDGET_SEC,
        "first_recall_correct_nonzero": first_recall_correct,
        "first_recall_in_steady_band": first_recall_in_steady_band,
        "steady_state_le_3s_by_recall3": steady_state_by_recall3,
        "recall_at_10_mean_ge_0_98": mean_recall_at_10 >= 0.98,
        # The recall@10 comparison is only meaningful if the reference dispatched
        # with the SAME structural state the daemon uses (community assignment
        # present, not cold_degrade). A cold-degraded reference is structure-blind
        # and would diverge at the k-boundary for reasons unrelated to the engine.
        "reference_structure_informed": bool(
            ref_recall.get("reference_structure_informed", False)
        ),
        "reference_validated": bool(reference_validation.get("ok", False)),
        "warm_reboot_correct": bool(warm_reboot.get("well_formed")) and warm_reboot.get("hit_count", 0) > 0,
        "sleep_smoke_pass_llm_free": bool(smoke.get("smoke_pass")) and bool(smoke.get("llm_free")),
        "sidecar_re_persisted": bool(smoke.get("sidecar_re_persisted")),
        "prod_untouched_by_harness": prod_untouched_by_harness,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"

    return {
        "prod_post": prod_post,
        "prod_bytes_unchanged": prod_bytes_unchanged,
        "prod_diff_attributable_to_independent_daemon": prod_diff_attributable_to_independent_daemon,
        "prod_untouched_by_harness": prod_untouched_by_harness,
        "per_cue_recall_at_10": per_cue_recall_at_10,
        "mean_recall_at_10": mean_recall_at_10,
        "gates": gates,
        "verdict": verdict,
    }


def run_proof(source: Path, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)

    print("== stage 1: copies + persist prep ==")
    copies = stage_copies(source, workdir)
    proof_root = Path(copies["proof_root"])
    _preflight_daemon_state_path(proof_root)

    print("== stage 2: reference build + validation ==")
    reference = stage_reference_build(proof_root, workdir)
    ref_root = Path(reference["ref_root"])

    if not reference["validation"].get("ok", False):
        print(
            "FAIL: reference validation did not pass -- refusing to proceed "
            "to recall stages (the reference is not a valid truth bar).",
            file=sys.stderr,
        )
        return {
            "cue_set_sha256": cue_set_sha256(),
            "cues": list(FIXED_CUES),
            "copies": copies,
            "reference": reference,
            "aborted": True,
            "abort_reason": "reference_validation_failed",
        }

    print("== stage 3: reference recall ==")
    ref_recall = stage_reference_recall(ref_root)

    print("== stage 4: daemon cold proof ==")
    daemon_cold = stage_daemon_proof(proof_root)

    print("== stage 5: warm reboot ==")
    warm_reboot = stage_warm_reboot(proof_root)

    print("== stage 6: sleep smoke ==")
    smoke = stage_sleep_smoke(proof_root)

    print("== stage 7: verdict ==")
    verdict = stage_verdict(
        copies["prod_pre"], daemon_cold, ref_recall, warm_reboot, smoke,
        reference["validation"],
    )

    return {
        "cue_set_sha256": cue_set_sha256(),
        "cues": list(FIXED_CUES),
        "copies": copies,
        "reference": reference,
        "reference_recall": ref_recall,
        "daemon_cold_proof": daemon_cold,
        "warm_reboot": warm_reboot,
        "sleep_smoke": smoke,
        "verdict": verdict,
        "aborted": False,
        "note": (
            "Disposable working copies are left on disk for manual "
            f"inspection: proof_root={proof_root}, ref_root={ref_root}. "
            "Not auto-deleted."
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end isolated daemon boot proof: a real daemon subprocess "
            "on a lilli copy of a production corpus, validated against a "
            "stdlib reference. NEVER live prod."
        )
    )
    parser.add_argument("--source", help="Source isolated lilli copy to duplicate")
    parser.add_argument("--workdir", help="Scratch directory for disposable working copies")
    parser.add_argument("--out", help="Optional path to also write the aggregate JSON")
    parser.add_argument("--stage-worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-in", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.stage_worker:
        _stage_worker_main(args.stage_worker, Path(args.worker_in), Path(args.worker_out))
        return

    if not args.source or not args.workdir:
        parser.error("--source and --workdir are required (unless --stage-worker)")

    source = Path(args.source).expanduser()
    workdir = Path(args.workdir).expanduser()

    _refuse_if_prod(source)
    _refuse_if_prod(workdir)

    aggregate = run_proof(source, workdir)
    print("\nJSON summary:")
    print(json.dumps(aggregate, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(aggregate, indent=2, default=str))


if __name__ == "__main__":
    main()
