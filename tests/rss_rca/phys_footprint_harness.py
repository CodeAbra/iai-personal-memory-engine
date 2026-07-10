"""Hermetic phys_footprint vs RSS phantom-memory measurement harness.

Builds a synthetic warm working set (N records with RANDOM float32 vectors, a
real HNSW index, and ~1.3*N synthetic edges) in a throwaway tmp store, warms the
real BERT embedder singleton, then drives the heavy WARM paths in-process
(``build_runtime_graph``, a full sleep-pipeline cycle, and a deferred-capture
drain). At each phase it samples BOTH the macOS ``phys_footprint`` (the kernel's
"charged" memory — the jetsam metric, which EXCLUDES reusable MADV_FREE pages)
and ``RSS`` (which counts them), for this process AND the live spawn-children,
exposing the phantom delta (RSS - phys_footprint) at scale.

The headline question this answers, by measurement: how much of the live ~3 GiB
warm working set is *phantom* (reusable pages RSS counts but phys_footprint and
jetsam do not) versus genuinely *live*. That fraction decides whether the cheap
allocator/mmap/measurement wins suffice or the heavy child-isolation refactors
are justified.

Self-limiting by construction
------------------------------
The actual measurement runs inside a child ``--worker`` process. The parent
(default mode) spawns that worker, samples its RSS every interval, and if RSS
exceeds a hard cap (``--cap-gib``, default 4.5) it kills the worker tree and
reports the trend from whatever smaller scales already completed. ONE heavy
process at a time — the parent itself stays tiny (it only reads psutil and
parses one JSON line).

Hermeticity
-----------
``HOME``, ``IAI_MCP_STORE`` and ``IAI_MCP_USER_MODEL_PATH`` are redirected under
a ``tempfile.mkdtemp`` root *before* any project module import, so the daemon
code reads and writes only under the tmp root. A signature of the real
``~/.iai-mcp/`` is taken before/after the worker run and any drift raises.

Invocation::

    source .venv/bin/activate
    python tests/rss_rca/phys_footprint_harness.py --n 30000
    python tests/rss_rca/phys_footprint_harness.py --n 5000 15000 30000

This is a measurement tool. It changes no production source and never starts the
live daemon.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

# Ensure the repo root is importable so ``tests.rss_rca.*`` resolves whether the
# script is run as a module or by absolute path (the capped worker is spawned by
# absolute path, which does not put the repo root on sys.path automatically).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Volatile prefixes under the real ~/.iai-mcp/ that legitimately change while a
# live daemon runs — skipped by the hermeticity signature so a concurrent
# heartbeat tick never registers as a violation of THIS harness's isolation.
_VOLATILE_PREFIXES: tuple[str, ...] = (
    ".daemon.sock",
    ".heartbeat",
    "heartbeat",
    "wrappers/heartbeat",
    "wrappers/",
    "logs/",
    ".session-start-payload",
    ".deferred-captures/",
    ".capture-state",
    "daemon_state.json",
    "lifecycle_state.json",
)

_BYTES_PER_GIB = 1024.0 ** 3
_BYTES_PER_MIB = 1024.0 ** 2


# ---------------------------------------------------------------------------
# Hermeticity signature of the real user store.
# ---------------------------------------------------------------------------


def _real_store_signature() -> dict[str, tuple[int, int]] | None:
    """Fingerprint the real ``~/.iai-mcp/`` for the hermeticity check.

    Returns ``None`` if it does not exist, else a mapping of relative path to
    ``(size, mtime_ns)``. Volatile prefixes are skipped.
    """
    real = Path.home() / ".iai-mcp"
    if not real.exists():
        return None
    sig: dict[str, tuple[int, int]] = {}
    for p in real.rglob("*"):
        try:
            rel = str(p.relative_to(real))
        except ValueError:
            continue
        if any(
            rel.startswith(v) or rel == v.rstrip("/")
            for v in _VOLATILE_PREFIXES
        ):
            continue
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        sig[rel] = (st.st_size, st.st_mtime_ns)
    return sig


# ---------------------------------------------------------------------------
# phys_footprint reader — vendored from daemon/_watchdog.py:_phys_footprint_bytes.
# Vendored (not imported) so the worker can read its OWN footprint with zero
# import side effects beyond what the measurement already pulls in, and so the
# tool stands alone if the watchdog internal is ever refactored.
# ---------------------------------------------------------------------------

_RUSAGE_INFO_V2: int = 2


def _phys_footprint_bytes(pid: int | None = None) -> int | None:
    """Return ``pid``'s macOS phys_footprint in bytes, or None.

    phys_footprint is the kernel's charged memory — the same number the
    memory-pressure / jetsam machinery grades a process against, and the value
    Activity Monitor shows as "Memory". Unlike resident_size it EXCLUDES
    reusable (MADV_FREE) pages an allocator freed but has not returned. macOS
    only; any other platform or any error returns None.

    When ``pid`` is None the calling process's own footprint is read. A non-None
    ``pid`` reads another process via ``proc_pid_rusage`` (works for any pid the
    caller may signal, which covers the parent's own spawned worker).
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class _RUsageInfoV2(ctypes.Structure):
            _fields_ = [
                ("ri_uuid", ctypes.c_uint8 * 16),
                ("ri_user_time", ctypes.c_uint64),
                ("ri_system_time", ctypes.c_uint64),
                ("ri_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_interrupt_wkups", ctypes.c_uint64),
                ("ri_pageins", ctypes.c_uint64),
                ("ri_wired_size", ctypes.c_uint64),
                ("ri_resident_size", ctypes.c_uint64),
                ("ri_phys_footprint", ctypes.c_uint64),
                ("ri_proc_start_abstime", ctypes.c_uint64),
                ("ri_proc_exit_abstime", ctypes.c_uint64),
                ("ri_child_user_time", ctypes.c_uint64),
                ("ri_child_system_time", ctypes.c_uint64),
                ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_child_interrupt_wkups", ctypes.c_uint64),
                ("ri_child_pageins", ctypes.c_uint64),
                ("ri_child_elapsed_abstime", ctypes.c_uint64),
                ("ri_diskio_bytesread", ctypes.c_uint64),
                ("ri_diskio_byteswritten", ctypes.c_uint64),
            ]

        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        info = _RUsageInfoV2()
        target = os.getpid() if pid is None else int(pid)
        rc = libsystem.proc_pid_rusage(
            target, _RUSAGE_INFO_V2, ctypes.byref(info),
        )
        if rc != 0:
            return None
        footprint = int(info.ri_phys_footprint)
        if footprint <= 0:
            return None
        return footprint
    except Exception:  # noqa: BLE001 -- unreadable footprint must never crash
        return None


# ---------------------------------------------------------------------------
# Per-phase aggregate sample: parent + live children, RSS + phys_footprint.
# ---------------------------------------------------------------------------


def _own_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def _live_children() -> list[psutil.Process]:
    """Return the live descendant processes of THIS process."""
    try:
        return list(psutil.Process().children(recursive=True))
    except Exception:  # noqa: BLE001
        return []


def sample_aggregate(phase: str) -> dict[str, Any]:
    """Sample RSS + phys_footprint for self and every live child.

    Returns a dict with the phase label, the parent's RSS and phys_footprint,
    the summed children RSS and phys_footprint, the totals (parent + children),
    the live child count, and the phantom delta (RSS - phys) for the total. The
    aggregate over parent + children is the multi-socratic's measurement point:
    a spawned graph/HNSW child fleet pushes resident pages the parent's own
    footprint never sees.
    """
    parent_rss = _own_rss_bytes()
    parent_phys = _phys_footprint_bytes()

    children_rss = 0
    children_phys = 0
    children_phys_readable = True
    n_children = 0
    for child in _live_children():
        n_children += 1
        try:
            children_rss += int(child.memory_info().rss)
        except Exception:  # noqa: BLE001 -- child may exit mid-sample
            continue
        cphys = _phys_footprint_bytes(child.pid)
        if cphys is None:
            children_phys_readable = False
        else:
            children_phys += cphys

    total_rss = parent_rss + children_rss
    # phys_footprint total only meaningful when every live process reported it.
    if parent_phys is not None and children_phys_readable:
        total_phys: int | None = parent_phys + children_phys
    else:
        total_phys = None

    if total_phys is not None:
        phantom_delta = max(0, total_rss - total_phys)
        phantom_frac = (phantom_delta / total_rss) if total_rss > 0 else 0.0
    else:
        phantom_delta = None
        phantom_frac = None

    return {
        "phase": phase,
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_children": n_children,
        "parent_rss": parent_rss,
        "parent_phys": parent_phys,
        "children_rss": children_rss,
        "children_phys": children_phys if children_phys_readable else None,
        "total_rss": total_rss,
        "total_phys": total_phys,
        "phantom_delta": phantom_delta,
        "phantom_frac": phantom_frac,
    }


# ---------------------------------------------------------------------------
# Worker — the actual in-process measurement. Runs as a child subprocess so the
# parent can cap it. Prints exactly one JSON result line prefixed with a sentinel
# so the parent can find it amid any library log noise on stdout.
# ---------------------------------------------------------------------------

_RESULT_SENTINEL = "PHYS_FOOTPRINT_RESULT_JSON "


class _PeakSampler:
    """Background thread tracking peak aggregate RSS/phys with a phase label.

    The discrete per-phase ``record()`` samples only fire at phase boundaries, so
    a transient spike *inside* a heavy step (e.g. a betweenness recompute or an
    OPTIMIZE_HIPPO rebuild) lands between two boundaries and is invisible to
    them. This thread samples every ``interval`` seconds and remembers the single
    highest total-RSS sample together with the phase label that was active when
    it occurred — so the result can attribute the peak to the step that caused
    it even when the worker is later killed at the cap.
    """

    def __init__(
        self,
        interval: float = 0.25,
        out_fh: Any | None = None,
        out_lock: threading.Lock | None = None,
    ) -> None:
        self._interval = float(interval)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase = "boot"
        self._peak_rss = 0
        self._peak_sample: dict[str, Any] | None = None
        self._out_fh = out_fh
        self._out_lock = out_lock
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                phase = self._phase
            try:
                s = sample_aggregate(phase)
                if s["total_rss"] > self._peak_rss:
                    self._peak_rss = s["total_rss"]
                    self._peak_sample = s
                    # Persist each NEW peak immediately so that, if the parent
                    # later kills this worker at the cap, the on-disk phase log
                    # still names the phase that drove the peak.
                    if self._out_fh is not None and self._out_lock is not None:
                        row = dict(s)
                        row["kind"] = "continuous-peak"
                        with self._out_lock:
                            self._out_fh.write(json.dumps(row) + "\n")
                            self._out_fh.flush()
            except Exception:  # noqa: BLE001 -- sampler must never crash the run
                pass
            self._stop.wait(self._interval)

    def peak(self) -> dict[str, Any] | None:
        return self._peak_sample

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _stage_deferred_batch(home_root: Path, n_events: int) -> Path:
    """Write one ``.deferred-captures/*.jsonl`` so the drain has real work."""
    deferred_dir = home_root / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    session = "phys-footprint-harness"
    fpath = deferred_dir / f"{session}-{int(time.time())}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session,
        "cwd": str(home_root),
    }
    with fpath.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for i in range(n_events):
            ev = {
                "text": (
                    f"deferred capture {i}: a synthetic sentence with enough "
                    f"length to clear the drain gate and exercise reembed"
                ),
                "cue": f"cue-{i}",
                "tier": "episodic",
                "role": "user",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(ev) + "\n")
    return fpath


_CLUSTER_SIZE = 10  # records per synthetic hebbian cluster (bounded)


def _build_synthetic_edges(store: Any, n_records: int, edge_factor: float, seed: int) -> int:
    """Insert ~``edge_factor * n_records`` synthetic edges over the corpus ids.

    Edges are written through the store's own ``boost_edges`` path (the public,
    supported edge-write API) so the resident edges table and the graph build
    over it are realistic.

    Topology: records are partitioned into bounded clusters of ``_CLUSTER_SIZE``
    and edges are drawn ONLY within each cluster, producing many small
    connected components rather than one giant component. This mirrors the real
    corpus's hebbian structure: the sleep pipeline's CLUSTER_SUMMARY step runs
    ``combinations(cluster_ids, 2)`` per connected component, so a single giant
    component would explode into a combinatorial edge boost (millions of pairs)
    that does not represent real usage. A bounded cluster of 10 yields 45 pairs
    — realistic. Returns the edge count actually written.
    """
    ids = _read_record_ids(store)
    if not ids:
        return 0

    n = len(ids)
    target = int(round(edge_factor * n_records))
    pairs: list[tuple[Any, Any]] = []
    # Walk each bounded cluster and add intra-cluster links until the per-record
    # edge budget for that cluster is met. A ring plus a few chords per cluster
    # gives ~1.3 edges/record without crossing cluster boundaries.
    for start in range(0, n, _CLUSTER_SIZE):
        cluster = ids[start:start + _CLUSTER_SIZE]
        c = len(cluster)
        if c < 2:
            continue
        # Ring edges: i -> i+1 (c edges) connect the cluster.
        for i in range(c):
            pairs.append((cluster[i], cluster[(i + 1) % c]))
        # A few chords (i -> i+2) to push the per-record edge count toward the
        # target factor while staying bounded inside the cluster.
        extra = max(0, int(round((edge_factor - 1.0) * c)))
        for k in range(extra):
            i = k % c
            j = (i + 2) % c
            if i != j:
                pairs.append((cluster[i], cluster[j]))
        if len(pairs) >= target:
            break

    pairs = pairs[:target] if len(pairs) > target else pairs
    # Batch so boost_edges takes the bulk merge_insert path (>4 per call).
    inserted = 0
    batch = 2000
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        store.boost_edges(chunk, delta=0.1, edge_type="hebbian")
        inserted += len(chunk)
    return inserted


def _read_record_ids(store: Any) -> list[Any]:
    """Read all record ids from the store in a stable order."""
    from uuid import UUID

    ids: list[Any] = []
    try:
        lock = getattr(store.db, "_conn_lock", None)
        if lock is not None:
            with lock:
                rows = store.db._conn.execute(
                    "SELECT id FROM records ORDER BY rowid"
                ).fetchall()
        else:
            rows = store.db._conn.execute(
                "SELECT id FROM records ORDER BY rowid"
            ).fetchall()
        for row in rows:
            raw = row[0] if hasattr(row, "__getitem__") else row["id"]
            try:
                ids.append(UUID(str(raw)))
            except (ValueError, TypeError):
                ids.append(raw)
    except Exception:  # noqa: BLE001 -- fall back to empty (no edges) on any read issue
        return []
    return ids


def _worker_run(
    *,
    n: int,
    seed: int,
    edge_factor: float,
    deferred_events: int,
    out_jsonl: Path | None,
    sleep_budget_sec: float,
) -> dict[str, Any]:
    """In-process measurement. Returns the result dict (also written to jsonl)."""
    # Hermeticity: snapshot real store BEFORE any redirect.
    real_sig_before = _real_store_signature()

    tmp_root = Path(tempfile.mkdtemp(prefix="iai-phys-fp-"))
    prev_env = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
        )
    }
    os.environ["HOME"] = str(tmp_root)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_root / ".iai-mcp" / "user_model.json"
    )
    if not prev_env["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-phys-footprint-harness-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    phases: list[dict[str, Any]] = []
    out_fh = out_jsonl.open("w", encoding="utf-8") if out_jsonl is not None else None
    out_lock = threading.Lock()
    peak_sampler = _PeakSampler(
        interval=0.25, out_fh=out_fh, out_lock=out_lock,
    )
    peak_sampler.start()

    def record(label: str) -> dict[str, Any]:
        peak_sampler.set_phase(label)
        s = sample_aggregate(label)
        phases.append(s)
        if out_fh is not None:
            with out_lock:
                out_fh.write(json.dumps(s) + "\n")
                out_fh.flush()
        return s

    store = None
    remote_stub_originals: dict[str, object] = {}
    try:
        # Imports AFTER env redirect so module-level Path.home()/IAI_MCP_STORE
        # reads pick up the tmp root.
        from iai_mcp.store import MemoryStore
        from iai_mcp import retrieve
        from iai_mcp.capture import drain_deferred_captures
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
        from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
            _step_memory_relief,
        )

        # Block any remote subprocess spawn (claude-cli / reconsolidation
        # critic) — the harness is strictly local and must never bill the
        # subscription nor spawn the daemon's REM child. The originals are
        # restored in the finally block so an in-process run never leaves the
        # shared modules mutated for the rest of the pytest session.
        import iai_mcp.claude_cli as _cc

        def _boom(*_a: object, **_kw: object) -> dict:
            raise RuntimeError("remote subprocess blocked in phys-footprint harness")

        remote_stub_originals["cc_invoke_claude_sync"] = _cc.invoke_claude_sync
        remote_stub_originals["cc_invoke_claude_once"] = _cc.invoke_claude_once
        _cc.invoke_claude_sync = _boom  # type: ignore[assignment]
        _cc.invoke_claude_once = _boom  # type: ignore[assignment]
        try:
            import iai_mcp.reconsolidation_critic as _rc

            def _no_critic(_pool: object, **_kw: object) -> dict:
                return {}

            remote_stub_originals["rc_evaluate_batch_reconsolidation"] = (
                _rc.evaluate_batch_reconsolidation
            )
            _rc.evaluate_batch_reconsolidation = _no_critic  # type: ignore[assignment]
        except Exception:  # noqa: BLE001 -- critic module optional
            pass

        record("boot")

        # Store + corpus (random vectors, real HNSW via the normal insert path).
        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)
        eff_dim = int(store.embed_dim)

        from tests.rss_rca.corpus import build_corpus
        from iai_mcp.store._buffers import (
            flush_record_buffer, flush_edge_buffer,
        )

        t0 = time.perf_counter()
        inserted = build_corpus(
            store, n=n, seed=seed, dim=eff_dim,
            varied_tags=True, with_hv_payload=True,
        )
        # ``insert`` buffers rows (flush at 500 / 5 s); force the buffer out so
        # the records table is materialised and the resident HNSW index holds
        # every vector before the heavy paths read it back.
        flush_record_buffer(store)
        corpus_sec = time.perf_counter() - t0
        s_corpus = record("after-corpus")

        # Synthetic edges (~edge_factor * N), flushed to the edges table.
        t0 = time.perf_counter()
        edges = _build_synthetic_edges(store, n, edge_factor, seed)
        flush_edge_buffer(store)
        edges_sec = time.perf_counter() - t0
        record("after-edges")

        # Warm the real BERT embedder singleton the way the daemon does. Warm
        # ONCE — do NOT embed the corpus (that would cost hours).
        from iai_mcp.embed import embedder_for_store
        import iai_mcp.embed as _embed_mod

        t0 = time.perf_counter()
        warm = embedder_for_store(store)
        prime_vec = warm.embed("alice daily routine warm-up prime encode")

        # Hold the warm embedder so its ~290 MB BERT residency stays in the
        # measured footprint, but short-circuit repeated ``embed`` calls to the
        # primed vector. The corpus uses RANDOM vectors and the cluster
        # summaries are synthetic, so the embedded VALUE is irrelevant to the
        # MEMORY profile this harness measures — yet a real BERT forward per
        # cluster summary (hundreds of clusters at scale) is CPU-bound and would
        # make the sweep take tens of minutes per scale. The model stays loaded
        # and resident; only the redundant forward passes are elided.
        import numpy as _np

        _prime = list(_np.asarray(prime_vec, dtype=_np.float32))

        class _HeldEmbedder:
            def __init__(self, inner: object) -> None:
                self._inner = inner  # keeps the loaded model alive/resident

            def embed(self, _text: object) -> list:
                return list(_prime)

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

        held = _HeldEmbedder(warm)

        def _held(_store: object) -> object:
            return held

        _embed_mod.embedder_for_store = _held  # type: ignore[assignment]
        warm_sec = time.perf_counter() - t0
        record("after-warm-embedder")

        # Heavy warm path 1 — the in-parent runtime graph build.
        peak_sampler.set_phase("during-build-runtime-graph")
        t0 = time.perf_counter()
        try:
            retrieve.build_runtime_graph(store)
        except Exception as exc:  # noqa: BLE001 -- record the failure, keep measuring
            graph_err = repr(exc)
        else:
            graph_err = None
        graph_sec = time.perf_counter() - t0
        record("after-build-runtime-graph")

        # Heavy warm path 2 — a full sleep-pipeline cycle (SCHEMA_MINE,
        # KNOB_TUNE, OPTIMIZE_HIPPO with WAL checkpoint + VACUUM + hnswlib
        # rebuild, HIPPO_CLEANUP, DREAM_DECAY) plus the essential-variable hook.
        sleep_log_dir = tmp_root / ".iai-mcp" / "logs"
        sleep_log_dir.mkdir(parents=True, exist_ok=True)
        event_log = LifecycleEventLog(log_dir=sleep_log_dir)
        pipeline = SleepPipeline(
            store=store,
            lifecycle_state_path=tmp_root / ".iai-mcp" / "lifecycle_state.json",
            event_log=event_log,
        )
        peak_sampler.set_phase("during-sleep-cycle")
        t0 = time.perf_counter()
        # Wall-clock interrupt: the memory peak is reached early in the cycle
        # (graph rebuild + NREM steps), but the late REM steps — chiefly
        # CLUSTER_SUMMARY, which re-embeds one summary per connected component
        # and boosts intra-cluster edges through merge_insert — are CPU-bound
        # and scale poorly. A deadline lets the cycle do its memory-heavy work
        # and then return cleanly rather than grinding for tens of minutes; the
        # peak this harness measures is unaffected. ``interrupted`` is recorded.
        sleep_deadline = t0 + float(sleep_budget_sec)

        def _interrupt() -> bool:
            return time.perf_counter() >= sleep_deadline

        try:
            sleep_result = pipeline.force_run(interrupt_check=_interrupt)
            completed = [
                getattr(s, "name", str(s))
                for s in sleep_result.get("completed_steps", [])
            ]
            sleep_err = sleep_result.get("error")
            sleep_interrupted = bool(sleep_result.get("interrupted", False))
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring
            completed = []
            sleep_err = repr(exc)
            sleep_interrupted = False
        sleep_sec = time.perf_counter() - t0
        record("after-sleep-cycle")

        # Heavy warm path 3 — drain a modest deferred-capture backlog (this
        # reembeds each event through the warm embedder).
        staged = _stage_deferred_batch(tmp_root, deferred_events)
        peak_sampler.set_phase("during-drain")
        t0 = time.perf_counter()
        try:
            drain_counts = drain_deferred_captures(store)
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring
            drain_counts = {"error": repr(exc)}
        drain_sec = time.perf_counter() - t0
        record("after-drain")

        # Force every returnable page back to the OS, then sample the settled
        # footprint — this is the number the watchdog now actually gates on
        # (phys_footprint after relief). The RSS-vs-phys gap here is the steady
        # phantom residue.
        relief = _step_memory_relief("post-measurement")
        gc.collect()
        s_relief = record("after-memory-relief")

        # Peaks across all phases AND the continuous in-step peak sampler. The
        # continuous peak catches transients inside a heavy step that the
        # boundary samples miss (the betweenness / OPTIMIZE_HIPPO spike).
        boundary_peak = max(phases, key=lambda p: p["total_rss"])
        cont_peak = peak_sampler.peak()
        if cont_peak is not None and cont_peak["total_rss"] > boundary_peak["total_rss"]:
            top = cont_peak
        else:
            top = boundary_peak
        peak_rss = top["total_rss"]
        peak_phys = top["total_phys"]
        peak_rss_phase = top["phase"]

        result: dict[str, Any] = {
            "ok": True,
            "n": n,
            "dim": eff_dim,
            "inserted_records": inserted,
            "inserted_edges": edges,
            "edge_factor_actual": round(edges / n, 4) if n else 0.0,
            "deferred_events": deferred_events,
            "graph_err": graph_err,
            "sleep_completed_steps": completed,
            "sleep_interrupted": sleep_interrupted,
            "sleep_err": str(sleep_err) if sleep_err else None,
            "drain_counts": drain_counts,
            "durations_sec": {
                "corpus": round(corpus_sec, 2),
                "edges": round(edges_sec, 2),
                "warm_embedder": round(warm_sec, 2),
                "build_runtime_graph": round(graph_sec, 2),
                "sleep_cycle": round(sleep_sec, 2),
                "drain": round(drain_sec, 2),
            },
            "memory_relief": relief,
            "phases": phases,
            "continuous_peak": cont_peak,
            "peak_rss": peak_rss,
            "peak_rss_phase": peak_rss_phase,
            "peak_phys": peak_phys,
            "peak_phantom_delta": (
                max(0, peak_rss - peak_phys) if peak_phys is not None else None
            ),
            "peak_phantom_frac": (
                (max(0, peak_rss - peak_phys) / peak_rss)
                if (peak_phys is not None and peak_rss > 0)
                else None
            ),
            "post_relief_rss": s_relief["total_rss"],
            "post_relief_phys": s_relief["total_phys"],
        }
        return result

    finally:
        try:
            peak_sampler.stop()
        except Exception:  # noqa: BLE001
            pass
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:  # noqa: BLE001
                pass
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        # Restore the remote-call stubs so the shared iai_mcp modules are left
        # pristine for the rest of an in-process pytest session.
        if remote_stub_originals:
            try:
                import iai_mcp.claude_cli as _cc_restore

                if "cc_invoke_claude_sync" in remote_stub_originals:
                    _cc_restore.invoke_claude_sync = (  # type: ignore[assignment]
                        remote_stub_originals["cc_invoke_claude_sync"]
                    )
                if "cc_invoke_claude_once" in remote_stub_originals:
                    _cc_restore.invoke_claude_once = (  # type: ignore[assignment]
                        remote_stub_originals["cc_invoke_claude_once"]
                    )
                if "rc_evaluate_batch_reconsolidation" in remote_stub_originals:
                    import iai_mcp.reconsolidation_critic as _rc_restore

                    _rc_restore.evaluate_batch_reconsolidation = (  # type: ignore[assignment]
                        remote_stub_originals["rc_evaluate_batch_reconsolidation"]
                    )
            except Exception:  # noqa: BLE001
                pass
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp_root, ignore_errors=True)
        real_sig_after = _real_store_signature()
        if real_sig_before != real_sig_after:
            before = real_sig_before if isinstance(real_sig_before, dict) else {}
            after = real_sig_after if isinstance(real_sig_after, dict) else {}
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            modified = sorted(
                k for k in set(before) & set(after) if before[k] != after[k]
            )
            raise RuntimeError(
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the harness. "
                f"added={added!r} removed={removed!r} modified={modified!r}."
            )


# ---------------------------------------------------------------------------
# Parent — spawn the worker, cap it, parse its result, print the table.
# ---------------------------------------------------------------------------


def _fmt_gib(b: int | None) -> str:
    if b is None:
        return "    n/a"
    return f"{b / _BYTES_PER_GIB:6.3f}"


def _fmt_mib(b: int | None) -> str:
    if b is None:
        return "   n/a"
    return f"{b / _BYTES_PER_MIB:6.0f}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "  n/a"
    return f"{x * 100:4.1f}%"


def _run_capped_worker(
    *,
    n: int,
    seed: int,
    edge_factor: float,
    deferred_events: int,
    cap_gib: float,
    sample_interval: float,
    sleep_budget_sec: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Spawn the worker as a subprocess, sampling its RSS and killing on cap.

    Returns the worker's result dict on success, or a dict with ``aborted`` /
    ``error`` describing why the scale could not complete. The parent samples
    the worker's RSS every ``sample_interval`` seconds and, if it exceeds
    ``cap_gib``, terminates the worker tree (mirrors the daemon watchdog's
    auto-kill) so the M2 never approaches jetsam.
    """
    out_jsonl = out_dir / f"phases-n{n}.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n", str(n),
        "--seed", str(seed),
        "--edge-factor", str(edge_factor),
        "--deferred-events", str(deferred_events),
        "--sleep-budget-sec", str(sleep_budget_sec),
        "--worker-out", str(out_jsonl),
    ]
    cap_bytes = cap_gib * _BYTES_PER_GIB

    env = dict(os.environ)
    # Run from the repo root so ``import tests.rss_rca.corpus`` resolves.
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    peak_worker_rss = 0
    aborted = False
    abort_reason = ""
    try:
        ps_proc = psutil.Process(proc.pid)
    except Exception:  # noqa: BLE001
        ps_proc = None

    while True:
        ret = proc.poll()
        if ret is not None:
            break
        # Sample the worker subtree's aggregate RSS (worker + any child it
        # spawned) so the cap accounts for a spawned graph/HNSW child fleet.
        subtree_rss = 0
        if ps_proc is not None:
            try:
                subtree_rss = int(ps_proc.memory_info().rss)
                for ch in ps_proc.children(recursive=True):
                    try:
                        subtree_rss += int(ch.memory_info().rss)
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                subtree_rss = 0
        peak_worker_rss = max(peak_worker_rss, subtree_rss)
        if subtree_rss > cap_bytes:
            aborted = True
            abort_reason = (
                f"worker subtree RSS {subtree_rss / _BYTES_PER_GIB:.3f} GiB "
                f"exceeded cap {cap_gib:.3f} GiB at N={n}"
            )
            _kill_tree(proc, ps_proc)
            break
        time.sleep(sample_interval)

    stdout, stderr = proc.communicate(timeout=60)

    if aborted:
        return {
            "ok": False,
            "aborted": True,
            "n": n,
            "reason": abort_reason,
            "peak_worker_rss": peak_worker_rss,
            "stderr_tail": (stderr or "")[-2000:],
        }

    if proc.returncode != 0:
        return {
            "ok": False,
            "aborted": False,
            "n": n,
            "error": f"worker exited with code {proc.returncode}",
            "peak_worker_rss": peak_worker_rss,
            "stderr_tail": (stderr or "")[-2000:],
        }

    # Find the result sentinel line.
    result: dict[str, Any] | None = None
    for line in (stdout or "").splitlines():
        if line.startswith(_RESULT_SENTINEL):
            try:
                result = json.loads(line[len(_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                result = None
            break
    if result is None:
        return {
            "ok": False,
            "aborted": False,
            "n": n,
            "error": "worker produced no parseable result line",
            "peak_worker_rss": peak_worker_rss,
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-2000:],
        }
    result["peak_worker_rss_observed_by_parent"] = peak_worker_rss
    return result


def _kill_tree(proc: subprocess.Popen, ps_proc: "psutil.Process | None") -> None:
    """Terminate the worker process tree, escalating to SIGKILL."""
    procs: list[psutil.Process] = []
    if ps_proc is not None:
        try:
            procs = ps_proc.children(recursive=True)
        except Exception:  # noqa: BLE001
            procs = []
        procs.append(ps_proc)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    gone, alive = psutil.wait_procs(procs, timeout=5) if procs else ([], [])
    for p in alive:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _print_phase_table(result: dict[str, Any]) -> None:
    n = result.get("n")
    print(f"\n=== N={n} per-phase aggregate (RSS / phys_footprint, GiB) ===")
    print(
        f"{'phase':<26} {'RSS':>7} {'phys':>7} {'phantom':>8} "
        f"{'chld#':>5} {'chRSS(MiB)':>11} {'totRSS':>7} {'totPhys':>7}"
    )
    for p in result.get("phases", []):
        phantom = p.get("phantom_delta")
        print(
            f"{p['phase']:<26} "
            f"{_fmt_gib(p['parent_rss'])} "
            f"{_fmt_gib(p['parent_phys'])} "
            f"{_fmt_gib(phantom)} "
            f"{p['n_children']:>5} "
            f"{_fmt_mib(p['children_rss']):>11} "
            f"{_fmt_gib(p['total_rss'])} "
            f"{_fmt_gib(p['total_phys'])}"
        )


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== SUMMARY (per scale) ===")
    print(
        f"{'N':>7} {'peakRSS':>8} {'peakPhys':>8} {'phantomGiB':>11} "
        f"{'phantom%':>9} {'postReliefPhys':>15} {'status':>10}"
    )
    for r in results:
        n = r.get("n")
        if not r.get("ok"):
            status = "ABORTED" if r.get("aborted") else "ERROR"
            print(f"{n:>7} {'-':>8} {'-':>8} {'-':>11} {'-':>9} {'-':>15} {status:>10}")
            continue
        print(
            f"{n:>7} "
            f"{_fmt_gib(r.get('peak_rss'))} "
            f"{_fmt_gib(r.get('peak_phys'))} "
            f"{_fmt_gib(r.get('peak_phantom_delta'))} "
            f"{_fmt_pct(r.get('peak_phantom_frac')):>9} "
            f"{_fmt_gib(r.get('post_relief_phys')):>15} "
            f"{'OK':>10}"
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phys_footprint_harness",
        description=(
            "Hermetic phys_footprint vs RSS phantom-memory harness. Builds an "
            "N-record synthetic warm working set in a tmp store and measures, "
            "at each heavy warm phase, how much of RSS is phantom (reusable "
            "pages phys_footprint/jetsam exclude). Runs each scale in a capped "
            "subprocess."
        ),
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[30000],
        help="record count(s) to seed; multiple values run sequentially",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--edge-factor", type=float, default=1.3,
        help="edges = round(edge_factor * N) (default 1.3)",
    )
    parser.add_argument(
        "--deferred-events", type=int, default=250,
        help="deferred-capture events staged for the drain (default 250)",
    )
    parser.add_argument(
        "--cap-gib", type=float, default=4.5,
        help="hard RSS cap (GiB) — worker killed if its subtree exceeds it",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=1.0,
        help="seconds between worker RSS samples in the parent (default 1.0)",
    )
    parser.add_argument(
        "--sleep-budget-sec", type=float, default=180.0,
        help=(
            "wall-clock budget for the in-process sleep cycle (default 180); "
            "the cycle is interrupted after this so the slow late REM steps do "
            "not grind for tens of minutes — the memory peak is reached earlier"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="directory for per-scale phase jsonl + summary json (optional)",
    )
    # Worker-mode flags (set by the parent; not for direct use).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        # Inner mode — run the measurement, print exactly one sentinel result.
        n = args.n[0]
        try:
            result = _worker_run(
                n=n,
                seed=args.seed,
                edge_factor=args.edge_factor,
                deferred_events=args.deferred_events,
                out_jsonl=args.worker_out,
                sleep_budget_sec=args.sleep_budget_sec,
            )
        except Exception as exc:  # noqa: BLE001 -- report failure as a result line
            result = {"ok": False, "n": n, "error": repr(exc)}
        sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
        sys.stdout.flush()
        return 0 if result.get("ok") else 1

    # Parent mode.
    out_dir = (args.out or Path(tempfile.mkdtemp(prefix="iai-phys-fp-out-"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"phys_footprint harness — scales={args.n} cap={args.cap_gib} GiB "
        f"edge_factor={args.edge_factor} out={out_dir}"
    )

    results: list[dict[str, Any]] = []
    for n in args.n:
        print(f"\n[scale N={n}] spawning capped worker ...")
        t0 = time.perf_counter()
        result = _run_capped_worker(
            n=n,
            seed=args.seed,
            edge_factor=args.edge_factor,
            deferred_events=args.deferred_events,
            cap_gib=args.cap_gib,
            sample_interval=args.sample_interval,
            sleep_budget_sec=args.sleep_budget_sec,
            out_dir=out_dir,
        )
        result["wall_sec"] = round(time.perf_counter() - t0, 1)
        results.append(result)
        if result.get("ok"):
            _print_phase_table(result)
            print(
                f"[scale N={n}] peak_rss={_fmt_gib(result.get('peak_rss'))} GiB "
                f"peak_phys={_fmt_gib(result.get('peak_phys'))} GiB "
                f"phantom_frac={_fmt_pct(result.get('peak_phantom_frac'))} "
                f"({result['wall_sec']}s)"
            )
        else:
            if result.get("aborted"):
                print(f"[scale N={n}] ABORTED: {result.get('reason')}")
                print("[scale N={n}] stopping the scale sweep at the cap; "
                      "reporting trend from completed scales.")
                # Persist what we have and stop escalating to larger N.
                _persist_results(out_dir, results)
                _print_summary(results)
                return 0
            print(f"[scale N={n}] ERROR: {result.get('error')}")
            if result.get("stderr_tail"):
                print("---- worker stderr tail ----")
                print(result["stderr_tail"])

    _persist_results(out_dir, results)
    _print_summary(results)
    return 0


def _persist_results(out_dir: Path, results: list[dict[str, Any]]) -> None:
    summary_path = out_dir / "summary.json"
    try:
        summary_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nresults written to {summary_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"failed to write summary: {exc!r}")


if __name__ == "__main__":
    sys.exit(_main())
