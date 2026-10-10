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
code reads and writes only under the tmp root. Hermeticity is then enforced BY
CONSTRUCTION: a ``sys.addaudithook`` installed immediately after the redirect
(and before any project import) records every ``open``/``os.open``/``os.mkdir``/
``os.rename``/``os.remove`` whose path falls outside a code-enumerated
allowlist (the tmp sandbox, the venv, the HF model cache, ``/dev``, and the
run's own out-dir); the worker asserts zero violations at exit. This replaces
an earlier before/after signature diff of the real ``~/.iai-mcp/``, which a
concurrent OS process (e.g. a session-index hook) could churn independently of
this harness and false-positive.

Invocation::

    source .venv/bin/activate
    python tests/rss_rca/phys_footprint_harness.py --n 30000
    python tests/rss_rca/phys_footprint_harness.py --n 5000 15000 30000

This is a measurement tool. It changes no production source and never starts the
live daemon.

Fixture cache (two-process build/measure split)
------------------------------------------------
``--build-fixture`` builds one (N, seed, edge_factor) corpus once and copies
the resulting store into ``~/.cache/iai-phys-fp-fixtures/<engine-generation>/
n<N>-seed<seed>-ef<edge_factor>/`` (override the root with ``--cache-dir``) --
a system cache location, mirroring pip/cargo cache placement, that is never
accidentally committed to the repo. The engine generation is a hash of the
compiled native-engine extension's own bytes, so a rebuild invalidates stale
byte-incompatible fixtures. ``--use-fixture`` then runs ``--trials`` (default
3) fresh measurement trials against copies of that cached fixture, skipping
the corpus-build cost every trial after the first.
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

# Module level, not inside _build_corpus_step: the import cost must stay
# outside every timed region. Below the sys.path mutation above by necessity.
from tests.rss_rca.corpus import build_corpus  # noqa: E402

_BYTES_PER_GIB = 1024.0 ** 3
_BYTES_PER_MIB = 1024.0 ** 2


# ---------------------------------------------------------------------------
# Hermeticity by construction: a sys.addaudithook records any filesystem
# write outside a code-enumerated allowlist. Installed once per worker
# process, immediately after the env redirect and before any project import.
# ---------------------------------------------------------------------------

_AUDITED_FS_EVENTS = frozenset({"open", "os.open", "os.mkdir", "os.rename", "os.remove"})

_WRITE_MODE_CHARS = frozenset("wax+")
_OS_OPEN_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _open_event_is_write(args: tuple[Any, ...]) -> bool:
    """True if an ``open`` audit event requests a write.

    Builtin ``open()``/``io.open()`` and ``os.open()`` share this ONE event
    (confirmed empirically -- there is no separate "os.open" event): args are
    ``(file, mode, flags)`` where builtin open sets a string ``mode`` and
    ``os.open()`` sets ``mode=None`` with the real int flags in position 2.
    A plain read is not a hermeticity concern -- reading stdlib/venv/model-
    cache files is expected. An unrecognized shape defaults to WRITE (the
    safer default for an unexplained access).
    """
    mode = args[1] if len(args) > 1 else None
    if isinstance(mode, str):
        return any(c in mode for c in _WRITE_MODE_CHARS)
    flags = args[2] if len(args) > 2 else None
    if isinstance(flags, int):
        return bool(flags & _OS_OPEN_WRITE_FLAGS)
    return True


def _resolve_dir_fd(dir_fd: Any) -> str | None:
    """Resolve an ``*at()``-style ``dir_fd`` argument to its absolute path.

    macOS only (``fcntl.F_GETPATH``); returns None otherwise (non-macOS,
    closed fd, or ``dir_fd`` is None/False) so the caller falls back to
    treating the path as relative to cwd, the OS's own interpretation absent
    a dir_fd.
    """
    if dir_fd is None or dir_fd is False or not isinstance(dir_fd, int):
        return None
    if sys.platform != "darwin":
        return None
    try:
        import fcntl
        buf = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, b"\0" * 1024)
        return buf.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
    except OSError:
        return None


def _paths_for_audit_event(event: str, args: tuple[Any, ...]) -> list[str]:
    """Extract the filesystem path argument(s) an audited event acts on.

    Handles each mutating event's ``dir_fd`` argument (``os.mkdir``:
    ``(path, mode, dir_fd)``; ``os.remove``: ``(path, dir_fd)``;
    ``os.rename``: ``(src, dst, src_dir_fd, dst_dir_fd)``) -- a relative path
    under a dir_fd is joined against the fd's own resolved directory rather
    than left bare, since a bare relative name resolves against the WRONG
    base (process cwd) and could misjudge an allowlisted write as a
    violation (observed: huggingface_hub's lock cleanup uses dir_fd-relative
    ``os.remove``).
    """
    if event == "os.mkdir":
        items: list[tuple[Any, Any]] = [(args[0], args[2] if len(args) > 2 else None)]
    elif event == "os.remove":
        items = [(args[0], args[1] if len(args) > 1 else None)]
    elif event == "os.rename":
        items = [
            (args[0], args[2] if len(args) > 2 else None),
            (args[1], args[3] if len(args) > 3 else None),
        ]
    else:
        items = [(args[0] if args else None, None)]

    paths: list[str] = []
    for raw, dir_fd in items:
        if raw is None or isinstance(raw, int):
            continue
        try:
            p = (
                raw.decode("utf-8", "surrogateescape")
                if isinstance(raw, bytes) else os.fspath(raw)
            )
        except TypeError:
            continue
        if not os.path.isabs(p):
            base = _resolve_dir_fd(dir_fd)
            if base is not None:
                p = os.path.join(base, p)
        paths.append(p)
    return paths


def _path_is_allowlisted(path: str, allowlist_roots: tuple[str, ...]) -> bool:
    """True if ``path`` falls under one of ``allowlist_roots``.

    Both sides are realpath'd: a dir_fd-relative path is resolved by the
    kernel (fcntl.F_GETPATH) to its CANONICAL form, so comparing against a
    non-canonical root would false-positive every write under it (observed:
    macOS $TMPDIR is under /var, itself a symlink to /private/var). Following
    symlinks is also the conservative choice for a hermeticity check -- a
    write through a symlink that escapes an allowlisted root must not be
    silently treated as safe.
    """
    real_path = os.path.realpath(path)
    return any(
        real_path == root or real_path.startswith(root.rstrip(os.sep) + os.sep)
        for root in allowlist_roots
        if root
    )


def _build_allowlist_roots(tmp_root: Path, out_dir: Path | None) -> tuple[str, ...]:
    """Enumerate the roots this worker process may legitimately write under.

    Code-enumerated, not discovered blindly: the worker's own sandbox, the
    venv PLUS its base interpreter install (``sys.prefix``/``sys.base_prefix``
    -- a venv's own stdlib is typically not copied, so bytecode-cache writes
    land under the base install, not the venv root), the HF model cache
    (resolved independent of ``$HOME`` by ``bert.rs``'s ``Cache::from_env`` --
    a redirected HOME does not relocate it), ``/dev``, and the run's own
    out-dir (the parent process writes ``phases-nN.jsonl``/``summary.json``
    there, outside tmp_root). Every root is realpath'd to match
    `_path_is_allowlisted`'s canonical comparison. Confirmed by one
    logging-only run at N=500: the only writes outside this allowlist were
    bytecode-cache compiles under the base interpreter install and the
    checked-out source tree (eliminated by `sys.dont_write_bytecode`) and
    numba's JIT cache (redirected via `NUMBA_CACHE_DIR`) -- mimalloc/native-
    ext writes all land under tmp_root already.
    """
    roots = [tmp_root, Path(sys.prefix), Path(sys.base_prefix), Path("/dev")]
    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if hf_cache:
        roots.append(Path(hf_cache))
    if out_dir is not None:
        roots.append(out_dir)
    return tuple(os.path.realpath(r) for r in roots)


def _install_audit_hook(
    allowlist_roots: tuple[str, ...], violations: list[tuple[str, str]],
) -> None:
    """Install a process-lifetime audit hook recording out-of-allowlist writes.

    Never raises from inside the hook -- raising here aborts the very
    filesystem operation being observed, which would corrupt a legitimate
    in-sandbox write. Violations are recorded and asserted at worker-run end
    instead. Scope note: this hook covers the WORKER process only. The RGC
    rebuild's ``multiprocessing.spawn`` child (runtime_graph_cache.py) is a
    fresh interpreter that does not inherit a hook installed after import,
    but it writes only through the Pipe/DB connection it is handed and
    inherits this same tmp_root env -- the parent-scoped assertion is
    sufficient.
    """
    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in _AUDITED_FS_EVENTS:
            return
        # "os.open" is checked for completeness (per the plan's enumerated
        # event set) but CPython actually routes os.open() through the SAME
        # "open" event as builtin open() -- confirmed empirically, see
        # _open_event_is_write's docstring; no live code path emits a
        # distinct "os.open" event today.
        if event in ("open", "os.open") and not _open_event_is_write(args):
            return
        for path in _paths_for_audit_event(event, args):
            if not _path_is_allowlisted(path, allowlist_roots):
                violations.append((event, path))

    sys.addaudithook(_hook)


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
        # Per-phase peak, keyed by the phase label active at sample time --
        # lets a caller isolate the peak DURING one labelled window (e.g. the
        # boot-sequence phase) even when a LATER phase (sleep-cycle, drain)
        # reaches a higher global RSS. The single global peak above stays for
        # existing consumers of ``peak()``.
        self._phase_peaks: dict[str, dict[str, Any]] = {}
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
                existing = self._phase_peaks.get(phase)
                if existing is None or s["total_rss"] > existing["total_rss"]:
                    self._phase_peaks[phase] = s
            except Exception:  # noqa: BLE001 -- sampler must never crash the run
                pass
            self._stop.wait(self._interval)

    def peak(self) -> dict[str, Any] | None:
        return self._peak_sample

    def peak_for_phase(self, phase: str) -> dict[str, Any] | None:
        """Highest-RSS continuous sample recorded while ``phase`` was active.

        Isolates one labelled window's own peak from the run's global peak --
        the metric a boot-only RSS bound must use, since a heavier LATER phase
        (sleep-cycle, drain) would otherwise dominate the global ``peak()``.
        """
        return self._phase_peaks.get(phase)

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


def run_boot_sequence_arm(
    store: Any, peak_sampler: "_PeakSampler | None" = None,
) -> tuple[dict[str, Any], Any, Any, list]:
    """Run the daemon boot graph-rebuild sequence on ``store``.

    Replicates `daemon._boot_preload_body`'s sequence exactly:
    `runtime_graph_cache._rebuild_and_save_rgc(store)` (DB->child streaming,
    never a full in-parent graph) THEN `retrieve.build_runtime_graph(store)`
    (the in-thread reconstruct, cache-hit on the just-saved compact payload).
    Sets ``peak_sampler``'s phase to ``"during-boot-sequence"`` for the
    duration when given, so a caller can read
    ``peak_sampler.peak_for_phase("during-boot-sequence")`` afterward.

    Returns ``(rgc_result, graph, assignment, rich_club)`` -- the graph triple
    was previously discarded; a caller now feeds it to `run_first_recall_arm`
    so a real recall exercises the SAME graph the boot sequence just built,
    rather than paying a second rebuild.

    Imports are LOCAL to this function (not module-level) so this harness
    module stays importable before a caller's own HOME/IAI_MCP_STORE
    redirect — an early top-level ``iai_mcp`` import would run before that
    redirect and defeat hermeticity.
    """
    from iai_mcp import retrieve, runtime_graph_cache

    if peak_sampler is not None:
        peak_sampler.set_phase("during-boot-sequence")
    rgc_result = runtime_graph_cache._rebuild_and_save_rgc(store)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    return rgc_result, graph, assignment, rich_club


_FIRST_RECALL_CUE = "alice daily routine recall probe"
_FIRST_RECALL_SESSION = "phys-footprint-harness-first-recall"


def run_first_recall_arm(
    store: Any,
    graph: Any,
    assignment: Any,
    rich_club: list,
    embedder: Any,
    peak_sampler: "_PeakSampler | None" = None,
) -> dict[str, Any]:
    """Run one production `pipeline.recall_for_response` inside the measured window.

    Resets the module-global recall-latency sentinel to ``None`` BEFORE the
    call so a degraded "retrieval_skipped" early return -- which returns
    before the timer write at pipeline.py's end of `recall_for_response` --
    leaves the sentinel untouched and is thereby DETECTABLE, not silently
    stale from a prior call. Wall-clock is timed by this harness's own
    `perf_counter` (authoritative); the module global is a diagnostic only.
    """
    import iai_mcp.pipeline as _pipeline

    if peak_sampler is not None:
        peak_sampler.set_phase("during-first-recall")

    _pipeline._last_recall_latency_ms = None
    t0 = time.perf_counter()
    response = _pipeline.recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        # _HeldEmbedder.embed() returns the primed vector regardless of text --
        # the cue string is cosmetic, result selectivity is fixed by
        # construction, not by cue content.
        cue=_FIRST_RECALL_CUE,
        session_id=_FIRST_RECALL_SESSION,
        budget_tokens=1500,
        mode="concept",
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    reported_ms = _pipeline._last_recall_latency_ms
    return {
        "response_hit_count": len(response.hits),
        "wall_ms": wall_ms,
        "reported_latency_ms": reported_ms,
        "degraded_early_return": reported_ms is None,
    }


_DISPATCH_SETTLED_SESSION = "phys-footprint-harness-dispatch-settled"
_DISPATCH_SETTLED_K = 32
_DISPATCH_SETTLED_SEED = 20260918


def run_dispatch_settled_arm(
    store: Any,
    dim: int,
    peak_sampler: "_PeakSampler | None" = None,
    k: int = _DISPATCH_SETTLED_K,
) -> dict[str, Any]:
    """Run `k` `core.dispatch("memory_recall", ...)` calls with `k` DISTINCT
    cue vectors (each hydrating a different candidate set) -- unlike
    `run_first_recall_arm`'s single shared cue, this is the only arm that
    attributes settled residency to the structure `_incident_edges_warm`'s
    hydrate-time decrypt-skip actually populates across a session.

    Reports the resident structure size beside the caller's own
    phys_footprint sample so a settled-bytes delta is attributable: probes
    BOTH an unbounded `_rank_builder_graph` and a capped
    `_RankIndexHandle.resident`, whichever the store actually carries, so
    the arm produces a non-vacuous number regardless of which residency
    mechanism the store under test implements.
    """
    import numpy as _np

    from iai_mcp import core as _core

    if peak_sampler is not None:
        peak_sampler.set_phase("during-dispatch-settled")

    rng = _np.random.default_rng(_DISPATCH_SETTLED_SEED)
    dispatch_err: str | None = None
    hit_counts: list[int] = []
    for i in range(k):
        vec = rng.standard_normal(dim).astype(_np.float32)
        norm = float(_np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        try:
            resp = _core.dispatch(store, "memory_recall", {
                "cue": f"dispatch-settled probe {i}",
                "session_id": _DISPATCH_SETTLED_SESSION,
                "budget_tokens": 1500,
                "cue_embedding": vec.tolist(),
            })
            hit_counts.append(len(resp.get("hits", [])))
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring the remaining cues
            dispatch_err = repr(exc)

    resident_count: int | None = None
    resident_source: str | None = None
    builder_graph = getattr(store, "_rank_builder_graph", None)
    if builder_graph is not None:
        try:
            resident_count = int(builder_graph.node_count())
            resident_source = "_rank_builder_graph"
        except Exception:  # noqa: BLE001 -- a broken probe must never fail the arm
            resident_count = None
    else:
        handle = getattr(store, "_rank_index_handle", None)
        resident = getattr(handle, "resident", None) if handle is not None else None
        if resident is not None:
            try:
                resident_count = len(resident)
                resident_source = "_rank_index_handle.resident"
            except Exception:  # noqa: BLE001 -- a broken probe must never fail the arm
                resident_count = None

    return {
        "k": k,
        "dispatch_err": dispatch_err,
        "hit_counts": hit_counts,
        "resident_count": resident_count,
        "resident_source": resident_source,
    }


def run_instance_count_probe(n: int) -> dict[str, Any]:
    """`gc.get_objects()` scan for the RSS decomposition probe: counts
    `MemoryGraph` instances that look corpus-shaped (`node_count() >=
    0.5 * n`, so a small ephemeral per-call candidate graph is excluded),
    plus `_RankIndexHandle` instance counts. A capped, non-graph
    resident-candidate cache should show up here as a warm bundle count of
    at most 1 and zero corpus-shaped builder graphs.

    The resident Rust `RankIndex` itself is a PyO3-native object with no
    Python-level container fields, so it is never added to the cyclic
    garbage collector's tracked set (`gc.is_tracked()` is False for it even
    while alive and referenced) -- `gc.get_objects()` structurally cannot
    see it, on any store shape. `rank_index_instances` below is therefore
    counted through the one Python-side object that DOES own it: a
    `_RankIndexHandle` whose `_index` has actually been built.
    """
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store._rank_index import _RankIndexHandle

    corpus_shaped_graphs = 0
    handle_count = 0
    rank_index_count = 0
    threshold = 0.5 * n
    for obj in gc.get_objects():
        try:
            if isinstance(obj, MemoryGraph):
                if obj.node_count() >= threshold:
                    corpus_shaped_graphs += 1
            elif isinstance(obj, _RankIndexHandle):
                handle_count += 1
                if obj._index is not None:
                    rank_index_count += 1
        except ReferenceError:  # noqa: BLE001 -- a GC'd-mid-scan weakref target is skipped
            continue
    return {
        "corpus_shaped_memory_graph_instances": corpus_shaped_graphs,
        "rank_index_handle_instances": handle_count,
        "rank_index_instances": rank_index_count,
    }


_REDIRECTED_ENV_KEYS = (
    "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
    "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND", "NUMBA_CACHE_DIR",
    "HF_HOME", "IAI_MCP_EMBED_OFFLINE",
)


def _real_home() -> Path:
    """The real (non-redirected) HOME, resolved via the OS password database
    so it stays correct regardless of any HOME override already active in
    this process's environment.
    """
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


def _redirect_env(tmp_root: Path) -> dict[str, str | None]:
    """Redirect HOME/IAI_MCP_STORE/model-path/numba-cache under ``tmp_root``.

    Returns the prior values for `_restore_env`. Also sets
    ``sys.dont_write_bytecode`` (numba's own disk cache is redirected via
    ``NUMBA_CACHE_DIR`` above, independent of that flag -- see
    `_build_allowlist_roots`'s docstring for the full write-elimination
    rationale).

    HF_HOME is pinned to the REAL machine's model cache -- never under
    ``tmp_root`` -- before HOME is overwritten below. The embedder falls back
    to ``~/.cache/huggingface`` when HF_HOME is unset, and ``~`` would
    otherwise resolve against the just-redirected HOME, forcing a cache miss
    and a live network re-download of the model on every worker invocation.
    IAI_MCP_EMBED_OFFLINE=1 turns a genuine cache miss into a loud error
    instead of a silent download that skews timing/RSS numbers.
    """
    prev_env = {k: os.environ.get(k) for k in _REDIRECTED_ENV_KEYS}
    real_hf_home = os.environ.get("HF_HOME") or str(
        _real_home() / ".cache" / "huggingface"
    )
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
    os.environ["NUMBA_CACHE_DIR"] = str(tmp_root / "numba-cache")
    os.environ["HF_HOME"] = real_hf_home
    os.environ["IAI_MCP_EMBED_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    return prev_env


def _restore_env(prev_env: dict[str, str | None]) -> None:
    for k, v in prev_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Fixture cache -- two-process build/measure split.
# ---------------------------------------------------------------------------

_DEFAULT_FIXTURE_CACHE_ROOT_NAME = "iai-phys-fp-fixtures"


def _default_fixture_cache_root() -> Path:
    return Path.home() / ".cache" / _DEFAULT_FIXTURE_CACHE_ROOT_NAME


def _engine_generation() -> str:
    """Fingerprint the native engine's compiled ABI for the fixture cache key.

    Hashes the compiled extension's own bytes -- NOT a version string, which
    does not bump on every ``maturin develop --release``. A rebuilt .so (any
    content change) must invalidate cached fixtures; byte-format
    compatibility across engine builds is not guaranteed.
    """
    import hashlib

    try:
        import iai_mcp_native

        so_path = Path(iai_mcp_native.__file__)
        return hashlib.sha256(so_path.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 -- engine not built/importable
        return "unknown"


def _fixture_cache_dir(
    cache_root: Path, n: int, seed: int, edge_factor: float,
) -> Path:
    """The cache directory for one (N, seed, edge_factor), keyed by engine generation."""
    key = f"n{n}-seed{seed}-ef{edge_factor}"
    return cache_root / _engine_generation() / key


def _checkpoint_wal(store: Any) -> None:
    """Force a full WAL checkpoint+truncate before close (mirrors OPTIMIZE_HIPPO).

    Lock-guarded per the shared-connection invariant: every ``conn.execute``
    on ``store.db._conn`` must hold ``_conn_lock``.
    """
    lock = getattr(store.db, "_conn_lock", None)
    if lock is not None:
        with lock:
            store.db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    else:
        store.db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# A checkpointed WAL/SHM sidecar may still exist as a header-only artifact
# (verified empirically against this engine: a full `wal_checkpoint(TRUNCATE)`
# leaves a 32-byte WAL file, the header with zero committed frames) -- that
# is the correct terminal state after close(), not an incomplete checkpoint.
_EMPTY_WAL_HEADER_MAX_BYTES = 32


def _assert_no_wal_shm_sidecar(store_path: Path) -> None:
    """Fail loudly if a ``-wal``/``-shm`` sidecar carries PENDING data.

    Only a sidecar LARGER than the empty-header size means real content
    remains uncheckpointed -- a fixture built from one would bleed that
    mutation into every trial that copies it.
    """
    offenders = sorted(
        f"{p}({p.stat().st_size}b)"
        for p in store_path.rglob("*")
        if (p.name.endswith("-wal") or p.name.endswith("-shm"))
        and p.stat().st_size > _EMPTY_WAL_HEADER_MAX_BYTES
    )
    if offenders:
        raise RuntimeError(
            "BUILD POSTCONDITION VIOLATION: -wal/-shm sidecar(s) carry "
            f"uncheckpointed data after store.close(): {offenders}"
        )


def _build_corpus_step(store: Any, *, n: int, seed: int, eff_dim: int) -> int:
    """Insert the (n, seed, dim) corpus into ``store`` and flush its buffer.

    Shared verbatim by `_build_fixture_store` and `_worker_run`'s non-fixture
    branch -- a future edit to this call shape that is not mirrored in BOTH
    callers would silently invalidate every cached fixture's claimed
    equivalence to a fresh build.
    """
    from iai_mcp.store._buffers import flush_record_buffer

    inserted = build_corpus(
        store, n=n, seed=seed, dim=eff_dim,
        varied_tags=True, with_hv_payload=True,
    )
    flush_record_buffer(store)
    return inserted


def _build_edges_step(store: Any, *, n: int, edge_factor: float, seed: int) -> int:
    """Insert ``~edge_factor * n`` synthetic edges into ``store`` and flush.

    Shared verbatim by `_build_fixture_store` and `_worker_run` -- see
    `_build_corpus_step`'s docstring for why this must not drift between them.
    """
    from iai_mcp.store._buffers import flush_edge_buffer

    edges = _build_synthetic_edges(store, n, edge_factor, seed)
    flush_edge_buffer(store)
    return edges


def _warm_embedder_step(store: Any) -> tuple[Any, Any]:
    """Warm the real BERT embedder singleton for ``store`` and return it plus
    the primer text's embedded vector.

    Shared verbatim by `_build_fixture_store` and `_worker_run` -- see
    `_build_corpus_step`'s docstring for why this must not drift between them.
    """
    from iai_mcp.embed import embedder_for_store

    warm = embedder_for_store(store)
    prime_vec = warm.embed("alice daily routine warm-up prime encode")
    return warm, prime_vec


def _build_fixture_store(
    *, n: int, seed: int, edge_factor: float, cache_dir: Path,
) -> None:
    """Build one (n, seed, edge_factor) corpus store and copy it into ``cache_dir``.

    Runs the SAME corpus-build + edges + embedder-warm sequence the
    measurement worker would, in a throwaway tmp store, then closes it,
    asserts the WAL/SHM-absence postcondition, and copies the tree into
    ``cache_dir`` -- overwriting any prior fixture at that key.
    """
    tmp_root = Path(os.path.realpath(tempfile.mkdtemp(prefix="iai-phys-fp-build-")))
    prev_env = _redirect_env(tmp_root)
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    store = None
    try:
        from iai_mcp.store import MemoryStore

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)
        eff_dim = int(store.embed_dim)

        _build_corpus_step(store, n=n, seed=seed, eff_dim=eff_dim)
        _build_edges_step(store, n=n, edge_factor=edge_factor, seed=seed)
        _warm_embedder_step(store)

        # MemoryStore.close() itself flushes the event/record/edge buffers as
        # its FIRST step (writing to the WAL), so a checkpoint issued before
        # close is invalidated by close's own writes -- verified empirically.
        # Checkpoint on a FRESH reopen after close instead; its buffers are
        # empty, so its own close() writes nothing new.
        store.close()
        store = None

        _reopened = MemoryStore(path=store_path)
        try:
            _checkpoint_wal(_reopened)
        finally:
            _reopened.close()

        _assert_no_wal_shm_sidecar(store_path)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        shutil.copytree(store_path, cache_dir)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        _restore_env(prev_env)
        shutil.rmtree(tmp_root, ignore_errors=True)


def _worker_run(
    *,
    n: int,
    seed: int,
    edge_factor: float,
    deferred_events: int,
    out_jsonl: Path | None,
    sleep_budget_sec: float,
    boot_sequence_only: bool = False,
    fixture_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """In-process measurement. Returns the result dict (also written to jsonl).

    ``fixture_cache_dir``, when given, skips the corpus-build + edges steps
    and copytree's a pre-built fixture (see `_build_fixture_store`) into the
    store path instead -- the two-process build/measure split's MEASURE half.
    """
    # realpath, not the raw mkdtemp string: on macOS $TMPDIR is under /var,
    # itself a symlink to /private/var, and *at()-family syscalls resolve
    # dir_fd to the CANONICAL path (fcntl.F_GETPATH) -- comparing against the
    # non-canonical form would false-positive every write under tmp_root.
    tmp_root = Path(os.path.realpath(tempfile.mkdtemp(prefix="iai-phys-fp-")))
    prev_env = _redirect_env(tmp_root)
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    # Hermeticity by construction -- installed BEFORE any iai_mcp import, so
    # every filesystem write this process makes for the rest of its life is
    # observed. See _install_audit_hook's docstring for the parent-scope note.
    _violations: list[tuple[str, str]] = []
    _out_dir_for_allowlist = out_jsonl.parent if out_jsonl is not None else None
    _install_audit_hook(
        _build_allowlist_roots(tmp_root, _out_dir_for_allowlist), _violations,
    )

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

        store_path = Path(os.environ["IAI_MCP_STORE"])
        if fixture_cache_dir is not None:
            # Two-process split, MEASURE half: the corpus was already built
            # once by --build-fixture. Never open the cached copy directly --
            # copytree into THIS trial's own tmp_root first, so mutation
            # (WAL growth, index touch) never bleeds into the cache source or
            # a sibling trial.
            store_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fixture_cache_dir, store_path)
            store = MemoryStore(path=store_path)
            eff_dim = int(store.embed_dim)
            inserted = n
            edges = int(round(edge_factor * n))
            corpus_sec = 0.0
            edges_sec = 0.0
            s_corpus = record("after-corpus")
            record("after-edges")
        else:
            # Store + corpus (random vectors, real HNSW via the normal insert
            # path).
            store_path.mkdir(parents=True, exist_ok=True)
            store = MemoryStore(path=store_path)
            eff_dim = int(store.embed_dim)

            t0 = time.perf_counter()
            # ``insert`` buffers rows (flush at 500 / 5 s); force the buffer
            # out so the records table is materialised and the resident
            # HNSW index holds every vector before the heavy paths read it
            # back.
            inserted = _build_corpus_step(store, n=n, seed=seed, eff_dim=eff_dim)
            corpus_sec = time.perf_counter() - t0
            s_corpus = record("after-corpus")

            # Synthetic edges (~edge_factor * N), flushed to the edges table.
            t0 = time.perf_counter()
            edges = _build_edges_step(store, n=n, edge_factor=edge_factor, seed=seed)
            edges_sec = time.perf_counter() - t0
            record("after-edges")

        # Warm the real BERT embedder singleton the way the daemon does. Warm
        # ONCE — do NOT embed the corpus (that would cost hours).
        import iai_mcp.embed as _embed_mod

        t0 = time.perf_counter()
        warm, prime_vec = _warm_embedder_step(store)

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

        def _held(_store: object, **_kwargs: object) -> object:
            # `**_kwargs` is load-bearing: memory_recall's dispatch always
            # passes `allow_identity_mismatch`/`build_timeout` to the real
            # embedder_for_store funnel (see tests/_helpers.py's identical
            # stub) -- a double that only accepts `_store` raises TypeError
            # on every call, which core.dispatch's memory_recall path
            # catches and silently degrades instead of using this double.
            return held

        _embed_mod.embedder_for_store = _held  # type: ignore[assignment]
        warm_sec = time.perf_counter() - t0
        s_warm_baseline = record("after-warm-embedder")

        # Heavy warm path 1 — the CONSOLIDATED daemon boot sequence:
        # runtime_graph_cache._rebuild_and_save_rgc(store) (DB->child streaming,
        # never a full in-parent graph) THEN retrieve.build_runtime_graph(store)
        # (in-thread reconstruct, cache-hit on the just-saved compact payload) --
        # exactly what daemon._boot_preload_body runs.
        t0 = time.perf_counter()
        graph = assignment = rich_club = None
        try:
            _rgc_result, graph, assignment, rich_club = run_boot_sequence_arm(
                store, peak_sampler,
            )
        except Exception as exc:  # noqa: BLE001 -- record the failure, keep measuring
            graph_err = repr(exc)
        else:
            graph_err = None
        graph_sec = time.perf_counter() - t0
        s_boot = record("after-boot-sequence")

        # The boot-sequence delta: peak_total_rss DURING the boot-sequence
        # phase minus the after-warm-embedder baseline -- phase-SCOPED, not
        # the run's global peak, so a heavier LATER phase (sleep-cycle/drain)
        # can never be misattributed to "boot".
        boot_seq_phase_peak = peak_sampler.peak_for_phase("during-boot-sequence")
        baseline_total_rss = s_warm_baseline["total_rss"]
        boot_sequence_peak_total_rss = (
            boot_seq_phase_peak["total_rss"] if boot_seq_phase_peak is not None
            else s_boot["total_rss"]
        )
        boot_sequence_delta_bytes = boot_sequence_peak_total_rss - baseline_total_rss

        # First-recall arm — a real pipeline.recall_for_response call inside
        # the measured window, part of boot+first-recall (not sleep/drain), so
        # it runs regardless of boot_sequence_only.
        t0 = time.perf_counter()
        if graph is not None:
            try:
                fr = run_first_recall_arm(
                    store, graph, assignment, rich_club, held,
                    peak_sampler=peak_sampler,
                )
                first_recall_err = None
            except Exception as exc:  # noqa: BLE001 -- record, keep measuring
                first_recall_err = repr(exc)
                fr = {
                    "response_hit_count": 0, "wall_ms": None,
                    "reported_latency_ms": None, "degraded_early_return": True,
                }
        else:
            first_recall_err = f"skipped: boot_sequence_arm failed: {graph_err}"
            fr = {
                "response_hit_count": 0, "wall_ms": None,
                "reported_latency_ms": None, "degraded_early_return": True,
            }
        first_recall_sec = time.perf_counter() - t0
        s_after_first_recall = record("after-first-recall")

        first_recall_phase_peak = peak_sampler.peak_for_phase("during-first-recall")
        first_recall_peak_total_rss = (
            first_recall_phase_peak["total_rss"] if first_recall_phase_peak is not None
            else s_after_first_recall["total_rss"]
        )
        first_recall_delta_bytes = (
            s_after_first_recall["total_rss"] - s_boot["total_rss"]
        )

        # Dispatch-settled arm — K DISTINCT-cue core.dispatch recalls (never
        # first_recall_arm's single shared cue), then a relief pass and a
        # settled sample -- a session-scoped residency metric, not yet
        # gated by a fixed ceiling. Runs regardless of boot_sequence_only
        # (same rationale as the first-recall arm above): it measures a
        # session-scoped residency term, not the sleep/drain paths.
        t0 = time.perf_counter()
        try:
            dispatch_settled = run_dispatch_settled_arm(
                store, eff_dim, peak_sampler=peak_sampler,
            )
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring
            dispatch_settled = {
                "k": 0, "dispatch_err": repr(exc), "hit_counts": [],
                "resident_count": None, "resident_source": None,
            }
        dispatch_settled_sec = time.perf_counter() - t0
        s_after_dispatch_settled = record("after-dispatch-settled")

        dispatch_settled_phase_peak = peak_sampler.peak_for_phase("during-dispatch-settled")
        dispatch_settled_peak_total_rss = (
            dispatch_settled_phase_peak["total_rss"] if dispatch_settled_phase_peak is not None
            else s_after_dispatch_settled["total_rss"]
        )
        settled_dispatch_delta_bytes = (
            s_after_dispatch_settled["total_rss"] - s_after_first_recall["total_rss"]
        )
        settled_dispatch_delta_mib_per_1k = (
            (settled_dispatch_delta_bytes / (1024 * 1024)) / (n / 1000.0)
            if n else 0.0
        )

        try:
            instance_counts = run_instance_count_probe(n)
        except Exception as exc:  # noqa: BLE001 -- record, keep measuring
            instance_counts = {"error": repr(exc)}

        if boot_sequence_only:
            # Lite mode: the sleep-cycle and drain are irrelevant to the
            # boot-sequence delta (they run strictly AFTER it) and are the
            # slowest part of a full-flow run — skip them so a scale sweep
            # completes in a small fraction of the time. Measurement-
            # equivalent by construction: the boot-sequence phase precedes
            # both, in either mode.
            completed = []
            sleep_err = None
            sleep_interrupted = False
            sleep_sec = 0.0
            drain_counts = {"skipped": "boot_sequence_only"}
            drain_sec = 0.0
        else:
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
                "boot_sequence": round(graph_sec, 2),
                "first_recall": round(first_recall_sec, 2),
                "dispatch_settled": round(dispatch_settled_sec, 2),
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
            # The boot-sequence-scoped delta -- peak_total_rss DURING the
            # "during-boot-sequence" phase minus the after-warm-embedder
            # baseline. Phase-scoped, not the run's global peak_rss above (see
            # peak_for_phase's docstring for why that distinction matters).
            "baseline_total_rss": baseline_total_rss,
            "boot_sequence_peak_total_rss": boot_sequence_peak_total_rss,
            "boot_sequence_delta_bytes": boot_sequence_delta_bytes,
            # First-recall arm — a real pipeline.recall_for_response call, the
            # gate keys off first_recall_wall_ms (harness perf_counter), never
            # first_recall_reported_latency_ms (module-global, diagnostic only).
            "first_recall_wall_ms": fr.get("wall_ms"),
            "first_recall_reported_latency_ms": fr.get("reported_latency_ms"),
            "first_recall_degraded_early_return": fr.get("degraded_early_return"),
            "first_recall_hit_count": fr.get("response_hit_count"),
            "first_recall_delta_bytes": first_recall_delta_bytes,
            "first_recall_peak_total_rss": first_recall_peak_total_rss,
            "first_recall_err": first_recall_err,
            # Dispatch-settled arm -- K distinct-cue core.dispatch recalls,
            # attributing settled residency to whichever structure the
            # store carries (see run_dispatch_settled_arm's docstring).
            "dispatch_settled_k": dispatch_settled.get("k"),
            "dispatch_settled_hit_counts": dispatch_settled.get("hit_counts"),
            "dispatch_settled_err": dispatch_settled.get("dispatch_err"),
            "dispatch_settled_resident_count": dispatch_settled.get("resident_count"),
            "dispatch_settled_resident_source": dispatch_settled.get("resident_source"),
            "dispatch_settled_peak_total_rss": dispatch_settled_peak_total_rss,
            "settled_dispatch_delta_bytes": settled_dispatch_delta_bytes,
            "settled_dispatch_delta_mib_per_1k": settled_dispatch_delta_mib_per_1k,
            # Instance-count probe: a capped, non-graph resident cache
            # should show warm bundle <= 1, builder 0, index 1.
            "instance_counts": instance_counts,
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
        _restore_env(prev_env)
        shutil.rmtree(tmp_root, ignore_errors=True)
        if _violations:
            detail = "; ".join(f"{event}:{path}" for event, path in _violations[:20])
            raise RuntimeError(
                f"HERMETICITY VIOLATION: {len(_violations)} filesystem write(s) "
                f"outside the worker's allowlist. First violations: {detail}"
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
    boot_sequence_only: bool = False,
    fixture_cache_dir: Path | None = None,
    out_jsonl_name: str | None = None,
) -> dict[str, Any]:
    """Spawn the worker as a subprocess, sampling its RSS and killing on cap.

    Returns the worker's result dict on success, or a dict with ``aborted`` /
    ``error`` describing why the scale could not complete. The parent samples
    the worker's RSS every ``sample_interval`` seconds and, if it exceeds
    ``cap_gib``, terminates the worker tree (mirrors the daemon watchdog's
    auto-kill) so the M2 never approaches jetsam. ``boot_sequence_only``
    forwards ``--boot-sequence-only`` to the worker (skips the sleep-cycle and
    drain phases, irrelevant to the boot-sequence delta). ``fixture_cache_dir``
    forwards ``--fixture-cache-dir`` (the two-process split's MEASURE half --
    skips the corpus-build cost). ``out_jsonl_name`` overrides the default
    ``phases-nN.jsonl`` name (used for per-trial filenames under
    ``--use-fixture``).
    """
    out_jsonl = out_dir / (out_jsonl_name or f"phases-n{n}.jsonl")
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
    if boot_sequence_only:
        cmd.append("--boot-sequence-only")
    if fixture_cache_dir is not None:
        cmd.extend(["--fixture-cache-dir", str(fixture_cache_dir)])
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
    parser.add_argument(
        "--boot-sequence-only", action="store_true",
        help=(
            "skip the sleep-cycle and drain phases (irrelevant to the "
            "boot-sequence delta, which is scoped to the boot-sequence phase "
            "only) so a scale sweep completes in a small fraction of the time"
        ),
    )
    parser.add_argument(
        "--build-fixture", action="store_true",
        help=(
            "build a cached fixture store per --n/--seed/--edge-factor and "
            "exit (no measurement) -- the two-process split's BUILD half"
        ),
    )
    parser.add_argument(
        "--use-fixture", action="store_true",
        help=(
            "measure against a cached fixture (built via --build-fixture) "
            "instead of building a fresh corpus per trial -- the "
            "two-process split's MEASURE half; implies --boot-sequence-only"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="override the default fixture cache root (~/.cache/iai-phys-fp-fixtures)",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="fresh-copy measurement trials per scale under --use-fixture (default 3)",
    )
    # Worker-mode flags (set by the parent; not for direct use).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fixture-cache-dir", type=Path, default=None, help=argparse.SUPPRESS)
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
                boot_sequence_only=args.boot_sequence_only,
                fixture_cache_dir=args.fixture_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- report failure as a result line
            result = {"ok": False, "n": n, "error": repr(exc)}
        sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
        sys.stdout.flush()
        return 0 if result.get("ok") else 1

    if args.build_fixture:
        cache_root = args.cache_dir or _default_fixture_cache_root()
        for n in args.n:
            cache_dir = _fixture_cache_dir(cache_root, n, args.seed, args.edge_factor)
            print(f"[fixture] building n={n} seed={args.seed} edge_factor={args.edge_factor} -> {cache_dir}")
            _build_fixture_store(
                n=n, seed=args.seed, edge_factor=args.edge_factor, cache_dir=cache_dir,
            )
            print(f"[fixture] built {cache_dir}")
        return 0

    # Parent mode.
    out_dir = (args.out or Path(tempfile.mkdtemp(prefix="iai-phys-fp-out-"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"phys_footprint harness — scales={args.n} cap={args.cap_gib} GiB "
        f"edge_factor={args.edge_factor} out={out_dir}"
    )

    cache_root = args.cache_dir or _default_fixture_cache_root()
    results: list[dict[str, Any]] = []
    for n in args.n:
        fixture_cache_dir: Path | None = None
        trial_count = 1
        if args.use_fixture:
            fixture_cache_dir = _fixture_cache_dir(cache_root, n, args.seed, args.edge_factor)
            if not fixture_cache_dir.exists():
                print(
                    f"[scale N={n}] ERROR: no cached fixture at {fixture_cache_dir} "
                    "-- run --build-fixture first"
                )
                results.append({
                    "ok": False, "n": n,
                    "error": f"missing fixture at {fixture_cache_dir}",
                })
                continue
            trial_count = max(1, args.trials)

        for trial in range(trial_count):
            label = f"N={n}" if not args.use_fixture else f"N={n} trial {trial + 1}/{trial_count}"
            out_jsonl_name = (
                f"phases-n{n}.jsonl" if not args.use_fixture
                else f"phases-n{n}-trial{trial}.jsonl"
            )
            print(f"\n[scale {label}] spawning capped worker ...")
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
                boot_sequence_only=args.boot_sequence_only or args.use_fixture,
                fixture_cache_dir=fixture_cache_dir,
                out_jsonl_name=out_jsonl_name,
            )
            result["wall_sec"] = round(time.perf_counter() - t0, 1)
            if args.use_fixture:
                result["trial"] = trial
            results.append(result)
            if result.get("ok"):
                _print_phase_table(result)
                print(
                    f"[scale {label}] peak_rss={_fmt_gib(result.get('peak_rss'))} GiB "
                    f"peak_phys={_fmt_gib(result.get('peak_phys'))} GiB "
                    f"phantom_frac={_fmt_pct(result.get('peak_phantom_frac'))} "
                    f"({result['wall_sec']}s)"
                )
            else:
                if result.get("aborted"):
                    print(f"[scale {label}] ABORTED: {result.get('reason')}")
                    print(f"[scale {label}] stopping the scale sweep at the cap; "
                          "reporting trend from completed scales.")
                    # Persist what we have and stop escalating to larger N.
                    _persist_results(out_dir, results)
                    _print_summary(results)
                    return 0
                print(f"[scale {label}] ERROR: {result.get('error')}")
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


# ---------------------------------------------------------------------------
# Bounded-marginal-cost gate.
#
# Three points cannot distinguish sub-linear from linear-with-a-good-constant
# -- the boot-open term itself scales ~linearly, which a naive "grows
# sub-linearly" curve-fit would wrongly FAIL. The gate shape is instead a
# per-metric marginal-cost ceiling: independent, never averaged, and it
# RAISES on an unset ceiling rather than vacuously passing.
# ---------------------------------------------------------------------------

# Current-measured marginal cost (worst segment across the committed baseline)
# plus jitter headroom; regression bounds, not target performance. A columnar
# storage optimization is expected to tighten these later. Settled-state and
# first-recall memory cost both scale roughly linearly with N today.
_MARGINAL_CEILINGS: dict[str, float | None] = {
    "first_recall_delta_mib_per_1k": 14.0,
    "first_recall_peak_mib_per_1k": 40.0,
    "boot_sequence_delta_mib_per_1k": 25.0,
    "first_recall_latency_ms": 2100.0,
}

_MARGINAL_COST_BYTE_METRICS: tuple[tuple[str, str], ...] = (
    ("first_recall_delta_bytes", "first_recall_delta_mib_per_1k"),
    ("first_recall_peak_total_rss", "first_recall_peak_mib_per_1k"),
    ("boot_sequence_delta_bytes", "boot_sequence_delta_mib_per_1k"),
)


def assert_bounded_marginal_cost(
    results: list[dict[str, Any]], ceilings: dict[str, float | None],
) -> None:
    """Assert BOUNDED MARGINAL COST across scales -- never a curve-fit.

    ``results`` are per-scale result dicts (the sweep's persisted
    ``summary.json`` shape -- a list of `_worker_run` return dicts).
    ``ceilings`` keys: ``first_recall_delta_mib_per_1k``,
    ``first_recall_peak_mib_per_1k``, ``boot_sequence_delta_mib_per_1k``,
    ``first_recall_latency_ms``.

    For each CONSECUTIVE pair of scales (sorted by N), computes
    ``marginal_mib_per_1k = (delta(N_high) - delta(N_low)) / ((N_high -
    N_low) / 1000)`` INDEPENDENTLY for each of the three byte-metrics and
    asserts it stays under its own ceiling -- a regression in ANY one metric
    fails on its own, never averaged against the others. Latency is checked
    once, at the largest scale, against ``first_recall_wall_ms`` (never
    ``first_recall_reported_latency_ms``, diagnostic only).

    Raises (never passes/skips) when a required ceiling is unset/None, fewer
    than 2 successful scales are given, or a metric is missing from a
    result -- a vacuously-passing gate is the exact false-comfort SC2 exists
    to prevent.
    """
    required = tuple(ceilings_key for _metric, ceilings_key in _MARGINAL_COST_BYTE_METRICS)
    required += ("first_recall_latency_ms",)
    missing = [k for k in required if ceilings.get(k) is None]
    if missing:
        raise RuntimeError(
            f"UNPINNED MARGINAL-COST CEILING(S): {missing} -- the gate "
            "cannot vacuously pass with an unset ceiling; pin from a "
            "measured baseline sweep first."
        )

    ok_results = sorted((r for r in results if r.get("ok")), key=lambda r: r["n"])
    if len(ok_results) < 2:
        raise RuntimeError(
            f"at least 2 successful scales are required to compute a "
            f"marginal cost; got {len(ok_results)}."
        )

    failures: list[str] = []
    for low, high in zip(ok_results, ok_results[1:]):
        delta_n_k = (high["n"] - low["n"]) / 1000.0
        if delta_n_k <= 0:
            raise RuntimeError(
                f"non-increasing scale pair (N={low['n']}, N={high['n']}) -- "
                "results must be strictly increasing in N."
            )
        for metric_key, ceiling_key in _MARGINAL_COST_BYTE_METRICS:
            lo_val = low.get(metric_key)
            hi_val = high.get(metric_key)
            if lo_val is None or hi_val is None:
                failures.append(
                    f"{metric_key} missing at N={low['n']} or N={high['n']}"
                )
                continue
            marginal_mib = (hi_val - lo_val) / _BYTES_PER_MIB / delta_n_k
            ceiling = ceilings[ceiling_key]
            if marginal_mib > ceiling:
                failures.append(
                    f"{metric_key} marginal {marginal_mib:.2f} MiB/1k > "
                    f"ceiling {ceiling:.2f} MiB/1k (N={low['n']}->N={high['n']})"
                )

    largest = ok_results[-1]
    wall_ms = largest.get("first_recall_wall_ms")
    latency_ceiling = ceilings["first_recall_latency_ms"]
    if wall_ms is None:
        failures.append(f"first_recall_wall_ms missing at N={largest['n']}")
    elif wall_ms > latency_ceiling:
        failures.append(
            f"first_recall_wall_ms {wall_ms:.1f}ms > ceiling "
            f"{latency_ceiling:.1f}ms at N={largest['n']}"
        )

    if failures:
        raise AssertionError(
            "BOUNDED MARGINAL COST VIOLATION(S):\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    sys.exit(_main())
