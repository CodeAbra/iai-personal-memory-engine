"""Hermetic multi-cycle RSS attribution harness.

Drives a synthetic corpus through repeated in-process WAKE→drain→SLEEP cycles,
captures per-cycle USS/RSS/PSS samples, tracemalloc snap diffs (Python + numpy
domain), macOS ``vmmap -resident`` dumps, and internal probe state (HNSW
counts, SQLite PRAGMAs, ``gc.get_objects()`` instance counts), then writes a
9-surface RCA findings document.

Hermeticity: every filesystem touch goes under a ``tempfile.mkdtemp`` root.
``HOME``, ``IAI_MCP_STORE``, and ``IAI_MCP_USER_MODEL_PATH`` are redirected
**before** any project module imports so the daemon code (in particular
``drain_deferred_captures`` which hard-codes ``Path.home() / ".iai-mcp"``)
reads and writes only under the tmp root. A ``_real_store_signature()``
fingerprint is taken before and after the run; any drift in the real
``~/.iai-mcp/`` raises ``RuntimeError("HERMETICITY VIOLATION: ...")``.

Invocation::

    source .venv/bin/activate
    python tests/rss_rca/harness.py \\
        --n 22000 --cycles 3 \\
        --out /tmp/rss-rca-artifacts/

The output directory contains ``RCA-FINDINGS.md`` (planner-consumed), plus
``rca-log.jsonl`` (sampler + per-cycle state) and ``vmmap-cycle-N.txt`` and
``tracemalloc-{top,diff}-cycle-N.txt`` per cycle (raw observability data).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil


# Add the repo's ``src/`` to sys.path so the harness can run as a standalone
# script (``python tests/rss_rca/harness.py ...``) before any project module
# has been imported. Also add the repo root so ``from tests.rss_rca.corpus
# import build_corpus`` resolves the sibling module when invoked via the CLI
# entrypoint rather than via pytest (which already configures rootdir for us).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = str(_REPO_ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)


# ---------------------------------------------------------------------------
# Hermeticity helpers — lifted in shape from the existing hermetic bench at
# ``bench/consolidation_rss_peak.py``. Volatile prefixes match verbatim so the
# signature contract is identical.
# ---------------------------------------------------------------------------


_VOLATILE_PREFIXES: tuple[str, ...] = (
    "wrappers/",
    "logs/",
    ".capture-state/",
    ".deferred-captures/",
    ".daemon.sock",
    ".daemon-state",
    ".session-start-payload",
    ".heartbeat",
    "wake.signal",
    ".wake.signal",
    ".locked",
    ".lock",
)


def _real_store_signature() -> object:
    """Fingerprint the real user ``~/.iai-mcp/`` for hermeticity checks.

    Returns ``None`` if the directory does not exist, otherwise a
    ``dict[str, tuple[int, int]]`` mapping relative paths to ``(size,
    mtime_ns)`` tuples. Volatile prefixes (daemon socket, heartbeat, capture
    state, etc.) are skipped because they legitimately change at any moment
    while the live daemon is running — they would produce false positives.
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


def _install_remote_stubs() -> dict[str, object]:
    """Patch claude-cli + reconsolidation critic so the harness never spawns a subprocess.

    Both modules are imported lazily so the patching only happens after
    ``run()`` is entered — module-level import of ``tests.rss_rca.harness``
    must NOT have side effects on ``iai_mcp.claude_cli`` because that would
    contaminate other tests in the same pytest session.

    Returns the original module attributes so ``run()`` can restore them in a
    ``finally`` block. Because ``run()`` executes in-process (driven by the
    smoke test), failing to restore would permanently leave the stubs on the
    shared modules and break every later test that resolves these functions.
    """
    def _boom(*_a: object, **_kw: object) -> dict:
        raise RuntimeError(
            "remote subprocess call attempted inside the RSS attribution "
            "harness (must never happen — harness is local-only)"
        )

    import iai_mcp.claude_cli as cc

    originals: dict[str, object] = {
        "cc_invoke_claude_sync": cc.invoke_claude_sync,
        "cc_invoke_claude_once": cc.invoke_claude_once,
    }
    cc.invoke_claude_sync = _boom  # type: ignore[assignment]
    cc.invoke_claude_once = _boom  # type: ignore[assignment]

    import iai_mcp.reconsolidation_critic as rc

    def _no_critic(_pool: object, **_kw: object) -> dict:
        return {}

    originals["rc_evaluate_batch_reconsolidation"] = (
        rc.evaluate_batch_reconsolidation
    )
    rc.evaluate_batch_reconsolidation = _no_critic  # type: ignore[assignment]
    return originals


def _restore_remote_stubs(originals: dict[str, object]) -> None:
    """Undo ``_install_remote_stubs`` so the shared modules are left pristine."""
    if not originals:
        return
    import iai_mcp.claude_cli as cc

    if "cc_invoke_claude_sync" in originals:
        cc.invoke_claude_sync = originals["cc_invoke_claude_sync"]  # type: ignore[assignment]
    if "cc_invoke_claude_once" in originals:
        cc.invoke_claude_once = originals["cc_invoke_claude_once"]  # type: ignore[assignment]

    if "rc_evaluate_batch_reconsolidation" in originals:
        import iai_mcp.reconsolidation_critic as rc

        rc.evaluate_batch_reconsolidation = (  # type: ignore[assignment]
            originals["rc_evaluate_batch_reconsolidation"]
        )


def _ru_maxrss_bytes() -> int:
    """Return ``getrusage`` peak RSS in bytes, handling macOS / Linux unit difference."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(r)
    return int(r) * 1024


# ---------------------------------------------------------------------------
# Sampler thread — psutil USS/RSS/PSS every interval_sec, written as JSONL.
# Slower (5 s) interval than the existing bench (50 ms) because multi-cycle
# runs are minutes long, not seconds; oversampling fills the log without
# improving signal quality.
# ---------------------------------------------------------------------------


class _Sampler:
    """Background psutil sampler — writes one JSONL row per tick."""

    def __init__(
        self,
        out_fh: io.TextIOBase,
        out_lock: threading.Lock,
        interval_sec: float = 5.0,
    ) -> None:
        self._fh = out_fh
        self._lock = out_lock
        self._interval = float(interval_sec)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._proc = psutil.Process()
        self._cycle_idx = -1
        self._step_label = "boot"
        self._samples: list[dict[str, Any]] = []

    def set_cycle(self, cycle_idx: int, step_label: str) -> None:
        self._cycle_idx = int(cycle_idx)
        self._step_label = str(step_label)

    def emit_marker(self, cycle_idx: int, step_label: str, rss_bytes: int) -> None:
        """Write one deterministic non-tick sampler row with an explicit RSS.

        Used by the driver to record a settled-footprint sample at the
        per-cycle boundary — a fixed point ("all returnable pages returned")
        rather than the next periodic tick, which lands at a random moment with
        large jitter. The row carries the same schema as the periodic samples so
        downstream parsers do not special-case it.
        """
        pa_pool_bytes = 0
        pa_pool_max = 0
        try:
            from iai_mcp.hippo import _schema as _pa
            _mp = _pa.default_memory_pool()
            pa_pool_bytes = int(_mp.bytes_allocated())
            pa_pool_max = int(_mp.max_memory())
        except Exception:
            pass
        try:
            mi = self._proc.memory_full_info()
            uss = int(getattr(mi, "uss", 0) or 0)
            pss = int(getattr(mi, "pss", 0) or 0)
        except Exception:
            uss = 0
            pss = 0
        row = {
            "kind": "sample",
            "t": time.monotonic(),
            "rss_bytes": int(rss_bytes),
            "uss_bytes": uss,
            "pss_bytes": pss,
            "pa_pool_bytes": pa_pool_bytes,
            "pa_pool_max": pa_pool_max,
            "cycle_idx": int(cycle_idx),
            "step": str(step_label),
        }
        self._samples.append(row)
        with self._lock:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                mi = self._proc.memory_full_info()
                # PyArrow pool bytes_allocated — falsifiable check vs Antigravity
                # hypothesis that mimalloc/jemalloc pool retention is the leak.
                pa_pool_bytes = 0
                pa_pool_max = 0
                try:
                    from iai_mcp.hippo import _schema as _pa
                    _mp = _pa.default_memory_pool()
                    pa_pool_bytes = int(_mp.bytes_allocated())
                    pa_pool_max = int(_mp.max_memory())
                except Exception:
                    pass
                row = {
                    "kind": "sample",
                    "t": time.monotonic(),
                    "rss_bytes": int(mi.rss),
                    "uss_bytes": int(getattr(mi, "uss", 0) or 0),
                    "pss_bytes": int(getattr(mi, "pss", 0) or 0),
                    "pa_pool_bytes": pa_pool_bytes,
                    "pa_pool_max": pa_pool_max,
                    "cycle_idx": self._cycle_idx,
                    "step": self._step_label,
                }
                self._samples.append(row)
                with self._lock:
                    self._fh.write(json.dumps(row) + "\n")
                    self._fh.flush()
            except Exception:
                # Best-effort; sampler must never crash the run.
                pass
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# vmmap dump (macOS only).
# ---------------------------------------------------------------------------


def _dump_vmmap(pid: int, out_path: Path) -> str | None:
    """Dump ``vmmap -resident`` for ``pid`` to ``out_path``; return the text or None."""
    if sys.platform != "darwin":
        out_path.write_text(
            "# vmmap not available on non-darwin; "
            "/proc/$pid/smaps is the Linux equivalent and is left to a "
            "follow-up phase\n",
            encoding="utf-8",
        )
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/vmmap", "-resident", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = result.stdout or ""
        out_path.write_text(text, encoding="utf-8")
        return text
    except (subprocess.TimeoutExpired, OSError):
        out_path.write_text(
            "# vmmap dump failed (timeout or OSError)\n",
            encoding="utf-8",
        )
        return None


# ---------------------------------------------------------------------------
# Per-cycle state capture.
# ---------------------------------------------------------------------------


def _capture_per_cycle_state(store: Any, pid: int, cycle_idx: int) -> dict[str, Any]:
    """Capture USS/RSS/PSS + HNSW counts + SQLite PRAGMAs + gc counters.

    Reads on ``store.db._conn`` are serialised under ``store.db._conn_lock``
    because the shared sqlite3 connection runs with ``check_same_thread=False``
    and the harness's sampler thread may otherwise interleave with a fetch.
    """
    proc = psutil.Process(pid)
    mi = proc.memory_full_info()

    state: dict[str, Any] = {
        "cycle": int(cycle_idx),
        "ts": time.monotonic(),
        "uss_bytes": int(getattr(mi, "uss", 0) or 0),
        "rss_bytes": int(mi.rss),
        "pss_bytes": int(getattr(mi, "pss", 0) or 0),
        "ru_maxrss_bytes": _ru_maxrss_bytes(),
    }

    # HNSW probes — direct attribute reads on the public-enough internal API.
    try:
        active = store.db._hnsw
        state["hnsw_active_count"] = int(active.get_current_count())
        state["hnsw_active_capacity"] = int(active.get_max_elements())
    except Exception:
        state["hnsw_active_count"] = None
        state["hnsw_active_capacity"] = None

    try:
        standby = getattr(store.db, "_hnsw_standby", None)
        if standby is not None:
            state["hnsw_standby_count"] = int(standby.get_current_count())
            state["hnsw_standby_capacity"] = int(standby.get_max_elements())
        else:
            state["hnsw_standby_count"] = None
            state["hnsw_standby_capacity"] = None
    except Exception:
        state["hnsw_standby_count"] = None
        state["hnsw_standby_capacity"] = None

    # SQLite probes — wrap under _conn_lock per the project's connection
    # thread-safety contract.
    try:
        lock = getattr(store.db, "_conn_lock", None)
        if lock is not None:
            with lock:
                state["sqlite_page_count"] = int(
                    store.db._conn.execute("PRAGMA page_count").fetchone()[0]
                )
                state["sqlite_page_size"] = int(
                    store.db._conn.execute("PRAGMA page_size").fetchone()[0]
                )
                state["sqlite_cache_size_pages"] = int(
                    store.db._conn.execute("PRAGMA cache_size").fetchone()[0]
                )
        else:
            state["sqlite_page_count"] = None
            state["sqlite_page_size"] = None
            state["sqlite_cache_size_pages"] = None
    except Exception:
        state["sqlite_page_count"] = None
        state["sqlite_page_size"] = None
        state["sqlite_cache_size_pages"] = None

    # gc counters.
    try:
        objs = gc.get_objects()
        state["gc_objects_total"] = len(objs)
        state["embedder_instances"] = sum(
            1 for o in objs if type(o).__name__ == "Embedder"
        )
    except Exception:
        state["gc_objects_total"] = None
        state["embedder_instances"] = None

    return state


# ---------------------------------------------------------------------------
# Deferred-captures staging — writes a fixed batch of jsonl files into the
# tmp HOME's ``.deferred-captures/`` so that the subsequent
# ``drain_deferred_captures(store)`` call materialises records into Hippo
# inside the tracemalloc window. Mirrors the file layout that the live
# daemon's session-stop hook writes.
# ---------------------------------------------------------------------------


def _stage_deferred_batch(
    home_root: Path,
    cycle_idx: int,
    n_events: int = 250,
) -> Path:
    """Write a single ``.deferred-captures/<session>-<cycle>.jsonl`` file.

    The file contains ``n_events`` lines after the header so that
    ``drain_deferred_captures`` has a meaningful workload to chew on. The
    session id includes ``cycle_idx`` so re-staging in a later cycle does
    not collide with the previous cycle's file.
    """
    deferred_dir = home_root / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    session = f"rss-rca-cycle-{cycle_idx}"
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
                    f"deferred capture {i} cycle {cycle_idx}: a synthetic "
                    f"sentence with enough length to clear the drain gate"
                ),
                "cue": f"cue-{cycle_idx}-{i}",
                "tier": "episodic",
                "role": "user",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(ev) + "\n")
    return fpath


# ---------------------------------------------------------------------------
# tracemalloc helpers.
# ---------------------------------------------------------------------------


def _take_snapshot_with_numpy() -> tuple[
    "tracemalloc.Snapshot", "tracemalloc.Snapshot",
]:
    """Take one default snapshot and one numpy-domain-filtered snapshot."""
    snap = tracemalloc.take_snapshot()
    try:
        np_domain = np.lib.tracemalloc_domain  # type: ignore[attr-defined]
        np_filter = [tracemalloc.Filter(True, "*", domain=np_domain)]
        snap_np = snap.filter_traces(np_filter)
    except (AttributeError, ImportError):
        # numpy may not have tracemalloc_domain on all builds; fall back to
        # the default snapshot so the harness still produces output.
        snap_np = snap
    return snap, snap_np


def _write_topn_to_file(
    snap: "tracemalloc.Snapshot",
    out_path: Path,
    *,
    top: int = 30,
) -> None:
    """Write the top-N file:line allocations to ``out_path``."""
    stats = snap.statistics("lineno")[:top]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# tracemalloc top {top} (lineno key)\n")
        for stat in stats:
            try:
                frame = stat.traceback[0]
                fh.write(
                    f"{stat.size / 1024.0:+.1f} KiB "
                    f"{frame.filename}:{frame.lineno} "
                    f"count={stat.count}\n"
                )
            except IndexError:
                fh.write(
                    f"{stat.size / 1024.0:+.1f} KiB "
                    f"<no-frame> count={stat.count}\n"
                )


def _write_diff_to_file(
    diff: "list[tracemalloc.StatisticDiff]",
    out_path: Path,
    *,
    top: int = 30,
    label: str = "diff",
) -> None:
    """Write the top-N diff entries (size_diff DESC) to ``out_path``."""
    sorted_diff = sorted(
        diff, key=lambda s: s.size_diff, reverse=True,
    )[:top]
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# {label} (top {top} by size_diff)\n")
        for stat in sorted_diff:
            try:
                frame = stat.traceback[0]
                fh.write(
                    f"{stat.size_diff / 1024.0:+.1f} KiB "
                    f"{frame.filename}:{frame.lineno} "
                    f"count_diff={stat.count_diff}\n"
                )
            except IndexError:
                fh.write(
                    f"{stat.size_diff / 1024.0:+.1f} KiB "
                    f"<no-frame> count_diff={stat.count_diff}\n"
                )


# ---------------------------------------------------------------------------
# Main run() — the driver. Hermetic. Tracemalloc + sampler + per-cycle state.
# ---------------------------------------------------------------------------


def run(
    *,
    n: int,
    cycles: int,
    out: Path,
    dim: int | None = None,
    seed: int = 42,
    deferred_events_per_cycle: int = 250,
) -> Path:
    """Drive ``cycles`` in-process WAKE→drain→SLEEP cycles against a tmp store.

    Returns the path to the rendered ``RCA-FINDINGS.md`` file under ``out``.
    """
    if n <= 0 or cycles <= 0:
        raise ValueError("--n and --cycles must be positive integers")

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Hermeticity: snapshot the real ~/.iai-mcp/ BEFORE any tmp redirect.
    real_sig_before = _real_store_signature()

    # Tmp roots — one for HOME (so drain_deferred_captures reads/writes here)
    # and the same root for IAI_MCP_STORE so MemoryStore writes here too.
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-rss-rca-"))
    prev_home = os.environ.get("HOME")
    prev_store = os.environ.get("IAI_MCP_STORE")
    prev_user_model = os.environ.get("IAI_MCP_USER_MODEL_PATH")
    prev_passphrase = os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
    prev_embed_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    prev_keyring = os.environ.get("PYTHON_KEYRING_BACKEND")

    os.environ["HOME"] = str(tmp_root)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_root / ".iai-mcp" / "user_model.json"
    )
    if not prev_passphrase:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-rss-rca-harness-deterministic-2026"
        )
    if dim is not None:
        os.environ["IAI_MCP_EMBED_DIM"] = str(int(dim))
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

    # Make the tmp HOME look real enough for any code that probes for it.
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    # Project imports happen AFTER the ENV redirects so all module-level
    # ``Path.home()`` and ``IAI_MCP_STORE`` reads pick up the tmp root.
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
        _step_memory_relief,
    )
    from iai_mcp.store import MemoryStore

    # Output artefacts.
    log_path = out / "rca-log.jsonl"
    log_fh = log_path.open("w", encoding="utf-8")
    log_lock = threading.Lock()

    store = None
    embedder = None
    sampler: _Sampler | None = None
    pid = os.getpid()
    findings_path = out / "RCA-FINDINGS.md"
    _remote_stub_originals: dict[str, object] = {}

    try:
        # Install remote-call stubs only after the imports — they monkey-patch
        # the just-imported module references. The originals are restored in
        # the finally block so an in-process run() never leaves the shared
        # modules mutated for the rest of the pytest session.
        _remote_stub_originals = _install_remote_stubs()

        # Build the store and the warm embedder.
        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)
        eff_dim = int(store.embed_dim)

        warmup_t0 = time.perf_counter()
        from iai_mcp.embed import Embedder
        embedder = Embedder()
        embedder.embed("alice daily routine warm-up prime encode")
        warmup_duration = time.perf_counter() - warmup_t0

        # Build the synthetic corpus.
        from tests.rss_rca.corpus import build_corpus
        from tests.rss_rca.surfaces import classify_all, render_findings_md

        corpus_t0 = time.perf_counter()
        inserted = build_corpus(
            store, n=n, seed=seed, dim=eff_dim, varied_tags=True,
            with_hv_payload=True,
        )
        corpus_duration = time.perf_counter() - corpus_t0
        with log_lock:
            log_fh.write(json.dumps({
                "kind": "corpus_built",
                "inserted": inserted,
                "duration_sec": round(corpus_duration, 3),
            }) + "\n")
            log_fh.flush()

        # Lifecycle event log — required by SleepPipeline constructor.
        sleep_log_dir = tmp_root / ".iai-mcp" / "logs"
        sleep_log_dir.mkdir(parents=True, exist_ok=True)
        event_log = LifecycleEventLog(log_dir=sleep_log_dir)

        # Start tracemalloc at depth 25 so per-frame attribution is faithful.
        tracemalloc.start(25)

        # Sampler thread.
        sampler = _Sampler(log_fh, log_lock, interval_sec=5.0)
        sampler.start()

        prior_state: dict[str, Any] | None = None
        all_rows_by_cycle: list[list[dict[str, Any]]] = []

        for cycle_idx in range(cycles):
            sampler.set_cycle(cycle_idx, "pre-drain")

            # Take the pre-cycle snapshot. Subsequent diffs key off this.
            snap_pre, snap_pre_np = _take_snapshot_with_numpy()

            # Step 1 — stage deferred captures and drain.
            sampler.set_cycle(cycle_idx, "drain")
            staged = _stage_deferred_batch(
                tmp_root, cycle_idx, n_events=deferred_events_per_cycle,
            )
            try:
                drain_counts = drain_deferred_captures(store)
            except Exception as exc:
                drain_counts = {"error": repr(exc)}
            snap_post_drain, snap_post_drain_np = _take_snapshot_with_numpy()
            state_post_drain = _capture_per_cycle_state(
                store, pid, cycle_idx,
            )
            state_post_drain["phase"] = "post-drain"
            state_post_drain["drain_counts"] = drain_counts
            state_post_drain["staged_file"] = str(staged)

            # Step 2 — drive the full sleep pipeline.
            sampler.set_cycle(cycle_idx, "sleep")
            pipeline = SleepPipeline(
                store=store,
                lifecycle_state_path=tmp_root / ".iai-mcp" / "lifecycle_state.json",
                event_log=event_log,
            )
            sleep_t0 = time.perf_counter()
            result = pipeline.run()
            sleep_duration = time.perf_counter() - sleep_t0
            snap_post_sleep, snap_post_sleep_np = _take_snapshot_with_numpy()
            state_post_sleep = _capture_per_cycle_state(
                store, pid, cycle_idx,
            )
            state_post_sleep["phase"] = "post-sleep"
            try:
                completed = [
                    getattr(s, "name", str(s))
                    for s in result.get("completed_steps", [])
                ]
            except Exception:
                completed = []
            state_post_sleep["completed_steps"] = completed
            state_post_sleep["sleep_duration_sec"] = round(sleep_duration, 3)

            # Deterministic settle-and-sample at the per-cycle boundary.
            # The periodic sampler's "last tick" lands at a random moment with
            # large jitter (a VACUUM transient or a mid-rebuild spike can be
            # caught), so two identical runs disagree by ~100 MiB. Instead, at
            # the cycle boundary force every returnable page back to the OS
            # (pyarrow pool release + gc.collect + macOS zone pressure relief)
            # and then take ONE RSS read. That "all returnable pages returned"
            # footprint is reproducible. This runs in the driver AFTER run()
            # returns — it observes the pipeline, it does not alter sleep
            # behaviour. ``_step_memory_relief`` performs the three relief
            # actions; the explicit psutil read after it yields integer bytes
            # matching the periodic-sample schema (its own rss_after is MB).
            settle = _step_memory_relief(f"post-cycle-{cycle_idx}")
            settled_rss = (
                int(psutil.Process(pid).memory_info().rss)
                or int(round(settle.get("rss_after_mb", 0.0) * 1e6))
            )
            sampler.emit_marker(cycle_idx, "post-cycle-settled", settled_rss)

            # Compute the diffs we need for attribution.
            diff_py_drain = snap_post_drain.compare_to(snap_pre, "lineno")
            diff_np_drain = snap_post_drain_np.compare_to(
                snap_pre_np, "lineno",
            )
            diff_py_sleep = snap_post_sleep.compare_to(
                snap_post_drain, "lineno",
            )
            diff_np_sleep = snap_post_sleep_np.compare_to(
                snap_post_drain_np, "lineno",
            )
            diff_py_total = snap_post_sleep.compare_to(snap_pre, "lineno")
            diff_np_total = snap_post_sleep_np.compare_to(
                snap_pre_np, "lineno",
            )

            # Write tracemalloc dumps for this cycle.
            top_path = out / f"tracemalloc-top-cycle-{cycle_idx}.txt"
            diff_path = out / f"tracemalloc-diff-cycle-{cycle_idx}.txt"
            _write_topn_to_file(snap_post_sleep, top_path, top=30)
            # Clear diff file then append three labelled sections.
            diff_path.write_text(
                "# tracemalloc diffs for this cycle\n", encoding="utf-8",
            )
            _write_diff_to_file(
                diff_py_drain, diff_path, top=30, label="post-drain vs pre",
            )
            _write_diff_to_file(
                diff_py_sleep, diff_path, top=30,
                label="post-sleep vs post-drain",
            )
            _write_diff_to_file(
                diff_py_total, diff_path, top=30, label="post-sleep vs pre",
            )

            # vmmap dump for this cycle.
            vmmap_path = out / f"vmmap-cycle-{cycle_idx}.txt"
            vmmap_text = _dump_vmmap(pid, vmmap_path)

            # Classify all 9 surfaces. The total diff (post-sleep vs pre)
            # is the primary attribution input; the post-sleep state is the
            # cycle's terminal probe snapshot.
            rows = classify_all(
                diff_py_total,
                diff_np_total,
                vmmap_text,
                state_post_sleep,
                prior_state,
                cycle_idx=cycle_idx,
            )
            all_rows_by_cycle.append(rows)

            # Persist per-cycle state and per-surface classification.
            with log_lock:
                log_fh.write(json.dumps({
                    "kind": "cycle",
                    "cycle": cycle_idx,
                    "state_post_drain": state_post_drain,
                    "state_post_sleep": state_post_sleep,
                    "surface_rows": [
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "classification": r["classification"],
                            "kb_per_cycle": r["kb_per_cycle"],
                            "evidence": r["evidence"],
                        }
                        for r in rows
                    ],
                }, default=str) + "\n")
                log_fh.flush()

            prior_state = state_post_sleep

            # gc.collect BETWEEN cycles to settle reference cycles before the
            # next snap_pre baseline. Never mid-cycle (would mask growth).
            gc.collect()

        # Render the deliverable findings doc using the last cycle's rows
        # (warm-state representative) but fall back if cycles==0.
        if all_rows_by_cycle:
            final_rows = all_rows_by_cycle[-1]
        else:
            final_rows = []

        run_metadata = {
            "iso_ts": datetime.now(timezone.utc).isoformat(),
            "n_records": int(n),
            "dim": int(eff_dim),
            "cycles": int(cycles),
            "pid": int(pid),
            "warmup_duration_sec": float(warmup_duration),
        }
        md_body = render_findings_md(final_rows, run_metadata)
        findings_path.write_text(md_body, encoding="utf-8")

        return findings_path

    finally:
        # Stop sampler before tearing down anything it might be sampling.
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass

        if tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except Exception:
                pass

        try:
            log_fh.close()
        except Exception:
            pass

        if store is not None:
            try:
                store.close()
            except Exception:
                pass

        # Drop the warm embedder + force a gc so the next harness call in the
        # same process does not double-count Embedder instances.
        embedder = None
        gc.collect()

        # Restore the remote-call stubs so the shared iai_mcp modules are left
        # pristine for the rest of an in-process pytest session.
        _restore_remote_stubs(_remote_stub_originals)

        # ENV restoration.
        def _restore_env(name: str, prev: str | None) -> None:
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

        _restore_env("HOME", prev_home)
        _restore_env("IAI_MCP_STORE", prev_store)
        _restore_env("IAI_MCP_USER_MODEL_PATH", prev_user_model)
        _restore_env("IAI_MCP_CRYPTO_PASSPHRASE", prev_passphrase)
        _restore_env("IAI_MCP_EMBED_DIM", prev_embed_dim)
        _restore_env("PYTHON_KEYRING_BACKEND", prev_keyring)

        # Tmp tree cleanup. Use ignore_errors so a transient file lock does
        # not mask the hermeticity check below.
        shutil.rmtree(tmp_root, ignore_errors=True)

        # Hermeticity post-check. Mirrors the verbatim bench message format.
        real_sig_after = _real_store_signature()
        if real_sig_before != real_sig_after:
            before = (
                real_sig_before if isinstance(real_sig_before, dict) else {}
            )
            after = (
                real_sig_after if isinstance(real_sig_after, dict) else {}
            )
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            modified = sorted(
                k for k in set(before) & set(after) if before[k] != after[k]
            )
            raise RuntimeError(
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the bench. "
                f"added={added!r} removed={removed!r} modified={modified!r}. "
                "The bench must never touch the real user store."
            )


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rss_rca_harness",
        description=(
            "Hermetic multi-cycle RSS attribution harness. Drives N "
            "synthetic records through repeated in-process WAKE→drain→SLEEP "
            "cycles, attributes per-cycle growth to the 9 candidate "
            "surfaces, and writes RCA-FINDINGS.md."
        ),
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="record count to seed",
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="sleep cycles to drive (default 3)",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help=(
            "output directory (RCA-FINDINGS.md + rca-log.jsonl + "
            "vmmap-cycle-N.txt + tracemalloc-{top,diff}-cycle-N.txt)"
        ),
    )
    parser.add_argument(
        "--dim", type=int, default=None,
        help="embedding dimension (default: store.embed_dim)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default 42)",
    )
    parser.add_argument(
        "--deferred-events-per-cycle", type=int, default=250,
        help="number of deferred-capture events staged before each cycle",
    )
    args = parser.parse_args(argv)

    if args.n <= 0 or args.cycles <= 0:
        parser.error("--n and --cycles must be positive integers")

    out = args.out.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if out == repo_root:
        parser.error(
            f"--out must be a subdirectory of the repo root ({repo_root}), "
            "not the repo root itself"
        )
    tmp_prefixes = ("/tmp/", "/private/tmp/", "/var/folders/")
    is_tmp = any(str(out).startswith(p) for p in tmp_prefixes)
    if repo_root not in out.parents and not is_tmp:
        parser.error(
            f"--out must be under the repo root ({repo_root}) "
            "or a tmp directory (/tmp/, /private/tmp/, /var/folders/)"
        )
    out.mkdir(parents=True, exist_ok=True)

    findings = run(
        n=args.n,
        cycles=args.cycles,
        out=out,
        dim=args.dim,
        seed=args.seed,
        deferred_events_per_cycle=args.deferred_events_per_cycle,
    )
    print(f"RCA-FINDINGS.md written to {findings}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
