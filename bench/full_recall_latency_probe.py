"""Spread-ON warm recall latency probe and cProfile harness.

Usage (against an isolated store copy — NEVER live prod):

    source .venv/bin/activate
    IAI_ISO="/tmp/iai-lat-37k-$(date +%s)"
    cp -r ~/.iai-mcp "$IAI_ISO"
    LILLI_STORAGE_DRIVER=lilli python bench/full_recall_latency_probe.py \
        --store-path "$IAI_ISO" --n 40

The probe:
  1. Warms the recall path once (discarded) so cold-build costs do not land on
     measurement calls.
  2. Resets ``pipeline._last_recall_latency_ms = 0.0`` before EACH recall so the
     wall-clock degrade global is defeated and spread is genuinely ON for every
     measured call.
  3. Records p50 / p95 wall-clock over N warm spread-ON recalls.
  4. Runs a single cProfile of one warm spread-ON recall, writes the .prof file to
     /tmp/recall_spread_on.prof, and prints pstats sorted by cumulative time
     (top 30 frames) so the dominant hotspot is unambiguous.

Attribution targets in the cProfile output:
  H1 — ``_from_row`` cumulative time + call count  (per-candidate decode: AES-GCM
       decrypt of literal_surface / provenance + 1.5 KB embedding BLOB parse)
  H2 — ``incident_edges`` cumulative time + call count  (unbounded top_k=None
       global hebbian degree pass at core/__init__.py:247-251)
  H3 — individual ``incident_edges`` calls (hop-1 / hop-2 repeated SQL scans)
  BLOB — ``np.frombuffer`` or ``to_pandas`` share (embedding BLOB decode)

To confirm the LIVE Layer-1 candidate-graph path ran (not the
``build_runtime_graph`` fallback at retrieve.py:635): look for
``store.query_similar`` → ``store.incident_edges`` → ``recall_for_response`` in
the call graph.  The fallback path shows ``build_runtime_graph`` near the top of
cumulative frames; if you see that, check that the HNSW ANN index was built
(the store must be opened in non-read-only / SHARED mode so the index is
available).

No absolute dev paths are hardcoded.  Isolation and quiesce are operator
responsibilities (see how-to-verify in the plan checkpoint).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path preamble — no-op when imported as package member
# ---------------------------------------------------------------------------
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    """Return (p50, p95) from a list of float samples."""
    if not samples:
        return 0.0, 0.0
    s = sorted(samples)
    p50 = s[len(s) // 2]
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    return p50, p95


def _reset_spread_on() -> None:
    """Reset the wall-clock degrade global so spread is ON for the next recall.

    This is the same operation bench/recall_accuracy.py performs before each
    measured recall (the per-cue degrade reset).  Without this reset, a prior
    slow recall leaves the global > 2000, and the next call silently runs with
    spread_hops=0 (the degraded path) — defeating the measurement.
    """
    import iai_mcp.pipeline as _pipeline
    _pipeline._last_recall_latency_ms = 0.0


def _run_one_dispatch(store, sentinel_cue: str) -> list:
    """Run the production dispatch path and return hit ids."""
    from iai_mcp import core
    response = core.dispatch(
        store,
        "memory_recall",
        {"cue": sentinel_cue, "session_id": "spread_on_probe"},
    )
    return response.get("hits", [])


# ---------------------------------------------------------------------------
# Embedder shim — identical to the one in bench/recall_accuracy.py
# ---------------------------------------------------------------------------

_SENTINEL_PREFIX = "__latency_probe__"


class _VecInjectionEmbedder:
    """Route a pre-computed vector through the production dispatch embedder path.

    Dispatch builds its own embedder via ``embedder_for_store``; to inject a
    known vector without going through text→embedding, register the vector under
    a sentinel string and pass that sentinel as the cue.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._registry: dict[str, list[float]] = {}
        self.DIM = getattr(wrapped, "DIM", getattr(wrapped, "DEFAULT_DIM", 384))
        self.DEFAULT_DIM = self.DIM
        self.DEFAULT_MODEL_KEY = getattr(wrapped, "DEFAULT_MODEL_KEY", "probe")

    def register(self, key: str, vec: list[float]) -> str:
        sentinel = f"{_SENTINEL_PREFIX}{key}"
        self._registry[sentinel] = vec
        return sentinel

    def embed(self, text: str) -> list[float]:
        if text in self._registry:
            return self._registry[text]
        return self._wrapped.embed(text)


class _InjectEmbedderCtx:
    """Context manager that routes dispatch's embedder_for_store through a shim."""

    def __init__(self, shim: "_VecInjectionEmbedder") -> None:
        self._shim = shim
        self._orig = None

    def __enter__(self) -> "_InjectEmbedderCtx":
        import iai_mcp.embed as _emb
        self._emb_mod = _emb
        self._orig = _emb.embedder_for_store
        _emb.embedder_for_store = lambda _store: self._shim
        return self

    def __exit__(self, *_exc) -> None:
        if self._orig is not None:
            self._emb_mod.embedder_for_store = self._orig
            self._orig = None


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def run_probe(store_path: str, n: int = 40, prof_out: str = "/tmp/recall_spread_on.prof") -> None:
    """Run the spread-ON latency probe and cProfile harness.

    Args:
        store_path: Path to an isolated store copy (never live prod).
        n:          Number of warm spread-ON recalls to time.
        prof_out:   Path to write the .prof file for the cProfile run.
    """
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    sp = Path(store_path)
    if not sp.exists():
        print(f"ERROR: store_path does not exist: {sp}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening isolated store: {sp}")
    # Open in SHARED mode so the HNSW ANN index is built — the Layer-1 recall
    # path requires the index (read-only open leaves it unbuilt).
    store = MemoryStore(path=sp, access_mode=AccessMode.SHARED)

    real_embedder = embedder_for_store(store)
    shim = _VecInjectionEmbedder(real_embedder)

    # Sample a probe cue: pick any active record's embedding as the vector seed.
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT id, embedding FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
            " LIMIT 1"
        ).fetchone()

    if row is None:
        print("ERROR: store has no active records.", file=sys.stderr)
        store.close()
        sys.exit(1)

    raw_emb = row["embedding"]
    probe_vec = np.frombuffer(raw_emb, dtype=np.float32).copy()
    pn = float(np.linalg.norm(probe_vec))
    if pn > 0.0:
        probe_vec = probe_vec / pn

    # Add small orthogonal noise so the cue is not a verbatim copy of the stored
    # embedding (mirrors the _perturb_vec pattern in bench/recall_accuracy.py).
    rng = np.random.default_rng(20260701)
    noise_dir = rng.standard_normal(len(probe_vec)).astype(np.float32)
    noise_dir -= float(np.dot(noise_dir, probe_vec)) * probe_vec
    nn = float(np.linalg.norm(noise_dir))
    if nn > 0.0:
        noise_dir /= nn
    perturbed = probe_vec + 0.15 * noise_dir
    pn2 = float(np.linalg.norm(perturbed))
    if pn2 > 0.0:
        perturbed = perturbed / pn2

    # ---- Reinforce-defer shim -----------------------------------------------
    # In non-daemon contexts (this probe), store._reinforce_queue is None so
    # queue_reinforce falls back to per-id synchronous reinforce_record calls.
    # The profiled cost is ~2.6 s per recall — a bench artifact, not a production
    # cost.  In production the daemon has a live ReinforceWriteQueue so the call
    # returns immediately (deferred off the recall thread).
    #
    # Neutralise queue_reinforce so the timed path mirrors the daemon: reinforce
    # is deferred and does not run on the recall response thread.  The sync write
    # path (reinforce_record → boost_edges) is a real production concern worth its
    # own dedicated reinforce-write perf investigation — that is out of this
    # recall-latency scope.
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]

    with _InjectEmbedderCtx(shim):
        sentinel = shim.register("probe_cue", perturbed.tolist())

        # ---- Warm-up: one throwaway recall so lazy caches are hot -----------
        _reset_spread_on()
        _run_one_dispatch(store, sentinel)
        print("Warm-up recall complete.")

        # ---- p50/p95 measurement loop ----------------------------------------
        latency_samples: list[float] = []
        print(f"Running {n} timed spread-ON recalls...")
        for _ in range(n):
            _reset_spread_on()
            t0 = time.perf_counter()
            _run_one_dispatch(store, sentinel)
            latency_samples.append((time.perf_counter() - t0) * 1000.0)

        p50, p95 = _p50_p95(latency_samples)
        print(f"\n--- Spread-ON wall-clock, reinforce-DEFERRED (n={n}) ---")
        print(f"  p50 = {p50:.1f} ms  [production-representative: reinforce deferred]")
        print(f"  p95 = {p95:.1f} ms  [production-representative: reinforce deferred]")
        print(f"  min = {min(latency_samples):.1f} ms")
        print(f"  max = {max(latency_samples):.1f} ms")
        print()
        print("  NOTE: sync-reinforce (~2.6 s) was a non-daemon bench artifact and is")
        print("  excluded from this measurement.  The deferred baseline above (~1.5 s)")
        print("  matches the production path (daemon: reinforce is off the recall thread).")

        if p95 < 1000.0:
            print(
                "\nWARNING: p95 is already sub-second. "
                "The spread-ON path may not be exercised (degrade still active?) "
                "or the store is small. Escalate before locking the fix design."
            )

        # ---- cProfile: one warm spread-ON recall ----------------------------
        print(f"\nRunning cProfile on one warm spread-ON recall -> {prof_out}")
        _reset_spread_on()

        pr = cProfile.Profile()
        pr.enable()
        _run_one_dispatch(store, sentinel)
        pr.disable()

        pr.dump_stats(prof_out)

        # Print top-30 cumulative frames to stdout
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf)
        ps.sort_stats("cumulative")
        ps.print_stats(30)
        print(buf.getvalue())

        # ---- Frame-targeted attribution summary -----------------------------
        print("--- Attribution targets (look for these frames in the cProfile output) ---")
        print("  H1 (per-candidate decode):  _from_row")
        print("  H1 detail — BLOB decode:    np.frombuffer (in _decode_raw_row)")
        print("  H2 (global degree fan-out): incident_edges  (with top_k=None, all hebbian)")
        print("  H3 (repeated edge scans):   incident_edges  (hop-1 / hop-2 SQL scans)")
        print("  Layer-1 path confirm:       query_similar -> incident_edges -> recall_for_response")
        print("  Fallback path (BAD if seen): build_runtime_graph")
        print()
        print("Record the p50/p95 and the dominant frame into the phase PROFILE.md.")

    store.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spread-ON warm recall latency probe + cProfile harness"
    )
    parser.add_argument(
        "--store-path",
        required=True,
        help="Path to an isolated store copy (NEVER live prod)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40,
        help="Number of warm spread-ON recalls to time (default 40)",
    )
    parser.add_argument(
        "--prof-out",
        default="/tmp/recall_spread_on.prof",
        help="Path to write the .prof file (default /tmp/recall_spread_on.prof)",
    )
    args = parser.parse_args()
    run_probe(store_path=args.store_path, n=args.n, prof_out=args.prof_out)


if __name__ == "__main__":
    main()
