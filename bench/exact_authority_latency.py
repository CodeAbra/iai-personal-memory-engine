"""Exact-authority resident-matrix build + scan + dispatch latency probe.

Usage (against an isolated store copy -- NEVER live prod):

    source .venv/bin/activate
    LILLI_STORAGE_DRIVER=lilli python -m bench.exact_authority_latency \
        --store-path ~/.iai-mcp-isotest

The probe measures, in order:
  1. Resident-matrix build time -- the first ``store.exact_top_k`` call on a
     cold index, reported as its own number, deliberately OUTSIDE every timed
     p50/p95 window below.
  2. Warm exact-scan p50/p95/max over N seeded random cues, calling
     ``store.exact_top_k(vec, k=10)`` directly (matrix already warm from
     step 1, so no cold-build cost lands in this window either).
  3. Production-dispatch p50/p95/max over M cues through
     ``core.dispatch(store, "memory_recall", ...)`` with the authority ON.
     Step 1 running first already provides the required warm-up so the
     one-time lazy matrix build can never land inside this timed window --
     the gate measures steady-state, not the cold build.
  4. Matrix resident size estimate (rows x dim x 4 bytes, float32) and row
     count.

Reinforce is deferred (mirrors the daemon's ReinforceWriteQueue) so the
~2.6s synchronous reinforce_record artifact never enters any timed window.

A machine-parsable JSON summary line is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path preamble -- no-op when imported as package member
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


def _p50_p95_max(samples: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, max) from a list of float samples."""
    if not samples:
        return 0.0, 0.0, 0.0
    s = sorted(samples)
    p50 = s[len(s) // 2]
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    return p50, p95, s[-1]


def _reset_spread_on() -> None:
    """Reset the wall-clock degrade global so spread is ON for the next recall.

    Same convention as bench/recall_accuracy.py and
    bench/full_recall_latency_probe.py: without this reset a prior slow
    recall could leave the global elevated, defeating steady-state
    measurement of the next call.
    """
    import iai_mcp.pipeline as _pipeline
    _pipeline._last_recall_latency_ms = 0.0


def _sample_active_embeddings(store, n: int, rng: "np.random.Generator") -> list["np.ndarray"]:
    """Sample up to n unit-normalized embeddings from the active corpus."""
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT embedding FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()

    if not rows:
        return []

    vecs = [np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows]
    idx = rng.permutation(len(vecs))
    picked = [vecs[i] for i in idx[: min(n, len(vecs))]]

    out: list["np.ndarray"] = []
    for v in picked:
        norm = float(np.linalg.norm(v))
        out.append(v / norm if norm > 0.0 else v)
    return out


def _perturb(vec: "np.ndarray", rng: "np.random.Generator", noise_fraction: float = 0.15) -> "np.ndarray":
    """Add small orthogonal noise so the cue is not a verbatim stored vector."""
    noise_dir = rng.standard_normal(len(vec)).astype(np.float32)
    noise_dir -= float(np.dot(noise_dir, vec)) * vec
    nn = float(np.linalg.norm(noise_dir))
    if nn > 0.0:
        noise_dir /= nn
    perturbed = vec + noise_fraction * noise_dir
    pn = float(np.linalg.norm(perturbed))
    return perturbed / pn if pn > 0.0 else perturbed


# ---------------------------------------------------------------------------
# Embedder shim -- routes a pre-computed vector through production dispatch
# ---------------------------------------------------------------------------

_SENTINEL_PREFIX = "__exact_authority_latency_probe__"


class _VecInjectionEmbedder:
    """Route a pre-computed vector through the production dispatch embedder path."""

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
        _emb.embedder_for_store = lambda _store, **_kw: self._shim
        return self

    def __exit__(self, *_exc) -> None:
        if self._orig is not None:
            self._emb_mod.embedder_for_store = self._orig
            self._orig = None


def _run_one_dispatch(store, sentinel_cue: str) -> list:
    """Run the production dispatch path and return hit ids."""
    from iai_mcp import core
    response = core.dispatch(
        store,
        "memory_recall",
        {"cue": sentinel_cue, "session_id": "exact_authority_latency_probe"},
    )
    return response.get("hits", [])


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def run_probe(
    store_path: str,
    n: int = 200,
    m: int = 50,
    k: int = 10,
    seed: int = 42,
) -> dict:
    """Run the exact-authority build/scan/dispatch latency probe.

    Args:
        store_path: Path to an isolated store copy (never live prod).
        n:          Number of warm exact-scan cues to time (default 200).
        m:          Number of production-dispatch cues to time (default 50).
        k:          Top-k cutoff for exact_top_k (default 10).
        seed:       Seed for the numpy RNG driving cue sampling/perturbation.

    Returns:
        dict summary (also printed as a JSON line).
    """
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    sp = Path(store_path)
    if not sp.exists():
        print(f"ERROR: store_path does not exist: {sp}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening isolated store: {sp}")
    # SHARED (non-read-only) mode: the production dispatch path (step 3)
    # requires the HNSW ANN index built at boot.
    store = MemoryStore(path=sp, access_mode=AccessMode.SHARED)

    # ---- Reinforce-defer shim (mirrors the daemon's ReinforceWriteQueue) ----
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]

    rng = np.random.default_rng(seed)

    active_vecs = _sample_active_embeddings(store, max(n, m) + 1, rng)
    if not active_vecs:
        print("ERROR: store has no active records.", file=sys.stderr)
        store.close()
        sys.exit(1)

    with store.db._conn_lock:
        row_count = store.db._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    corpus_size = int(row_count["n"]) if row_count else 0

    dim = int(len(active_vecs[0]))

    # ---- (a) Resident-matrix build time -- first exact_top_k call, cold. ----
    # Deliberately OUTSIDE every timed p50/p95 window below.
    build_cue = _perturb(active_vecs[0], rng)
    t0 = time.perf_counter()
    _ = store.exact_top_k(build_cue.tolist(), k=k)
    build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Resident-matrix build (cold, one-time): {build_ms:.1f} ms  [excluded from all p50/p95 gates below]")

    # ---- (b) Warm exact-scan p50/p95/max over N seeded cues. ----------------
    scan_samples: list[float] = []
    n_scan = min(n, len(active_vecs))
    for i in range(n_scan):
        cue = _perturb(active_vecs[i % len(active_vecs)], rng)
        t0 = time.perf_counter()
        _ = store.exact_top_k(cue.tolist(), k=k)
        scan_samples.append((time.perf_counter() - t0) * 1000.0)
    scan_p50, scan_p95, scan_max = _p50_p95_max(scan_samples)
    print(f"Warm exact-scan (n={n_scan}): p50={scan_p50:.2f}ms p95={scan_p95:.2f}ms max={scan_max:.2f}ms")

    # ---- (c) Production-dispatch p50/p95/max over M cues, authority ON. -----
    # The matrix is already warm from step (a)/(b) above -- this satisfies the
    # steady-state requirement: the cold build can never land inside this
    # timed window.
    real_embedder = embedder_for_store(store)
    shim = _VecInjectionEmbedder(real_embedder)
    dispatch_samples: list[float] = []
    m_dispatch = min(m, len(active_vecs))
    with _InjectEmbedderCtx(shim):
        # Explicit throwaway warm-up dispatch call, kept even though (a)/(b)
        # already warmed the matrix -- belt-and-suspenders steady-state proof.
        _reset_spread_on()
        _warm_cue = _perturb(active_vecs[0], rng)
        _warm_sentinel = shim.register("__warmup__", _warm_cue.tolist())
        _run_one_dispatch(store, _warm_sentinel)
        print("Dispatch warm-up call complete (discarded; matrix already warm from step a/b).")

        for i in range(m_dispatch):
            cue = _perturb(active_vecs[i % len(active_vecs)], rng)
            sentinel = shim.register(str(i), cue.tolist())
            _reset_spread_on()
            t0 = time.perf_counter()
            _run_one_dispatch(store, sentinel)
            dispatch_samples.append((time.perf_counter() - t0) * 1000.0)

    dispatch_p50, dispatch_p95, dispatch_max = _p50_p95_max(dispatch_samples)
    print(f"Production dispatch, steady-state (m={m_dispatch}): p50={dispatch_p50:.2f}ms p95={dispatch_p95:.2f}ms max={dispatch_max:.2f}ms")

    # ---- (d) Matrix resident size estimate. ---------------------------------
    matrix_bytes = corpus_size * dim * 4  # float32
    matrix_mb = matrix_bytes / (1024.0 * 1024.0)
    print(f"Matrix resident size estimate: {corpus_size} rows x {dim} dims x 4 bytes = {matrix_mb:.2f} MB")

    store.close()

    result = {
        "corpus_size": corpus_size,
        "matrix_dim": dim,
        "matrix_resident_mb": round(matrix_mb, 3),
        "build_ms_one_time": round(build_ms, 2),
        "scan_n": n_scan,
        "scan_p50_ms": round(scan_p50, 2),
        "scan_p95_ms": round(scan_p95, 2),
        "scan_max_ms": round(scan_max, 2),
        "dispatch_m": m_dispatch,
        "dispatch_p50_ms": round(dispatch_p50, 2),
        "dispatch_p95_ms": round(dispatch_p95, 2),
        "dispatch_max_ms": round(dispatch_max, 2),
        "k": k,
        "seed": seed,
    }
    print("\nJSON summary:")
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact-authority resident-matrix build + warm-scan + production-dispatch "
            "latency probe. Run against an isolated store copy, never live prod."
        )
    )
    parser.add_argument(
        "--store-path",
        required=True,
        help="Path to an isolated store copy (NEVER live prod)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of warm exact-scan cues to time (default 200)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=50,
        help="Number of production-dispatch cues to time (default 50)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k cutoff for exact_top_k (default 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for cue sampling/perturbation (default 42)",
    )
    args = parser.parse_args()
    run_probe(store_path=args.store_path, n=args.n, m=args.m, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
