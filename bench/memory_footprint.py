from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
        "iai-mcp-bench-falsifiability-deterministic-2026"
    )

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

THRESHOLD_MB = 1600.0


def _threshold_mb_for_n(n: int) -> float:
    if n <= 1000:
        return THRESHOLD_MB
    import math
    return THRESHOLD_MB * (1.0 + 0.25 * math.log10(n / 1000.0))


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(r) / 1024.0 / 1024.0
    return float(r) / 1024.0


def _make_noise_record(i: int, rng: np.random.Generator, dim: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    vec = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"bench noise record {i}",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["bench", "ops-11"],
        language="en",
    )


def _seed_store(
    store: MemoryStore, n: int, dim: int, seed: int, *, concurrent: bool = False
) -> int:
    rng = np.random.default_rng(seed)
    records = [_make_noise_record(i, rng, dim=dim) for i in range(n)]
    if not concurrent:
        for r in records:
            store.insert(r)
        return len(records)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=256) as pool:
        list(pool.map(store.insert, records))
    return len(records)


def run_memory_footprint(
    n: int = 10_000,
    store_path: Path | str | None = None,
    dim: int = EMBED_DIM,
    seed: int = 42,
    *,
    skip_graph: bool = False,
    async_writes: bool = False,
) -> dict:
    cleanup: tempfile.TemporaryDirectory | None = None
    if store_path is None:
        cleanup = tempfile.TemporaryDirectory(prefix="iai-bench-ops11-")
        path = Path(cleanup.name)
    else:
        path = Path(store_path)
        path.mkdir(parents=True, exist_ok=True)

    prev_embed_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if dim != EMBED_DIM:
        os.environ["IAI_MCP_EMBED_DIM"] = str(dim)

    try:
        store = MemoryStore(path=path)
        eff_dim = store.embed_dim

        if async_writes:
            import asyncio as _asyncio

            async def _enable():
                await store.enable_async_writes()

            _asyncio.run(_enable())

        t0 = time.perf_counter()
        seed_n = _seed_store(
            store, n, dim=eff_dim, seed=seed, concurrent=async_writes,
        )
        seed_ms = (time.perf_counter() - t0) * 1000.0

        if async_writes:
            import asyncio as _asyncio

            async def _disable():
                await store.disable_async_writes()

            _asyncio.run(_disable())

        graph_built = False
        graph_ms = 0.0
        if not skip_graph:
            from iai_mcp import retrieve

            t1 = time.perf_counter()
            try:
                _graph, _assignment, _rc = retrieve.build_runtime_graph(store)
                graph_built = True
            except Exception:
                graph_built = False
            graph_ms = (time.perf_counter() - t1) * 1000.0

        gc.collect()
        rss_mb_peak = _rss_mb()

        threshold_mb = _threshold_mb_for_n(n)
        return {
            "n": n,
            "rss_mb_peak": round(rss_mb_peak, 2),
            "threshold_mb": round(threshold_mb, 2),
            "passed": rss_mb_peak <= threshold_mb,
            "platform": sys.platform,
            "stage_ms": {
                "seed": round(seed_ms, 2),
                "graph": round(graph_ms, 2),
            },
            "seed_n": seed_n,
            "graph_built": graph_built,
            "dim": eff_dim,
            "async_writes": bool(async_writes),
        }
    finally:
        if dim != EMBED_DIM:
            if prev_embed_dim is None:
                os.environ.pop("IAI_MCP_EMBED_DIM", None)
            else:
                os.environ["IAI_MCP_EMBED_DIM"] = prev_embed_dim
        if cleanup is not None:
            cleanup.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.memory_footprint",
        description=(
            "OPS-11 / M-03 RAM bench. Seeds N records, optionally builds "
            "the runtime graph, reports peak RSS. Target (D26-01): "
            "<=1600 MB at N=1000 on a 16+ GB host. The N=10k path is "
            "deferred (DEF-25-F LanceDB native hang)."
        ),
    )
    parser.add_argument(
        "--n", "--n-records", dest="n", type=int, default=1_000,
        help="record count to seed (default 1000 per D26-01)",
    )
    parser.add_argument(
        "--dim", type=int, default=EMBED_DIM,
        help=f"embedding dimension (default {EMBED_DIM}; tests use 32/64 for speed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (default 42)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip build_runtime_graph; isolate store-only RSS",
    )
    parser.add_argument(
        "--async-writes", action="store_true",
        help=(
            "Enable MemoryStore.enable_async_writes() before the "
            "seed loop so inserts go through the coalescing AsyncWriteQueue. "
            "Target: amortise the ~0.3 MB/insert LanceDB buffer overhead by "
            "batching 128 inserts per flush."
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write the JSON result to this file (in addition to stdout).",
    )
    args = parser.parse_args(argv)
    result = run_memory_footprint(
        n=args.n, dim=args.dim, seed=args.seed,
        skip_graph=args.skip_graph, async_writes=args.async_writes,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
