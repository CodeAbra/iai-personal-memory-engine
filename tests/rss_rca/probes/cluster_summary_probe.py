"""Hermetic diagnostic probe for the CLUSTER_SUMMARY pairing/grind pathology.

Builds two tmp stores with different hebbian topologies and reports, for each,
the connected-component (cluster) size distribution that ``_build_hebbian_clusters``
produces and the total ``combinations(cluster_ids, 2)`` pair count the current
``_process_cluster_summaries`` would materialize, plus how many per-cluster
``boost_edges`` calls it would issue.

Topologies
----------
1. realistic-multi-cluster: bounded clusters of ~10 records, intra-cluster edges
   only (the harness's bounded model — many small components).
2. giant-component: a single similarity CHAIN a~b~c~... that transitively bridges
   every record into ONE connected component (the pattern-separation insert path
   links each new record to its high-cosine neighbor, so a slow-topic-drift corpus
   forms exactly this chain on a real store).

Run::

    source .venv/bin/activate
    python tests/rss_rca/probes/cluster_summary_probe.py --n 3000
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from uuid import uuid4

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _mk_store(tmp_root: Path):
    os.environ["HOME"] = str(tmp_root)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(tmp_root / ".iai-mcp" / "user_model.json")
    os.environ.setdefault(
        "IAI_MCP_CRYPTO_PASSPHRASE", "cluster-summary-probe-deterministic-2026"
    )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    store_path = Path(os.environ["IAI_MCP_STORE"])
    store_path.mkdir(parents=True, exist_ok=True)
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=store_path)


def _seed_records(store, n: int) -> list:
    """Insert ``n`` records with RANDOM orthogonal-ish vectors (build_corpus path).

    Random vectors clear the pattern-separation near-dup SKIP gate, so all ``n``
    records persist — unlike degenerate identical embeddings, which collapse to a
    handful of survivors and would mask the real cluster topology. We then read the
    persisted ids back so the edge topologies reference only rows that exist.
    """
    from tests.rss_rca.corpus import build_corpus
    from iai_mcp.store._buffers import flush_record_buffer

    build_corpus(store, n=n, seed=7, dim=int(store.embed_dim))
    flush_record_buffer(store)

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
    from uuid import UUID

    return [UUID(str(r[0])) for r in rows]


def _measure(store) -> dict:
    from iai_mcp.sleep import _build_hebbian_clusters, CLUSTER_MIN_SIZE
    from iai_mcp.store._buffers import flush_edge_buffer

    flush_edge_buffer(store)
    clusters = _build_hebbian_clusters(store)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    eligible = [c for c in clusters if len(c) >= CLUSTER_MIN_SIZE]
    total_pairs = sum(len(list(combinations(c, 2))) for c in eligible)
    return {
        "n_clusters": len(clusters),
        "n_eligible_clusters": len(eligible),
        "largest_cluster": sizes[0] if sizes else 0,
        "total_combinations_pairs": total_pairs,
        "per_cluster_boost_edges_calls": len(eligible),
        "size_hist_top5": sizes[:5],
    }


def topology_realistic(store, ids: list, cluster_size: int = 10) -> None:
    from iai_mcp.store._buffers import flush_edge_buffer

    pairs = []
    for start in range(0, len(ids), cluster_size):
        cluster = ids[start:start + cluster_size]
        for i in range(len(cluster)):
            pairs.append((cluster[i], cluster[(i + 1) % len(cluster)]))
    for s in range(0, len(pairs), 2000):
        store.boost_edges(pairs[s:s + 2000], delta=0.1, edge_type="hebbian")
    flush_edge_buffer(store)


def topology_giant_chain(store, ids: list) -> None:
    """A single similarity chain a~b~c~... → ONE giant connected component."""
    from iai_mcp.store._buffers import flush_edge_buffer

    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    for s in range(0, len(pairs), 2000):
        store.boost_edges(pairs[s:s + 2000], delta=0.1, edge_type="hebbian")
    flush_edge_buffer(store)


def _run(n: int) -> None:
    for label, build in (
        ("realistic-multi-cluster", topology_realistic),
        ("giant-component-chain", topology_giant_chain),
    ):
        tmp_root = Path(tempfile.mkdtemp(prefix="iai-cluster-probe-"))
        try:
            store = _mk_store(tmp_root)
            ids = _seed_records(store, n)
            build(store, ids)
            m = _measure(store)
            print(f"\n=== topology={label} N={n} ===")
            for k, v in m.items():
                print(f"  {k:32} {v}")
            store.close()
        finally:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    _run(args.n)
