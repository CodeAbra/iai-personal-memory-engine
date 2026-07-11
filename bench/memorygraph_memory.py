from __future__ import annotations

import argparse
import gc
import resource
import sys
from pathlib import Path
from uuid import uuid4


_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bench.memorygraph_memory",
        description="Measure peak RSS during MemoryGraph build.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of nodes to add (default: 10000).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5000,
        help="Number of edges to add (default: 5000).",
    )
    args = parser.parse_args()

    from iai_mcp.graph import MemoryGraph

    gc.collect()
    before = rss_mb()

    g = MemoryGraph()
    nodes = [uuid4() for _ in range(args.n)]
    for n in nodes:
        g.add_node(n, community_id=None, embedding=[0.0] * 384)

    edges_added = 0
    i = 0
    while edges_added < args.m and i < len(nodes) - 1:
        g.add_edge(nodes[i], nodes[i + 1], weight=1.0)
        edges_added += 1
        i += 2

    gc.collect()
    after = rss_mb()
    delta = after - before
    print(
        f"RSS delta @ N={args.n} m={args.m}: {delta:.1f} MB "
        f"(before={before:.1f} MB, after={after:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
