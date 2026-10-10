"""The ONE sanctioned iai_mcp_native import site in the tools subsystem.

Asymmetry, load-bearing: this module is the only file under
src/iai_mcp/tools/ (and the console-script modules tools_cli.py,
cli/_tools.py) permitted to import iai_mcp_native. Every other tools
module MUST stay brain-free -- test_embed_worker_isolation.py statically
scans the subsystem's source for this exact invariant.

Run this as a detached subprocess (`python -m iai_mcp.tools.embed_worker`),
never import it from a hot search/index/exec module -- importing it loads
the brain's native extension into the caller's process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

# 512 is the embedder's own tokenizer window (see rust/iai_mcp_embed_core);
# whitespace-token count is a proxy for that count, not an exact match.
_TRUNCATION_TOKEN_PROXY_LIMIT = 512


def _token_count_proxy(text: str) -> int:
    return len(text.split())


def embed_paths(paths: "list[str]", store_root: "Path | None" = None) -> int:
    """Encodes each path's whole-file text (unit v1: one vector per file,
    accepting the embedder's head-truncation bias past the token window)
    and publishes one vector segment under `store_root`. Returns the
    count actually indexed (files decodable as UTF-8 text).
    """
    from iai_mcp_native import embed

    import iai_mcp_tools_core as _native
    from iai_mcp.tools import default_vector_store_root

    root = store_root or default_vector_store_root()
    root.mkdir(parents=True, exist_ok=True)

    embedder = embed.Embedder()
    vectors: "list[list[float]]" = []
    decoded_paths: "list[str]" = []
    truncated_flags: "list[bool]" = []
    for p in paths:
        path = Path(p)
        try:
            text = path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        vectors.append(embedder.encode(text))
        decoded_paths.append(str(path))
        truncated_flags.append(_token_count_proxy(text) > _TRUNCATION_TOKEN_PROXY_LIMIT)

    if not vectors:
        return 0
    _native.index_vectors(str(root), vectors, decoded_paths, truncated_flags)
    return len(decoded_paths)


def embed_query(text: str) -> "list[float]":
    from iai_mcp_native import embed

    return embed.Embedder().encode(text)


def _percentiles(samples: "list[float]") -> "dict[str, float]":
    """Pure nearest-rank median (p50) + p95 over a wall-clock sample list.

    A floor for background-run sizing, never a latency SLA -- nothing is
    gated on the returned numbers.
    """
    if not samples:
        raise ValueError("no samples")
    ordered = sorted(samples)
    n = len(ordered)

    def _nearest_rank(p: float) -> float:
        idx = max(0, min(n - 1, round(p * (n - 1))))
        return ordered[idx]

    return {"median": _nearest_rank(0.5), "p95": _nearest_rank(0.95)}


def warm_latency(
    paths: "list[str]", encode: "Callable[[str], object] | None" = None
) -> "dict[str, object]":
    """Corpus-representative WARM per-file embed latency: one discarded
    cold call, then one timed call per path in `paths`. Returns a floor
    for background-run sizing (median_ms, p95_ms, n, floor=True) --
    never a latency SLA, nothing gated on the numbers.

    `encode` is injectable so this sampling STRUCTURE (warm-up discard +
    per-file timing + percentile emission) is exercisable without loading
    the brain: a stub returning a fixed vector drives the full path. It
    defaults to the real embedder's single-text `encode()` call.

    A full-corpus warm run over a real repo belongs in a background job
    with its full output redirected to a log; an in-test run stays a
    bounded sample, under 300s.
    """
    if not paths:
        raise ValueError("no paths")
    if encode is None:
        from iai_mcp_native import embed

        encode = embed.Embedder().encode

    texts = [Path(p).read_text(encoding="utf-8") for p in paths]

    encode(texts[0])  # discarded cold call -- warm-up, never timed

    samples: "list[float]" = []
    for text in texts:
        start = time.perf_counter()
        encode(text)
        samples.append(time.perf_counter() - start)

    pct = _percentiles(samples)
    return {
        "median_ms": pct["median"] * 1000.0,
        "p95_ms": pct["p95"] * 1000.0,
        "n": len(samples),
        "floor": True,
    }


def _cmd_index(args: argparse.Namespace) -> int:
    count = embed_paths(args.paths, store_root=Path(args.store_root))
    print(f"embedded {count} files under {args.store_root}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    print(json.dumps(embed_query(args.text)))
    return 0


def _cmd_time(args: argparse.Namespace) -> int:
    print(json.dumps(warm_latency(args.paths)))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed_worker")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_index = sub.add_parser(
        "index", help="embed every path's whole-file text and publish one vector segment"
    )
    p_index.add_argument("store_root", help="vector store root to publish the segment under")
    p_index.add_argument("paths", nargs="+", help="file paths to embed")
    p_index.set_defaults(func=_cmd_index)

    p_query = sub.add_parser(
        "query", help="embed one query string and print its vector as a JSON array"
    )
    p_query.add_argument("text", help="query text to embed")
    p_query.set_defaults(func=_cmd_query)

    p_time = sub.add_parser(
        "time",
        help="record a WARM per-file embed-latency floor (median + p95 ms) over a bounded path sample",
    )
    p_time.add_argument("paths", nargs="+", help="file paths to sample")
    p_time.set_defaults(func=_cmd_time)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
