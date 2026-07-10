"""Measure the pure-Python hypervector-kernel baseline.

Times the reference (pure-Python) implementations of the projection-apply, the
BSC majority-vote bundle, and the popcount Hamming distance with ``timeit``, and
writes the measured nanoseconds-per-op to ``fixtures/golden/hdc/py_baseline.json``.

This file is the real denominator the native bench divides into: the speedup is
``py_ns / rust_ns`` per kernel, so the baseline must be measured here, never
invented. Run with the pure-Python backend forced so the timings reflect the
reference path, not a delegated native call::

    .venv/bin/python scripts/bench/measure_hd_py_baseline.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import timeit
from pathlib import Path

# Force the pure-Python reference path for the measurement.
os.environ["IAI_MCP_HD_BACKEND"] = "python"

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from iai_mcp.lilli.core.similarity import hamming  # noqa: E402
from iai_mcp.lilli.crossmodal import embed_to_hv  # noqa: E402
from iai_mcp.lilli.tiers import bsc  # noqa: E402

_OUT = _REPO / "fixtures" / "golden" / "hdc" / "py_baseline.json"

# Iteration counts chosen so each measured loop runs well above timer noise
# while staying quick. Reported ns/op = total_seconds / n_iters * 1e9.
_FROM_EMBEDDING_ITERS = 2_000
_BSC_BUNDLE_ITERS = 5_000
_BSC_BUNDLE_VOTE_ITERS = 5_000
_HAMMING_ITERS = 50_000


def _bundle_vote_numpy(bound_bytes: list[bytes], n: int) -> bytes:
    """The pure-Python majority vote over already-bound HVs.

    This is exactly the numeric op the native kernel replaces — no role lookups
    and no binds, so it is the apples-to-apples denominator for the native
    vote-only kernel.
    """
    stacked = np.stack([np.frombuffer(hv, dtype=np.uint8) for hv in bound_bytes])
    bits = np.unpackbits(stacked, axis=1).astype(np.int32)
    sums = bits.sum(axis=0)
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()


def _ns_per_op(stmt, *, number: int) -> float:
    total = timeit.timeit(stmt, number=number)
    return total / number * 1e9


def main() -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal(384).astype(np.float32).tolist()

    # A mid-capacity BSC pair set (8 of the 10-pair cap at D=4096).
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT",
        "COMMUNITY_ID", "TEMPORAL_POSITION", "ACTOR", "OBJECT",
    ]
    pairs = [(r, bsc.filler_hv(f"v{i}")) for i, r in enumerate(roles)]

    a = bsc.filler_hv("hamming-a")
    b = bsc.filler_hv("hamming-b")

    # Pre-bind the pairs once so the vote-only measurement times exactly the
    # numpy majority vote the native kernel replaces.
    n_pairs = len(pairs)
    bound_bytes = [bsc.bind(bsc.role_hv(r), f) for r, f in pairs]

    from_embedding_ns = _ns_per_op(
        lambda: embed_to_hv.from_embedding(emb), number=_FROM_EMBEDDING_ITERS
    )
    bsc_bundle_ns = _ns_per_op(
        lambda: bsc.bundle(pairs), number=_BSC_BUNDLE_ITERS
    )
    bsc_bundle_vote_ns = _ns_per_op(
        lambda: _bundle_vote_numpy(bound_bytes, n_pairs),
        number=_BSC_BUNDLE_VOTE_ITERS,
    )
    hamming_ns = _ns_per_op(lambda: hamming(a, b), number=_HAMMING_ITERS)

    payload = {
        "from_embedding_ns": from_embedding_ns,
        "bsc_bundle_ns": bsc_bundle_ns,
        "bsc_bundle_vote_ns": bsc_bundle_vote_ns,
        "hamming_ns": hamming_ns,
        "method": "timeit",
        "iters": {
            "from_embedding": _FROM_EMBEDDING_ITERS,
            "bsc_bundle": _BSC_BUNDLE_ITERS,
            "bsc_bundle_vote": _BSC_BUNDLE_VOTE_ITERS,
            "hamming": _HAMMING_ITERS,
        },
        "machine": f"{platform.machine()} {platform.system()}",
        "python": platform.python_version(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
