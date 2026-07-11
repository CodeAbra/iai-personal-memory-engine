from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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

from iai_mcp.core import _seed_l0_identity
from iai_mcp.retrieve import recall
from iai_mcp.store import EMBED_DIM, MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord

ACCURACY_FLOOR = 0.99
NOISE_SEED = 20260416


def _make_pinned(text: str, dim: int = EMBED_DIM) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] * dim,
        community_id=None,
        centrality=0.0,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["benchmark", "pinned"],
        language="en",
    )


def _random_unit_vector(rng: np.random.Generator, dim: int = EMBED_DIM) -> list[float]:
    v = rng.standard_normal(dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_noise(i: int, rng: np.random.Generator, dim: int = EMBED_DIM) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"unrelated session noise record #{i}: " + ("y " * 20),
        aaak_index="",
        embedding=_random_unit_vector(rng, dim=dim),
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def run_verbatim_bench(
    store: MemoryStore | None = None,
    n_records: int = 20,
    session_gap: int = 20,
    noise_per_session: int = 10,
    seed: int = NOISE_SEED,
    *,
    skip_l0_seed: bool = False,
    storage_direct: bool = False,
    k: int | None = None,
) -> dict:
    s = store if store is not None else MemoryStore()
    if not skip_l0_seed:
        _seed_l0_identity(s)

    dim = s.embed_dim

    pinned_texts = [
        f"alice said on day {i}: verbatim phrase #{i}-{'x' * 10}"
        for i in range(n_records)
    ]
    pinned_records = [_make_pinned(t, dim=dim) for t in pinned_texts]
    for r in pinned_records:
        s.insert(r)

    rng = np.random.default_rng(seed)
    for session_idx in range(session_gap):
        for j in range(noise_per_session):
            s.insert(_make_noise(session_idx * noise_per_session + j, rng, dim=dim))

    flush_record_buffer(s)

    cue_emb = [1.0] * dim
    effective_k = k if k is not None else max(n_records + 10, 20)
    hits_exact = 0
    for text in pinned_texts:
        if storage_direct:
            raw = s.query_similar(cue_emb, k=effective_k)
            literal_surfaces = [rec.literal_surface for rec, _score in raw]
        else:
            resp = recall(
                store=s,
                cue_embedding=cue_emb,
                cue_text=text,
                session_id="bench-verbatim",
                budget_tokens=5000,
                k_hits=effective_k,
                k_anti=3,
                mode="concept",
            )
            literal_surfaces = [h.literal_surface for h in resp.hits]
        if text in literal_surfaces:
            hits_exact += 1

    accuracy = hits_exact / n_records if n_records > 0 else 0.0
    return {
        "accuracy": accuracy,
        "n_records": n_records,
        "session_gap": session_gap,
        "noise_per_session": noise_per_session,
        "hits_exact": hits_exact,
        "passed": accuracy >= ACCURACY_FLOOR,
        "floor": ACCURACY_FLOOR,
        "noise_mode": "random-unit-vectors",
        "noise_seed": seed,
        "skip_l0_seed": bool(skip_l0_seed),
        "storage_direct": bool(storage_direct),
        "k": int(effective_k),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.verbatim",
        description="verbatim recall benchmark + diagnostics",
    )
    parser.add_argument(
        "--skip-l0-seed",
        action="store_true",
        help="D5-01 diagnostic: skip _seed_l0_identity to isolate L0 crowding effect",
    )
    parser.add_argument(
        "--storage-direct",
        action="store_true",
        help="D5-01 diagnostic: bypass recall(), call store.query_similar directly",
    )
    parser.add_argument(
        "--n", "--n-records",
        dest="n_records",
        type=int,
        default=20,
        help="pinned record count (default 20)",
    )
    parser.add_argument(
        "--gap", "--session-gap",
        dest="session_gap",
        type=int,
        default=20,
        help="session gap -- how many noise sessions between writes and recall (default 20)",
    )
    parser.add_argument(
        "--noise-per-session",
        type=int,
        default=10,
        help="noise records per simulated session (default 10)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="override k_hits (default: max(n_records + 10, 20))",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = run_verbatim_bench(
        n_records=args.n_records,
        session_gap=args.session_gap,
        noise_per_session=args.noise_per_session,
        skip_l0_seed=args.skip_l0_seed,
        storage_direct=args.storage_direct,
        k=args.k,
    )
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
