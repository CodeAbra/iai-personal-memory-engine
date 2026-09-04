"""Offline builder for a live-daemon priming fixture: a fully-committed lilli
store on disk, planted BEFORE any daemon process starts (the priming-cache
loader memoizes per store-instance, so a freshly spawned daemon subprocess
reads whatever is already on disk).

The fixture plants a seed record whose literal surface equals the frozen cue
text (a guaranteed recall seed), a pool of filler records sized into the
hundreds so the widening cost this fixture exists to expose actually fires,
and a candidate target record built as a cue-orthogonal blend at a chosen
cosine -- never a frozen literal string, since a target sharing the cue's
wording inherits lexical-match bonuses that make it served in every arm
regardless of priming and defeats set-exclusivity by construction.

Two independent builder flags compose:
  - ``require_band``: False plants a single fixed-cosine target (the pool
    widening this exposes does not depend on the target ever being served);
    True sweeps a small cosine ladder and accepts the first candidate whose
    OBSERVED two-arm serving outcome satisfies the acceptance predicate below,
    raising loudly if none does, rather than silently persisting a candidate
    that would make the ON/OFF comparison vacuous.
  - ``plant``: whether the winning target actually gets wired into the
    priming cache before the store closes. False builds a byte-identical
    corpus with an empty priming cache, so the same measurement machinery can
    demonstrate its own negative result on an unplanted copy.

Acceptance predicate for a band-verified candidate (all three required):
  1. absent from the served set with priming off,
  2. present in the served set with priming on (default nudge),
  3. absent again with priming on but the nudge multiplier neutralized to
     1.0 -- isolating that the SPECIFIC score nudge, not mere pool widening,
     is what puts the candidate over the served-set line.
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from iai_mcp import prime_cache
from iai_mcp.embed import Embedder
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

from tests._synthetic_cue_corpus import insert_corpus
from tests._warm_recall_repro_support import _assert_not_prod_path

CUE = "how does the archive cleanup job handle stale sessions this month"

_POOL_FLOOR = 500
_N_PLAIN_FILLERS = 640
_N_DECOY_FILLERS = 60
_ALPHA_LADDER = (0.84, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93)
_SINGLE_FIXED_ALPHA = 0.86
_DRIVER = "lilli"


class BandSearchExhaustedError(RuntimeError):
    """No candidate in the alpha ladder satisfied the three-arm predicate."""


@dataclass(frozen=True)
class BuiltFixture:
    root: Path
    pool_count: int
    src_id: str
    dst_id: str
    chunk_id: str
    cue: str
    dst_cosine: float


def _unit(v) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n) if n > 0 else arr


def _orthogonal(rng: np.random.Generator, basis_unit: np.ndarray, dim: int) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    v = v - basis_unit * float(np.dot(v, basis_unit))
    return _unit(v)


def _blend(cue_unit: np.ndarray, ortho: np.ndarray, alpha: float) -> list[float]:
    beta = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
    return _unit(alpha * cue_unit + beta * ortho).tolist()


def _mk_record(
    literal_surface: str, embedding: list[float], created_at: datetime, *, tier: str = "episodic",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid.uuid4(), tier=tier, literal_surface=literal_surface, aaak_index="",
        embedding=embedding, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=created_at, updated_at=created_at, tags=[], language="en",
    )


def _make_fillers(rng: np.random.Generator, cue_unit: np.ndarray, dim: int, ts: datetime) -> "list[MemoryRecord]":
    out: list[MemoryRecord] = []
    for i in range(_N_PLAIN_FILLERS):
        alpha = float(rng.uniform(0.55, 0.85))
        ortho = _orthogonal(rng, cue_unit, dim)
        emb = _blend(cue_unit, ortho, alpha)
        text = f"unrelated filler passage number {i} static log entry lorem qux zephyr"
        out.append(_mk_record(text, emb, ts))
    for i in range(_N_DECOY_FILLERS):
        alpha = float(rng.uniform(0.92, 0.97))
        ortho = _orthogonal(rng, cue_unit, dim)
        emb = _blend(cue_unit, ortho, alpha)
        text = (
            f"related note {i} about archive cleanup and stale session handling "
            f"covered by a separate maintenance procedure"
        )
        out.append(_mk_record(text, emb, ts))
    return out


@contextmanager
def _env_scope(**kv: "str | None") -> Iterator[None]:
    prev = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _served_ids(
    store, graph, assignment, rich_club, embedder, cue_unit, *,
    prime: str, boost: "str | None" = None,
) -> "set[str]":
    from iai_mcp.pipeline import recall_for_response

    with _env_scope(IAI_MCP_PROC_PRIME=prime, IAI_MCP_PROC_PRIME_BOOST=boost):
        graph._records_view_cache = None
        resp = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=CUE, session_id=f"fixture-build-{prime}-{boost}",
            budget_tokens=1500, mode="concept", cue_embedding=cue_unit.tolist(),
            use_rust_scorer=True,
        )
        return {str(h.record_id) for h in resp.hits}


def _select_band_verified_dst(
    store, src: MemoryRecord, candidates: "list[tuple[float, MemoryRecord]]", embedder, cue_unit,
    graph, assignment, rich_club,
) -> "tuple[MemoryRecord, str, float]":
    tried: list[float] = []
    for alpha, candidate in candidates:
        chunk_id = uuid.uuid4().hex
        assert prime_cache.save(store, {
            "seed_to_chunks": {str(src.id): [chunk_id]},
            "chunk_members": {chunk_id: [str(src.id), str(candidate.id)]},
        }) is True
        prime_cache.invalidate(store)
        tried.append(alpha)

        off_ids = _served_ids(store, graph, assignment, rich_club, embedder, cue_unit, prime="0")
        if str(candidate.id) in off_ids:
            continue
        on_ids = _served_ids(store, graph, assignment, rich_club, embedder, cue_unit, prime="1")
        if str(candidate.id) not in on_ids:
            continue
        on_neutral_ids = _served_ids(
            store, graph, assignment, rich_club, embedder, cue_unit, prime="1", boost="1.0",
        )
        if str(candidate.id) in on_neutral_ids:
            continue

        return candidate, chunk_id, alpha

    raise BandSearchExhaustedError(
        "no candidate target satisfied the three-arm acceptance predicate "
        "(absent from served(PRIME=0), present in served(PRIME=1), absent again "
        "with the priming nudge neutralized) against the frozen cue; alphas "
        f"attempted: {tried}"
    )


def build_fixture(
    root: Path, *, plant: bool = True, require_band: bool = False, driver: str = _DRIVER,
) -> BuiltFixture:
    """Build a fully-committed store at ``root`` (must not already exist).

    ``plant`` controls whether the winning target's chunk is left wired into
    the on-disk priming cache; ``require_band`` controls whether the target
    is a single fixed-cosine plant (widening-only) or a band-verified pick
    from the acceptance-predicate ladder (set-exclusivity, attributable to
    the priming nudge specifically). ``driver`` selects the storage backend
    the fixture is built and reopened against -- the live-daemon p95 gate
    itself stays on the default (lilli) driver; the parameter exists for a
    store-round-trip parity check across both supported drivers.
    """
    _assert_not_prod_path(root)
    if root.exists():
        raise ValueError(f"fixture root must be fresh (absent): {root}")

    with _env_scope(IAI_MCP_STORE=str(root), LILLI_STORAGE_DRIVER=driver):
        embedder = Embedder()
        store = MemoryStore(path=root)
        try:
            recent = datetime.now(timezone.utc) - timedelta(hours=1)
            cue_unit = _unit(embedder.embed(CUE))
            dim = cue_unit.shape[0]
            rng = np.random.default_rng(0)

            src = _mk_record(CUE, cue_unit.tolist(), recent)
            fillers = _make_fillers(rng, cue_unit, dim, recent)
            dst_ortho = _orthogonal(rng, cue_unit, dim)

            if require_band:
                candidates = []
                for alpha in _ALPHA_LADDER:
                    emb = _blend(cue_unit, dst_ortho, alpha)
                    text = f"generic candidate surface marker position {alpha:.2f} neutral"
                    candidates.append((alpha, _mk_record(text, emb, recent)))
                all_records = [src, *fillers, *[rec for _, rec in candidates]]
            else:
                emb = _blend(cue_unit, dst_ortho, _SINGLE_FIXED_ALPHA)
                text = "generic candidate surface marker single fixed position neutral"
                fixed_dst = _mk_record(text, emb, recent)
                all_records = [src, *fillers, fixed_dst]

            insert_corpus(store, all_records)
            pool_count = len(all_records)
            if pool_count < _POOL_FLOOR:
                raise RuntimeError(
                    f"built pool count {pool_count} is below the required floor of {_POOL_FLOOR}"
                )

            # Persist a warm runtime-graph cache to disk as part of the offline
            # build, exactly as a real capture-then-consolidate cycle would
            # leave the store -- a freshly spawned daemon booting against a
            # store with no persisted cache pays a cold structural rebuild on
            # its first calls, which the require_band search below already
            # triggers as a side effect but the single-target path would not.
            from iai_mcp.retrieve import build_runtime_graph

            graph, assignment, rich_club = build_runtime_graph(store)

            if require_band:
                dst, chunk_id, dst_cosine = _select_band_verified_dst(
                    store, src, candidates, embedder, cue_unit,
                    graph, assignment, rich_club,
                )
            else:
                dst, chunk_id, dst_cosine = fixed_dst, uuid.uuid4().hex, _SINGLE_FIXED_ALPHA

            if plant:
                assert prime_cache.save(store, {
                    "seed_to_chunks": {str(src.id): [chunk_id]},
                    "chunk_members": {chunk_id: [str(src.id), str(dst.id)]},
                }) is True
            else:
                assert prime_cache.save(
                    store, {"seed_to_chunks": {}, "chunk_members": {}},
                ) is True
            prime_cache.invalidate(store)
        finally:
            store.close()

    return BuiltFixture(
        root=root, pool_count=pool_count, src_id=str(src.id), dst_id=str(dst.id),
        chunk_id=chunk_id, cue=CUE, dst_cosine=dst_cosine,
    )
