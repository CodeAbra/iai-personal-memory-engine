"""High-volume capture burst through dedup + buffer flush + edge merge_insert.

Drives several hundred capture-shaped inserts through the real write-time
pattern-separation gate (``store.insert``) with a KNOWN mix of exact
duplicates, near-duplicates at/above ``near_dup_threshold``, and distinct
records, then deliberately re-boosts colliding edges across buffered and
flushed batches. Every synthetic embedding is placed at an EXACT, controlled
cosine (a disjoint 2-d rotation plane per record group, mutually orthogonal
across groups) rather than drawn from a real or hash-seeded embedder, so
every dedup verdict is a reproducible, known number rather than an emergent
one -- the same technique ``tests/_dedup_eval_fixture.py`` uses for the
per-pair eval set, extended here to a volume burst.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.rss_rca.harness import _real_store_signature

# ---------------------------------------------------------------------------
# Burst shape. near_dup_threshold defaults to 0.92 (daemon_config.py); the
# near-dup cosines below are chosen strictly above it so every one is a
# guaranteed SKIP-merge, and the singleton axes carry no companion at all so
# they are guaranteed to stay distinct.
# ---------------------------------------------------------------------------
_N_CLUSTERS = 40
_EXACT_DUPS_PER_CLUSTER = 4
_NEAR_DUP_COSINES = (0.995, 0.97, 0.93)
_N_SINGLETONS = 20

_N_GROUPS = _N_CLUSTERS + _N_SINGLETONS
_EXPECTED_DISTINCT_COUNT = _N_GROUPS
_EMBED_DIM = 2 * _N_GROUPS  # one disjoint (axis, axis+1) rotation plane per group


def _make_embedding_at_cosine(
    cos_target: float, axis_index: int, embed_dim: int = _EMBED_DIM,
) -> list[float]:
    """A unit vector at exact cosine ``cos_target`` against axis ``axis_index``,
    confined to the (axis_index, axis_index + 1) plane -- zero everywhere
    else, so vectors on disjoint axis pairs are exactly orthogonal.
    """
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    vec = [0.0] * embed_dim
    vec[axis_index] = cos_target
    vec[axis_index + 1] = residual
    return vec


def _reference_embedding(axis_index: int, embed_dim: int = _EMBED_DIM) -> list[float]:
    vec = [0.0] * embed_dim
    vec[axis_index] = 1.0
    return vec


def _make_record(*, embedding: list[float], literal_surface: str) -> "MemoryRecord":  # noqa: F821
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["bench"],
        language="en",
    )


def _build_burst() -> list[tuple[list[float], str]]:
    """Return ``(embedding, literal_surface)`` pairs in a known distinct-vs-
    duplicate mix: each of ``_N_CLUSTERS`` groups contributes one canonical
    record, ``_EXACT_DUPS_PER_CLUSTER`` byte-identical-embedding duplicates,
    and one entry per ``_NEAR_DUP_COSINES`` value (all >= near_dup_threshold,
    same axis, different text) -- every one of these must SKIP-merge into the
    cluster's single surviving record. ``_N_SINGLETONS`` further groups each
    contribute exactly one record on their own orthogonal axis -- these must
    stay distinct.
    """
    burst: list[tuple[list[float], str]] = []
    for c in range(_N_CLUSTERS):
        axis = 2 * c
        canonical_text = f"alice cluster {c} canonical capture note"
        canonical_vec = _reference_embedding(axis)
        burst.append((canonical_vec, canonical_text))
        for d in range(_EXACT_DUPS_PER_CLUSTER):
            burst.append((canonical_vec, canonical_text))
        for k, cos in enumerate(_NEAR_DUP_COSINES):
            burst.append((
                _make_embedding_at_cosine(cos, axis),
                f"alice cluster {c} paraphrase {k}",
            ))
    for s in range(_N_SINGLETONS):
        axis = 2 * (_N_CLUSTERS + s)
        burst.append((
            _reference_embedding(axis),
            f"alice distinct singleton capture {s}",
        ))
    return burst


def _redirect_env(tmp_path: Path) -> dict[str, str | None]:
    prev = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
            "IAI_MCP_EMBED_DIM", "IAI_MCP_PATSEP_DRY_RUN",
            "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        )
    }
    os.environ["HOME"] = str(tmp_path)
    os.environ["IAI_MCP_STORE"] = str(tmp_path / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_path / ".iai-mcp" / "user_model.json"
    )
    if not prev["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-capture-dedup-volume-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    os.environ["IAI_MCP_EMBED_DIM"] = str(_EMBED_DIM)
    # The write-time pattern-separation gate defaults to dry-run under pytest
    # (PYTEST_CURRENT_TEST is set) -- a SKIP verdict would still fall through
    # to a full insert in dry-run mode, so count == distinct-input could never
    # hold without this. Enforcement, not observation, is the point here.
    os.environ["IAI_MCP_PATSEP_DRY_RUN"] = "false"
    # The tightest near-dup cosine (0.93) sits only 0.01 above the daemon
    # default (0.92); pin the threshold explicitly so an ambient override
    # (developer shell, CI) cannot push it past 0.93 and break the merge.
    os.environ["IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD"] = "0.92"
    (tmp_path / ".iai-mcp").mkdir(parents=True, exist_ok=True)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.timeout(180)
def test_high_volume_capture_burst_dedup_and_edge_merge_hold(tmp_path: Path) -> None:
    """A several-hundred-turn capture burst neither leaks a duplicate nor
    drops a distinct record, never raises IntegrityError from a deliberately
    colliding edge merge_insert burst, and drains both write buffers to
    empty. The timeout above is the deadlock assertion: a hang fails this
    gate deterministically instead of hanging the suite.
    """
    sig_before = _real_store_signature()
    prev_env = _redirect_env(tmp_path)
    store = None
    try:
        from iai_mcp.errors import IntegrityError
        from iai_mcp.store import MemoryStore
        from iai_mcp.store._buffers import (
            _edge_buffer,
            _record_buffer,
            flush_edge_buffer,
            flush_record_buffer,
        )

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(
            path=store_path, user_id="alice",
            read_consistency_interval=timedelta(seconds=0),
        )

        burst = _build_burst()
        assert len(burst) >= 300, (
            f"burst volume {len(burst)} too small to exercise a high-volume "
            f"buffer flush + colliding edge merge"
        )

        surviving_ids: list[UUID] = []
        for embedding, text in burst:
            rec = _make_record(embedding=embedding, literal_surface=text)
            store.insert(rec)
            # insert() mutates rec.id in place to the existing record's id on
            # a SKIP-merge, so this always reads the SURVIVING id.
            surviving_ids.append(rec.id)

        flush_record_buffer(store)

        distinct_ids = sorted(set(surviving_ids), key=str)
        assert len(distinct_ids) == _EXPECTED_DISTINCT_COUNT, (
            f"expected {_EXPECTED_DISTINCT_COUNT} distinct surviving records "
            f"({_N_CLUSTERS} clusters + {_N_SINGLETONS} singletons), got "
            f"{len(distinct_ids)} from a {len(burst)}-turn burst"
        )

        active_count = store.active_records_count()
        assert active_count == _EXPECTED_DISTINCT_COUNT, (
            f"active_records_count() == {active_count}, expected "
            f"{_EXPECTED_DISTINCT_COUNT} -- dedup leaked a duplicate or "
            f"dropped a distinct record"
        )

        # Deliberate edge-merge collision stress: boost the SAME ring of
        # (src, dst) pairs across THREE separate calls -- the first lands
        # while the prior boosts (insert()'s own self-loop + pattern-
        # separation-seed edges) are still buffered, the second collides
        # with the first's own still-buffered rows, and the third (after an
        # explicit flush) collides with rows now genuinely committed to the
        # edges table. boost_edges/flush_edge_buffer both go through
        # merge_insert, never tbl.add -- none of these calls may raise
        # IntegrityError.
        ring_pairs = [
            (distinct_ids[i], distinct_ids[(i + 1) % len(distinct_ids)])
            for i in range(len(distinct_ids))
        ]
        try:
            store.boost_edges(ring_pairs, delta=0.1, edge_type="hebbian")
            store.boost_edges(ring_pairs, delta=0.1, edge_type="hebbian")
            flush_edge_buffer(store)
            store.boost_edges(ring_pairs, delta=0.1, edge_type="hebbian")
            flush_edge_buffer(store)
        except IntegrityError as exc:  # pragma: no cover -- must never fire
            pytest.fail(
                f"boost_edges raised IntegrityError under a deliberately "
                f"colliding burst: {exc}"
            )

        assert list(_record_buffer.get(id(store), [])) == [], (
            "record buffer did not drain to empty after flush"
        )
        assert list(_edge_buffer.get(id(store), [])) == [], (
            "edge buffer did not drain to empty after flush"
        )

    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        _restore_env(prev_env)
        sig_after = _real_store_signature()
        if sig_before != sig_after:
            raise RuntimeError(
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the test."
            )
