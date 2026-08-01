"""Field-for-field equivalence between single-row get() and batched get_batch().

get_batch() must be a drop-in substitute for get() in the recall hot-path: the
record it returns has to be byte-identical across EVERY MemoryRecord field,
including the V5-codec structure fields (structure_hv, structure_hv_payload,
hv_tier) that travel through distinct projection paths. This holds under both
storage drivers.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, STRUCTURE_HV_BYTES, MemoryRecord


def _record(
    *,
    text: str,
    tier: str = "episodic",
    detail: int = 2,
    pinned: bool = False,
    never_merge: bool = False,
    language: str = "en",
    structure_hv_payload: bytes = b"",
    hv_tier: str = "bsc",
    provenance: list | None = None,
    centrality: float = 0.0,
    s5_trust_score: float = 0.5,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.013] * EMBED_DIM,
        community_id=None,
        centrality=centrality,
        detail_level=detail,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=never_merge,
        provenance=provenance if provenance is not None else [],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language=language,
        s5_trust_score=s5_trust_score,
        structure_hv_payload=structure_hv_payload,
        hv_tier=hv_tier,
    )


def _corpus() -> list[MemoryRecord]:
    """Records that exercise every projection-sensitive field path."""
    return [
        # Plain record (structure_hv auto-bound on insert -> non-empty).
        _record(text="hello world"),
        # Verbatim torture: NUL, emoji, CJK, RTL.
        _record(text="line1\x00line2 \U0001f9e0 漢字 العربية mixed"),
        # Pinned + never_merge (these flags must round-trip exactly).
        _record(text="pinned fact", pinned=True, never_merge=True),
        # Non-empty V5 structure_hv_payload on a non-bsc tier — the field the
        # projection must not silently zero.
        _record(
            text="codec payload record",
            hv_tier="fhrr",
            structure_hv_payload=bytes(range(64)),
        ),
        # Populated provenance (encrypted JSON round-trip).
        _record(
            text="provenance record",
            provenance=[{"session_id": "sess-42", "captured_at": "2026-06-26"}],
        ),
        # Non-default scalars.
        _record(text="scored record", centrality=0.731, s5_trust_score=0.875),
        # semantic tier with a distinct embedding vector.
        _record(
            text="semantic record",
            tier="semantic",
            embedding=[float(i % 7) * 0.01 for i in range(EMBED_DIM)],
        ),
    ]


_FIELDS = [f.name for f in dataclasses.fields(MemoryRecord)]


def _assert_field_identical(got, batched, field: str) -> None:
    a = getattr(got, field)
    b = getattr(batched, field)
    if field == "embedding":
        assert len(a) == len(b), f"embedding length differs for {field}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert x == y, f"embedding[{i}] differs: {x!r} != {y!r}"
        return
    assert a == b, f"field {field!r} differs: get={a!r} get_batch={b!r}"
    assert type(a) is type(b), (
        f"field {field!r} type differs: {type(a).__name__} vs {type(b).__name__}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_get_batch_field_identical_to_get(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    records = _corpus()
    for r in records:
        store.insert(r)

    ids = [r.id for r in records]

    # Single-id batch: get_batch([id])[id] must equal get(id) field-for-field.
    for rid in ids:
        got = store.get(rid)
        assert got is not None, f"get({rid}) returned None"
        batch = store.get_batch([rid])
        assert rid in batch, f"get_batch([{rid}]) missing the id"
        batched = batch[rid]
        for field in _FIELDS:
            _assert_field_identical(got, batched, field)

    # structure_hv must be non-empty after insert auto-binds it, and must
    # survive the batched projection (the gate the consensus flagged).
    for rid in ids:
        got = store.get(rid)
        batched = store.get_batch([rid])[rid]
        assert got.structure_hv, f"structure_hv unexpectedly empty for {rid}"
        assert len(got.structure_hv) == STRUCTURE_HV_BYTES
        assert batched.structure_hv == got.structure_hv, (
            "get_batch silently zeroed structure_hv"
        )

    # Multi-id batch in one call: every record identical to its get().
    full = store.get_batch(ids)
    assert set(full.keys()) == set(ids)
    for rid in ids:
        got = store.get(rid)
        for field in _FIELDS:
            _assert_field_identical(got, full[rid], field)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_get_batch_missing_id_is_skipped(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)
    r = _record(text="present")
    store.insert(r)
    absent = uuid4()
    batch = store.get_batch([r.id, absent])
    assert r.id in batch
    assert absent not in batch  # identical to the loop's `if rec is None: skip`
