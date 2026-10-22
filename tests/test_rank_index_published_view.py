"""`_RankIndexHandle.published_view` in isolation: slot-map coverage, exact
embedding equality against the store's own decode path, the pending
all-zero placeholder, and memo identity across repeated calls at the same
generation. Exercises the accessor directly -- `_collect_graph_pool` no
longer wires through it (see the pipeline revert this test file replaces
coverage for)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.store._rank_index import rank_index_for
from iai_mcp.types import MemoryRecord

_DIM = 16  # small synthetic dim -- keeps this a single-file, native-embedder-free test


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    ts = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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
        created_at=ts,
        updated_at=ts,
        tags=["t"],
        language="en",
    )


@pytest.fixture
def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def test_slot_map_covers_every_resident_id(_store: MemoryStore):
    recs = [_make_rec(uuid4(), f"surface-{i}", _unit_vec(i)) for i in range(5)]
    for rec in recs:
        _store.insert(rec)

    _matrix, slot_map = rank_index_for(_store).published_view()

    assert set(slot_map.keys()) == {rec.id.int for rec in recs}


def test_matrix_row_equals_store_decoded_embedding_exactly(_store: MemoryStore):
    rec = _make_rec(uuid4(), "gamma", _unit_vec(3))
    _store.insert(rec)

    matrix, slot_map = rank_index_for(_store).published_view()

    slot = slot_map[rec.id.int]
    store_decoded = _store.get_batch([rec.id])[rec.id].embedding
    expected = np.asarray(store_decoded, dtype=np.float32)

    assert np.array_equal(matrix[slot], expected), (
        "both sides decode the same underlying embedding column through "
        "the same _decode_raw_row path -- exact equality, not np.allclose, "
        "is the correct proof of byte identity"
    )


def test_pending_row_published_matrix_row_is_all_zero(_store: MemoryStore):
    pending_id = uuid4()
    now = datetime.now(timezone.utc).isoformat()
    _store.insert_pending(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="a pending row with no embedding yet",
        tags_json="[]",
        provenance_json="[]",
        created_at=now,
        updated_at=now,
    )

    matrix, slot_map = rank_index_for(_store).published_view()

    slot = slot_map[pending_id.int]
    assert not np.any(matrix[slot])


def test_memo_returns_identical_object_at_same_generation(_store: MemoryStore):
    rec = _make_rec(uuid4(), "alpha", _unit_vec(1))
    _store.insert(rec)

    handle = rank_index_for(_store)
    first_matrix, first_slots = handle.published_view()
    second_matrix, second_slots = handle.published_view()

    assert first_matrix is second_matrix
    assert first_slots is second_slots
