"""A resident, freshness-checked full-corpus graph already holds a
boot-decrypted `literal_surface` per node. `_RankIndexHandle._build` reuses
that plaintext by reference for every bundle-resident id instead of paying
its own independent decrypt, collapsing the two whole-corpus decrypt passes
(boot + first rank-index build) into one. A bundle-absent id (no warm graph,
or a row the warm graph never saw) still pays its own decrypt exactly as
before.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore, flush_record_buffer
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


def _build_corpus(store: MemoryStore, n: int = 3) -> dict[str, UUID]:
    ids: dict[str, UUID] = {}
    for i in range(n):
        rid = uuid4()
        surface = f"a distinct surface about topicmarker{i} for record {i}"
        store.insert(_make_rec(rid, surface, _unit_vec(i + 1)))
        ids[f"rec_{i}"] = rid
    flush_record_buffer(store)
    return ids


def _decrypt_spy(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> dict[UUID, int]:
    """Wraps `store._decrypt_for_record` with a per-id call counter, still
    delegating to the original implementation so decrypted content stays
    correct."""
    calls: dict[UUID, int] = {}
    original = store._decrypt_for_record

    def _spy(record_id: UUID, value: str) -> str:
        calls[record_id] = calls.get(record_id, 0) + 1
        return original(record_id, value)

    monkeypatch.setattr(store, "_decrypt_for_record", _spy)
    return calls


def test_build_reuses_warm_bundle_surface_zero_decrypt_calls(
    _store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
):
    ids = _build_corpus(_store)
    calls = _decrypt_spy(_store, monkeypatch)

    # Boot: populates store._warm_graph_bundle with fully-decrypted surfaces.
    graph, _assignment, _rich_club = retrieve.build_runtime_graph(_store)
    assert calls, "fixture self-check: the boot pass must decrypt at least once"
    calls.clear()

    handle = rank_index_for(_store)
    handle._build()

    assert calls == {}, (
        "the rank-index build must perform zero decrypt calls for ids "
        f"already resident in the warm bundle -- got calls: {calls}"
    )

    _generation, rust_ids_list, _matrix, _degree, postings = handle.snapshot(
        tokens=["topicmarker1"],
    )
    rust_ids = {UUID(int=i) for i in rust_ids_list}
    for rid in ids.values():
        assert rid in rust_ids, "a bundle-resident id must be indexed"

    assert ids["rec_1"].int in postings.get("topicmarker1", {}), (
        "the surface reused from the warm bundle must tokenize identically "
        "to a freshly-decrypted surface -- a mismatched token means the "
        "bundle-sourced string diverged from the true decrypted content"
    )


def test_build_falls_back_to_decrypt_when_no_warm_bundle(
    _store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
):
    ids = _build_corpus(_store)
    calls = _decrypt_spy(_store, monkeypatch)

    # No retrieve.build_runtime_graph call -- store._warm_graph_bundle is
    # absent, so the fallback path must decrypt every row itself, exactly
    # as before this change.
    handle = rank_index_for(_store)
    handle._build()

    assert len(calls) == len(ids), (
        "with no warm bundle resident, every row must still pay its own "
        f"decrypt -- expected {len(ids)} distinct ids decrypted, got {calls}"
    )

    _generation, _rust_ids, _matrix, _degree, postings = handle.snapshot(
        tokens=["topicmarker0"],
    )
    assert ids["rec_0"].int in postings.get("topicmarker0", {})


def test_build_divergence_bundle_authoritative_over_a_failing_independent_decrypt(
    _store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
):
    """Closes the open decrypt-divergence question: for a bundle-resident
    id, a build-time decrypt attempt that would independently fail (e.g. a
    transient crypto error unrelated to the bytes the boot pass already
    proved decryptable) must never run, and must never change indexed
    membership -- the merge makes the boot decrypt authoritative for
    bundle-resident rows instead of requiring a second, independently
    successful decrypt.
    """
    ids = _build_corpus(_store)
    target_id = ids["rec_1"]

    graph, _assignment, _rich_club = retrieve.build_runtime_graph(_store)
    assert str(target_id) in graph._node_payload, (
        "fixture self-check: the target id must be bundle-resident after boot"
    )

    original = _store._decrypt_for_record
    divergent_calls: list[UUID] = []

    def _divergent(record_id: UUID, value: str) -> str:
        if record_id == target_id:
            divergent_calls.append(record_id)
            raise ValueError("simulated independent-decrypt divergence")
        return original(record_id, value)

    monkeypatch.setattr(_store, "_decrypt_for_record", _divergent)

    handle = rank_index_for(_store)
    handle._build()  # must not raise -- the divergent path is never taken

    assert divergent_calls == [], (
        "a bundle-resident id must never reach the independent decrypt "
        "call at all, divergent or not"
    )

    _generation, rust_ids_list, _matrix, _degree, postings = handle.snapshot(
        tokens=["topicmarker1"],
    )
    rust_ids = {UUID(int=i) for i in rust_ids_list}
    assert target_id in rust_ids, (
        "a bundle-resident id must stay indexed even though its own "
        "independent decrypt would have raised -- the bundle's already-"
        "proven-successful boot decrypt is authoritative here"
    )
    assert target_id.int in postings.get("topicmarker1", {}), (
        "the indexed row must carry the correct (bundle-sourced) surface "
        "content, not an empty or corrupted string"
    )
