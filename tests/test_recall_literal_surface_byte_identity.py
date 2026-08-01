"""SC-4 guard: literal_surface byte-identity across a recall that fires
reinforce_record(is_retrieval=True).

Every successful ``memory_recall`` dispatch calls ``store.queue_reinforce``
on the returned hit ids (``core/__init__.py:770-777``), which — absent a
background queue — synchronously calls ``reinforce_record(rid,
is_retrieval=True)`` (``_store.py:1304-1323``). That call sets
``labile_until`` (a datetime metadata column) and boosts an edge weight; it
MUST NEVER touch ``literal_surface`` (Mottron EPF / double_empathy passive
invariant — verbatim/lossless recall is non-negotiable).

This test locks that invariant by reading ``literal_surface`` via
``store.get()`` (the exact AES-GCM decrypt round-trip — no whitespace or
unicode normalization anywhere on this path) before and after a recall that
returns the record as a hit, and asserting byte equality.

This closes a coverage gap the 182/183 test suites do not cover: neither
phase exercised the retrieval-reconsolidation branch specifically against a
literal_surface byte-identity assertion.

Runs GREEN today — this is a regression lock, not a RED guard. Any future
plan touching the reconsolidation path must keep this test green.

Dual-driver: parametrizes ``LILLI_STORAGE_DRIVER`` since the assertion
depends on a real store round-trip (insert, recall, re-fetch).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder

# Verbatim byte content deliberately including whitespace runs, unicode, and
# mixed-width characters — anything a lossy normalizer would alter.
_VERBATIM_SURFACES: list[str] = [
    "line one\n\n  line two with   extra   spaces\t\ttabbed",
    "Ünïcödé çhârs — em dash, füll-width Ａ, and a ¥ sign",
    "trailing whitespace kept exactly as-is   ",
]


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


@pytest.fixture(autouse=True)
def _clear_authority_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder — a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
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
        created_at=now,
        updated_at=now,
        tags=["capture"],
        language="en",
    )


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    import iai_mcp.embed as _embed_mod

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _StubEmbedder(vec))


def _dispatch_recall(store: MemoryStore, cue_vec: list[float]) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "byte-identity guard test cue, embedder is stubbed",
        "session_id": "byte-identity-guard-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _make_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_literal_surface_byte_identical_after_reinforcing_recall(driver, _make_store, monkeypatch):
    """Insert records with known verbatim literal_surface bytes, issue a
    recall that returns them as hits (triggering queue_reinforce ->
    reinforce_record(is_retrieval=True) on each), then re-fetch and assert
    byte-for-byte equality against the original strings — no normalization,
    no truncation, no alteration of any kind."""
    store = _make_store(driver)
    cue_vec = _seeded_vec(31)

    record_ids: list[UUID] = []
    for i, surface in enumerate(_VERBATIM_SURFACES):
        rid = uuid4()
        # First record shares the cue vector exactly so it is guaranteed to
        # be the top hit; the others get a slight perturbation so they are
        # still plausible candidates without being identical vectors.
        vec = cue_vec if i == 0 else _seeded_vec(31 + i)
        store.insert(_make_rec(rid, surface, vec))
        record_ids.append(rid)
    flush_record_buffer(store)

    before = {rid: store.get(rid).literal_surface for rid in record_ids}
    for rid, surface in zip(record_ids, _VERBATIM_SURFACES):
        assert before[rid] == surface, "pre-recall literal_surface already diverges from input"

    _stub_embedder_for_store(monkeypatch, cue_vec)
    resp = _dispatch_recall(store, cue_vec)
    assert resp.get("hits"), "recall returned no hits — cannot exercise the reinforce path"

    after = {rid: store.get(rid).literal_surface for rid in record_ids}
    for rid, surface in zip(record_ids, _VERBATIM_SURFACES):
        assert after[rid] == surface, (
            f"literal_surface changed after a reinforcing recall for record {rid}: "
            f"before={before[rid]!r} after={after[rid]!r}"
        )
        assert after[rid] == before[rid], (
            f"literal_surface diverged from its pre-recall value for record {rid}"
        )
