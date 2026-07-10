from __future__ import annotations

# Historical context for this file:
#   It originally pinned the contract of a per-store AESGCM cache that the
#   store-level decrypt wrapper used to amortize cipher construction across
#   read loops. The codec/zstd-envelope refactor collapsed that wrapper into a
#   3-line delegate onto the canonical crypto.decrypt_field primitive, which
#   constructs its own AEAD context per call. The cache + its invalidate hook
#   are gone by design (single chokepoint > micro-cache).
#
# Per-test disposition under the new contract:
#   - The "uses cached AESGCM" assertion is gone (cache deleted); rewritten to
#     prove the delegate routes through crypto.decrypt_field instead.
#   - "is actually cached" / "invalidate clears" — quarantined; assert on
#     symbols that no longer exist.
#   - "byte-identical output" / "unique per-record nonce" — kept; durable
#     round-trip invariants independent of caching.
#   - "skips cache for plaintext passthrough" — rewritten as a plaintext
#     short-circuit guard (is_encrypted() must fire before any decrypt call).

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

def _make(
    text: str = "hello world",
    tier: str = "episodic",
    tags: list[str] | None = None,
    detail: int = 2,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags if tags is not None else [],
        language=language,
    )

@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")

def test_decrypt_for_record_delegates_to_crypto_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store-level decrypt MUST route through the canonical crypto primitive
    so the codec rules (envelope routing, fail-loud on corruption) apply
    uniformly. The cache that used to short-circuit this path is gone; the
    delegate now constructs the AEAD context inside crypto.decrypt_field per
    call. The durable contract is that the delegate fires."""
    from iai_mcp import crypto as crypto_mod
    from iai_mcp.store import _store as store_mod

    calls: list[str] = []
    real_decrypt = crypto_mod.decrypt_field

    def spy(value, key, associated_data=b""):
        calls.append(str(value)[:25])
        return real_decrypt(value, key, associated_data=associated_data)

    monkeypatch.setattr(crypto_mod, "decrypt_field", spy)
    monkeypatch.setattr(store_mod, "decrypt_field", spy, raising=False)

    store_local = MemoryStore(path=tmp_path / "lancedb")
    for i in range(5):
        store_local.insert(_make(text=f"record-{i}"))

    calls.clear()
    records = store_local.all_records()
    assert len(records) == 5
    assert calls, (
        "store-level decrypt wrapper bypassed crypto.decrypt_field — the "
        "codec route would never fire on this read path"
    )

def test_decrypt_for_record_output_byte_identical(
    store: MemoryStore,
) -> None:
    """Round-trip byte-identity is the lossless-recall invariant; the codec
    must not lose a single byte on any path."""
    verbatim = "ハロー世界 — ground truth literal"
    rec = _make(text=verbatim)
    store.insert(rec)

    via_all = store.all_records()
    assert len(via_all) == 1
    assert via_all[0].literal_surface == verbatim

    via_get = store.get(rec.id)
    assert via_get is not None
    assert via_get.literal_surface == verbatim

def test_cached_aesgcm_property_removed(store: MemoryStore) -> None:
    """The per-store AESGCM cache and its invalidate hook were removed when
    the decrypt wrapper collapsed to a delegate. This test pins their absence
    so the cache cannot drift back in without the contract change being made
    explicit again."""
    assert not hasattr(store, "_cached_aesgcm"), (
        "_cached_aesgcm property reappeared on MemoryStore — codec route "
        "would silently skip the canonical primitive"
    )
    assert not hasattr(store, "_invalidate_aesgcm_cache"), (
        "_invalidate_aesgcm_cache helper reappeared on MemoryStore — its "
        "only previous caller (cmd_crypto_rotate) was updated to drop the "
        "explicit invalidation, since key rotation now propagates through "
        "store._crypto_key reassignment"
    )

def test_decrypt_handles_unique_per_record_nonce(store: MemoryStore) -> None:
    """Each insert picks a fresh nonce; multiple records with identical
    plaintext must still round-trip independently."""
    duplicate = "duplicate"
    recs = [_make(text=duplicate) for _ in range(3)]
    for r in recs:
        store.insert(r)

    for original in recs:
        got = store.get(original.id)
        assert got is not None
        assert got.literal_surface == duplicate

def test_decrypt_for_record_short_circuits_for_plaintext_passthrough(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The is_encrypted() short-circuit must fire BEFORE the delegate calls
    into the primitive; plaintext values cost nothing on the read path. We
    pin this by spying on crypto.decrypt_field and asserting it never gets
    invoked when the input lacks the ciphertext prefix."""
    from iai_mcp import crypto as crypto_mod
    from iai_mcp.store import _store as store_mod

    calls: list[str] = []
    real_decrypt = crypto_mod.decrypt_field

    def spy(value, key, associated_data=b""):
        calls.append(str(value)[:25])
        return real_decrypt(value, key, associated_data=associated_data)

    monkeypatch.setattr(crypto_mod, "decrypt_field", spy)
    monkeypatch.setattr(store_mod, "decrypt_field", spy, raising=False)

    rid = uuid4()
    plaintext = "plaintext that has no iai:enc:v1: prefix"

    out = store._decrypt_for_record(rid, plaintext)
    assert out == plaintext
    assert not calls, (
        "is_encrypted() short-circuit must fire BEFORE crypto.decrypt_field; "
        "plaintext passthrough should not pay AEAD construction cost"
    )
