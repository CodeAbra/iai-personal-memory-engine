from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    store_for_test: dict[tuple[str, str], str] = {}

    def fake_get(service: str, username: str):
        return store_for_test.get((service, username))

    def fake_set(service: str, username: str, password: str) -> None:
        store_for_test[(service, username)] = password

    def fake_delete(service: str, username: str) -> None:
        store_for_test.pop((service, username), None)

    monkeypatch.setattr(_keyring, "get_password", fake_get)
    monkeypatch.setattr(_keyring, "set_password", fake_set)
    monkeypatch.setattr(_keyring, "delete_password", fake_delete)
    yield store_for_test

def _make(text: str = "hello", language: str = "en", detail: int = 2):
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
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
        provenance=[{"ts": "2026-04-17T12:00:00Z", "cue": "original cue", "session_id": "s1"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["topic:test"],
        language=language,
        profile_modulation_gain={"learnedKnob": 0.42},
    )

def test_insert_writes_encrypted_literal_surface_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make(text="top-secret Russian phrase: Привет")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["literal_surface"].startswith("iai:enc:v1:")

def test_insert_writes_encrypted_provenance_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")

def test_insert_writes_encrypted_profile_modulation_gain_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["profile_modulation_gain_json"].startswith("iai:enc:v1:")

def test_embedding_remains_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    emb = list(row["embedding"])
    assert len(emb) == store.embed_dim
    assert emb[0] == pytest.approx(0.1)

def test_language_remains_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make(language="ru", text="Привет")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["language"] == "ru"

def test_tags_remain_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    tags = json.loads(row["tags_json"])
    assert tags == ["topic:test"]

def test_get_decrypts_literal_surface(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    text = "Alice said: пусть каждое слово сохранится точно"
    rec = _make(text=text)
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == text

def test_get_decrypts_provenance(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.provenance == rec.provenance

def test_get_decrypts_profile_modulation_gain(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.profile_modulation_gain == rec.profile_modulation_gain

def test_all_records_decrypts_all_rows(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    r1 = _make(text="first")
    r2 = _make(text="второй")
    store.insert(r1)
    store.insert(r2)

    all_r = store.all_records()
    texts = {r.literal_surface for r in all_r}
    assert "first" in texts
    assert "второй" in texts

def test_query_similar_still_works_after_encryption(tmp_path):
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM
    store = MemoryStore(path=tmp_path)
    rec = _make(text="probe me")
    store.insert(rec)
    hits = store.query_similar([0.1] * EMBED_DIM, k=5)
    assert len(hits) >= 1
    assert hits[0][0].literal_surface == "probe me"

def test_encrypted_row_cannot_be_decrypted_with_wrong_key(tmp_path, monkeypatch):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make(text="sensitive")
    store.insert(rec)

    store._crypto_key = b"\xff" * 32  # type: ignore[attr-defined]
    with pytest.raises(Exception):
        store.get(rec.id)

def test_ad_binding_prevents_row_swap(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.store import _uuid_literal

    store = MemoryStore(path=tmp_path)
    r_a = _make(text="row A secret")
    r_b = _make(text="row B secret")
    store.insert(r_a)
    store.insert(r_b)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    ct_a = df[df["id"] == str(r_a.id)].iloc[0]["literal_surface"]

    tbl.update(
        where=f"id = '{_uuid_literal(r_b.id)}'",
        values={"literal_surface": ct_a},
    )

    with pytest.raises(Exception):
        store.get(r_b.id)

def test_get_passes_through_plaintext_rows(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.store import _uuid_literal

    store = MemoryStore(path=tmp_path)
    rec = _make(text="plaintext-legacy")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where=f"id = '{_uuid_literal(rec.id)}'",
        values={
            "literal_surface": "plaintext-legacy",
            "provenance_json": json.dumps(rec.provenance),
            "profile_modulation_gain_json": json.dumps(rec.profile_modulation_gain),
        },
    )

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == "plaintext-legacy"
    assert got.provenance == rec.provenance
    assert got.profile_modulation_gain == rec.profile_modulation_gain

def test_append_provenance_batch_still_writes_encrypted(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    new_entry = {"ts": "2026-04-17T13:00:00Z", "cue": "batch cue", "session_id": "s2"}
    store.append_provenance_batch([(rec.id, new_entry)])

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")

    got = store.get(rec.id)
    assert got is not None
    cues = [p["cue"] for p in got.provenance]
    assert "batch cue" in cues

def test_append_provenance_single_still_writes_encrypted(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)
    store.append_provenance(rec.id, {"ts": "x", "cue": "y", "session_id": "z"})

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")

def test_reopen_store_with_same_keyring_decrypts(tmp_path):
    from iai_mcp.store import MemoryStore
    s1 = MemoryStore(path=tmp_path)
    rec = _make(text="persistent secret")
    s1.insert(rec)
    del s1

    s2 = MemoryStore(path=tmp_path)
    got = s2.get(rec.id)
    assert got is not None
    assert got.literal_surface == "persistent secret"


# --------------------------------------------------------------------------- #
# Envelope contract — store-level integration tests for the compression       #
# codec that lives inside the AES-GCM envelope on the 3 record columns.       #
#                                                                             #
# These tests fail RED on the current tree (no codec, _decrypt_for_record     #
# still inlines AESGCM and bypasses the canonical decrypt primitive) and      #
# turn GREEN once the codec lands and _decrypt_for_record delegates.          #
# --------------------------------------------------------------------------- #

ENVELOPE_VERSION_UNCOMPRESSED: int = 0x00
ENVELOPE_VERSION_ZSTD: int = 0x01


def _wrap_payload_as_ciphertext(payload: bytes, key: bytes, ad: bytes) -> str:
    """AEAD-encrypt a hand-built envelope payload (version byte + body) and
    return the wire form. Mirrors the codec write-side AEAD step, used to
    inject hand-crafted legacy 0x00 rows and AEAD-valid-but-body-corrupt
    0x01 rows directly into the on-disk records table."""
    import base64
    import secrets

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ct_with_tag = AESGCM(key).encrypt(nonce, payload, ad or None)
    return "iai:enc:v1:" + base64.b64encode(nonce + ct_with_tag).decode("ascii")


def _aead_plaintext_of(ciphertext_b64: str, key: bytes, ad: bytes) -> bytes:
    """Decrypt the AEAD layer directly and return the raw envelope payload
    (version byte + body). Used to inspect on-disk wire after a write."""
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    prefix = "iai:enc:v1:"
    assert ciphertext_b64.startswith(prefix)
    raw = base64.b64decode(ciphertext_b64[len(prefix):])
    nonce, ct_with_tag = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ct_with_tag, ad or None)


def _direct_sql_update(store, table: str, record_id, column: str, value: str) -> None:
    """Write a column on a specific row by going straight to the SQLite
    connection. Bypasses the encrypt-on-write chokepoint, which is exactly
    what we need to seed a legacy 0x00 row or to inject corruption."""
    rid = str(record_id).lower()
    with store.db._conn_lock:
        store.db._conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE id = ?",
            (value, rid),
        )


def test_decrypt_for_record_delegates_to_crypto_decrypt_field(tmp_path, monkeypatch):
    """The store-level decrypt wrapper must route through the canonical
    crypto.decrypt_field primitive so a 0x01 record column reads correctly.
    On the current tree the wrapper inlines AESGCM and bypasses the codec —
    so the spy never fires and a 0x01 row would UnicodeDecodeError on read."""
    from iai_mcp import crypto as crypto_mod
    from iai_mcp.store import MemoryStore, _store as store_mod

    calls: list[str] = []
    real_decrypt = crypto_mod.decrypt_field

    def spy(value, key, associated_data=b""):
        calls.append(str(value)[:25])
        return real_decrypt(value, key, associated_data=associated_data)

    # Patch BOTH the crypto module attribute and the store-module-local name
    # so the spy catches the call regardless of how the codec delegate imports
    # the primitive after refactor.
    monkeypatch.setattr(crypto_mod, "decrypt_field", spy)
    monkeypatch.setattr(store_mod, "decrypt_field", spy, raising=False)

    store = MemoryStore(path=tmp_path)
    rec = _make(text="payload that will compress on a real codec path")
    store.insert(rec)
    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == rec.literal_surface
    assert calls, (
        "store-level decrypt wrapper bypassed crypto.decrypt_field — a 0x01 "
        "compressed envelope would UnicodeDecodeError through this path"
    )


def test_no_double_encryption_via_is_encrypted_guard(tmp_path):
    """Feeding an already-encrypted value through the encrypt wrapper must
    short-circuit (is_encrypted guard). Encrypt-of-ciphertext on a write
    cascade would silently corrupt verbatim recall."""
    from uuid import uuid4

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    some_uuid = uuid4()
    real_ct = encrypt_field(
        "anything",
        store._key(),
        associated_data=str(some_uuid).lower().encode("ascii"),
    )
    out = store._encrypt_for_record(some_uuid, real_ct)
    assert out == real_ct, (
        "encrypt wrapper double-encrypted a value that was already ciphertext"
    )


def test_record_corrupt_zstd_frame_raises_hippo_integrity_error(tmp_path):
    """A record column carrying an AEAD-valid envelope whose body fails to
    decompress is on-disk corruption of authenticated content. Record reads
    must fail-loud (HippoIntegrityError or HippoDecryptError), never silently
    degrade or return garbage."""
    import pytest

    from iai_mcp.hippo import HippoDecryptError, HippoIntegrityError
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make(text="record we will corrupt")
    store.insert(rec)

    # AEAD-valid wire whose plaintext is a 0x01 envelope with a body that
    # carries the zstd magic prefix but garbage afterwards — passes the AEAD
    # tag check, fails the codec decompress step.
    bad_body = b"\x28\xb5\x2f\xfd" + b"\x00" * 16 + b"garbage-not-a-real-zstd-frame"
    bad_payload = bytes([ENVELOPE_VERSION_ZSTD]) + bad_body
    ad = str(rec.id).lower().encode("ascii")
    bad_wire = _wrap_payload_as_ciphertext(bad_payload, store._key(), ad=ad)

    _direct_sql_update(store, RECORDS_TABLE, rec.id, "literal_surface", bad_wire)

    with pytest.raises((HippoIntegrityError, HippoDecryptError)):
        store.get(rec.id)


def test_mixed_corpus_legacy_and_new_read_identically(tmp_path):
    """A pre-97 legacy row (raw utf-8, no version byte) and a newly-written
    0x01 zstd row must both round-trip identically through the store read path.
    Mixed corpus is the permanent steady-state during organic migration.

    Legacy rows have NO version prefix — the AEAD plaintext is the raw utf-8
    content.  This is the actual on-disk shape from the old writer; a synthetic
    0x00-prefixed envelope is NOT a real legacy row (the old writer never
    emitted 0x00).
    """
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)

    # Seed a legacy row: AEAD plaintext = raw utf-8, no version prefix.
    # This mirrors exactly what the old pre-97 writer produced.
    legacy = _make(text="placeholder for legacy row")
    store.insert(legacy)
    legacy_text = "legacy value — no version byte prefix"
    legacy_raw_utf8 = legacy_text.encode("utf-8")
    legacy_ad = str(legacy.id).lower().encode("ascii")
    legacy_wire = _wrap_payload_as_ciphertext(
        legacy_raw_utf8, store._key(), ad=legacy_ad
    )
    _direct_sql_update(
        store, RECORDS_TABLE, legacy.id, "literal_surface", legacy_wire
    )

    modern = _make(text="modern compressed value")
    store.insert(modern)

    got_legacy = store.get(legacy.id)
    got_modern = store.get(modern.id)
    assert got_legacy is not None
    assert got_modern is not None
    assert got_legacy.literal_surface == legacy_text, (
        "legacy raw-utf8 row must round-trip byte-identical"
    )
    assert got_modern.literal_surface == "modern compressed value"


def test_natural_rewrite_legacy_0x00_upgrades_to_0x01_on_update(tmp_path):
    """A legacy 0x00 row, when re-encrypted via store.update, must emit a
    fresh 0x01 envelope. Natural-rewrite migration relies on this exact
    promotion happening transparently at every write entry point."""
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make(text="will be rewritten")
    store.insert(rec)

    # Replace the on-disk literal_surface with a hand-crafted 0x00 envelope
    # so the next plaintext re-encryption is observable as a real promotion.
    legacy_payload = bytes([ENVELOPE_VERSION_UNCOMPRESSED]) + b"legacy literal"
    ad = str(rec.id).lower().encode("ascii")
    legacy_wire = _wrap_payload_as_ciphertext(legacy_payload, store._key(), ad=ad)
    _direct_sql_update(
        store, RECORDS_TABLE, rec.id, "literal_surface", legacy_wire
    )

    rec.literal_surface = "rewritten literal"
    store.update(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    new_wire = df[df["id"] == str(rec.id)].iloc[0]["literal_surface"]
    new_payload = _aead_plaintext_of(new_wire, store._key(), ad=ad)
    assert new_payload[0] == ENVELOPE_VERSION_ZSTD, (
        f"natural rewrite via update must emit compressed envelope; got "
        f"version byte {new_payload[0]:#x}"
    )


def test_natural_rewrite_legacy_0x00_upgrades_to_0x01_on_append_provenance(tmp_path):
    """append_provenance is the second main natural-rewrite trigger. A 0x00
    provenance column must promote to 0x01 the moment a new provenance entry
    is appended (which re-encrypts the JSON list)."""
    import json

    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    legacy_plain = json.dumps([{"ts": "x", "cue": "legacy", "session_id": "s0"}])
    legacy_payload = bytes([ENVELOPE_VERSION_UNCOMPRESSED]) + legacy_plain.encode("utf-8")
    ad = str(rec.id).lower().encode("ascii")
    legacy_wire = _wrap_payload_as_ciphertext(legacy_payload, store._key(), ad=ad)
    _direct_sql_update(
        store, RECORDS_TABLE, rec.id, "provenance_json", legacy_wire
    )

    store.append_provenance(
        rec.id, {"ts": "y", "cue": "fresh", "session_id": "s1"}
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    new_wire = df[df["id"] == str(rec.id)].iloc[0]["provenance_json"]
    new_payload = _aead_plaintext_of(new_wire, store._key(), ad=ad)
    assert new_payload[0] == ENVELOPE_VERSION_ZSTD, (
        f"natural rewrite via append_provenance must emit compressed "
        f"envelope; got version byte {new_payload[0]:#x}"
    )


# --------------------------------------------------------------------------- #
# Adversarial legacy store round-trip — WR-01 regression guards.              #
#                                                                             #
# These tests seed literal_surface values as true legacy rows (raw utf-8,    #
# no version prefix, the way the old pre-97 writer actually wrote them) and  #
# verify that store.get returns the content byte-identical.  The four cases   #
# are the exact shapes that broke under the naive byte-0 heuristic.           #
# --------------------------------------------------------------------------- #

def _seed_legacy_literal(store, rec, text: str) -> None:
    """Inject a literal_surface value as a true legacy row: AEAD plaintext is
    raw utf-8 with no version prefix, mirroring the old writer."""
    from iai_mcp.store import RECORDS_TABLE

    raw_utf8 = text.encode("utf-8")
    ad = str(rec.id).lower().encode("ascii")
    wire = _wrap_payload_as_ciphertext(raw_utf8, store._key(), ad=ad)
    _direct_sql_update(store, RECORDS_TABLE, rec.id, "literal_surface", wire)


def test_store_legacy_nul_leading_literal_surface_roundtrip(tmp_path):
    """Legacy literal_surface beginning with U+0000 (NUL) must round-trip
    byte-identical.  The old codec mis-routes this as ENVELOPE_VERSION_UNCOMPRESSED
    and silently drops the first character on read."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="placeholder")
    store.insert(rec)

    content = "\x00 nul-leading memory — must survive verbatim"
    _seed_legacy_literal(store, rec, content)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == content, (
        f"NUL-leading legacy literal_surface corrupted: got {got.literal_surface!r}"
    )
    assert got.literal_surface.encode("utf-8") == content.encode("utf-8")


def test_store_legacy_soh_leading_literal_surface_roundtrip(tmp_path):
    """Legacy literal_surface beginning with U+0001 (SOH) must round-trip
    byte-identical.  The old codec mis-routes this as ENVELOPE_VERSION_ZSTD and
    raises ZstdError → HippoIntegrityError, making the record permanently
    unreadable."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="placeholder")
    store.insert(rec)

    content = "\x01 soh-leading memory — must not be fed to zstd"
    _seed_legacy_literal(store, rec, content)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == content, (
        f"SOH-leading legacy literal_surface corrupted: got {got.literal_surface!r}"
    )
    assert got.literal_surface.encode("utf-8") == content.encode("utf-8")


def test_store_legacy_soh_zstd_magic_lookalike_literal_surface_roundtrip(tmp_path):
    """Legacy literal_surface beginning with SOH + zstd-magic bytes must
    round-trip byte-identical.  This is the most adversarial pattern: the
    byte sequence looks like a genuine zstd frame to the naive heuristic, but
    it is raw user text written before the codec existed."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="placeholder")
    store.insert(rec)

    content = "\x01\x28\xb5\x2f\xfd this looks like a zstd header but it is not"
    _seed_legacy_literal(store, rec, content)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == content, (
        f"SOH+magic-lookalike legacy literal_surface corrupted: got {got.literal_surface!r}"
    )


def test_store_new_write_nul_and_soh_leading_roundtrip(tmp_path):
    """New-write path (store.insert) must also round-trip content beginning
    with NUL or SOH byte-identical.  These are written as genuine 0x01+zstd
    envelopes and decoded correctly on the read side."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    nul_rec = _make(text="\x00 nul at start — new write")
    store.insert(nul_rec)
    got_nul = store.get(nul_rec.id)
    assert got_nul is not None
    assert got_nul.literal_surface == "\x00 nul at start — new write"

    soh_rec = _make(text="\x01 soh at start — new write")
    store.insert(soh_rec)
    got_soh = store.get(soh_rec.id)
    assert got_soh is not None
    assert got_soh.literal_surface == "\x01 soh at start — new write"
