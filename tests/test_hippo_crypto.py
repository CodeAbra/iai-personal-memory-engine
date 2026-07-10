from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tests._store_raw import open_store_raw

import numpy as np
import pytest

from iai_mcp.hippo import (
    HippoDB,
    HippoDecryptError,
)
from iai_mcp.types import EMBED_DIM


def _rng_unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(
    *,
    rid: str | None = None,
    literal_surface: str = "test surface",
    provenance_json: str | None = None,
    profile_modulation_gain_json: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": rid or str(uuid4()),
        "tier": "episodic",
        "literal_surface": literal_surface,
        "provenance_json": provenance_json or '{"src": "test"}',
        "profile_modulation_gain_json": profile_modulation_gain_json or '{"g": 1.0}',
        "embedding": embedding or _rng_unit_vec(),
        "created_at": now,
    }


def _event_row(*, eid: str | None = None, data_json: str = '{"ev": "test"}') -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": eid or str(uuid4()),
        "kind": "test_event",
        "severity": "info",
        "domain": "storage",
        "ts": now,
        "data_json": data_json,
        "session_id": None,
        "source_ids_json": None,
    }


_RECORDS_SELECT: dict[str, str] = {
    "id":                            "SELECT id FROM records WHERE id = ?",
    "tier":                          "SELECT tier FROM records WHERE id = ?",
    "literal_surface":               "SELECT literal_surface FROM records WHERE id = ?",
    "provenance_json":               "SELECT provenance_json FROM records WHERE id = ?",
    "profile_modulation_gain_json":  "SELECT profile_modulation_gain_json FROM records WHERE id = ?",
    "embedding":                     "SELECT embedding FROM records WHERE id = ?",
    "created_at":                    "SELECT created_at FROM records WHERE id = ?",
}

_EVENTS_SELECT: dict[str, str] = {
    "id":               "SELECT id FROM events WHERE id = ?",
    "kind":             "SELECT kind FROM events WHERE id = ?",
    "severity":         "SELECT severity FROM events WHERE id = ?",
    "domain":           "SELECT domain FROM events WHERE id = ?",
    "ts":               "SELECT ts FROM events WHERE id = ?",
    "data_json":        "SELECT data_json FROM events WHERE id = ?",
    "session_id":       "SELECT session_id FROM events WHERE id = ?",
    "source_ids_json":  "SELECT source_ids_json FROM events WHERE id = ?",
}

_RECORDS_UPDATE: dict[str, str] = {
    "literal_surface":               "UPDATE records SET literal_surface = ? WHERE id = ?",
    "provenance_json":               "UPDATE records SET provenance_json = ? WHERE id = ?",
    "profile_modulation_gain_json":  "UPDATE records SET profile_modulation_gain_json = ? WHERE id = ?",
}

_EVENTS_UPDATE: dict[str, str] = {
    "data_json":        "UPDATE events SET data_json = ? WHERE id = ?",
    "source_ids_json":  "UPDATE events SET source_ids_json = ? WHERE id = ?",
}


def _raw_records_col(db_path: Path, col: str, row_id: str) -> str | None:
    stmt = _RECORDS_SELECT[col]
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(stmt, (row_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _raw_events_col(db_path: Path, col: str, row_id: str) -> str | None:
    stmt = _EVENTS_SELECT[col]
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(stmt, (row_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _raw_records_set(db_path: Path, col: str, row_id: str, value: str) -> None:
    stmt = _RECORDS_UPDATE[col]
    conn = open_store_raw(db_path)
    try:
        conn.execute(stmt, (value, row_id))
        conn.commit()
    finally:
        conn.close()


def _raw_events_set(db_path: Path, col: str, row_id: str, value: str) -> None:
    stmt = _EVENTS_UPDATE[col]
    conn = open_store_raw(db_path)
    try:
        conn.execute(stmt, (value, row_id))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def test_key() -> bytes:
    return os.urandom(32)


@pytest.fixture()
def hippo_with_key(tmp_path: Path, test_key: bytes):
    provider = lambda: test_key  # noqa: E731
    db = HippoDB(tmp_path, crypto_key_provider=provider)
    yield db
    db.close()


@pytest.fixture()
def hippo_no_key(tmp_path: Path):
    db = HippoDB(tmp_path)
    yield db
    db.close()


@pytest.fixture()
def brain_db_path(tmp_path: Path) -> Path:
    return tmp_path / "hippo" / "brain.sqlite3"


def test_records_literal_surface_encrypted_on_disk(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row = _record_row(literal_surface="secret phrase")
    hippo_with_key.open_table("records").add([row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw!r}"

    raw_tier = _raw_records_col(brain_db_path, "tier", row["id"])
    assert raw_tier == "episodic"
    raw_id = _raw_records_col(brain_db_path, "id", row["id"])
    assert raw_id == row["id"]


def test_records_literal_surface_decrypted_on_read(
    hippo_with_key: HippoDB,
) -> None:
    row = _record_row(literal_surface="secret phrase")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])
    df = tbl.to_pandas()
    match = df[df["id"] == row["id"]]
    assert not match.empty
    raw_val = match.iloc[0]["literal_surface"]
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "secret phrase"


def test_provenance_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    payload = '{"source": "unit-test", "confidence": 0.99}'
    row = _record_row(provenance_json=payload)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "provenance_json", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["provenance_json"]
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "provenance_json", result)
    assert plaintext == payload


def test_profile_modulation_gain_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    payload = '{"gain": 2.5, "decay": 0.1}'
    row = _record_row(profile_modulation_gain_json=payload)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "profile_modulation_gain_json", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["profile_modulation_gain_json"]
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "profile_modulation_gain_json", result)
    assert plaintext == payload


def test_events_data_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    payload = '{"event": "login", "user": "alice"}'
    ev = _event_row(data_json=payload)
    tbl = hippo_with_key.open_table("events")
    tbl.add([ev])

    raw = _raw_events_col(brain_db_path, "data_json", ev["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == ev["id"]].iloc[0]["data_json"]
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_event_field(ev["id"], "data_json", result)
    assert plaintext == payload


def test_no_crypto_provider_is_plaintext(hippo_no_key: HippoDB) -> None:
    row = _record_row(literal_surface="plain text")
    tbl = hippo_no_key.open_table("records")
    tbl.add([row])
    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    assert result == "plain text"


def test_idempotent_encrypt_does_not_double_encrypt(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    import base64
    dummy_payload = base64.b64encode(b"\x00" * 44).decode("ascii")
    fake_ct = "iai:enc:v1:" + dummy_payload
    row = _record_row(literal_surface=fake_ct)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw == fake_ct


def test_plaintext_passthrough_on_decrypt(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = open_store_raw(brain_db_path)
    embedding_blob = np.zeros(EMBED_DIM, dtype=np.float32).tobytes()
    try:
        conn.execute(
            "INSERT INTO records (id, tier, literal_surface, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (row_id, "episodic", "raw plaintext value", embedding_blob, now),
        )
        conn.commit()
    finally:
        conn.close()

    df = hippo_with_key.open_table("records").to_pandas()
    result = df[df["id"] == row_id].iloc[0]["literal_surface"]
    assert result == "raw plaintext value"


def test_records_cross_row_aad_swap_raises_decrypt_error(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row_a = _record_row(literal_surface="Record A content")
    row_b = _record_row(literal_surface="Record B content")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row_a])
    tbl.add([row_b])

    ct_a = _raw_records_col(brain_db_path, "literal_surface", row_a["id"])
    assert ct_a is not None and ct_a.startswith("iai:enc:v1:")
    _raw_records_set(brain_db_path, "literal_surface", row_b["id"], ct_a)

    df = tbl.to_pandas()
    with pytest.raises(HippoDecryptError):
        tbl._decrypt_df(df)


def test_records_decrypt_failure_emits_audit_event(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row_a = _record_row(literal_surface="Source ciphertext")
    row_b = _record_row(literal_surface="Target to corrupt")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row_a])
    tbl.add([row_b])

    ct_a = _raw_records_col(brain_db_path, "literal_surface", row_a["id"])
    _raw_records_set(brain_db_path, "literal_surface", row_b["id"], ct_a)

    df = tbl.to_pandas()
    with pytest.raises(HippoDecryptError):
        tbl._decrypt_df(df)

    events_tbl = hippo_with_key.open_table("events")
    events_df_raw = events_tbl.to_pandas()
    events_df = events_tbl._decrypt_df(events_df_raw)
    audit_rows = events_df[events_df["kind"] == "record_decrypt_failed"]
    assert not audit_rows.empty, "Expected at least one record_decrypt_failed event"

    found = False
    for _, ev_row in audit_rows.iterrows():
        try:
            payload = json.loads(ev_row["data_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if payload.get("record_id") == row_b["id"]:
            found = True
            break
    assert found, (
        f"record_decrypt_failed event did not reference record id {row_b['id']!r}; "
        f"events: {audit_rows['data_json'].tolist()!r}"
    )


def test_events_cross_row_aad_swap_returns_fallback(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    ev_a = _event_row(data_json='{"msg": "event_a"}')
    ev_b = _event_row(data_json='{"msg": "event_b"}')
    tbl = hippo_with_key.open_table("events")
    tbl.add([ev_a])
    tbl.add([ev_b])

    ct_a = _raw_events_col(brain_db_path, "data_json", ev_a["id"])
    assert ct_a is not None and ct_a.startswith("iai:enc:v1:")
    _raw_events_set(brain_db_path, "data_json", ev_b["id"], ct_a)

    df = tbl.to_pandas()
    decrypted_df = tbl._decrypt_df(df)
    result_b = decrypted_df[decrypted_df["id"] == ev_b["id"]].iloc[0]["data_json"]
    assert result_b == "{}", f"Expected lenient fallback '{{}}', got {result_b!r}"


def test_search_results_decrypt_literal_surface(
    hippo_with_key: HippoDB,
) -> None:
    texts = ["Alpha memory", "Beta memory", "Gamma memory"]
    embeddings = [_rng_unit_vec(i) for i in range(3)]
    rows = [
        _record_row(literal_surface=t, embedding=e)
        for t, e in zip(texts, embeddings)
    ]
    tbl = hippo_with_key.open_table("records")
    for r in rows:
        tbl.add([r])

    result_df = tbl.search(vector=embeddings[0]).limit(3).to_pandas()
    assert not result_df.empty
    raw_vals = result_df["literal_surface"].tolist()
    for val in raw_vals:
        if isinstance(val, str) and val:
            assert val.startswith("iai:enc:v1:"), f"Expected ciphertext from ANN, got: {val!r}"
    decrypted_df = tbl._decrypt_df(result_df)
    retrieved = set(decrypted_df["literal_surface"].tolist())
    for text in texts:
        if text in retrieved:
            break
    else:
        pytest.fail(f"No expected plaintext found after manual decrypt: {retrieved!r}")


def test_merge_insert_encrypts_records(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row = _record_row(literal_surface="original content")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    updated_row = dict(row)
    updated_row["literal_surface"] = "merge surface content"
    tbl.merge_insert(["id"]).when_matched_update_all().execute([updated_row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"merge_insert did not encrypt; got: {raw!r}"

    df = tbl.to_pandas()
    raw_val = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "merge surface content"


def test_update_id_keyed_with_encrypted_value(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    row = _record_row(literal_surface="original text")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    tbl.update(
        where=f"id = '{row['id']}'",
        values={"literal_surface": "updated text"},
    )

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"Update did not encrypt; got: {raw!r}"

    df = tbl.to_pandas()
    raw_val = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "updated text"


def test_update_non_id_keyed_encrypted_column_raises(
    hippo_with_key: HippoDB,
) -> None:
    row = _record_row(literal_surface="some text")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    with pytest.raises(ValueError, match="id-keyed WHERE"):
        tbl.update(
            where="tier = 'episodic'",
            values={"literal_surface": "danger zone"},
        )


def test_compressed_aead_envelope_survives_store_round_trip_byte_identical(
    hippo_with_key: HippoDB, brain_db_path: Path, test_key: bytes
) -> None:
    """A compressed-encrypted (zstd-in-AEAD) blob must round-trip through the
    store byte-for-byte.

    The store holds the ciphertext as an opaque value; it must never truncate
    or alter the zstd frame. This is the constitutional verbatim/lossless
    invariant for the compressed-AEAD envelope: the exact wire bytes written are
    the exact wire bytes read back, and they still decrypt+decompress to the
    original plaintext. A highly compressible, multi-kilobyte plaintext is used
    so the zstd frame is non-trivial and any dropped/added byte would change the
    decrypted result or fail the AEAD tag.
    """
    from iai_mcp.crypto import decrypt_field, encrypt_field

    # Large, highly compressible plaintext with embedded control characters and
    # multi-byte sequences so a single-byte drift is unambiguous.
    plaintext = ("verbatim-block 🧠 " + "A" * 4096 + "\x00\x01\x02 日本語 ") * 8

    # Build the compressed-AEAD envelope exactly as the store's encrypt path does,
    # keyed by the row uuid (the associated data the store binds per row).
    row_id = str(uuid4())
    aad = row_id.encode("utf-8")
    ciphertext = encrypt_field(plaintext, test_key, associated_data=aad)
    assert ciphertext.startswith("iai:enc:v1:")

    # Store the pre-built ciphertext as the record's literal_surface. The store's
    # own encrypt path is idempotent on an already-encrypted value, so what lands
    # on disk is exactly this ciphertext.
    row = _record_row(rid=row_id, literal_surface=ciphertext)
    hippo_with_key.open_table("records").add([row])

    # Read the raw at-rest value back through the driver-aware raw open.
    raw = _raw_records_col(brain_db_path, "literal_surface", row_id)
    assert raw is not None
    raw_str = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw

    # BYTE-IDENTICAL: the stored ciphertext equals the written ciphertext exactly.
    assert raw_str == ciphertext, (
        "compressed-AEAD envelope was altered by the store round-trip — "
        "the verbatim/lossless invariant is violated"
    )

    # And it still decrypts + decompresses back to the exact original plaintext.
    recovered = decrypt_field(raw_str, test_key, associated_data=aad)
    assert recovered == plaintext
