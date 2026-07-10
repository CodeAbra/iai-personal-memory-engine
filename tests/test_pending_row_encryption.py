"""At-rest confidentiality of deferred-embed (pending) rows.

A pending row is written before its embedding exists, on a fast path that does
not go through the normal record-insert encryption. On an encrypted store the
confidential record columns (``literal_surface``, ``provenance_json``) must be
ciphertext at rest exactly like a fully-embedded row, and the later re-embed
must operate on the recovered plaintext so the vector is of the message, never
of the ciphertext. On an unencrypted store the same path stores plaintext and
embeds it directly, unchanged.

These tests are hermetic: a tmp store path, and the encrypted case supplies its
own key provider directly (no daemon, no real home store, no socket).
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tests._store_raw import open_store_raw

import pytest

from iai_mcp.crypto import is_encrypted
from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


_LITERAL = "User secret pending phrase that must never sit in plaintext"
_PROVENANCE = '[{"src": "unit-test", "role": "user"}]'


class _DeterministicEmbedder:
    """Maps text to a stable unit vector by content.

    Distinct texts produce near-orthogonal vectors, so ``embed(plaintext)`` and
    ``embed(ciphertext)`` are easily distinguished by cosine. Identity is exact
    for identical input, which is what the plaintext-vector proof relies on.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        import numpy as np

        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _brain_db_path(root: Path) -> Path:
    return root / "hippo" / "brain.sqlite3"


def _raw_col(db_path: Path, col: str, row_id: str) -> str | None:
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(
            f"SELECT {col} FROM records WHERE id = ?", (row_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _insert_pending(db: HippoDB, record_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    db.insert_pending_row(
        record_id=record_id,
        tier="episodic",
        literal_surface=_LITERAL,
        tags_json="[]",
        provenance_json=_PROVENANCE,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture()
def test_key() -> bytes:
    import os

    return os.urandom(32)


@pytest.fixture()
def encrypted_db(tmp_path: Path, test_key: bytes):
    db = HippoDB(tmp_path, crypto_key_provider=lambda: test_key)
    yield db
    db.close()


@pytest.fixture()
def plaintext_db(tmp_path: Path):
    # No key provider: the store is unencrypted; the pending fast path must be
    # unchanged (plaintext at rest, correct vector).
    db = HippoDB(tmp_path)
    yield db
    db.close()


def test_encrypted_store_pending_row_columns_encrypted_at_rest(
    encrypted_db: HippoDB, tmp_path: Path
) -> None:
    """(a) The at-rest leak is closed: confidential columns are ciphertext."""
    record_id = str(uuid4())
    _insert_pending(encrypted_db, record_id)

    db_path = _brain_db_path(tmp_path)
    raw_literal = _raw_col(db_path, "literal_surface", record_id)
    raw_prov = _raw_col(db_path, "provenance_json", record_id)

    assert raw_literal is not None
    assert is_encrypted(raw_literal), (
        f"literal_surface stored as plaintext on an encrypted store: {raw_literal!r}"
    )
    assert raw_literal != _LITERAL
    assert raw_prov is not None
    assert is_encrypted(raw_prov), (
        f"provenance_json stored as plaintext on an encrypted store: {raw_prov!r}"
    )
    assert raw_prov != _PROVENANCE

    # The encrypted column still round-trips to the original plaintext.
    assert (
        encrypted_db._decrypt_record_field(record_id, "literal_surface", raw_literal)
        == _LITERAL
    )
    assert (
        encrypted_db._decrypt_record_field(record_id, "provenance_json", raw_prov)
        == _PROVENANCE
    )


def test_encrypted_store_reembed_uses_plaintext_not_ciphertext(
    encrypted_db: HippoDB, tmp_path: Path
) -> None:
    """(b) End-to-end: the wake re-embed vector is of the plaintext, not the
    ciphertext, and the pending flag clears."""
    record_id = str(uuid4())
    _insert_pending(encrypted_db, record_id)

    embedder = _DeterministicEmbedder()
    result = encrypted_db.pending_embeddings_wake_sequence(embedder=embedder)
    assert result["action"] == "wake_sequence"
    assert result["reembed_count"] == 1

    db_path = _brain_db_path(tmp_path)
    # Read the stored embedding blob and the cleared pending flag directly.
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(
            "SELECT embedding, embedding_pending FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    blob, pending = row[0], row[1]
    assert pending == 0, "embedding_pending must flip to 0 after the wake sequence"

    import struct

    n = len(blob) // 4
    stored_vec = list(struct.unpack(f"<{n}f", blob))

    raw_literal = _raw_col(db_path, "literal_surface", record_id)
    assert raw_literal is not None and is_encrypted(raw_literal)

    plaintext_vec = embedder.embed(_LITERAL)
    ciphertext_vec = embedder.embed(raw_literal)

    cos_plain = _cosine(stored_vec, plaintext_vec)
    cos_cipher = _cosine(stored_vec, ciphertext_vec)

    assert cos_plain > 0.9999, (
        f"stored vector is not the plaintext embedding (cos={cos_plain:.6f})"
    )
    assert cos_cipher < 0.5, (
        "stored vector matches the ciphertext embedding — re-embed read raw "
        f"ciphertext instead of decrypting (cos={cos_cipher:.6f})"
    )


def test_unencrypted_store_pending_row_plaintext_and_correct_vector(
    plaintext_db: HippoDB, tmp_path: Path
) -> None:
    """(c) No key provider: plaintext at rest and a correct vector, unchanged."""
    record_id = str(uuid4())
    _insert_pending(plaintext_db, record_id)

    db_path = _brain_db_path(tmp_path)
    raw_literal = _raw_col(db_path, "literal_surface", record_id)
    raw_prov = _raw_col(db_path, "provenance_json", record_id)
    assert raw_literal == _LITERAL, "unencrypted store must keep plaintext at rest"
    assert not is_encrypted(raw_literal)
    assert raw_prov == _PROVENANCE
    assert not is_encrypted(raw_prov)

    embedder = _DeterministicEmbedder()
    result = plaintext_db.pending_embeddings_wake_sequence(embedder=embedder)
    assert result["action"] == "wake_sequence"
    assert result["reembed_count"] == 1

    conn = open_store_raw(db_path)
    try:
        row = conn.execute(
            "SELECT embedding, embedding_pending FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    blob, pending = row[0], row[1]
    assert pending == 0

    import struct

    n = len(blob) // 4
    stored_vec = list(struct.unpack(f"<{n}f", blob))
    plaintext_vec = embedder.embed(_LITERAL)
    assert _cosine(stored_vec, plaintext_vec) > 0.9999
