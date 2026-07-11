"""At-rest encryption guard for the daemon-down direct-write path.

The direct write (`iai capture` while no daemon is up) must encrypt
`literal_surface` and `provenance_json` at rest exactly like the normal
MemoryStore path, so verbatim user content is never stored as plaintext in
brain.sqlite3. Covers BOTH write branches: the deferred-embed branch
(`insert_pending_row`) and the eager-embed branch (`_insert_row_with_embedding`).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests._store_raw import open_store_raw


_SECRET = "top-secret direct-write phrase: alice's passphrase Привет 42"


def _raw_row_for(db_path: Path, fragment: str):
    conn = open_store_raw(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT id, literal_surface, provenance_json FROM records "
            "WHERE literal_surface LIKE ? OR literal_surface IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (f"%{fragment}%",),
        ).fetchone()
    finally:
        conn.close()


def test_direct_write_deferred_encrypts_at_rest(hermetic_store: Path) -> None:
    """Deferred-embed branch (insert_pending_row) stores ciphertext at rest."""
    from iai_mcp.direct_write import write_turn_direct

    result = write_turn_direct(
        store_root=hermetic_store,
        text=_SECRET,
        session_id="enc-session",
        role="user",
        deferred_embedding=True,
    )
    assert result["status"] == "inserted", result

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    row = _raw_row_for(db_path, "top-secret direct-write")
    assert row is not None, "direct-written row not found at rest"

    # At-rest content MUST be ciphertext, not the verbatim plaintext.
    assert row["literal_surface"].startswith("iai:enc:v1:"), (
        "literal_surface stored as PLAINTEXT at rest — AES-256-GCM bypassed"
    )
    assert row["provenance_json"].startswith("iai:enc:v1:"), (
        "provenance_json stored as PLAINTEXT at rest — AES-256-GCM bypassed"
    )
    assert _SECRET not in (row["literal_surface"] or ""), (
        "plaintext secret leaked in literal_surface at rest"
    )


def test_direct_write_eager_encrypts_at_rest(hermetic_store: Path, monkeypatch) -> None:
    """Eager-embed branch (_insert_row_with_embedding) stores ciphertext too."""
    import iai_mcp.direct_write as dw
    from iai_mcp.types import EMBED_DIM

    # Force the eager path: hand write_turn_direct a real embedding so it routes
    # through _insert_row_with_embedding instead of insert_pending_row.
    monkeypatch.setattr(
        dw, "_try_get_embedding_fast",
        lambda text, cue: [0.01] * EMBED_DIM,
    )

    result = dw.write_turn_direct(
        store_root=hermetic_store,
        text=_SECRET,
        session_id="enc-eager-session",
        role="user",
        deferred_embedding=False,
    )
    assert result["status"] == "inserted", result

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = open_store_raw(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT literal_surface, provenance_json, embedding_pending "
            "FROM records ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "eager-embedded row not found at rest"
    assert row["embedding_pending"] == 0, "eager path must not defer embedding"
    assert row["literal_surface"].startswith("iai:enc:v1:"), (
        "eager-path literal_surface stored as PLAINTEXT — AES-256-GCM bypassed"
    )
    assert row["provenance_json"].startswith("iai:enc:v1:"), (
        "eager-path provenance_json stored as PLAINTEXT — AES-256-GCM bypassed"
    )
    assert _SECRET not in (row["literal_surface"] or "")


def test_direct_write_decrypts_back_byte_exact(hermetic_store: Path) -> None:
    """A normal encrypted read decrypts the at-rest ciphertext back verbatim."""
    from iai_mcp.direct_write import write_turn_direct
    from iai_mcp.direct_recency import read_recent_user_turns_direct

    write_turn_direct(
        store_root=hermetic_store,
        text=_SECRET,
        session_id="enc-roundtrip-session",
        role="user",
        deferred_embedding=True,
    )

    turns = read_recent_user_turns_direct(hermetic_store, n=10)
    surfaces = [t.literal_surface for t in turns]
    assert any(s == _SECRET for s in surfaces), (
        "verbatim content not recovered byte-exact after encrypted round-trip; "
        f"got surfaces={surfaces!r}"
    )
