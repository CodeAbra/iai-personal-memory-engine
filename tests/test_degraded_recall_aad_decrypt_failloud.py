"""Degraded recall (daemon-down rail): AAD parity + fail-loud on decrypt failure.

The daemon-down degraded recall rail must derive the AES AAD identically to the
write side (lowercased canonical id) so an encrypted literal_surface decrypts
byte-exact. On a decrypt failure (corrupt/undecryptable ciphertext), the rail
must SKIP the row — never return raw ciphertext as if it were user-facing
content (constitutional: decrypt failure is fail-loud, never silent ciphertext).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from tests._store_raw import open_store_raw


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")


def _make_record(literal_surface: str, seed: int = 7):
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    rng = np.random.RandomState(seed=seed)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=rng.randn(EMBED_DIM).tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "s", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


def _seed(store_root: Path, surface: str, seed: int = 7) -> str:
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(store_root)
    try:
        rec = _make_record(surface, seed=seed)
        store.insert(rec)
        flush_record_buffer(store)
        return str(rec.id)
    finally:
        store.close()


def test_degraded_recall_decrypts_byte_exact(hermetic_store: Path):
    from iai_mcp.hippo._recall import degraded_semantic_recall

    surface = "雪 degraded-rail verbatim 🧠 {\"k\": 1} byte-exact probe"
    _seed(hermetic_store, surface, seed=11)

    # Confirm the row is encrypted at rest (AAD-bound to the lowercase id).
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(
            "SELECT id, literal_surface FROM records LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    stored_surface = row[1]
    assert isinstance(stored_surface, str) and stored_surface.startswith("iai:enc:v1:"), (
        "seeded row must be encrypted at rest"
    )

    hits = degraded_semantic_recall(hermetic_store, cue="probe", limit=10)
    surfaces = [h.get("literal_surface", "") for h in hits]
    assert surface in surfaces, (
        "degraded rail must decrypt the AAD-bound literal_surface byte-exact; "
        f"got {surfaces!r}"
    )
    # Ciphertext must never leak through as the user-facing surface.
    assert not any(s.startswith("iai:enc:v1:") for s in surfaces), (
        "degraded rail returned raw ciphertext as a surface"
    )


def test_degraded_recall_skips_undecryptable_row_never_surfaces_ciphertext(
    hermetic_store: Path,
):
    from iai_mcp.hippo._recall import degraded_semantic_recall

    good_surface = "good decryptable degraded probe row"
    _seed(hermetic_store, good_surface, seed=12)

    # Read the at-rest ciphertext for the seeded row, then corrupt its payload
    # so AEAD validation fails (InvalidTag) on the degraded read.
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = open_store_raw(db_path)
    try:
        rid, ct = conn.execute(
            "SELECT id, literal_surface FROM records LIMIT 1"
        ).fetchone()
        assert ct.startswith("iai:enc:v1:"), "row must be encrypted to corrupt it"
        # Flip characters in the base64 body to break the AEAD tag while keeping
        # the prefix intact, so is_encrypted() still routes it into decrypt.
        prefix = "iai:enc:v1:"
        body = ct[len(prefix):]
        corrupted_body = ("A" if body[0] != "A" else "B") + body[1:-1] + (
            "A" if body[-1] != "A" else "B"
        )
        corrupted = prefix + corrupted_body
        conn.execute(
            "UPDATE records SET literal_surface = ? WHERE id = ?",
            (corrupted, rid),
        )
        conn.commit()
    finally:
        conn.close()

    hits = degraded_semantic_recall(hermetic_store, cue="probe", limit=10)
    surfaces = [h.get("literal_surface", "") for h in hits]

    # Fail-loud: the undecryptable row is skipped, NOT surfaced as ciphertext.
    assert corrupted not in surfaces, (
        "degraded rail surfaced corrupted ciphertext as content (must skip)"
    )
    assert not any(
        s.startswith("iai:enc:v1:") for s in surfaces
    ), f"degraded rail leaked raw ciphertext: {surfaces!r}"
    assert good_surface not in surfaces, (
        "sanity: the only seeded row was corrupted, so it must not decrypt"
    )


def test_degraded_temporal_recall_decrypts_byte_exact(hermetic_store: Path):
    from iai_mcp.hippo._recall import degraded_temporal_recall

    surface = "temporal degraded verbatim probe — byte exact"
    _seed(hermetic_store, surface, seed=13)

    as_of = datetime.now(timezone.utc).isoformat()
    hits = degraded_temporal_recall(hermetic_store, cue="probe", as_of=as_of, limit=10)
    surfaces = [h.get("literal_surface", "") for h in hits]
    assert surface in surfaces, (
        f"temporal degraded rail must decrypt byte-exact; got {surfaces!r}"
    )
    assert not any(s.startswith("iai:enc:v1:") for s in surfaces), (
        "temporal degraded rail leaked raw ciphertext"
    )
