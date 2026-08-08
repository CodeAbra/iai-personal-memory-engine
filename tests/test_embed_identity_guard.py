"""One store, one vector space: the embed-identity guard.

Two 384d models produce mutually meaningless vectors, and the native backend
honors environment model overrides the Python layer would otherwise never
see. The store records the identity of the embedder whose vectors it holds;
any embedder whose identity differs is refused, because mixing generations
turns cosine into noise silently.

The identity is written ONLY by a completed re-embed migration. It is never
adopted on first use: an implicit adopt on a store whose vectors are already
foreign would stamp a false attestation the guard could never see past.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


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


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _meta_identity(store):
    from iai_mcp.embed import EMBED_IDENTITY_META_KEY
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (EMBED_IDENTITY_META_KEY,),
        ).fetchone()
    return row["value"] if row is not None else None


def _insert_record(store, tier, text):
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    dim = int(store.embed_dim)
    vec = [0.0] * dim
    vec[0] = 1.0
    rec = MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "test", "session_id": "-",
                     "role": "user"}],
        created_at=now,
        updated_at=now,
        tags=["capture"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=5,
    )
    store.insert(rec)
    return rec.id


def test_unstamped_store_passes_and_is_never_written(tmp_path, monkeypatch):
    from iai_mcp.embed import embedder_for_store

    store = _fresh_store(tmp_path, monkeypatch)
    embedder_for_store(store)
    assert _meta_identity(store) is None
    # A different-model runtime also passes while no stamp exists: legacy
    # tolerance — protection begins at the first stamp, never before.
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "intfloat/multilingual-e5-small")
    embedder_for_store(store)
    assert _meta_identity(store) is None


def test_dry_run_migration_does_not_stamp(tmp_path, monkeypatch):
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    _insert_record(store, "episodic", "a record whose vector is deliberately wrong")
    result = mig.migrate_reembed_from_text(store, dry_run=True)
    assert result["dry_run"] is True
    assert _meta_identity(store) is None


def test_stamped_store_refuses_model_override(tmp_path, monkeypatch):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    _insert_record(store, "episodic", "the guard record")
    mig.migrate_reembed_from_text(store)
    assert _meta_identity(store) is not None
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "intfloat/multilingual-e5-small")
    with pytest.raises(ValueError, match="vector generations"):
        embedder_for_store(store)


def test_stamped_store_refuses_text_prefix_change(tmp_path, monkeypatch):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    _insert_record(store, "episodic", "the guard record")
    mig.migrate_reembed_from_text(store)
    monkeypatch.setenv("IAI_MCP_EMBED_TEXT_PREFIX", "query: ")
    with pytest.raises(ValueError, match="vector generations"):
        embedder_for_store(store)


def test_allow_mismatch_lets_migration_in_and_restamp_moves_identity(
    tmp_path, monkeypatch
):
    from iai_mcp.embed import (
        effective_model_identity,
        embedder_for_store,
        stamp_store_embed_identity,
    )
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    _insert_record(store, "episodic", "the guard record")
    mig.migrate_reembed_from_text(store)
    old = _meta_identity(store)
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "intfloat/multilingual-e5-small")
    embedder = embedder_for_store(store, allow_identity_mismatch=True)
    # obtaining the embedder for a migration must not move the stamp
    assert _meta_identity(store) == old
    stamp_store_embed_identity(store, embedder)
    assert _meta_identity(store) == effective_model_identity(embedder)
    embedder_for_store(store)  # the once-foreign identity is now the store's


def test_reembed_covers_all_tiers_and_stamps(tmp_path, monkeypatch):
    import numpy as np

    from iai_mcp.embed import effective_model_identity, embedder_for_store
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    embedder = embedder_for_store(store)
    ids = {
        tier: _insert_record(store, tier, f"a {tier} record about the vector guard")
        for tier in ("episodic", "semantic", "procedural")
    }

    result = mig.migrate_reembed_from_text(store)
    assert result["reembedded"] >= len(ids)

    by_id = {rec.id: rec for rec in store.all_records()}
    for tier, rid in ids.items():
        rec = by_id[rid]
        got = np.asarray(rec.embedding, dtype=np.float32)
        want = np.asarray(embedder.embed(rec.literal_surface), dtype=np.float32)
        assert float(np.dot(got, want) /
                     (np.linalg.norm(got) * np.linalg.norm(want))) > 0.999, tier

    assert _meta_identity(store) == effective_model_identity(embedder)


def test_zero_write_run_still_rebuilds_index_and_clears_checkpoint(
    tmp_path, monkeypatch
):
    from iai_mcp.migrate import _reembed_from_text as mig

    store = _fresh_store(tmp_path, monkeypatch)
    _insert_record(store, "episodic", "the only record")
    mig.migrate_reembed_from_text(store)

    # Simulate the interrupted-after-final-window resume: cursor points past
    # every row, so the walk writes nothing — the derived structures must
    # still be rebuilt and the checkpoint must not survive.
    mig._save_resume_cursor(
        store, "ffffffff-ffff-ffff-ffff-ffffffffffff",
        {"reembedded": 1, "skipped": 0, "total": 1},
    )
    calls = {"rebuild": 0}
    orig = store.db._rebuild_index_from_sqlite

    def counting_rebuild():
        calls["rebuild"] += 1
        return orig()

    monkeypatch.setattr(store.db, "_rebuild_index_from_sqlite", counting_rebuild)
    result = mig.migrate_reembed_from_text(store, resume=True)
    assert result["reembedded"] == 0
    assert calls["rebuild"] == 1
    import os
    assert not os.path.exists(mig._progress_path(store))
