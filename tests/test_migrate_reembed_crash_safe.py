from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

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


class _DimEmbedder:

    def __init__(self, dim: int):
        self.DIM = dim
        self.model_key = f"fake-dim-{dim}"

    def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.DIM
        for i, ch in enumerate(text or ""):
            vec[i % self.DIM] += ord(ch) / 256.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _fresh_store(tmp_path, dim: int, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(dim))
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_records(store, embedder, n: int = 10) -> list[UUID]:
    from iai_mcp.types import MemoryRecord
    ids = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        rid = uuid4()
        text = f"Crash-safe seed record #{i:02d} with literal surface content."
        rec = MemoryRecord(
            id=rid,
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=embedder.embed(text),
            structure_hv=b"",
            community_id="",
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.5,
            difficulty=0.3,
            last_reviewed=now,
            never_decay=False,
            never_merge=False,
            provenance=[
                {"ts": "2026-04-30T00:00:00+00:00", "cue": f"seed-{i}", "session_id": "seed"}
            ],
            created_at=now,
            updated_at=now,
            tags=["test", "crash-safe"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=4,
        )
        store.insert(rec)
        ids.append(rid)
    return ids


def test_successful_migration_promotes_old_to_records(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    target = _DimEmbedder(1024)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=20)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    result = migrate_reembed_to_current_dim(store, target)
    assert result["target_dim"] == 1024
    assert result["source_dim"] == 384

    names = set(store.db.table_names())
    assert "records" in names, "records table must exist after swap"
    assert "records_v_new" not in names, (
        "records_v_new must be cleaned after atomic swap"
    )
    old_tables = [n for n in names if n.startswith("records_old_")]
    assert len(old_tables) == 1, (
        f"exactly one records_old_<ts> expected (deferred cleanup); got {old_tables}"
    )

    assert store.db.open_table("records").count_rows() >= 19


def test_migration_preserves_structural_metadata_and_pending_flag(
    tmp_path,
    monkeypatch,
):
    from iai_mcp.types import MemoryRecord, STRUCTURE_HV_BYTES

    src = _DimEmbedder(384)
    target = _DimEmbedder(1024)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    rid = uuid4()
    now = datetime.now(timezone.utc)
    structure_hv = b"\x02" * STRUCTURE_HV_BYTES
    structure_payload = b'{"source":"migration-test"}'
    rec = MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface="Pending structured record survives reembed migration.",
        aaak_index="",
        embedding=src.embed("Pending structured record survives reembed migration."),
        structure_hv=structure_hv,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["test"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=4,
        hv_tier="sparse_vsa",
        structure_hv_payload=structure_payload,
        embedding_pending=1,
    )
    store.insert(rec)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    migrate_reembed_to_current_dim(store, target)

    migrated = store.get(rid)
    assert migrated is not None
    assert len(migrated.embedding) == target.DIM
    assert migrated.structure_hv == structure_hv
    assert migrated.hv_tier == "sparse_vsa"
    assert migrated.structure_hv_payload == structure_payload
    assert migrated.embedding_pending == 1


def test_mid_migration_kill_preserves_old_table(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt("simulated mid-migration kill")
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    names = set(store.db.table_names())
    assert "records" in names
    assert store.db.open_table("records").count_rows() == 10, (
        "Original records table must stay intact when kill fires mid-stage"
    )
    assert "records_v_new" in names, (
        "records_v_new must exist with the partial set after kill"
    )
    assert store.db.open_table("records_v_new").count_rows() == 3, (
        "records_v_new must hold the 3 successfully-staged rows"
    )
    progress_path = Path(store.root) / "migration_progress.json"
    assert progress_path.exists(), (
        "migration_progress.json must be written on each successful row"
    )


def test_rollback_handler_restores_from_old(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt()
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim, _rollback
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    rc = _rollback(store.db, store)
    assert rc == 0, "rollback must succeed on a clean kill-mid-stage state"

    names = set(store.db.table_names())
    assert "records" in names, "records must still exist (it was never dropped)"
    assert store.db.open_table("records").count_rows() == 10, (
        "records must hold the original 10 rows after rollback"
    )
    assert "records_v_new" not in names, "records_v_new must be dropped by rollback"
    assert not any(n.startswith("records_old_") for n in names), (
        "no records_old_<ts> in this scenario (kill fired before swap)"
    )
    progress_path = Path(store.root) / "migration_progress.json"
    assert not progress_path.exists(), "rollback must drop the progress file"


def test_resume_handler_continues_from_checkpoint(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=10)

    target = _DimEmbedder(1024)
    call_count = {"n": 0}
    real_embed = target.embed

    def embed_or_kill(text):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise KeyboardInterrupt()
        return real_embed(text)

    monkeypatch.setattr(target, "embed", embed_or_kill)

    from iai_mcp.migrate import migrate_reembed_to_current_dim, _resume
    with pytest.raises(KeyboardInterrupt):
        migrate_reembed_to_current_dim(store, target)

    monkeypatch.setattr(target, "embed", real_embed)
    rc = _resume(store.db, store, target)
    assert rc == 0, "resume must succeed on a recoverable partial state"

    assert store.db.open_table("records").count_rows() == 10, (
        "all 10 rows present after resume + atomic swap"
    )
    assert "records_v_new" not in set(store.db.table_names()), (
        "records_v_new cleaned after the swap completes"
    )
    progress_path = Path(store.root) / "migration_progress.json"
    assert not progress_path.exists(), "resume must drop the progress file on success"


def test_idempotency_rerun_after_success(tmp_path, monkeypatch):
    from iai_mcp.events import query_events
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=5)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    migrate_reembed_to_current_dim(store, _DimEmbedder(1024))
    migrate_reembed_to_current_dim(store, _DimEmbedder(1024))

    events = query_events(store, kind="migration_reembed", limit=5)
    assert len(events) >= 2, (
        f"both runs must emit a migration_reembed event; got {len(events)}"
    )
    no_op_events = [e for e in events if e["data"].get("no_op") is True]
    assert len(no_op_events) >= 1, (
        "second run at same dim must emit a migration_reembed event with no_op=True"
    )
