from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4



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


def _seed_records(store, embedder, n: int = 3) -> list[UUID]:
    from iai_mcp.types import MemoryRecord
    ids = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        rid = uuid4()
        text = f"Record #{i} with literal surface content that must survive migration."
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
            provenance=[{"ts": "2026-04-17T00:00:00+00:00", "cue": f"seed-{i}", "session_id": "seed"}],
            created_at=now,
            updated_at=now,
            tags=["test", "migration"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=4,
        )
        store.insert(rec)
        ids.append(rid)
    return ids


def test_reembed_upgrades_dim_and_preserves_all_non_embedding_fields(tmp_path, monkeypatch):
    src_embedder = _DimEmbedder(384)
    target_embedder = _DimEmbedder(1024)

    store = _fresh_store(tmp_path, 384, monkeypatch)
    assert store.embed_dim == 384
    seeded_ids = _seed_records(store, src_embedder, n=3)
    pre = {rid: store.get(rid) for rid in seeded_ids}

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    result = migrate_reembed_to_current_dim(store, target_embedder)
    assert result["target_dim"] == 1024
    assert result["source_dim"] == 384
    assert result["updated"] == 3

    assert store.embed_dim == 1024

    for rid in seeded_ids:
        post = store.get(rid)
        assert post is not None
        assert post.literal_surface == pre[rid].literal_surface, "literal_surface byte-identical"
        assert post.tier == pre[rid].tier
        assert post.tags == pre[rid].tags
        assert post.language == pre[rid].language
        assert post.schema_version == pre[rid].schema_version
        assert post.s5_trust_score == pre[rid].s5_trust_score
        assert post.pinned == pre[rid].pinned
        assert post.detail_level == pre[rid].detail_level
        assert post.never_decay == pre[rid].never_decay
        assert post.never_merge == pre[rid].never_merge
        assert post.provenance == pre[rid].provenance
        assert len(post.embedding) == 1024, "new embedding must have target dim"
        assert len(pre[rid].embedding) == 384, "old embedding was at source dim"
        assert post.embedding != pre[rid].embedding, "embedding must be re-computed"


def test_reembed_idempotent_same_dim_no_op(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=2)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    result = migrate_reembed_to_current_dim(store, _DimEmbedder(384))
    assert result["updated"] == 0
    assert result["skipped"] == 2 or result.get("no_op") is True
    assert store.embed_dim == 384


def test_reembed_dry_run_reports_without_mutating(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    seeded = _seed_records(store, src, n=2)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    result = migrate_reembed_to_current_dim(store, _DimEmbedder(1024), dry_run=True)
    assert result["would_update"] == 2
    assert store.embed_dim == 384
    post = store.get(seeded[0])
    assert len(post.embedding) == 384


def test_reembed_emits_migration_event(tmp_path, monkeypatch):
    from iai_mcp.events import query_events
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=1)

    from iai_mcp.migrate import migrate_reembed_to_current_dim
    migrate_reembed_to_current_dim(store, _DimEmbedder(1024))

    events = query_events(store, kind="migration_reembed", limit=5)
    assert len(events) >= 1
    data = events[0]["data"]
    assert data.get("source_dim") == 384
    assert data.get("target_dim") == 1024
    assert data.get("updated") == 1
