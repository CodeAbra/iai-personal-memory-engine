from __future__ import annotations

import json
from iai_mcp import errors
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest


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
            provenance=[
                {
                    "ts": "2026-04-17T00:00:00+00:00",
                    "cue": f"seed-{i}",
                    "session_id": "seed",
                }
            ],
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


def test_reembed_upgrades_dim_and_preserves_all_non_embedding_fields(
    tmp_path, monkeypatch
):
    src_embedder = _DimEmbedder(384)
    target_embedder = _DimEmbedder(1024)

    store = _fresh_store(tmp_path, 384, monkeypatch)
    assert store.embed_dim == 384
    seeded_ids = _seed_records(store, src_embedder, n=3)
    pre = {rid: store.get(rid) for rid in seeded_ids}
    with store.db.ro_conn() as conn:
        stored_before = dict(
            conn.execute(
                "SELECT * FROM records WHERE id = ?",
                (str(seeded_ids[0]),),
            ).fetchone()
        )
    encrypted_before = stored_before["literal_surface"]
    from iai_mcp.crypto import is_encrypted

    assert is_encrypted(encrypted_before)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, target_embedder)
    assert result["target_dim"] == 1024
    assert result["source_dim"] == 384
    assert result["updated"] == 3

    assert store.embed_dim == 1024
    with store.db.ro_conn() as conn:
        stored_after = dict(
            conn.execute(
                "SELECT * FROM records WHERE id = ?",
                (str(seeded_ids[0]),),
            ).fetchone()
        )
    encrypted_after = stored_after["literal_surface"]
    assert encrypted_after == encrypted_before, (
        "migration must preserve encrypted storage bytes exactly"
    )
    assert {
        key: value for key, value in stored_after.items() if key != "embedding"
    } == {key: value for key, value in stored_before.items() if key != "embedding"}

    for rid in seeded_ids:
        post = store.get(rid)
        assert post is not None
        assert post.literal_surface == pre[rid].literal_surface, (
            "literal_surface byte-identical"
        )
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


def test_reembed_table_swap_and_dimension_metadata_are_atomic(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, _DimEmbedder(384), n=2)

    # Both storage-driver connection wrappers reject instance-attribute
    # assignment on `execute` (delegate to a read-only-attribute driver
    # object); patch the wrapper CLASS instead, which class-level setattr
    # reaches directly, and delegate every non-matching statement to the
    # original bound method unchanged.
    conn_cls = type(store.db._conn)
    _orig_execute = conn_cls.execute

    def _flatten(values):
        for v in values:
            if isinstance(v, (tuple, list)):
                yield from _flatten(v)
            else:
                yield v

    def _raise_on_meta_update(self, sql, *args, **kwargs):
        bound = list(_flatten(args)) + list(_flatten(kwargs.values()))
        if (
            isinstance(sql, str)
            and sql.lstrip().upper().startswith("UPDATE")
            and "_hippo_meta" in sql
            and ("embed_dim" in sql or "embed_dim" in bound)
        ):
            raise errors.IntegrityError("simulated metadata failure")
        return _orig_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(conn_cls, "execute", _raise_on_meta_update)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    with pytest.raises(errors.IntegrityError, match="simulated metadata failure"):
        migrate_reembed_to_current_dim(store, _DimEmbedder(1024))

    assert store.db.open_table("records").count_rows() == 2
    assert store.db.open_table("records_v_new").count_rows() == 2
    with store.db.ro_conn() as conn:
        stored_dim = conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = 'embed_dim'"
        ).fetchone()[0]
    assert stored_dim == "384"
    assert store.embed_dim == 384


def test_reopen_rebuilds_stale_hnsw_after_dimension_migration(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    target = _DimEmbedder(1024)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    seeded = _seed_records(store, source, n=3)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    migrate_reembed_to_current_dim(store, target)
    store.db.close()  # persists the still-live pre-restart 384d HNSW buffer

    from iai_mcp.store import MemoryStore

    reopened = MemoryStore()
    try:
        hits = reopened.query_similar(
            target.embed(
                "Record #0 with literal surface content that must survive migration."
            ),
            k=3,
        )
        assert seeded[0] in {record.id for record, _score in hits}
        with reopened.db.ro_conn() as conn:
            row = conn.execute(
                "SELECT vec_label, embedding FROM records WHERE id = ?",
                (str(seeded[0]),),
            ).fetchone()
        stored = np.frombuffer(row["embedding"], dtype=np.float32)
        indexed = reopened.db._hnsw.get_items([int(row["vec_label"])])[0]
        assert indexed.shape == (1024,)
        assert np.allclose(indexed, stored, rtol=1e-4, atol=1e-5)
    finally:
        reopened.db.close()


def test_reembed_uses_real_http_provider_batch_path(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, _DimEmbedder(384), n=3)

    class Response:
        def __init__(self, body: bytes):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            return self.body

    requests: list[dict] = []

    def fake_urlopen(request, *, timeout):
        assert timeout == 30.0
        payload = json.loads(request.data)
        requests.append(payload)
        vectors = [[1.0, 0.0, 0.0] for _ in payload["texts"]]
        return Response(
            json.dumps(
                {
                    "model": "test-http-model",
                    "dimensions": 3,
                    "vectors": vectors,
                }
            ).encode()
        )

    monkeypatch.setenv("IAI_MCP_EMBED_PROVIDER", "http")
    monkeypatch.setenv("IAI_MCP_EMBED_URL", "http://127.0.0.1:4488/embed")
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "3")
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "test-http-model")
    monkeypatch.setattr("iai_mcp.embed.urlopen", fake_urlopen)

    from iai_mcp.embed import Embedder
    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, Embedder())

    assert result["updated"] == 3
    assert store.embed_dim == 3
    assert len(requests) == 1
    assert requests[0]["input_type"] == "document"
    assert len(requests[0]["texts"]) == 3


def test_reembed_flushes_records_buffered_in_same_process(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    seeded_ids = _seed_records(store, source, n=3)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, _DimEmbedder(1024))

    assert result["updated"] == 3
    assert {record.id for record in store.all_records()} == set(seeded_ids)


def test_reembed_idempotent_same_dim_no_op(tmp_path, monkeypatch):
    src = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, src, n=2)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, _DimEmbedder(384))
    assert result["updated"] == 0
    assert result["skipped"] == 2 or result.get("no_op") is True
    assert store.embed_dim == 384


def test_reembed_can_force_model_change_at_same_dimension(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    seeded = _seed_records(store, source, n=2)
    before = {rid: store.get(rid).embedding for rid in seeded}

    class Replacement(_DimEmbedder):
        model_key = "test-model"

        def embed(self, text: str) -> list[float]:
            vector = super().embed(text)
            return list(reversed(vector))

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, Replacement(384), force=True)

    assert result["updated"] == 2
    assert all(store.get(rid).embedding != before[rid] for rid in seeded)


def test_reembed_batches_provider_calls(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, source, n=10)

    class BatchSpy(_DimEmbedder):
        def __init__(self, dim: int):
            super().__init__(dim)
            self.batch_sizes: list[int] = []
            self.supports_batch = True

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_sizes.append(len(texts))
            return super().embed_batch(texts)

    target = BatchSpy(1024)
    from iai_mcp.migrate import migrate_reembed_to_current_dim

    result = migrate_reembed_to_current_dim(store, target, batch_size=4)

    assert result["updated"] == 10
    assert target.batch_sizes == [4, 4, 2]


def test_reembed_checkpoints_once_per_provider_batch(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, source, n=10)

    class BatchEmbedder(_DimEmbedder):
        supports_batch = True

    from iai_mcp.migrate import _reembed

    writes: list[dict] = []
    real_write = _reembed._progress_write

    def spy_write(store, state):
        writes.append(state)
        real_write(store, state)

    monkeypatch.setattr(_reembed, "_progress_write", spy_write)
    result = _reembed.migrate_reembed_to_current_dim(
        store, BatchEmbedder(1024), batch_size=4
    )

    assert result["updated"] == 10
    assert len(writes) == 3
    assert [state["row_index"] for state in writes] == [3, 7, 9]
    assert all("staged_ids" not in state for state in writes)


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


def test_reembed_refuses_swap_when_any_record_fails(tmp_path, monkeypatch):
    source = _DimEmbedder(384)
    store = _fresh_store(tmp_path, 384, monkeypatch)
    _seed_records(store, source, n=10)

    class OneFailure(_DimEmbedder):
        def embed(self, text: str) -> list[float]:
            if "#4 " in text:
                raise ValueError("simulated record failure")
            return super().embed(text)

    from iai_mcp.migrate import migrate_reembed_to_current_dim

    with pytest.raises(RuntimeError, match="refusing to swap"):
        migrate_reembed_to_current_dim(store, OneFailure(1024))

    assert store.db.open_table("records").count_rows() == 10
    assert store.db.open_table("records_v_new").count_rows() == 9
    assert store.embed_dim == 384


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
