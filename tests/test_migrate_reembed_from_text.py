"""Hermetic tests for the re-embed-from-text migration.

The capture path embedded the provenance cue label instead of the message
content, collapsing the stored vector space. This migration rebuilds each
episodic record's embedding from its intact ``literal_surface`` text and
rebuilds the recall index.

These tests build a throwaway store, seed records whose stored embedding is a
deliberate label-embedding (reproducing the defect), run the migration, and
prove: vectors match ``embed(literal_surface)``, the random-pair cosine drops
materially, idempotency holds, and ``literal_surface`` is never modified.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.write import cosine


# Distinct, semantically-unrelated contents so correct embeddings spread out
# in vector space and the collapsed-vs-corrected cosine gap is unambiguous.
_CONTENTS = [
    "The migration must rebuild embeddings from the verbatim message text.",
    "Sourdough fermentation depends on wild yeast and ambient temperature.",
    "Orbital mechanics govern how satellites maintain a stable trajectory.",
    "The cellist tuned her instrument before the evening rehearsal began.",
    "Compiler optimizations can reorder instructions across basic blocks.",
    "Coral reefs bleach when ocean temperatures stay elevated for too long.",
]

# The single positional cue label every record was wrongly embedded from.
_CUE_LABEL = "session sess-test turn 0"


@pytest.fixture
def reembed_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-reembed-from-text-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_broken(store, embedder) -> list[UUID]:
    """Insert episodic records whose stored embedding is the cue-label vector.

    literal_surface is the real distinct content; embedding is deliberately
    embed(cue_label) for every record -- exactly the collapsed state the bug
    produced. Returns the inserted ids in content order.
    """
    from iai_mcp.types import MemoryRecord

    label_vec = list(embedder.embed(_CUE_LABEL))
    now = datetime.now(timezone.utc)
    ids: list[UUID] = []
    for i, content in enumerate(_CONTENTS):
        rid = uuid4()
        rec = MemoryRecord(
            id=rid,
            tier="episodic",
            literal_surface=content,
            aaak_index="",
            embedding=label_vec,
            structure_hv=b"",
            community_id="",
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{
                "ts": now.isoformat(),
                "cue": _CUE_LABEL,
                "session_id": "sess-test",
                "role": "user",
            }],
            created_at=now,
            updated_at=now,
            tags=["capture", "role:user"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=5,
        )
        store.insert(rec)
        ids.append(rid)
    return ids


def _stored_embedding(store, rid: UUID) -> list[float]:
    rec = store.get(rid)
    assert rec is not None
    return list(rec.embedding)


def _mean_pairwise_cosine(vecs: list[list[float]]) -> float:
    pairs = list(itertools.combinations(range(len(vecs)), 2))
    return sum(cosine(vecs[i], vecs[j]) for i, j in pairs) / len(pairs)


def test_reembed_corrects_vectors_and_drops_collapsed_cosine(reembed_home):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)

    # Pre-migration: every record carries the same label vector -> mean
    # pairwise cosine is ~1.0 (the collapsed state).
    pre_vecs = [_stored_embedding(store, rid) for rid in ids]
    pre_cos = _mean_pairwise_cosine(pre_vecs)
    assert pre_cos > 0.99, (
        f"seed should reproduce the collapsed state; pre-migration mean "
        f"pairwise cosine was {pre_cos:.4f}, expected ~1.0"
    )

    result = migrate_reembed_from_text(store)
    assert result["reembedded"] == len(ids), result
    assert result["skipped"] == 0, result
    assert result["total"] == len(ids), result

    # Each record's new embedding equals embed(its own literal_surface).
    for rid, content in zip(ids, _CONTENTS):
        new_vec = _stored_embedding(store, rid)
        expected = list(embedder.embed(content))
        assert cosine(new_vec, expected) > 0.9999, (
            f"record {rid} was not re-embedded from its own text "
            f"(cosine vs embed(literal_surface) = "
            f"{cosine(new_vec, expected):.6f})"
        )
        # literal_surface must be byte-identical -- never modified.
        assert store.get(rid).literal_surface == content

    # Post-migration: distinct contents spread out, so mean pairwise cosine
    # drops materially from the collapsed ~1.0 baseline.
    post_vecs = [_stored_embedding(store, rid) for rid in ids]
    post_cos = _mean_pairwise_cosine(post_vecs)
    assert post_cos < pre_cos - 0.2, (
        f"random-pair cosine did not drop materially: "
        f"pre={pre_cos:.4f} post={post_cos:.4f}"
    )


def test_reembed_is_idempotent(reembed_home):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)

    migrate_reembed_from_text(store)
    first = {str(rid): _stored_embedding(store, rid) for rid in ids}

    result = migrate_reembed_from_text(store)
    assert result["reembedded"] == len(ids), result

    for rid in ids:
        second = _stored_embedding(store, rid)
        assert cosine(first[str(rid)], second) > 0.999999, (
            f"second migration changed vectors for {rid}; re-embedding the "
            f"same text must be a no-op in effect"
        )


def test_reembed_skips_records_with_empty_text(reembed_home):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text
    from iai_mcp.store import _uuid_literal, RECORDS_TABLE

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)

    # Blank out literal_surface on one record at the storage layer (encrypted
    # empty string). The migration must skip it -- never fabricate a vector --
    # while re-embedding the rest.
    blanked = ids[0]
    tbl = store.db.open_table(RECORDS_TABLE)
    ct = store._encrypt_for_record(blanked, "")
    tbl.update(
        where=f"id = '{_uuid_literal(blanked)}'",
        values={"literal_surface": ct},
    )

    result = migrate_reembed_from_text(store)
    assert result["skipped"] == 1, result
    assert result["reembedded"] == len(ids) - 1, result
    assert result["total"] == len(ids), result


def test_reembed_invalidates_exact_index(reembed_home, monkeypatch):
    """The migration rewrites embeddings in place; the resident exact-cosine
    matrix (if warm) must be invalidated alongside the hnsw rebuild and the
    runtime graph cache drop, so a subsequent exact_top_k call rebuilds from
    the corrected vectors rather than serving a stale pre-correction
    snapshot."""
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    _seed_broken(store, embedder)

    calls: list[bool] = []
    real_invalidate = store.invalidate_exact_index

    def _spy_invalidate():
        calls.append(True)
        return real_invalidate()

    monkeypatch.setattr(store, "invalidate_exact_index", _spy_invalidate)

    result = migrate_reembed_from_text(store)
    assert result["reembedded"] > 0, result

    assert calls, (
        "migrate_reembed_from_text must call store.invalidate_exact_index() "
        "after rewriting embeddings in place"
    )


def test_reembed_dry_run_changes_nothing(reembed_home):
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)

    before = {str(rid): _stored_embedding(store, rid) for rid in ids}
    result = migrate_reembed_from_text(store, dry_run=True)
    assert result["dry_run"] is True
    assert result["reembedded"] == len(ids), result

    for rid in ids:
        after = _stored_embedding(store, rid)
        assert cosine(before[str(rid)], after) > 0.999999, (
            f"dry-run must not modify stored vectors; {rid} changed"
        )


def test_reembed_uses_batch_embed_once_per_window(reembed_home, monkeypatch):
    """The window path must call embed_batch once per id-window, never the
    per-record embed() for the corpus loop. With 6 records and a window of 3,
    that is exactly 2 batch calls of 3 texts each and zero per-record calls."""
    from iai_mcp.embed import Embedder, embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)
    assert len(ids) == 6

    batch_calls: list[int] = []
    single_calls: list[str] = []

    real_batch = Embedder.embed_batch
    real_embed = Embedder.embed

    def counting_batch(self, texts):
        batch_calls.append(len(texts))
        return real_batch(self, texts)

    def counting_embed(self, text):
        single_calls.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed_batch", counting_batch)
    monkeypatch.setattr(Embedder, "embed", counting_embed)

    result = migrate_reembed_from_text(store, batch_size=3)
    assert result["reembedded"] == 6, result

    assert batch_calls == [3, 3], (
        f"expected one embed_batch call per 3-record window (windows of 3), "
        f"got batch sizes {batch_calls}"
    )
    assert single_calls == [], (
        f"corpus loop must not call per-record embed(); saw {len(single_calls)} "
        f"single calls"
    )


def test_reembed_resume_skips_committed_windows(reembed_home, monkeypatch):
    """Interrupt after the first window commits, then resume: the resumed run
    must not re-read or re-embed window 1, must finish the rest, and the final
    state must equal a clean full run (every vector == embed(its own text))."""
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.migrate import migrate_reembed_from_text

    store = _open_store()
    embedder = embedder_for_store(store)
    ids = _seed_broken(store, embedder)
    assert len(ids) == 6

    # Window size 3 -> two windows. Interrupt window 2 at its encode step
    # (before its transaction opens): window 1 has already committed its write
    # and saved its checkpoint, window 2 raises before any write -- a faithful
    # crash after window 1 committed but before window 2 landed. The raise
    # propagates (the migration's fail-safe wraps only embed_batch, not encode).
    import iai_mcp.hippo as _hippo

    real_encode = _hippo._encode_embedding
    encode_windows: list[int] = []

    def fail_on_window2_encode(vec):
        # First three encode calls = window 1; raise on the fourth (window 2).
        encode_windows.append(1)
        if len(encode_windows) > 3:
            raise RuntimeError("simulated interrupt at window 2 encode")
        return real_encode(vec)

    monkeypatch.setattr(_hippo, "_encode_embedding", fail_on_window2_encode)

    with pytest.raises(RuntimeError):
        migrate_reembed_from_text(store, batch_size=3)

    monkeypatch.setattr(_hippo, "_encode_embedding", real_encode)

    # Window 1 (the first three ids, id-ordered) is corrected; window 2 is not.
    ordered = sorted(ids, key=lambda r: str(r))
    win1, win2 = ordered[:3], ordered[3:]
    for rid in win1:
        content = _CONTENTS[ids.index(rid)]
        got = _stored_embedding(store, rid)
        expected = list(embedder.embed(content))
        assert cosine(got, expected) > 0.9999, (
            f"window-1 record {rid} should be corrected after the interrupt"
        )

    # Resume with the encode restored; window 1 must NOT be re-read.
    reread: list[str] = []
    real_decrypt = store._decrypt_for_record

    def counting_decrypt(record_id, value):
        reread.append(str(record_id))
        return real_decrypt(record_id, value)

    monkeypatch.setattr(store, "_decrypt_for_record", counting_decrypt)

    result = migrate_reembed_from_text(store, batch_size=3, resume=True)
    assert result["reembedded"] == 3, result
    assert result["total"] == 3, result

    win1_ids = {str(r) for r in win1}
    assert not (win1_ids & set(reread)), (
        f"resume re-read already-committed window-1 ids: {win1_ids & set(reread)}"
    )

    # Final state: every record == embed(its own literal_surface).
    for rid in ids:
        content = _CONTENTS[ids.index(rid)]
        got = _stored_embedding(store, rid)
        expected = list(embedder.embed(content))
        assert cosine(got, expected) > 0.9999, (
            f"after resume, record {rid} is not embed(its own text)"
        )
