"""A clean pending-drain pass must NOT rebuild the whole ANN index: flips are
fed into the live buffer per row, so the unconditional rebuild re-added the
entire corpus for a handful of flipped rows on every ambient drain — the
daemon burned multiple cores re-indexing a store that had not diverged."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM


@pytest.fixture(autouse=True)
def _crypto(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


class DetEmb:
    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()


def _make_rec(seed: int, store):
    from iai_mcp.types import MemoryRecord

    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface=f"wake incremental probe {seed}",
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=["t"],
        language="en",
    )


def _insert_pending(store, surface: str) -> str:
    pid = str(uuid4())
    store.db.insert_pending_row(
        record_id=pid, tier="episodic", literal_surface=surface,
        tags_json=json.dumps(["t"]), provenance_json=json.dumps([]),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return pid


def test_clean_flip_pass_skips_the_full_rebuild(tmp_path):
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    try:
        for i in range(8):
            store.insert(_make_rec(i, store))
        flush_record_buffer(store)

        pid = _insert_pending(store, "incremental ann flip surface")
        emb = DetEmb()
        result = store.db.pending_embeddings_wake_sequence(embedder=emb)

        assert result["action"] == "wake_sequence", result
        assert result["reembed_count"] == 1, result
        assert result["rebuild"]["action"] == "skip", (
            f"a clean flip pass must not re-add the whole corpus: {result}"
        )

        # The flipped row must be ANN-findable through the incremental add.
        cue = emb.embed("incremental ann flip surface")
        got = store.query_similar(list(cue), k=3)
        assert any(str(r.id) == pid for r, _ in got), (
            "flipped row absent from ANN after the incremental add"
        )

        # A clean follow-up pass with nothing pending is a pure skip.
        again = store.db.pending_embeddings_wake_sequence(embedder=emb)
        assert again["action"] == "skip", again
    finally:
        store.close()


def test_mismatch_still_routes_through_the_full_rebuild(tmp_path):
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    try:
        for i in range(6):
            store.insert(_make_rec(100 + i, store))
        flush_record_buffer(store)

        # Manufacture an index/corpus divergence: drop one label-map entry.
        assert store.db._label_map, "seeded corpus must be indexed"
        dropped = next(iter(store.db._label_map))
        store.db._label_map.pop(dropped)

        result = store.db.pending_embeddings_wake_sequence(embedder=DetEmb())
        assert result["action"] == "wake_sequence", result
        assert result["rebuild"]["action"] == "rebuild", (
            f"an index/corpus mismatch must self-heal via the full rebuild: {result}"
        )
        assert dropped in store.db._label_map, (
            "the rebuild must restore the dropped label"
        )
    finally:
        store.close()
