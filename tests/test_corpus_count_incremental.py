"""Hot write paths shift the cached corpus counts by their exact delta instead
of dropping them: an invalidation there forced the next reader into a full
filtered COUNT — on the lilli engine a scan of every leaf page, re-paid on
every ambient capture around the clock."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _crypto(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


def _make_rec(seed: int, store):
    from datetime import datetime, timezone
    from uuid import uuid4

    import numpy as np

    from iai_mcp.types import MemoryRecord

    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface=f"incremental count probe {seed}",
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=["t"],
        language="en",
    )


class TestAdjustUnit:
    def test_adjust_shifts_a_cached_value(self):
        from iai_mcp.store._corpus_count_cache import CorpusCountCache

        cc = CorpusCountCache()
        cc.put("active", 100)
        cc.adjust("active", 3)
        assert cc.get("active") == 103
        cc.adjust("active", -1)
        assert cc.get("active") == 102

    def test_adjust_on_a_miss_stays_a_miss(self):
        from iai_mcp.store._corpus_count_cache import CorpusCountCache

        cc = CorpusCountCache()
        cc.adjust("active", 5)
        assert cc.get("active") is None

    def test_adjust_refuses_an_in_flight_stale_recompute(self):
        """A recompute that snapshotted its generation BEFORE the adjusted
        write committed may have counted the pre-write state — put_if_gen must
        refuse it exactly as it would after an invalidate."""
        from iai_mcp.store._corpus_count_cache import CorpusCountCache

        cc = CorpusCountCache()
        gen = cc.generation()
        cc.adjust("active", 1)
        cc.put_if_gen("active", 999, gen)
        assert cc.get("active") is None

    def test_adjust_fires_the_staleness_hook(self):
        from iai_mcp.store._corpus_count_cache import CorpusCountCache

        cc = CorpusCountCache()
        fired = []
        cc.on_invalidate = lambda: fired.append(1)
        cc.put("active", 10)
        cc.adjust("active", 1)
        assert fired, "adjusted writes must reach RO staleness signals too"


class TestStoreEndToEnd:
    def test_record_flush_keeps_active_count_live_without_recount(self, tmp_path):
        from iai_mcp.store import MemoryStore, flush_record_buffer

        store = MemoryStore(path=tmp_path / "store")
        try:
            for i in range(5):
                store.insert(_make_rec(i, store))
            flush_record_buffer(store)
            primed = store.active_records_count()
            assert store._corpus_count_cache.get("active") == primed

            for i in range(3):
                store.insert(_make_rec(100 + i, store))
            flush_record_buffer(store)

            # A PRESENT cached value is the proof: the flush shifted it in
            # place instead of dropping it for a full filtered recount.
            assert store._corpus_count_cache.get("active") == primed + 3
            assert store.active_records_count() == primed + 3
        finally:
            store.close()

    def test_pending_insert_and_flip_track_exactly(self, tmp_path):
        import json
        from datetime import datetime, timezone
        from uuid import uuid4

        import numpy as np

        from iai_mcp.store import MemoryStore, flush_record_buffer
        from iai_mcp.types import EMBED_DIM

        class DetEmb:
            def embed(self, text):
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                v = rng.standard_normal(EMBED_DIM).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                return v.tolist()

        store = MemoryStore(path=tmp_path / "store")
        try:
            for i in range(4):
                store.insert(_make_rec(i, store))
            flush_record_buffer(store)
            active0 = store.active_records_count()
            pending0 = store.pending_records_count()
            assert store._corpus_count_cache.get("active") == active0
            assert store._corpus_count_cache.get("pending") == pending0

            store.db.insert_pending_row(
                record_id=str(uuid4()), tier="episodic",
                literal_surface="pending probe surface",
                tags_json=json.dumps(["t"]), provenance_json=json.dumps([]),
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            assert store._corpus_count_cache.get("pending") == pending0 + 1

            store.db.pending_embeddings_wake_sequence(embedder=DetEmb())
            assert store._corpus_count_cache.get("active") == active0 + 1
            assert store._corpus_count_cache.get("pending") == pending0

            # Ground truth: force a recount and confirm the shifted values
            # were exact, not merely self-consistent.
            store._corpus_count_cache.clear()
            assert store.active_records_count() == active0 + 1
            assert store.pending_records_count() == pending0
        finally:
            store.close()

    def test_bare_hippodb_without_store_still_works(self, tmp_path):
        """No MemoryStore means no adjust callback: the pending-insert fire
        site must fall back cleanly (no crash, no callback)."""
        import json
        from datetime import datetime, timezone
        from uuid import uuid4

        from iai_mcp.hippo import HippoDB

        db = HippoDB(tmp_path)
        try:
            db.insert_pending_row(
                record_id=str(uuid4()), tier="episodic",
                literal_surface="bare pending surface",
                tags_json=json.dumps([]), provenance_json=json.dumps([]),
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            db.close()
