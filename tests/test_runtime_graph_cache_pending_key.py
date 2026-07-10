from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from test_semantic_degraded_path import _make_normal_record  # noqa: E402

from iai_mcp.types import EMBED_DIM  # noqa: E402


class _DeterministicEmbedder:
    """Map text to a stable unit vector by content (no model load)."""

    def embed(self, text: str) -> list[float]:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()


def _insert_pending(store, surface: str) -> str:
    pid = str(uuid.uuid4())
    store.db.insert_pending_row(
        record_id=pid,
        tier="episodic",
        literal_surface=surface,
        tags_json=json.dumps(["role:user"]),
        provenance_json=json.dumps([]),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return pid


def test_pending_to_embedded_flip_changes_cache_key(hermetic_store: Path) -> None:
    """A pending->embedded transition must change _cache_key, driven through the
    PRODUCTION write paths (not a raw out-of-band SQLite write).

    With no pending rows the key carries pending_count 0; inserting a pending row
    via ``insert_pending_row`` bumps it to 1; the ``pending_embeddings_wake_sequence``
    re-embed flips it back to 0 — proving the raw pending term is not absorbed by
    the count window. Each transition fires the corpus-count invalidation callback,
    so ``_cache_key`` (which now reads the count cache, not a live probe) observes
    every transition.
    """
    from iai_mcp.runtime_graph_cache import _cache_key
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        # Seed one ordinary embedded row so the corpus is non-empty.
        store.insert(_make_normal_record("alice baseline embedded row", seed=7))
        flush_record_buffer(store)

        key_no_pending = _cache_key(store)

        # Pending appearance through the production write path (fires cb('pending')).
        _insert_pending(store, "pending probe surface carrying real text")
        key_with_pending = _cache_key(store)

        # Re-embed flip 1->0 through the production wake sequence
        # (fires cb('active','pending')).
        result = store.db.pending_embeddings_wake_sequence(
            embedder=_DeterministicEmbedder()
        )
        assert result.get("action") == "wake_sequence", result
        assert result.get("reembed_count") == 1, result

        key_after_reembed = _cache_key(store)

        # The pending appearance changed the key.
        assert key_with_pending != key_no_pending
        # The load-bearing assertion: the 1->0 re-embed flip MUST change the key.
        assert key_after_reembed != key_with_pending
    finally:
        store.close()


def test_ordinary_insert_within_window_keeps_key_stable(
    hermetic_store: Path,
) -> None:
    """With zero pending rows throughout, adding ordinary records that stay
    within the same count window does NOT change the key — the pending term is
    identically 0, so the fix does not thrash ordinary inserts.
    """
    from iai_mcp.runtime_graph_cache import _cache_key
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        store.insert(_make_normal_record("alice first row", seed=1))
        flush_record_buffer(store)
        key_a = _cache_key(store)

        store.insert(
            _make_normal_record("bob second row within the same window", seed=2)
        )
        flush_record_buffer(store)
        key_b = _cache_key(store)

        # Two records, both inside the same count window, no pending rows.
        assert key_b == key_a
    finally:
        store.close()


def test_cache_key_parity_triple_still_addressable(hermetic_store: Path) -> None:
    """The parity components (schema, embed_dim, cache_version) sit at the new
    positions [3],[4],[5] and CACHE_VERSION is still the last element — the
    insertion-point invariant the legacy [:-1] slice and the parity gates rely on.
    """
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.runtime_graph_cache import _cache_key, _parity_components
    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        key = _cache_key(store)
        assert len(key) == 6
        schema, embed_dim, cache_version = _parity_components(store)
        assert key[3] == schema
        assert key[4] == embed_dim
        assert key[5] == cache_version
        assert key[-1] == rgc.CACHE_VERSION
    finally:
        store.close()
