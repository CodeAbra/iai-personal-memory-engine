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


def test_pending_lifecycle_freshness_travels_by_unlink_not_key(hermetic_store: Path) -> None:
    """Pending rows are deliberately INVISIBLE to _cache_key: they are never
    graph nodes, and keying on them re-keyed the warm bundle on every ambient
    capture. The flip's freshness demand travels as an explicit unlink of the
    warm-graph cache instead — driven through the PRODUCTION write paths.
    """
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.runtime_graph_cache import _cache_key
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        # Seed one ordinary embedded row so the corpus is non-empty.
        store.insert(_make_normal_record("alice baseline embedded row", seed=7))
        flush_record_buffer(store)

        key_no_pending = _cache_key(store)

        # Pending appearance through the production write path: NO key change.
        _insert_pending(store, "pending probe surface carrying real text")
        assert _cache_key(store) == key_no_pending, (
            "an ambient capture landing as pending must not re-key the bundle"
        )

        # A cache file exists so the unlink assertion below has teeth.
        rgc._cache_path(store).parent.mkdir(parents=True, exist_ok=True)
        rgc._cache_path(store).write_text("{}", encoding="utf-8")

        # Re-embed flip 1->0 through the production wake sequence.
        result = store.db.pending_embeddings_wake_sequence(
            embedder=_DeterministicEmbedder()
        )
        assert result.get("action") == "wake_sequence", result
        assert result.get("reembed_count") == 1, result

        # The load-bearing assertion: the flip UNLINKS the warm-graph cache,
        # which no key term can miss and no stale serve can survive.
        assert not rgc._cache_path(store).exists(), (
            "the wake sequence must unlink the warm-graph cache after a flip"
        )
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
    """The parity components (schema, embed_dim, cache_version) are the key's
    TAIL whatever its arity, and CACHE_VERSION is the last element — the
    invariant the legacy [:-1] slice and the shape-agnostic parity gates use.
    """
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.runtime_graph_cache import _cache_key, _parity_components
    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        key = _cache_key(store)
        assert len(key) >= 3
        assert tuple(key[-3:]) == tuple(_parity_components(store))
        assert key[-1] == rgc.CACHE_VERSION
    finally:
        store.close()
