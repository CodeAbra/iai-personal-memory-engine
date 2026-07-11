"""AES-fence guard for the recall spread/final-hit path.

Any candidate that enters the final hit set must be fully decoded
(``literal_surface`` present) before it is returned. This guards the
invariant that no undecrypted spread candidate can leak into a hit with
empty content.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.types import EMBED_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_store_with_edges(tmp_path: Path):
    """Build a minimal store with a 2-hop chain for spread-candidate testing."""
    from datetime import datetime, timezone
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    store = MemoryStore(path=tmp_path / "decode_store")

    # Seed: high-cosine ANN candidate
    seed_uuid = UUID(int=0xBEEF_0001)
    seed_vec = _unit_vec(10)
    store.insert(MemoryRecord(
        id=seed_uuid,
        tier="episodic",
        literal_surface="User seed record",
        aaak_index="",
        embedding=seed_vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    ))

    # Spread candidate: reachable only via hop, not via ANN top-k
    spread_uuid = UUID(int=0xBEEF_0002)
    rng = np.random.default_rng(20260701)
    base = np.array(seed_vec, dtype=np.float32)
    noise = rng.standard_normal(EMBED_DIM).astype(np.float32)
    noise -= float(np.dot(noise, base)) * base
    noise /= float(np.linalg.norm(noise))
    spread_vec = (0.02 * base + np.sqrt(1 - 0.02**2) * noise)
    spread_vec = spread_vec / float(np.linalg.norm(spread_vec))
    store.insert(MemoryRecord(
        id=spread_uuid,
        tier="episodic",
        literal_surface="User spread-only candidate",
        aaak_index="",
        embedding=spread_vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    ))

    store.boost_edges([(seed_uuid, spread_uuid)], edge_type="hebbian", delta=[5.0])
    return store, seed_uuid, spread_uuid, seed_vec


# ---------------------------------------------------------------------------
# Test: final hits are always fully decoded (AES-fence)
# ---------------------------------------------------------------------------


def test_final_hit_receives_full_decode(tmp_path, monkeypatch):
    """Any candidate that enters the final hit set is fully decoded (AES-fence).

    A candidate that becomes a final hit must have a non-empty
    literal_surface (the full _decrypt_for_record ran for it).
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "d.sock"))

    store, seed_uuid, spread_uuid, seed_vec = _build_store_with_edges(tmp_path)

    from iai_mcp import runtime_graph_cache as _rgc
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response

    graph, assignment, rc = build_runtime_graph(store)
    _rgc.save(store, assignment, rc)

    import iai_mcp.pipeline as _pl
    _pl._last_recall_latency_ms = 0.0

    embedder = Embedder()
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=embedder,
        cue="User seed record",
        session_id="test",
        budget_tokens=3000,
        mode="concept",
    )

    _pl._last_recall_latency_ms = 0.0
    store.close()

    # Every final hit must have a non-empty, fully-decoded literal_surface
    assert len(resp.hits) > 0, "recall must return at least one hit"
    for hit in resp.hits:
        assert hit.literal_surface, (
            f"hit {hit.record_id} has empty/None literal_surface — "
            "the AES-fence was not applied; an undecrypted candidate entered the hit set"
        )
