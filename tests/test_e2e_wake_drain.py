from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.provenance_buffer import (
    _BUFFER_FILENAME,
    flush_deferred_provenance,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord

class _PerfEmbedder:

    DIM = EMBED_DIM

    def __init__(self, base_seed: int = 0) -> None:
        self._base_seed = base_seed

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random
        digest = hashlib.sha256(
            f"{self._base_seed}:{text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _make_rec(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
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
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )

def _seed_small_store(path, n: int = 20, seed: int = 0):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=path)
    embedder = _PerfEmbedder(base_seed=seed)
    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    for i in range(n):
        vec = embedder.embed(f"seed-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"synthetic fact {i}", tags=tags)
        store.insert(rec)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, embedder, graph, assignment, rich_club

def test_buffer_drains_on_wake(tmp_path, monkeypatch):
    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_small_store(
        tmp_path, n=20, seed=0,
    )

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    session = "wake-drain-test"

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id=session,
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1, "recall must produce at least one hit"

    buffer_path = Path(store.root) / _BUFFER_FILENAME
    assert buffer_path.exists(), "defer_provenance must have created the buffer"
    pre_lines = buffer_path.read_text().strip().splitlines()
    assert len(pre_lines) > 0, (
        f"buffer must contain entries from the recall defer; got {len(pre_lines)}"
    )

    drained = flush_deferred_provenance(store)
    assert drained == len(pre_lines), (
        f"flush must drain all pending entries; expected {len(pre_lines)}, got {drained}"
    )

    post_text = buffer_path.read_text().strip()
    assert post_text == "", (
        f"buffer must be truncated after flush; got {len(post_text)} bytes"
    )

    for h in resp.hits:
        rec = store.get(h.record_id)
        assert rec is not None
        matching = [p for p in rec.provenance if p.get("session_id") == session]
        assert len(matching) == 1, (
            f"record {h.record_id} has {len(matching)} provenance entries "
            f"for session '{session}'; expected exactly 1. prov={rec.provenance}"
        )

def test_flush_idempotent_on_empty(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    buffer_path = Path(store.root) / _BUFFER_FILENAME

    assert not buffer_path.exists(), "fresh store must not have a buffer file"
    count_missing = flush_deferred_provenance(store)
    assert count_missing == 0

    buffer_path.write_text("")
    count_empty = flush_deferred_provenance(store)
    assert count_empty == 0

def test_grep_production_callsite_exists():
    repo_root = Path(__file__).resolve().parent.parent
    daemon_src = repo_root / "src" / "iai_mcp" / "daemon" / "__init__.py"
    if not daemon_src.exists():
        pytest.skip("daemon.py source not on disk (sdist/wheel install)")
    text = daemon_src.read_text()
    assert "flush_deferred_provenance" in text, (
        "daemon.py MUST reference flush_deferred_provenance; otherwise the "
        "MEM-05 deferred-provenance buffer is never drained in production "
        "(wake-drain regression)."
    )
