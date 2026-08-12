"""End-to-end witness for thread-following recall: a later turn's cue must
surface an earlier turn's fact byte-for-byte, dispatched through the same
production entry point (``core.dispatch``) the daemon serves recall through.

Uses a small committed synthetic fixture (alice-only) so the witness is
portable and never depends on a local, never-committed labelled corpus.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from iai_mcp import core
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "uat_thread_alice.json"


def _load_fixture() -> dict:
    with _FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _seed_thread(store: MemoryStore, turns: list[dict]) -> None:
    embedder = embedder_for_store(store)
    now = datetime.now(timezone.utc)
    for turn in turns:
        text = turn["text"]
        vector = list(embedder.embed(text))
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=vector,
            community_id=None,
            centrality=0.3,
            detail_level=3,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "uat-thread", "turn": turn["turn"]}],
            created_at=now,
            updated_at=now,
            tags=[f"role:{turn['role']}"],
            language="en",
            role=turn["role"],
        )
        store.insert(rec)


def test_thread_following_recalls_turn_one_verbatim(tmp_path):
    data = _load_fixture()
    assert data.get("schema_version") == 1

    store = MemoryStore(path=tmp_path)
    _seed_thread(store, data["turns"])

    turn_one_text = data["turns"][0]["text"]
    final_cue = data["final_cue"]

    response = core.dispatch(
        store,
        "memory_recall",
        {"cue": final_cue, "session_id": "uat-thread", "budget_tokens": 2000},
    )
    hits = response.get("hits", [])
    assert hits, f"expected non-empty hits for cue {final_cue!r}, got {response!r}"

    # Verbatim = exact string compare against a hit's stored payload, not a
    # fuzzy substring-in-hits proxy -- this is what proves lossless recall.
    surfaces = [h.get("literal_surface") for h in hits]
    assert turn_one_text in surfaces, (
        f"turn-1 fact {turn_one_text!r} did not surface verbatim in recall "
        f"hits for cue {final_cue!r}; got surfaces={surfaces!r}"
    )
