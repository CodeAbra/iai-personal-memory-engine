"""Behavioral tests for the time-held-out, episodic-tail-gold labelling
script: the repo-tree write refusal, tail-member reconstruction from real
``consolidated_from`` edges, and the natural-cue mining exclusion."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

import scripts.label_holdout_tail_cues as label_holdout_tail_cues_module
from scripts.label_holdout_tail_cues import (
    _REPO_ROOT,
    _write_fixture,
    cluster_tail_members,
    mine_holdout_cues,
)
from tests.test_eval_copy_store_warm_baseline import (
    _reset_graph_cache_generation_epoch,  # noqa: F401 -- autouse fixture
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _dense_vec(active_idx, dim: int = EMBED_DIM, mag: float = 0.9, base: float = 0.02) -> list[float]:
    vec = [base] * dim
    for i in active_idx:
        vec[i] = mag
    return vec


def _rec(text: str, vec, *, tier: str = "episodic", tags=None, age_days: int = 0) -> MemoryRecord:
    now = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags or [],
        language="en",
    )


def _flush(store) -> None:
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    flush_record_buffer(store)
    flush_edge_buffer(store)


# ---------------------------------------------------------------------------
# Repo-tree write refusal
# ---------------------------------------------------------------------------


def test_write_fixture_refuses_repo_tree_path(tmp_path) -> None:
    in_repo_path = _REPO_ROOT / "tmp-holdout-fixture-should-never-land.json"
    with pytest.raises(SystemExit) as exc_info:
        _write_fixture([{"cue_id": "x"}], in_repo_path, force_in_repo=False)
    assert exc_info.value.code == 2
    assert not in_repo_path.exists(), "a refused write must never touch the repo tree"


def test_write_fixture_force_in_repo_override_writes(tmp_path, monkeypatch) -> None:
    """--force-in-repo must actually bypass the refusal for a GENUINELY
    in-repo path -- not merely succeed on a path that was never refused
    in the first place. Uses a fake repo root under tmp_path so the
    refusal/override logic is exercised without ever touching the real
    tree -- a killed run cannot leave a stray file in the actual repo."""
    fake_repo_root = tmp_path.resolve()
    monkeypatch.setattr(label_holdout_tail_cues_module, "_REPO_ROOT", fake_repo_root)
    in_repo_path = fake_repo_root / "tmp-holdout-fixture-force-override-check.json"

    with pytest.raises(SystemExit) as exc_info:
        _write_fixture([{"cue_id": "x"}], in_repo_path, force_in_repo=False)
    assert exc_info.value.code == 2
    assert not in_repo_path.exists(), "the SAME path must be genuinely refused without the flag"

    _write_fixture([{"cue_id": "x"}], in_repo_path, force_in_repo=True)

    assert in_repo_path.exists()


def test_write_fixture_writes_local_path(tmp_path) -> None:
    out_path = tmp_path / "eval-fixtures" / "holdout.json"
    _write_fixture(
        [{"cue_id": "cue_0001", "gold_record_id": "abc", "cluster_id": "def"}],
        out_path, force_in_repo=False,
    )
    assert out_path.exists()
    import json

    payload = json.loads(out_path.read_text())
    assert payload["cues"][0]["gold_record_id"] == "abc"


# ---------------------------------------------------------------------------
# Tail-member reconstruction from real consolidated_from edges
# ---------------------------------------------------------------------------


def test_cluster_tail_members_excludes_first_five_by_recency(tmp_path) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")

    summary = _rec("Cluster summary of seven records", _dense_vec(range(0, 20)), tier="semantic")
    store.insert(summary)

    # Seven members, oldest to newest by age_days descending (age 6 = oldest).
    members = [
        _rec(f"member note {i}", _dense_vec(range(i, i + 20)), age_days=6 - i)
        for i in range(7)
    ]
    for m in members:
        store.insert(m)
    _flush(store)

    store.boost_edges(
        [(summary.id, m.id) for m in members], edge_type="consolidated_from", delta=1.0,
    )
    _flush(store)

    tails = cluster_tail_members(store)
    assert summary.id in tails

    # Recency-descending order matches insertion order 6,5,4,3,2,1,0 (member
    # index i has age_days=6-i, so member 6 is the newest/age 0, member 0 is
    # oldest/age 6). Sorted descending by recency: member6..member0. The
    # first 5 (member6,5,4,3,2) are embedded by the summary; the tail
    # (rank 6+) is member1, member0 -- the two OLDEST records.
    tail_ids = {r.id for r in tails[summary.id]}
    expected_tail_ids = {members[1].id, members[0].id}
    assert tail_ids == expected_tail_ids, (
        f"expected the two oldest members {expected_tail_ids} as the tail, "
        f"got {tail_ids}"
    )

    store.close()


def test_cluster_tail_members_empty_for_small_cluster(tmp_path) -> None:
    """A cluster with <= 5 members has no tail -- every member is within
    the summary's own first-5 embed window."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    summary = _rec("Small cluster summary", _dense_vec(range(0, 20)), tier="semantic")
    store.insert(summary)
    members = [_rec(f"small member {i}", _dense_vec(range(i, i + 20))) for i in range(3)]
    for m in members:
        store.insert(m)
    _flush(store)

    store.boost_edges(
        [(summary.id, m.id) for m in members], edge_type="consolidated_from", delta=1.0,
    )
    _flush(store)

    tails = cluster_tail_members(store)
    assert summary.id not in tails

    store.close()


# ---------------------------------------------------------------------------
# Natural-cue mining excludes tail-gold candidates
# ---------------------------------------------------------------------------


def test_mine_holdout_cues_excludes_tail_ids(tmp_path) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    cue_rec = _rec("what did alice say about the project kickoff", _dense_vec(range(0, 20)), tags=["role:user"])
    tail_rec = _rec("tail member also tagged role:user", _dense_vec(range(30, 50)), tags=["role:user"])
    store.insert(cue_rec)
    store.insert(tail_rec)
    _flush(store)

    mined = mine_holdout_cues(store, sample_n=10, excluded_ids={tail_rec.id})
    mined_ids = {r.id for r in mined}
    assert cue_rec.id in mined_ids
    assert tail_rec.id not in mined_ids

    store.close()
