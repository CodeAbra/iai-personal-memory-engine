"""RED test for the capture-time half of the never_merge hard lock: it must
hold at the capture-time cosine gate, not just the write-time gate.

Confirmed live gap (capture.py, the cosine loop below the exact-key branch):
the capture-time gate's cosine loop calls ``store.reinforce_record(record.id)``
unconditionally on any ``score >= DEDUP_COS_THRESHOLD`` hit, with NO
``never_merge`` check on the existing neighbour — a pinned record can be
silently reinforced.

Ground truth for the neighbour's cosine score is controlled by monkeypatching
``store.query_similar`` directly (never depending on the real embedder's
actual similarity output — mirrors the eval fixture's synthetic-cosine
discipline), so this test is deterministic regardless of embedder behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.brainview import BrainView
from iai_mcp.capture import DEDUP_COS_THRESHOLD, capture_turn
from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from iai_mcp import pipeline


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _make_record(
    rid: UUID,
    surface: str = "topic",
    *,
    tier: str = "episodic",
    embedding: list[float] | None = None,
    never_merge: bool = False,
    salience_level: str = "unflagged",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding) if embedding is not None else [0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        salience_level=salience_level,
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pinned_neighbour_never_reinforced_by_capture_turn(driver, store, monkeypatch):
    """A never_merge=True existing neighbour at cos >= DEDUP_COS_THRESHOLD is
    NEVER reinforced by capture_turn — the turn falls through to a normal
    insert instead.

    RED today: capture.py's cosine loop (~lines 390-402) calls
    store.reinforce_record(record.id) unconditionally on any qualifying
    cosine hit, with no never_merge guard.
    """
    _select_driver(driver, monkeypatch)

    pinned_id = uuid4()
    pinned_record = _make_record(
        pinned_id,
        "alice identity anchor — never merge",
        tier="semantic",
        never_merge=True,
    )
    store.insert(pinned_record)

    reinforce_calls: list[UUID] = []
    orig_reinforce = store.reinforce_record

    def _spy_reinforce(record_id, *args, **kwargs):
        reinforce_calls.append(record_id)
        return orig_reinforce(record_id, *args, **kwargs)

    monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

    # Deterministic neighbour score at/above DEDUP_COS_THRESHOLD, independent
    # of the real embedder's actual cosine output.
    def _fake_query_similar(embedding, k=3, tier=None):
        return [(pinned_record, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result = capture_turn(
        store=store,
        text="alice's identity anchor restated in slightly different words",
        cue="identity probe",
        tier="semantic",
        session_id="s1",
        role="user",
    )

    assert pinned_id not in reinforce_calls, (
        f"[driver={driver}] capture_turn must NEVER reinforce a never_merge=True "
        f"neighbour; reinforce_record was called with {reinforce_calls}"
    )
    assert result["status"] != "reinforced", (
        f"[driver={driver}] capture_turn must not report 'reinforced' against "
        f"a pinned neighbour; got {result}"
    )

    # The pinned record's hebbian self-loop edge weight must be unchanged —
    # reinforce_record was never invoked, so no edge-weight mutation occurred.
    # The turn must fall through to a normal insert (which the write-time
    # gate then independently protects).
    rows = list(store.iter_records())
    surfaces = [r.literal_surface for r in rows]
    assert "alice identity anchor — never merge" in surfaces, (
        "the pinned record's literal_surface must be untouched"
    )
    non_pinned_rows = [r for r in rows if r.id != pinned_id]
    assert len(non_pinned_rows) >= 1, (
        f"[driver={driver}] the incoming turn must fall through to a normal "
        f"insert rather than being silently absorbed into the pinned record; "
        f"rows={[r.literal_surface for r in rows]}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_pinned_neighbour_still_reinforced_regression(driver, store, monkeypatch):
    """Regression guard: a NON-pinned neighbour at cos >= DEDUP_COS_THRESHOLD
    is still reinforced and 'reinforced' is returned — unchanged behaviour.
    This must stay GREEN before and after the never_merge guard lands.
    """
    _select_driver(driver, monkeypatch)

    existing_id = uuid4()
    existing_record = _make_record(
        existing_id, "the daemon restarts nightly for consolidation", tier="semantic",
        never_merge=False,
    )
    store.insert(existing_record)

    reinforce_calls: list[UUID] = []
    orig_reinforce = store.reinforce_record

    def _spy_reinforce(record_id, *args, **kwargs):
        reinforce_calls.append(record_id)
        return orig_reinforce(record_id, *args, **kwargs)

    monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

    def _fake_query_similar(embedding, k=3, tier=None):
        return [(existing_record, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result = capture_turn(
        store=store,
        text="the daemon restarts nightly for consolidation, restated",
        cue="daemon probe",
        tier="semantic",
        session_id="s1",
        role="user",
    )

    assert existing_id in reinforce_calls, (
        f"[driver={driver}] a non-pinned neighbour must still be reinforced "
        f"(unchanged behaviour); reinforce_record was called with {reinforce_calls}"
    )
    assert result["status"] == "reinforced", (
        f"[driver={driver}] expected 'reinforced' for a non-pinned near-dup "
        f"neighbour; got {result}"
    )


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_salience_correction_recoverable(tmp_path):
    """A wrongly-set salience_level is recoverable: correcting it back to
    unflagged removes the rank boost on the next recall, with never_merge and
    pinned untouched throughout the correction.
    """
    store = MemoryStore(path=tmp_path / "salience-correction-store")
    embedder = Embedder()
    target_vec = list(embedder.embed("a decision worth remembering clearly"))
    shared_text = "the deployment decision alice made this morning"

    critical = _make_record(
        uuid4(), shared_text, embedding=target_vec, salience_level="critical"
    )
    unflagged = _make_record(
        uuid4(), shared_text, embedding=target_vec, salience_level="unflagged"
    )
    store.insert(critical)
    store.insert(unflagged)

    fillers = [
        _make_record(uuid4(), f"unrelated filler record {i}", embedding=_random_vec(9000 + i))
        for i in range(12)
    ]
    for f in fillers:
        store.insert(f)

    graph = MemoryGraph()
    for rec in (critical, unflagged, *fillers):
        graph.add_node(rec.id, None, rec.embedding)
    assignment = CommunityAssignment()
    target_ids = {critical.id, unflagged.id}

    def _scores() -> dict:
        pipeline._last_recall_latency_ms = 0.0
        result = pipeline.recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=embedder, cue="an unrelated grocery list cue phrase",
            session_id="s1", budget_tokens=4000, mode="concept",
            cue_embedding=target_vec,
        )
        return {h.record_id: h.score for h in result.hits if h.record_id in target_ids}

    before = _scores()
    assert len(before) == 2, before
    assert before[critical.id] > before[unflagged.id], (
        f"critical must outrank its equal-cosine unflagged twin before correction; got {before}"
    )

    stored_before = store.get(critical.id)
    assert stored_before.never_merge is False
    assert stored_before.pinned is False

    view = BrainView(store, pinned_direct=True)
    result = view.salience_direct(record_id=str(critical.id), level="unflagged")
    assert result["status"] == "salience_set"
    assert result["level"] == "unflagged"

    stored_after = store.get(critical.id)
    assert stored_after.salience_level == "unflagged"
    assert stored_after.never_merge is False
    assert stored_after.pinned is False

    after = _scores()
    assert len(after) == 2, after
    assert abs(after[critical.id] - after[unflagged.id]) < 1e-6, (
        f"after correction the boost must be gone -- scores must tie within "
        f"float32 noise; got {after}"
    )


def test_salience_wrong_record_isolated_from_others(tmp_path):
    """A salience_level set on the wrong record never alters, merges, or
    drops any OTHER record -- neither its stored row nor its recall score.
    """
    store = MemoryStore(path=tmp_path / "salience-isolation-store")
    embedder = Embedder()

    a_text = "record A — flagged by mistake instead of B"
    b_text = "record B — the intended salience target"
    a_vec = list(embedder.embed(a_text))
    b_vec = list(embedder.embed(b_text))

    record_a = _make_record(uuid4(), a_text, embedding=a_vec)
    record_b = _make_record(uuid4(), b_text, embedding=b_vec)
    store.insert(record_a)
    store.insert(record_b)

    fillers = [
        _make_record(uuid4(), f"unrelated filler record {i}", embedding=_random_vec(7000 + i))
        for i in range(12)
    ]
    for f in fillers:
        store.insert(f)

    graph = MemoryGraph()
    for rec in (record_a, record_b, *fillers):
        graph.add_node(rec.id, None, rec.embedding)
    assignment = CommunityAssignment()

    def _b_hit():
        pipeline._last_recall_latency_ms = 0.0
        result = pipeline.recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=embedder, cue=b_text,
            session_id="s1", budget_tokens=4000, mode="concept",
            cue_embedding=b_vec,
        )
        matches = [h for h in result.hits if h.record_id == record_b.id]
        assert len(matches) == 1, (
            f"record B must surface as a hit; got "
            f"{[(h.record_id, h.reason) for h in result.hits]}"
        )
        return matches[0]

    snapshot_b_before = store.get(record_b.id)
    b_hit_before = _b_hit()
    assert "xsalience" not in b_hit_before.reason

    view = BrainView(store, pinned_direct=True)
    result = view.salience_direct(record_id=str(record_a.id), level="critical")
    assert result["status"] == "salience_set"

    snapshot_b_after = store.get(record_b.id)
    for field in (
        "literal_surface", "salience_level", "never_merge", "pinned", "tier",
        "detail_level", "centrality", "stability", "difficulty", "never_decay",
        "tags", "language", "community_id",
    ):
        assert getattr(snapshot_b_before, field) == getattr(snapshot_b_after, field), (
            f"record B's {field} must be byte-identical after A's mis-flag; "
            f"before={getattr(snapshot_b_before, field)!r} "
            f"after={getattr(snapshot_b_after, field)!r}"
        )
    assert snapshot_b_after.salience_level == "unflagged", (
        "record B must remain unflagged -- the mis-flag landed on A only"
    )

    b_hit_after = _b_hit()
    assert "xsalience" not in b_hit_after.reason, (
        f"A's flag must never appear in B's reason string; got {b_hit_after.reason!r}"
    )
    assert abs(b_hit_after.score - b_hit_before.score) < 1e-6, (
        f"A's flag must contribute zero to B's score; "
        f"before={b_hit_before.score} after={b_hit_after.score}"
    )


def test_salience_correction_writes_only_salience_column(tmp_path):
    """salience_direct writes ONLY the salience_level column -- every other
    field, specifically pinned and never_merge, is unchanged.
    """
    store = MemoryStore(path=tmp_path / "salience-column-scope-store")
    rid = uuid4()
    record = _make_record(rid, "a record with everything set to a non-default baseline")
    store.insert(record)

    before = store.get(rid)
    assert before.salience_level == "unflagged"
    assert before.pinned is False
    assert before.never_merge is False

    view = BrainView(store, pinned_direct=True)
    result = view.salience_direct(record_id=str(rid), level="critical")
    assert result["status"] == "salience_set"
    assert result["level"] == "critical"

    after = store.get(rid)
    assert after.salience_level == "critical"

    # provenance is expected to change: salience_direct best-effort appends a
    # traceability entry, mirroring pin_direct -- it is not part of the
    # `values={"salience_level": level}` UPDATE dict itself.
    changed_fields = [
        f for f in (
            "literal_surface", "tier", "embedding", "community_id", "centrality",
            "detail_level", "pinned", "stability", "difficulty", "last_reviewed",
            "never_decay", "never_merge", "tags", "language",
        )
        if getattr(before, f) != getattr(after, f)
    ]
    assert changed_fields == [], (
        f"salience_direct must write ONLY salience_level (+ best-effort "
        f"provenance); unexpected column drift: {changed_fields}"
    )
    assert after.pinned is False
    assert after.never_merge is False
