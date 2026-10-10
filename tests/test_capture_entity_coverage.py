"""Real-route proof and coverage measurement for capture-time entity tagging.

Two independent surfaces, never conflated: entity-tag coverage feeds the
anchor field inside the AAAK index (the entity/anchor-overlap ranking term),
while salience_level is a separate weighted-OR composite over directive,
epistemic_status, and entity_tags. Per the shipped weights a lone entity-tag
signal stays below the composite's notable threshold, so entity coverage on
its own never raises salience_level -- these tests assert only the
aaak/entity surface and never claim otherwise.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.migrate._salience_backfill import backfill_salience
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    return MemoryStore(path=tmp_path)


# ---------------------------------------------------------------------------
# Real-route proof
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_entity_bearing_turn_earns_entity_tags_and_anchor_field(driver, store, monkeypatch):
    """A turn carrying a code-shaped token through the real capture route
    must earn an entity: tag and a populated anchor (E:) field inside the
    minted AAAK index -- proving the shipped extractor fires on the single
    production choke point, not a bypassable side path."""
    _select_driver(driver, monkeypatch)
    from iai_mcp.aaak import parse_aaak_index

    result = capture_turn(
        store=store, cue="c",
        text="alice pushed the fix to `nimbus-cluster` before lunch",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    entity_tags = [t for t in rec.tags if t.startswith("entity:")]
    assert entity_tags, "an entity-bearing turn must earn an entity: tag on the real capture route"
    assert rec.aaak_index
    parsed = parse_aaak_index(rec.aaak_index)
    assert parsed["entities"], (
        "the anchor overlap field must be populated, not just the aaak_index "
        "string -- the index string is structurally non-empty on every "
        "capture, so the entities field is the signal that actually moves"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_plain_prose_turn_earns_no_entity_tags(driver, store, monkeypatch):
    """Plain declarative prose with no code-shaped token and no mid-sentence
    proper noun legitimately earns no entity tag and an empty anchor field
    -- precision over recall is the designed behavior, not a defect."""
    _select_driver(driver, monkeypatch)
    from iai_mcp.aaak import parse_aaak_index

    result = capture_turn(
        store=store, cue="c",
        text="alice reviewed the quarterly numbers this afternoon",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    entity_tags = [t for t in rec.tags if t.startswith("entity:")]
    assert not entity_tags
    parsed = parse_aaak_index(rec.aaak_index)
    assert not parsed["entities"]


# ---------------------------------------------------------------------------
# Coverage-rate measurement (dry-run only, hermetic corpus)
# ---------------------------------------------------------------------------


def _emb(i: int) -> "list[float]":
    vec = [0.0] * EMBED_DIM
    vec[i % EMBED_DIM] = 1.0
    vec[(i * 37 + 11) % EMBED_DIM] = 0.5
    return vec


def _legacy_summary(i: int, text: str) -> MemoryRecord:
    """A pre-existing semantic-tier summary minted before entity extraction
    ran over it: no entity: tag, no anchor field -- the stale-row shape a
    live corpus accumulates over time."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=_emb(i),
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
        tags=["semantic", "cls_summary"],
        language="en",
        directive=False,
        epistemic_status="unknown",
        salience_level="unflagged",
    )


# (entity-bearing, epistemic_status, text) for the ten conversational rows
# minted through the real capture route -- five carry a code-shaped token,
# five are plain declarative prose.
_CONVERSATIONAL_ROWS = [
    (True, "fact", "alice confirmed the `nimbus-cluster` deploy finished clean"),
    (False, "fact", "alice confirmed the deploy finished without incident"),
    (True, "estimate", "ping @alice_ops about the nimbus_gateway rollout window"),
    (False, "estimate", "the rollout window should wrap up by early afternoon"),
    (True, "hypothesis", "maybe the payments-worker queue is causing the backlog"),
    (False, "hypothesis", "maybe the delay was caused by unrelated approvals"),
    (True, "opinion", "bob merged the release.branch a bit too quickly this morning"),
    (False, "opinion", "bob merged too quickly without proper review this morning"),
    (True, "unknown", "check the `data-migrator` logs before the next attempt"),
    (False, "unknown", "check the logs again before the next attempt tomorrow"),
]


def test_dry_run_measures_entity_and_aaak_coverage_rate_on_mixed_corpus(tmp_path, monkeypatch):
    """Builds a self-contained live-shape corpus -- conversational turns
    minted through the real capture route (mixed entity-bearing/plain-prose,
    mixed epistemic_status) plus pre-existing semantic summaries minted
    before entity extraction ran over them -- and measures coverage in
    dry-run only. Never opens or mutates any store outside tmp_path."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store = MemoryStore(path=tmp_path)

    record_ids: "list[UUID]" = []
    for i, (entity_bearing, epistemic_status, text) in enumerate(_CONVERSATIONAL_ROWS):
        result = capture_turn(
            store=store, cue="c", text=text,
            session_id=f"s{i}", role="user",
            epistemic_status=epistemic_status,
        )
        assert result["status"] == "inserted", result
        rid = UUID(result["record_id"])
        record_ids.append(rid)
        rec = store.get(rid)
        has_entity = any(t.startswith("entity:") for t in rec.tags)
        assert has_entity is entity_bearing, (
            "corpus construction assumption violated for row "
            f"{i}: expected entity-bearing={entity_bearing}"
        )

    legacy_rows = [
        _legacy_summary(100, "a summary capturing rollout discussion across earlier sessions"),
        _legacy_summary(101, "a summary noting the payments-worker incident review outcome"),
    ]
    for row in legacy_rows:
        store.insert(row)
        record_ids.append(row.id)
    flush_record_buffer(store)

    summary = backfill_salience(store, apply=False, store_path=tmp_path)
    assert summary["mode"] == "dry-run"
    assert summary["records_scanned"] == len(record_ids)

    entity_tag_rate = summary["entity_tag_rate"]
    aaak_nonempty_rate = summary["aaak_nonempty_rate"]
    salience_distribution_current = summary["salience_distribution_current"]

    assert entity_tag_rate == pytest.approx(5 / 12)
    # aaak_nonempty_rate is structural, not an entity-coverage signal: every real-route row
    # gets a populated skeleton regardless of entities; only the 2 legacy-shaped rows are empty.
    assert aaak_nonempty_rate == pytest.approx(10 / 12)
    assert salience_distribution_current == {"unflagged": 10, "notable": 2, "critical": 0}

    epistemic_status_distribution = Counter(
        store.get(rid).epistemic_status for rid in record_ids
    )

    corpus_shape = f"hermetic-fixture-shaped (self-built, n={len(record_ids)}, mixed episodic+semantic)"
    print(f"coverage_rate_corpus_shape={corpus_shape}")
    print(f"coverage_rate_entity_tag_rate={entity_tag_rate:.4f}")
    print(f"coverage_rate_aaak_nonempty_rate={aaak_nonempty_rate:.4f}")
    print(f"coverage_rate_salience_distribution_current={salience_distribution_current}")
    print(f"coverage_rate_epistemic_status_distribution={dict(epistemic_status_distribution)}")
