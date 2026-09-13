"""Store-wide salience + entity/AAAK backfill (operator command).

Covers both mutation classes the sweep applies on ``--apply``: the
capture-time weighted-OR composite for ordinary rows, and the cluster-size
rule (existing ``cls_summary`` rows -> notable) for summaries minted before
that rule existed. Dry-run is the default and must never write.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

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


def _emb(i: int) -> list[float]:
    vec = [0.0] * EMBED_DIM
    vec[i % EMBED_DIM] = 1.0
    vec[(i * 37 + 11) % EMBED_DIM] = 0.5
    return vec


def _rec(
    i: int,
    *,
    text: str,
    tier: str = "episodic",
    tags: "list[str] | None" = None,
    directive: bool = False,
    epistemic_status: str = "unknown",
    salience_level: str = "unflagged",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        tags=list(tags or []),
        language="en",
        directive=directive,
        epistemic_status=epistemic_status,
        salience_level=salience_level,
    )


def _seed_world(store: MemoryStore):
    """One directive row (composite must promote), one entity-only row
    (composite must NOT promote), one pre-existing cls_summary row minted
    before the cluster-size rule (must be promoted by the tag rule alone),
    and one already-critical cls_summary row (monotonic floor)."""
    directive_row = _rec(
        0, text="deploy the release before the end of the day",
        directive=True,
    )
    entity_only_row = _rec(
        1, text="see the `nimbus-cluster` dashboard for details",
    )
    old_summary = _rec(
        2, text="alice and bob discussed rollout timing across sessions",
        tier="semantic", tags=["semantic", "cls_summary"],
    )
    critical_summary = _rec(
        3, text="a summary already marked critical by an earlier pass",
        tier="semantic", tags=["semantic", "cls_summary"],
        salience_level="critical",
    )
    for r in (directive_row, entity_only_row, old_summary, critical_summary):
        store.insert(r)
    flush_record_buffer(store)
    return directive_row, entity_only_row, old_summary, critical_summary


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dry_run_makes_zero_writes(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    directive_row, entity_only_row, old_summary, critical_summary = _seed_world(store)

    before = {
        r.id: (r.salience_level, list(r.tags), r.aaak_index)
        for r in (directive_row, entity_only_row, old_summary, critical_summary)
    }

    summary = backfill_salience(store, apply=False)
    assert summary["mode"] == "dry-run"
    assert summary["snapshot_dir"] is None
    assert summary["salience_would_promote"] >= 2, (
        "the directive row and the old summary must both be counted as would-promote"
    )
    assert summary["salience_promoted"] == 0

    for rid, (level, tags, aaak) in before.items():
        row = store.get(rid)
        assert row is not None
        assert row.salience_level == level, "dry-run must never write salience_level"
        assert row.tags == tags, "dry-run must never write tags"
        assert row.aaak_index == aaak, "dry-run must never write aaak_index"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_composite_promotes_directive_row_entity_only_stays_unflagged(
    driver, tmp_path, monkeypatch,
):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    directive_row, entity_only_row, _old_summary, _critical_summary = _seed_world(store)

    summary = backfill_salience(store, apply=True, store_path=tmp_path)
    assert summary["mode"] == "apply"

    promoted = store.get(directive_row.id)
    assert promoted is not None
    assert promoted.salience_level == "notable", (
        "a directive row must reach notable via the composite"
    )

    entity_only = store.get(entity_only_row.id)
    assert entity_only is not None
    assert entity_only.salience_level == "unflagged", (
        "an entity-tags-only row must stay unflagged (0.5 < 1.0 threshold)"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_promotes_existing_cls_summary_via_cluster_size_rule(
    driver, tmp_path, monkeypatch,
):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _directive_row, _entity_only_row, old_summary, _critical_summary = _seed_world(store)

    backfill_salience(store, apply=True, store_path=tmp_path)

    row = store.get(old_summary.id)
    assert row is not None
    assert row.epistemic_status == "unknown"
    assert row.directive is False
    assert row.salience_level == "notable", (
        "an existing cls_summary row with directive=False/epistemic_status=unknown "
        "must be promoted to notable by the cluster-size rule, not left unflagged "
        "by the composite alone -- this is the exact regression the composite-only "
        "path would cause"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_never_lowers_an_already_higher_mark(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _directive_row, _entity_only_row, _old_summary, critical_summary = _seed_world(store)

    backfill_salience(store, apply=True, store_path=tmp_path)

    row = store.get(critical_summary.id)
    assert row is not None
    assert row.salience_level == "critical", (
        "a row already above the target level must never be lowered"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_metadata_only_literal_surface_byte_unchanged(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    seeded = _seed_world(store)
    before = {r.id: r.literal_surface for r in seeded}

    backfill_salience(store, apply=True, store_path=tmp_path)

    for rid, text in before.items():
        row = store.get(rid)
        assert row is not None
        assert row.literal_surface == text, "the sweep must never touch literal_surface"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_writes_snapshot_before_mutating(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_world(store)

    summary = backfill_salience(store, apply=True, store_path=tmp_path)
    assert summary["snapshot_dir"] is not None
    assert Path(summary["snapshot_dir"]).exists()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_idempotent_second_apply_run_is_noop(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _directive_row, entity_only_row, _old_summary, _critical_summary = _seed_world(store)

    first = backfill_salience(store, apply=True, store_path=tmp_path)
    assert first["salience_promoted"] >= 2

    tags_after_first = list(store.get(entity_only_row.id).tags)

    second = backfill_salience(store, apply=True, store_path=tmp_path)
    assert second["salience_would_promote"] == 0
    assert second["salience_promoted"] == 0
    assert second["entity_backfill"]["records_written"] == 0

    tags_after_second = list(store.get(entity_only_row.id).tags)
    assert tags_after_second == tags_after_first, (
        "a second run must not grow the tag set (no duplicate tags)"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_apply_promotion_invalidates_graph_cache(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    # A row with no extractable entities and no existing tags keeps the
    # nested entity-anchor sweep a no-op, so any cache invalidation
    # observed here is attributable to the salience-promotion branch
    # under test, not the entity/AAAK sweep's own cache guard.
    directive_row = _rec(
        0, text="deploy the release before the end of the day", directive=True,
    )
    store.insert(directive_row)
    flush_record_buffer(store)
    stale_cache = tmp_path / "runtime_graph_cache.json"
    stale_cache.write_text("{}")

    summary = backfill_salience(store, apply=True, store_path=tmp_path)
    assert summary["entity_backfill"]["graph_cache_invalidated"] is False, (
        "the entity/AAAK sweep must stay a no-op for this seed, or it would "
        "invalidate the cache itself and the assertion below would not "
        "isolate the salience-promotion branch"
    )
    assert summary["salience_promoted"] >= 1
    assert summary["graph_cache_invalidated"] is True
    assert not stale_cache.exists()


def test_cli_dry_run_prints_would_change_and_resulting_numbers(tmp_path, capsys):
    from iai_mcp.cli import _build_parser

    store = MemoryStore(path=tmp_path)
    directive_row, _entity_only_row, _old_summary, _critical_summary = _seed_world(store)

    parser = _build_parser()
    args = parser.parse_args(["salience-backfill", "--store-path", str(tmp_path)])
    rc = args.func(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert "salience-backfill [dry-run]" in out
    assert "salience would-promote:" in out
    assert "salience distribution after:" in out
    assert "AAAK non-empty rate (current):" in out
    assert "entity tag rate (current):" in out

    unchanged = store.get(directive_row.id)
    assert unchanged is not None
    assert unchanged.salience_level == "unflagged", (
        "dry-run via the CLI must not mutate the store"
    )


def test_cli_apply_triggers_snapshot_and_write_path(tmp_path, capsys):
    from iai_mcp.cli import _build_parser

    store = MemoryStore(path=tmp_path)
    directive_row, _entity_only_row, _old_summary, _critical_summary = _seed_world(store)

    parser = _build_parser()
    args = parser.parse_args(
        ["salience-backfill", "--apply", "--store-path", str(tmp_path)]
    )
    rc = args.func(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert "salience-backfill [apply]" in out
    assert "snapshot directory:" in out
    assert "AAAK non-empty rate (resulting):" in out
    assert "entity tag rate (resulting):" in out

    promoted = store.get(directive_row.id)
    assert promoted is not None
    assert promoted.salience_level == "notable"
