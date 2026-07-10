"""RED tests for the labelled eval-set assertions: zero false-merge, and
strictly-stronger-than-the-old-0.95-threshold near-dup detection.

Exercises the write-time pattern-separation gate (``_pattern_separation_gate_with_hits``,
called unconditionally from ``store.insert()``) against the labelled eval set in
``tests/_dedup_eval_fixture.py``. Every assertion reads the ACTUAL records-table
row count, never the ``pattern_separation_pass`` event's ``action`` field alone —
the event fires with ``dry_run_mode=True`` even when nothing was actually skipped.

RED today: the "strengthened" assertions (the (near_dup_threshold, 0.95)
band) depend on a two-stage exact-confirm the write-time gate does not yet
have — the ANN-approximate top-8 prefilter alone is not guaranteed to surface
every candidate in that band the way an exact scan would. The distinct-stays-
distinct and verbatim-byte-identity assertions are regression guards; they may
already pass against current code.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _dedup_eval_fixture import (  # noqa: E402 — test helper; sys.path manipulated above
    DEDUP_EVAL_PAIRS,
    DISTINCT_PAIRS,
    STRICTLY_STRONGER_THAN_95_PAIRS,
    EMBED_DIM,
    _make_embedding_at_cosine,
    _reference_embedding,
)

from iai_mcp.events import query_events
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _make_record(
    *, embedding: list[float], literal_surface: str, never_merge: bool = False,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )


@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    # MANDATORY: dry-run defaults True under pytest; the SKIP row-count effect
    # is invisible otherwise.
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("pair", DEDUP_EVAL_PAIRS, ids=lambda p: p.name)
def test_eval_pair_row_count_matches_verdict(driver, pair, tmp_path, monkeypatch):
    """Each labelled pair's actual row-count effect matches its expected verdict.

    "duplicate" => SKIP-merge, row count unchanged (still 1).
    "distinct"  => INSERT, row count becomes 2.

    RED for the strengthened-merge band today: an ANN-approximate-only gate's
    top-8 prefilter may not treat a (near_dup_threshold, 0.95)-band candidate
    as strongly as an exact-confirm stage would.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        anchor = _make_record(
            embedding=pair.anchor_embedding, literal_surface=pair.anchor_surface,
        )
        store.insert(anchor)

        candidate = _make_record(
            embedding=pair.candidate_embedding, literal_surface=pair.candidate_surface,
        )
        store.insert(candidate)

        tbl = store.db.open_table(RECORDS_TABLE)
        row_count = tbl.count_rows()

        if pair.verdict == "duplicate":
            assert row_count == 1, (
                f"[driver={driver}] pair={pair.name} cos={pair.cosine} labelled "
                f"'duplicate' must SKIP-merge (row count 1); got {row_count}"
            )
        else:
            assert row_count == 2, (
                f"[driver={driver}] pair={pair.name} cos={pair.cosine} labelled "
                f"'distinct' must INSERT as its own row (row count 2); got {row_count}"
            )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_zero_false_merge_across_distinct_pairs(driver, tmp_path, monkeypatch):
    """Every 'distinct'-labelled pair produces TWO rows (false-merge rate == 0)."""
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        for pair in DISTINCT_PAIRS:
            anchor = _make_record(
                embedding=pair.anchor_embedding, literal_surface=pair.anchor_surface,
            )
            store.insert(anchor)
            candidate = _make_record(
                embedding=pair.candidate_embedding, literal_surface=pair.candidate_surface,
            )
            store.insert(candidate)

            tbl = store.db.open_table(RECORDS_TABLE)
            all_rows = tbl.count_rows()
            assert all_rows > 0

        events = query_events(store, kind="pattern_separation_pass", limit=1000)
        skip_events = [e for e in events if e["data"].get("action") == "skip"]
        assert not skip_events, (
            f"[driver={driver}] false-merge rate must be 0: distinct-labelled "
            f"pairs must never SKIP-merge; got {len(skip_events)} skip events: "
            f"{[e['data'] for e in skip_events]}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_strictly_stronger_than_old_capture_threshold(driver, tmp_path, monkeypatch):
    """A (near_dup_threshold, 0.95)-band pair the OLD 0.95 capture
    threshold would have INSERTED is SKIP-merged by the strengthened
    write-time gate — this is the trivial-corpus case where the ANN top-k
    prefilter already sees the true near-dup (no distractors to push it out
    of the window). This assertion is a regression guard: it may already
    pass today (the ANN prefilter alone suffices at tiny corpus size), and
    it MUST still pass once the exact-confirm stage is wired in.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        assert STRICTLY_STRONGER_THAN_95_PAIRS, "eval set must have a strengthened-band pair"
        for pair in STRICTLY_STRONGER_THAN_95_PAIRS:
            assert pair.cosine < 0.95, (
                f"pair={pair.name} must be BELOW the old 0.95 capture threshold "
                f"to demonstrate strengthening, got cos={pair.cosine}"
            )
            anchor = _make_record(
                embedding=pair.anchor_embedding, literal_surface=pair.anchor_surface,
            )
            store.insert(anchor)
            candidate = _make_record(
                embedding=pair.candidate_embedding, literal_surface=pair.candidate_surface,
            )
            store.insert(candidate)

            tbl = store.db.open_table(RECORDS_TABLE)
            row_count = tbl.count_rows()
            assert row_count == 1, (
                f"[driver={driver}] strictly-stronger-than-0.95 evidence: "
                f"pair={pair.name} cos={pair.cosine} is in the "
                f"(near_dup_threshold, 0.95) band that the OLD 0.95 capture "
                f"threshold would have INSERTED as a new row; the strengthened "
                f"write-time gate must SKIP-merge it (row count 1); got {row_count}"
            )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ann_prefilter_total_miss_still_caught_by_exact_confirm(driver, tmp_path, monkeypatch):
    """The actual failure mode this gate must close: a true near-dup that
    the ANN top-k prefilter's approximate window misses entirely (pushed
    out of the top-``cfg.top_k`` window by closer distractor records) is still
    exact-confirmed and SKIP-merged, because the exact-cosine authority scans
    the WHOLE corpus, not just the ANN's approximate top-k.

    Construction: seed ``cfg.top_k`` (default 8) distractor records strictly
    CLOSER to the anchor than the true near-dup candidate (cosines between
    the candidate's cosine and 1.0), so the ANN top-k prefilter's window is
    fully occupied by distractors and the true candidate falls outside it —
    while the exact-confirm scan (over the full corpus) still finds it.

    RED today: the write-time gate's near-dup branch only consults
    ``hits = self.query_similar(..., k=cfg.top_k)`` — an ANN top-k query —
    with no second-stage exact-confirm. When the true near-dup is pushed
    outside that top-k window, today's code treats ``hits[0]`` (a distractor)
    as the top hit; if that distractor's cosine is also >= near_dup_threshold,
    the record collapses into the WRONG neighbour rather than exact-confirming
    against the ANN candidate specifically evaluated — or, if distractors
    themselves are all above threshold, the candidate is never compared
    against its true near-dup at all. This test targets the exact-confirm
    mechanism directly, independent of trivial-corpus behaviour.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        from _dedup_eval_fixture import _reference_embedding, _make_embedding_at_cosine

        distractor_axis = 16
        anchor = _make_record(
            embedding=_reference_embedding(axis_index=distractor_axis, embed_dim=EMBED_DIM),
            literal_surface="the write-time gate is the universal choke point for every insert",
        )
        store.insert(anchor)

        # 8 distractors at cosines strictly between the true candidate's
        # cosine (0.93) and 1.0, all pointing at the SAME anchor axis so they
        # compete for the ANN top-k window ahead of the true candidate.
        distractor_cosines = [0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.9935, 0.9931]
        for i, cos in enumerate(distractor_cosines):
            distractor = _make_record(
                embedding=_make_embedding_at_cosine(cos, axis_index=distractor_axis, embed_dim=EMBED_DIM),
                literal_surface=f"distractor record {i} unrelated to the anchor's actual content",
                never_merge=True,  # distractors must never merge with each other or the anchor
            )
            store.insert(distractor)

        tbl = store.db.open_table(RECORDS_TABLE)
        row_count_before_candidate = tbl.count_rows()
        assert row_count_before_candidate == 9, (
            f"expected anchor + 8 distractors = 9 rows before the true candidate; "
            f"got {row_count_before_candidate}"
        )

        true_candidate = _make_record(
            embedding=_make_embedding_at_cosine(0.93, axis_index=distractor_axis, embed_dim=EMBED_DIM),
            literal_surface="the write-time gate is the universal choke point for every insert, roughly",
        )
        store.insert(true_candidate)

        tbl = store.db.open_table(RECORDS_TABLE)
        row_count_after = tbl.count_rows()
        assert row_count_after == 9, (
            f"[driver={driver}] the true near-dup candidate (cos=0.93 to the anchor, "
            f"in the near_dup_threshold..0.95 band) is pushed outside the ANN top-8 "
            f"window by 8 closer distractors; the exact-confirm stage must still "
            f"catch it and SKIP-merge (row count stays 9, not 10); got {row_count_after}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_verbatim_literal_surface_untouched_after_merge(driver, tmp_path, monkeypatch):
    """After a MERGE, the surviving record's literal_surface is byte-identical
    to what was first stored — mirrors test_promotion_preserves_literal_surface.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        anchor_surface = "alice prefers tea over coffee"
        anchor = _make_record(
            embedding=_reference_embedding(), literal_surface=anchor_surface,
        )
        store.insert(anchor)
        anchor_id = anchor.id

        near_dup_embedding = _make_embedding_at_cosine(0.99)
        candidate = _make_record(
            embedding=near_dup_embedding,
            literal_surface="alice prefers tea over coffee",
        )
        store.insert(candidate)

        record = store.get(anchor_id)
        assert record is not None
        assert record.literal_surface == anchor_surface, (
            f"[driver={driver}] surviving record's literal_surface must stay "
            f"byte-identical after a SKIP-merge; got {record.literal_surface!r}, "
            f"expected {anchor_surface!r}"
        )
    finally:
        store.close()
