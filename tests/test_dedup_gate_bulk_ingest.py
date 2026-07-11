"""RED test for sequential bulk-like stream dedup.

No separate bulk-ingest API exists in the tree — no ``bulk_ingest``/
``direct_teach``/``textbook`` symbol anywhere in ``src/``. A future bulk-ingest
loop calling ``store.insert()`` sequentially per item already gets intra-batch
dedup for free, because ``insert()`` is the sole write choke point and the
write-time pattern-separation gate queries the CURRENT corpus (including
everything already committed earlier in the same loop) on every call. This
test exercises that property directly, with NO new bulk-insert code path — a
plain Python ``for`` loop over ``store.insert()``.

RED today where the stream is large enough that a near-dupe is pushed outside
the ANN top-k window by closer distractors (mirrors
test_dedup_gate_eval_set.py's ANN-total-miss mechanism) — the write-time
gate's near-dup branch has no exact-confirm stage yet.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _dedup_eval_fixture import (  # noqa: E402 — test helper; sys.path manipulated above
    EMBED_DIM,
    _make_embedding_at_cosine,
    _reference_embedding,
)

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
def test_small_stream_collapses_near_dupes_retains_distinct(driver, tmp_path, monkeypatch):
    """A small mixed near-dup/distinct stream, inserted sequentially via the
    sole write entry point (store.insert()), collapses near-dupes to one row
    each while every distinct item keeps its own row.

    Small enough that the ANN top-k prefilter alone suffices (no distractor
    contention) — a regression guard, may already pass today.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        # Three independent "paragraph groups", one axis pair each so they
        # never cross-contaminate cosine similarity with each other.
        stream = [
            # group 0: two near-dup paragraphs (cos 0.99) + one genuinely distinct
            _make_record(
                embedding=_reference_embedding(axis_index=0, embed_dim=EMBED_DIM),
                literal_surface="the exact-cosine authority is a resident float32 matrix",
            ),
            _make_record(
                embedding=_make_embedding_at_cosine(0.99, axis_index=0, embed_dim=EMBED_DIM),
                literal_surface="the exact-cosine authority is a resident float32 matrix, restated",
            ),
            # group 1: a genuinely distinct paragraph
            _make_record(
                embedding=_reference_embedding(axis_index=2, embed_dim=EMBED_DIM),
                literal_surface="never touch time machine on this machine",
            ),
            # group 2: two more near-dups (cos 0.97) + a distinct
            _make_record(
                embedding=_reference_embedding(axis_index=4, embed_dim=EMBED_DIM),
                literal_surface="the sensory tier formalizes the existing live jsonl mechanism",
            ),
            _make_record(
                embedding=_make_embedding_at_cosine(0.97, axis_index=4, embed_dim=EMBED_DIM),
                literal_surface="the sensory tier formalizes the existing live jsonl mechanism, again",
            ),
            _make_record(
                embedding=_reference_embedding(axis_index=6, embed_dim=EMBED_DIM),
                literal_surface="bge-small-en-v1.5 produces 384-dimensional embeddings",
            ),
        ]
        for rec in stream:
            store.insert(rec)

        tbl = store.db.open_table(RECORDS_TABLE)
        row_count = tbl.count_rows()
        # 4 independent groups (g0's anchor, g1's standalone distinct, g2's
        # anchor, g3's standalone distinct) each survive as exactly 1 row;
        # g0's and g2's near-dup companions collapse into their group anchor.
        expected = 4
        assert row_count == expected, (
            f"[driver={driver}] sequential stream must collapse near-dupes to 1 "
            f"row per group and retain every distinct group's own row; expected "
            f"{expected}, got {row_count}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_bulk_stream_ann_window_miss_still_collapses_via_exact_confirm(driver, tmp_path, monkeypatch):
    """A near-dupe pushed outside the ANN top-k window by closer distractors
    (a realistic bulk-ingest / direct-teaching failure mode: many similar
    chunks from the same source topic) still collapses via the exact-confirm
    stage, because exact_top_k scans the WHOLE corpus, not just the ANN's
    approximate top-k.

    RED today: the write-time gate's near-dup branch has no exact-confirm
    stage; a true near-dup outside the ANN top-cfg.top_k window is not
    caught.
    """
    _select_driver(driver, monkeypatch)
    store = _make_store(tmp_path)
    try:
        distractor_axis = 8
        anchor = _make_record(
            embedding=_reference_embedding(axis_index=distractor_axis, embed_dim=EMBED_DIM),
            literal_surface="chapter 3 discusses hippocampal replay during sleep consolidation",
        )
        store.insert(anchor)

        distractor_cosines = [0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.9935, 0.9931]
        for i, cos in enumerate(distractor_cosines):
            distractor = _make_record(
                embedding=_make_embedding_at_cosine(cos, axis_index=distractor_axis, embed_dim=EMBED_DIM),
                literal_surface=f"chapter {i} covers an unrelated neuroscience topic",
                never_merge=True,
            )
            store.insert(distractor)

        tbl = store.db.open_table(RECORDS_TABLE)
        assert tbl.count_rows() == 9

        near_dup_chunk = _make_record(
            embedding=_make_embedding_at_cosine(0.93, axis_index=distractor_axis, embed_dim=EMBED_DIM),
            literal_surface="chapter 3 discusses hippocampal replay during sleep consolidation, "
                             "repeated due to a chunking-boundary bug upstream",
        )
        store.insert(near_dup_chunk)

        tbl = store.db.open_table(RECORDS_TABLE)
        row_count = tbl.count_rows()
        assert row_count == 9, (
            f"[driver={driver}] a near-dup chunk pushed outside the ANN top-8 "
            f"window by 8 closer distractors must still collapse via the "
            f"exact-confirm stage (row count stays 9, not 10); got {row_count}"
        )
    finally:
        store.close()
