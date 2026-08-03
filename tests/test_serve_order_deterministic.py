"""Served hit ordering is a pure function of (score, record_id).

Python's sort is stable, so a score-only key lets equal-scored hits keep
whatever candidate scan order produced them — the same cue could serve
different orderings across calls. ``sort_served_hits`` breaks ties on
``record_id`` and is the single sorter for every serve path.
"""
from __future__ import annotations

import random
from uuid import UUID

from iai_mcp.retrieve import MemoryHit, sort_served_hits


def _hit(record_id: str, score: float) -> MemoryHit:
    return MemoryHit(
        record_id=UUID(record_id),
        score=score,
        reason="test",
        literal_surface="x",
        adjacent_suggestions=[],
    )


def _ids(hits: list[MemoryHit]) -> list[str]:
    return [str(h.record_id) for h in hits]


def test_equal_scores_order_independent_of_scan_order() -> None:
    ids = [f"00000000-0000-0000-0000-{i:012x}" for i in range(8)]
    hits = [_hit(i, 0.5) for i in ids]
    rng = random.Random(7)
    baseline = _ids(sort_served_hits(list(hits)))
    for _ in range(20):
        shuffled = list(hits)
        rng.shuffle(shuffled)
        assert _ids(sort_served_hits(shuffled)) == baseline


def test_score_still_dominates_the_tiebreak() -> None:
    lo = _hit("00000000-0000-0000-0000-000000000001", 0.1)
    hi = _hit("ffffffff-ffff-ffff-ffff-ffffffffffff", 0.9)
    assert _ids(sort_served_hits([lo, hi])) == [str(hi.record_id), str(lo.record_id)]


def test_serve_paths_use_the_shared_sorter() -> None:
    import re
    from pathlib import Path

    score_only = re.compile(
        r"(?:\.sort\(|sorted\()[^\n]*key=lambda (\w+): \1\.score\b"
    )
    root = Path(__file__).resolve().parents[1] / "src" / "iai_mcp"
    offenders = []
    for name in ("pipeline.py", "retrieve.py", "core/__init__.py"):
        text = (root / name).read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if score_only.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "score-only sort on a serve path — equal scores would inherit scan "
        "order; route it through sort_served_hits or add the record_id "
        "tiebreak:\n  " + "\n  ".join(offenders)
    )
