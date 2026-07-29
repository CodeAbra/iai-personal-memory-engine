from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from iai_mcp.retrieve import _SUPERSEDE_CAP_EPSILON, apply_supersede_cap
from iai_mcp.types import MemoryHit

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


def _hit(score: float, valid_to: "datetime | None" = None) -> MemoryHit:
    return MemoryHit(
        record_id=uuid4(),
        score=score,
        reason="test",
        literal_surface="x",
        adjacent_suggestions=[],
        valid_to=valid_to,
    )


def test_stale_src_capped_below_best_corrector() -> None:
    stale = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    corrector = _hit(1.1)
    other = _hit(0.5)
    outgoing = {str(stale.record_id): [str(corrector.record_id)]}

    apply_supersede_cap([stale, corrector, other], outgoing, now=NOW)

    assert stale.score < corrector.score
    assert abs(stale.score - (1.1 - _SUPERSEDE_CAP_EPSILON)) < 1e-9
    assert other.score == 0.5


def test_not_yet_stale_src_untouched() -> None:
    src = _hit(2.0, valid_to=NOW + timedelta(hours=1))
    corrector = _hit(1.0)
    outgoing = {str(src.record_id): [str(corrector.record_id)]}

    apply_supersede_cap([src, corrector], outgoing, now=NOW)

    assert src.score == 2.0


def test_historical_intent_skips_cap() -> None:
    stale = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    corrector = _hit(1.0)
    outgoing = {str(stale.record_id): [str(corrector.record_id)]}

    apply_supersede_cap(
        [stale, corrector], outgoing, now=NOW, cue_intent="historical_verbatim",
    )

    assert stale.score == 2.0


def test_corrector_absent_from_hits_leaves_src() -> None:
    stale = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    outgoing = {str(stale.record_id): [str(uuid4())]}

    apply_supersede_cap([stale], outgoing, now=NOW)

    assert stale.score == 2.0


def test_chain_caps_transitively() -> None:
    # A corrected by B, B corrected by C; both A and B are stale.
    a = _hit(3.0, valid_to=NOW - timedelta(hours=2))
    b = _hit(2.5, valid_to=NOW - timedelta(hours=1))
    c = _hit(1.0)
    outgoing = {
        str(a.record_id): [str(b.record_id)],
        str(b.record_id): [str(c.record_id)],
    }

    apply_supersede_cap([a, b, c], outgoing, now=NOW)

    assert c.score == 1.0
    assert b.score < c.score
    assert a.score < b.score


def test_src_already_below_corrector_untouched() -> None:
    stale = _hit(0.4, valid_to=NOW - timedelta(hours=1))
    corrector = _hit(1.0)
    outgoing = {str(stale.record_id): [str(corrector.record_id)]}

    apply_supersede_cap([stale, corrector], outgoing, now=NOW)

    assert stale.score == 0.4


def test_best_of_multiple_correctors_wins() -> None:
    stale = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    weak = _hit(0.3)
    strong = _hit(1.2)
    outgoing = {
        str(stale.record_id): [str(weak.record_id), str(strong.record_id)],
    }

    apply_supersede_cap([stale, weak, strong], outgoing, now=NOW)

    assert abs(stale.score - (1.2 - _SUPERSEDE_CAP_EPSILON)) < 1e-9


def test_corrector_outside_serving_window_leaves_src() -> None:
    # A legitimately matching stale hit must not sink under the noise floor
    # when its corrector is irrelevant to the cue (ranked below the window).
    stale = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    noise = [_hit(1.0 - 0.01 * i) for i in range(12)]
    corrector = _hit(0.05)
    outgoing = {str(stale.record_id): [str(corrector.record_id)]}

    apply_supersede_cap([stale, *noise, corrector], outgoing, now=NOW)

    assert stale.score == 2.0

    # Same shape but the corrector ranks inside the window -> cap fires.
    stale2 = _hit(2.0, valid_to=NOW - timedelta(hours=1))
    corrector2 = _hit(1.5)
    outgoing2 = {str(stale2.record_id): [str(corrector2.record_id)]}
    apply_supersede_cap([stale2, *noise, corrector2], outgoing2, now=NOW)
    assert stale2.score < corrector2.score


def test_empty_inputs_are_noops() -> None:
    assert apply_supersede_cap([], {"a": ["b"]}, now=NOW) == []
    hit = _hit(1.0, valid_to=NOW - timedelta(hours=1))
    apply_supersede_cap([hit], None, now=NOW)
    apply_supersede_cap([hit], {}, now=NOW)
    assert hit.score == 1.0
