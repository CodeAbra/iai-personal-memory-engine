"""Reproduces the admission-window-narrow failure shape: a topically
relevant but differently-worded record crowded entirely out of the primary
raw-cosine ANN window by many near-duplicate, register-matched phrasings of
a short/vague cue -- at the REAL, unwidened production admission window
(never a test-harness-widened one).

Extends the three-session served-foresight harness (session setup,
real-embedder wiring, real capture-and-consolidate helpers) from
tests/test_baseline_knowledge_vs_chatter.py, but deliberately does NOT reuse
that harness's own calibration/order-bury helpers, which widen the
admission window for a different (ranking-only) study. Admission is never
mocked: every candidate set below comes from a real store.query_similar
call over a real embedder.

Construction-only, bounded attempts: a run that cannot construct a genuine
admission-window-narrow pair within the attempt cap is INCONCLUSIVE and
must be escalated, never a silently vacuous pass.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from tests.test_baseline_knowledge_vs_chatter import (
    SESSION_CHATTER,
    SESSION_KNOWLEDGE,
    SESSION_QUERY,
    _effective_admission_window,
    _select_driver,
)

# Deliberately no pytest.mark.bench here: unlike the harness this module
# extends, this reproduction must run under the plan default-gate command
# (no --bench flag) -- marking it bench-opt-in would silently skip every
# test and turn a real construction attempt into a vacuous green.

# A short, register-generic cue with no capitalized proper noun and no
# distinctive >=4-letter word: the derived-cue reserve lane's entity lane
# (capitalized tokens only) and rare-token lane (IDF-gated) must both find
# nothing to probe with, so the primary admission window is the ONLY path
# a candidate can reach the served pack through -- exactly the shape a
# short/vague real-world turn produces.
_QUERY_TEXT = "hey, is it done yet?"
_GREETINGS = [
    "hey", "hi", "yo", "hey there", "so hey", "hiya",
    "hey hey", "ok so", "alright so", "well hey", "yo yo", "hmm hey",
]
_ASKS = [
    "is it done yet",
    "is that done yet",
    "did it finish yet",
    "is this finished yet",
    "did that wrap up yet",
]

# Topically distant from the vague cue's own vocabulary on purpose: the
# on-topic relevance is that it directly ANSWERS a completion-status
# question, but it shares no distinctive token with the cue or the
# distractor pool, so no reserve lane can rescue it by keyword overlap --
# only the primary admission window could ever surface it.
_KNOWLEDGE_TEXT = (
    "The greenhouse irrigation controller passed its final acceptance test "
    "and ships to the client site next week."
)

_ATTEMPT_LEVELS = [24, 28, 32, 36, 40, 44, 48, 52]


def _distractor_texts() -> "list[str]":
    out = []
    for greeting in _GREETINGS:
        for ask in _ASKS:
            text = f"{greeting}, {ask}?"
            if text != _QUERY_TEXT:
                out.append(text)
    return out


_ALL_DISTRACTORS = _distractor_texts()
assert len(_ALL_DISTRACTORS) >= max(_ATTEMPT_LEVELS), (
    "distractor pool must cover every authored attempt level -- a mismatch "
    "silently truncates the highest attempts to a no-op"
)


@dataclass
class _AdmissionWindowAttempt:
    distractor_count: int
    target_in_window: bool
    target_wide_cos: "float | None"
    knowledge_id: UUID
    cue_embedding: list


def _run_admission_window_attempt(
    store: MemoryStore, level: int, window_k: int,
    *, directive: bool = False, salience_level: "str | None" = None,
) -> _AdmissionWindowAttempt:
    """directive/salience_level default to unflagged (this module's own
    red-before shape); a later green-after fixture re-runs this exact
    construction with one of them set, to measure whether a signalled
    record survives where an identical unflagged one does not."""
    result = capture_turn(
        store, cue="", text=_KNOWLEDGE_TEXT,
        tier="episodic", session_id=SESSION_KNOWLEDGE, role="user",
        live_turn=False, directive=directive, salience_level=salience_level,
    )
    flush_record_buffer(store)
    knowledge_id = UUID(result["record_id"])

    for text in _ALL_DISTRACTORS[:level]:
        capture_turn(
            store, cue="", text=text,
            tier="episodic", session_id=SESSION_CHATTER, role="user",
            live_turn=False,
        )
    flush_record_buffer(store)

    embedder = embedder_for_store(store)
    cue_embedding = list(embedder.embed(_QUERY_TEXT))

    primary = store.query_similar(cue_embedding, k=window_k)
    primary_ids = [rec.id for rec, _cos in primary]
    in_window = knowledge_id in primary_ids

    wide_cos = None
    if not in_window:
        # The production window alone cannot say how far outside it the
        # target sits -- a wide probe over the whole (small) corpus proves
        # the target is a real, reachable embedding-space neighbor and not
        # simply missing/malformed.
        wide = store.query_similar(cue_embedding, k=level + 1)
        wide_by_id = {rec.id: cos for rec, cos in wide}
        wide_cos = wide_by_id.get(knowledge_id)

    return _AdmissionWindowAttempt(
        distractor_count=level, target_in_window=in_window,
        target_wide_cos=wide_cos, knowledge_id=knowledge_id,
        cue_embedding=cue_embedding,
    )


def _construct_admission_window_probe(
    driver: str, *, directive: bool = False, salience_level: "str | None" = None,
) -> dict:
    """Iterate _ATTEMPT_LEVELS in order, at most len(_ATTEMPT_LEVELS) times,
    until the on-topic knowledge record is excluded from the real, unwidened
    production admission window. Once excluded, STOP tuning -- call
    refresh_pack exactly once on the winning attempt and record the served
    outcome as-measured. If the cap is exhausted without exclusion, record
    the closest attempt as a first-class INCONCLUSIVE outcome, never a
    fabricated one.

    directive/salience_level default to unflagged, reproducing this
    module's own red-before shape; a later fixture may pass one of them to
    measure whether an identical, otherwise-unflagged record survives once
    signalled."""
    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)

        window_k = _effective_admission_window()
        assert window_k == 24, (
            "this reproduction must run at the real, unwidened production "
            f"admission window (24); got {window_k} -- a widening override "
            "leaked into this reproduction"
        )

        base = Path(tempfile.mkdtemp(prefix=f"iai-admission-window-{driver}-"))
        attempts_log: "list[dict]" = []
        winner: "_AdmissionWindowAttempt | None" = None
        winner_store: "MemoryStore | None" = None

        for level in _ATTEMPT_LEVELS:
            store = MemoryStore(path=base / f"level-{level}")
            attempt = _run_admission_window_attempt(
                store, level, window_k,
                directive=directive, salience_level=salience_level,
            )
            attempts_log.append({
                "distractor_count": attempt.distractor_count,
                "target_in_window": attempt.target_in_window,
                "target_wide_cos": attempt.target_wide_cos,
            })
            if not attempt.target_in_window:
                winner = attempt
                winner_store = store
                break
            store.close()

        if winner is None:
            return {
                "outcome": "inconclusive",
                "reason": (
                    "the on-topic knowledge record stayed inside the "
                    "production admission window at every attempted "
                    "distractor count"
                ),
                "effective_admission_window": window_k,
                "attempts": attempts_log,
            }

        try:
            report = foresight.refresh_pack(
                winner_store, cue_text=_QUERY_TEXT,
                cue_embedding=winner.cue_embedding, session_id=SESSION_QUERY,
            )
        finally:
            winner_store.close()

        packed_ids_raw = report.get("packed_ids") or []
        packed_ids = [pid for pid in packed_ids_raw if not pid.startswith("q:")]
        knowledge_served = str(winner.knowledge_id) in packed_ids
        other_served = [pid for pid in packed_ids if pid != str(winner.knowledge_id)]
        knowledge_index = (
            packed_ids.index(str(winner.knowledge_id)) if knowledge_served else -1
        )

        return {
            "outcome": "constructed",
            "distractor_count": winner.distractor_count,
            "target_wide_cos": winner.target_wide_cos,
            "effective_admission_window": window_k,
            "knowledge_served": knowledge_served,
            "distractor_served_count": len(other_served),
            "packed_len": len(packed_ids_raw),
            "packed": report["packed"],
            "attempts": attempts_log,
            "packed_ids": packed_ids,
            "packed_ids_raw": packed_ids_raw,
            "knowledge_index": knowledge_index,
        }
    finally:
        mp.undo()


_PROBE_CACHE: "dict[tuple, dict]" = {}


def _get_admission_window_probe(
    driver: str, *, directive: bool = False, salience_level: "str | None" = None,
) -> dict:
    key = (driver, directive, salience_level)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = _construct_admission_window_probe(
            driver, directive=directive, salience_level=salience_level,
        )
    return _PROBE_CACHE[key]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_admission_window_excludes_far_register_record_at_production_defaults(driver):
    """Non-vacuous reproduction: a short/vague cue crowded with many
    near-duplicate register-matched phrasings excludes an on-topic but
    differently-worded knowledge record from the real 24-candidate
    production admission window entirely, and a register-matched
    distractor is served in its place. A run that cannot construct this
    within the bounded attempt cap is INCONCLUSIVE, recorded and failed
    loudly -- never silently treated as "no bug"."""
    result = _get_admission_window_probe(driver)
    print(json.dumps({"driver": driver, **result}, indent=2, default=str))

    if result["outcome"] == "inconclusive":
        pytest.fail(
            f"[{driver}] INCONCLUSIVE: could not construct a genuine "
            f"admission-window-narrow reproduction within the bounded "
            f"attempt cap -- escalate, do not treat as a refutation: "
            f"{result}"
        )

    assert not result["knowledge_served"], (
        f"[{driver}] the on-topic knowledge record must be ABSENT from the "
        f"served foresight pack for a non-vacuous admission-window "
        f"reproduction: {result}"
    )
    assert result["distractor_served_count"] > 0, (
        f"[{driver}] at least one register-matched distractor must SURFACE "
        f"on the served pack while the on-topic knowledge record is "
        f"excluded -- both-absent is degenerate and proves nothing: "
        f"{result}"
    )
