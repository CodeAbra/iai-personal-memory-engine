"""Incident regression guard, distinct from the generic served-order gate:
proves the standing-rule-vs-chatter shape specifically -- a directive-marked
standing rule (captured via REAL capture_turn(directive=True), promoted by
the REAL classify_salience path, never a hand-set salience_level) is served
AHEAD of phrasing-similar chatter for the SAME relevant action, on the
`<iai-mcp-foresight>` push surface, at the real unwidened production
admission window. Admission is never mocked: every candidate set below
comes from a real store.query_similar call over a real embedder.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from tests.test_baseline_knowledge_vs_chatter import (
    _effective_admission_window,
    _select_driver,
)

_RULE_SESSION = "standing-rule-capture-session"
_CHATTER_SESSION = "standing-rule-chatter-session"
_QUERY_SESSION = "standing-rule-query-session"

# A short, register-generic query about the SAME action the rule governs,
# sharing no distinctive token with the rule's own wording -- mirrors the
# live incident's shape (a phatic question about the current action, not
# the rule itself).
_QUERY_TEXT = "yo, is this ready to ship or what"

_STANDING_RULE_TEXT = (
    "alice's standing rule: always confirm the deploy window with the "
    "on-call engineer before shipping."
)

_GREETINGS = [
    "hey", "hi", "yo", "hey there", "so hey", "hiya",
    "hey hey", "ok so", "alright so", "well hey", "yo yo", "hmm hey",
]
_ASKS = [
    "are we shipping today",
    "are we deploying soon",
    "is the deploy happening yet",
    "did we ship that yet",
    "are we good to ship this",
]

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


def _construct_standing_rule_probe(driver: str) -> dict:
    """Iterate _ATTEMPT_LEVELS until the standing rule is excluded from the
    real, unwidened production admission window, exactly the sibling
    admission-window-narrow construction's own methodology (real captures,
    real embedder, real store.query_similar) -- self-contained here because
    this incident's rule/chatter text is distinct from the sibling gate's."""
    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)

        window_k = _effective_admission_window()
        assert window_k == 24, (
            "this reproduction must run at the real, unwidened production "
            f"admission window (24); got {window_k} -- a widening override "
            "leaked into this reproduction"
        )

        base = Path(tempfile.mkdtemp(prefix=f"iai-standing-rule-{driver}-"))
        attempts_log: "list[dict]" = []
        winner_store: "MemoryStore | None" = None
        winner_level = None
        winner_cue = None
        winner_rule_id: "UUID | None" = None

        for level in _ATTEMPT_LEVELS:
            store = MemoryStore(path=base / f"level-{level}")
            result = capture_turn(
                store, cue="", text=_STANDING_RULE_TEXT,
                tier="episodic", session_id=_RULE_SESSION, role="user",
                live_turn=False, directive=True,
            )
            flush_record_buffer(store)
            rule_id = UUID(result["record_id"])

            # Discriminating precondition: the reserve lane only discovers
            # records whose salience_level is not "unflagged" -- if the real
            # classify_salience path did not promote this capture, the rest
            # of this probe would measure nothing about the relocation.
            rule_rec = store.get(rule_id)
            assert rule_rec is not None, "captured standing rule vanished"
            assert rule_rec.salience_level != "unflagged", (
                "capture_turn(directive=True) must promote salience_level "
                f"off 'unflagged' via the real classify_salience path; got "
                f"{rule_rec.salience_level!r} -- without this promotion the "
                "reserve lane has no signal to discover, independent of any "
                "served-order relocation"
            )

            for text in _ALL_DISTRACTORS[:level]:
                capture_turn(
                    store, cue="", text=text,
                    tier="episodic", session_id=_CHATTER_SESSION, role="user",
                    live_turn=False,
                )
            flush_record_buffer(store)

            embedder = embedder_for_store(store)
            cue_embedding = list(embedder.embed(_QUERY_TEXT))

            primary = store.query_similar(cue_embedding, k=window_k)
            primary_ids = [rec.id for rec, _cos in primary]
            in_window = rule_id in primary_ids
            attempts_log.append({
                "distractor_count": level, "target_in_window": in_window,
            })
            if not in_window:
                winner_store = store
                winner_level = level
                winner_cue = cue_embedding
                winner_rule_id = rule_id
                break
            store.close()

        if winner_store is None:
            return {
                "outcome": "inconclusive",
                "reason": (
                    "the standing rule stayed inside the production "
                    "admission window at every attempted distractor count"
                ),
                "effective_admission_window": window_k,
                "attempts": attempts_log,
            }

        try:
            report = foresight.refresh_pack(
                winner_store, cue_text=_QUERY_TEXT,
                cue_embedding=winner_cue, session_id=_QUERY_SESSION,
            )
        finally:
            winner_store.close()

        packed_ids_raw = report.get("packed_ids") or []
        packed_ids = [pid for pid in packed_ids_raw if not pid.startswith("q:")]
        rule_served = str(winner_rule_id) in packed_ids
        rule_index = packed_ids.index(str(winner_rule_id)) if rule_served else -1
        distractor_indices = [
            i for i, pid in enumerate(packed_ids) if pid != str(winner_rule_id)
        ]

        return {
            "outcome": "constructed",
            "distractor_count": winner_level,
            "effective_admission_window": window_k,
            "rule_served": rule_served,
            "rule_index": rule_index,
            "distractor_served_count": len(distractor_indices),
            "min_distractor_index": min(distractor_indices) if distractor_indices else None,
            "packed_len": len(packed_ids_raw),
            "packed": report["packed"],
            "attempts": attempts_log,
        }
    finally:
        mp.undo()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_standing_rule_served_ahead_of_phrasing_similar_chatter(driver):
    """The incident shape: a directive-marked standing rule, excluded from
    the real admission window on raw cosine alone, is served AHEAD of every
    phrasing-similar chatter distractor for the same relevant action, with
    chatter still present and the per-turn ceiling held."""
    result = _construct_standing_rule_probe(driver)
    print(json.dumps({"driver": driver, **result}, indent=2, default=str))

    if result["outcome"] == "inconclusive":
        pytest.fail(
            f"[{driver}] INCONCLUSIVE: could not construct a genuine "
            f"standing-rule-excluded reproduction within the bounded "
            f"attempt cap -- escalate, do not treat as a refutation: "
            f"{result}"
        )

    assert result["rule_served"], (
        f"[{driver}] the directive-marked standing rule must be served: "
        f"{result}"
    )
    assert result["distractor_served_count"] > 0, (
        f"[{driver}] phrasing-similar chatter must remain present alongside "
        f"the served rule -- both-served, not rule-displacing-chatter: "
        f"{result}"
    )
    min_distractor_index = result["min_distractor_index"]
    assert min_distractor_index is not None, (
        f"[{driver}] no distractor index recorded despite a non-zero "
        f"distractor count: {result}"
    )
    assert result["rule_index"] < min_distractor_index, (
        f"[{driver}] the standing rule must be served AHEAD of every "
        f"phrasing-similar chatter distractor (rule_index < every "
        f"distractor index): {result}"
    )
    ceiling = (
        foresight.FORESIGHT_MAX_ITEMS_DEFAULT
        + foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_DEFAULT
    )
    assert result["packed"] <= ceiling, (
        f"[{driver}] served count must not exceed the per-turn ceiling "
        f"({ceiling}): {result}"
    )
