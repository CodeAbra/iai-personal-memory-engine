"""Permanent guards for the served-order relocation: an unflagged corpus
must be a byte-identical no-op with the reserve lane at its new position,
and a flagged record that also clears the primary admission window must be
served exactly once, rendered by the reserve lane, within the per-turn
ceiling.

Deviation from a literal before/after UUID comparison: capture_turn mints a
random id per record, so two independently constructed corpora can never
share literal record ids -- byte-identity is proved here on a SINGLE shared
store queried under two distinct session ids instead, which makes the two
arms' packed_ids directly and meaningfully comparable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp import foresight
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store
from tests.test_baseline_foresight_admission import (
    _QUERY_TEXT,
    _run_admission_window_attempt,
)
from tests.test_baseline_knowledge_vs_chatter import (
    _effective_admission_window,
    _select_driver,
)
from tests.test_value_signal_astra_probe import _MappedEmbedder, _mix, _unit


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_unflagged_corpus_is_byte_identical_with_reserve_lane_relocated(
    driver, tmp_path, monkeypatch,
):
    """On a single shared, unflagged corpus the relocated reserve lane's
    indexed discovery finds no salience/directive-tagged rows, so its new
    earlier position must not perturb the pack: the served ids are the same
    whether the lane is on (default) or forced off."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"fence-unflagged-{driver}")
    try:
        window_k = _effective_admission_window()
        attempt = _run_admission_window_attempt(store, 24, window_k)

        flagged_rows = list(store.iter_records(
            where="salience_level != 'unflagged' AND tombstoned_at IS NULL",
        ))
        assert not flagged_rows, (
            f"[{driver}] this fixture must be genuinely unflagged for the "
            f"no-op proof to mean anything: {flagged_rows}"
        )

        report_on = foresight.refresh_pack(
            store, cue_text=_QUERY_TEXT, cue_embedding=attempt.cue_embedding,
            session_id=f"fence-unflagged-on-{driver}",
        )
        print(json.dumps(
            {"driver": driver, "arm": "on", **report_on}, indent=2, default=str,
        ))

        monkeypatch.setenv(foresight.FORESIGHT_RESERVE_LANE_OFF_ENV, "1")
        report_off = foresight.refresh_pack(
            store, cue_text=_QUERY_TEXT, cue_embedding=attempt.cue_embedding,
            session_id=f"fence-unflagged-off-{driver}",
        )
        print(json.dumps(
            {"driver": driver, "arm": "off", **report_off}, indent=2, default=str,
        ))

        assert report_on["packed_ids"], (
            f"[{driver}] the ON arm must be non-vacuous (some distractor "
            f"served) or this comparison proves nothing: {report_on}"
        )
        assert report_off["packed_ids"], (
            f"[{driver}] the OFF arm must be non-vacuous: {report_off}"
        )
        assert report_on["packed_ids"] == report_off["packed_ids"], (
            f"[{driver}] the relocated (empty on this corpus) reserve lane "
            f"must not perturb the served pack: on={report_on['packed_ids']} "
            f"off={report_off['packed_ids']}"
        )
    finally:
        store.close()


def _flagged_rec(
    text: str, emb: "list[float]", *, salience_level: str, session_id: str,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        salience_level=salience_level,
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_flagged_in_window_record_served_once_with_reserve_marker(
    driver, tmp_path, monkeypatch,
):
    """A directive/salience-flagged record that ALSO clears the primary
    admission window must not be rendered twice: the reserve lane (now
    first) renders it with its own marker, and the primary lane's
    seen_surfaces dedup skips the later duplicate -- within the per-turn
    ceiling."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _select_driver(driver, monkeypatch)

    query_text = "so hey, anything new worth flagging on your end lately"
    flagged_text = (
        "alice's standing rule: always confirm the deploy window with the "
        "on-call engineer before shipping."
    )
    chatter_texts = [
        "yeah so anyway, nothing much new on my end, just been keeping busy",
        "honestly just the usual stuff, no real updates from me today",
    ]

    query_vec = _unit(0)
    flagged_vec = _mix(0, 1, 0.75)
    chatter_vecs = [_mix(0, 2, 0.62), _mix(0, 3, 0.61)]
    mapping = {query_text: query_vec, flagged_text: flagged_vec}
    for text, vec in zip(chatter_texts, chatter_vecs):
        mapping[text] = vec
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))

    capture_session = f"fence-flagged-capture-{driver}"
    query_session = f"fence-flagged-query-{driver}"

    store = MemoryStore(path=tmp_path / f"fence-flagged-{driver}")
    try:
        flagged = _flagged_rec(
            flagged_text, flagged_vec,
            salience_level="notable", session_id=capture_session,
        )
        store.insert(flagged)
        for text, vec in zip(chatter_texts, chatter_vecs):
            store.insert(_flagged_rec(
                text, vec, salience_level="unflagged", session_id=capture_session,
            ))
        flush_record_buffer(store)

        report = foresight.refresh_pack(
            store, cue_text=query_text, cue_embedding=query_vec,
            session_id=query_session,
        )
        print(json.dumps(
            {"driver": driver, **report}, indent=2, default=str,
        ))

        packed_ids = [pid for pid in report["packed_ids"] if not pid.startswith("q:")]
        flagged_id = str(flagged.id)
        occurrences = packed_ids.count(flagged_id)
        assert occurrences == 1, (
            f"[{driver}] the flagged in-window record must be served EXACTLY "
            f"once (no duplication across lanes): occurrences={occurrences} "
            f"packed_ids={packed_ids}"
        )
        ceiling = (
            foresight.FORESIGHT_MAX_ITEMS_DEFAULT
            + foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_DEFAULT
        )
        assert report["packed"] <= ceiling, (
            f"[{driver}] served count must not exceed the per-turn ceiling "
            f"({ceiling}): {report['packed']}"
        )

        pack = foresight.pack_path(store, query_session)
        pack_text = pack.read_text() if pack.is_file() else ""
        flagged_line = next(
            (ln for ln in pack_text.splitlines() if "on-call engineer" in ln), None,
        )
        assert flagged_line is not None, (
            f"[{driver}] the flagged record's rendered line was not found in "
            f"the published pack: {pack_text!r}"
        )
        assert "◆" in flagged_line, (
            f"[{driver}] the flagged record must render with the reserve "
            f"lane's own marker (the reserve renderer must win, not the "
            f"primary/derived-cue/assistant-tail renderer): {flagged_line!r}"
        )
    finally:
        store.close()
