"""End-to-end witness for ambient capture: a correction turn survives the
no-save path and is recallable, while a short chit-chat turn is dropped by
the drain's length floor -- neither requires the user (or Claude) to call
an explicit "save" tool, and no daemon source is touched.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from iai_mcp import core
from iai_mcp.capture import drain_deferred_captures, write_deferred_captures
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore


def _write_transcript(path: Path, turns: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(t) for t in turns) + "\n")


def test_correction_survives_no_save_chit_chat_dropped(tmp_path, monkeypatch):
    # _spool_root() resolves Path.home()/".iai-mcp" unconditionally -- a
    # scratch HOME isolates the deferred-captures spool from the real
    # operator home.
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    correction_text = "actually, use bge-small not e5, alice decided"
    chit_chat_text = "lol ok"  # 6 chars, below the 12-char capture-length floor

    transcript_path = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript_path,
        [
            {"type": "user", "message": {"role": "user", "content": correction_text}},
            {"type": "assistant", "message": {"role": "assistant", "content": chit_chat_text}},
        ],
    )

    out_path = write_deferred_captures(
        session_id="uat-ambient",
        transcript_path=transcript_path,
        cwd=str(tmp_path),
    )
    assert out_path.exists()

    store = MemoryStore(path=tmp_path / "store")
    counts = drain_deferred_captures(store)

    # The correction is genuinely new content -> inserted. The chit-chat
    # line clears the noise filter (it's not a structural marker) but is
    # too short to carry signal -- the drain's MIN_CAPTURE_LEN floor skips
    # it as "intentional", proving the drop happened by design, not by
    # accident.
    assert counts.get("events_inserted", 0) >= 1, counts
    assert counts.get("events_skipped_intentional", 0) >= 1, counts

    # Direct store scan: prove the chit-chat text never landed as a record,
    # rather than inferring the drop from an unrelated recall cue's miss.
    stored_ids = store.db.open_table("records").to_pandas()["id"].tolist()
    stored_surfaces = [store.get(UUID(rid)).literal_surface for rid in stored_ids]
    assert correction_text in stored_surfaces, (
        f"correction {correction_text!r} missing from store records: {stored_surfaces!r}"
    )
    assert chit_chat_text not in stored_surfaces, (
        f"chit-chat {chit_chat_text!r} leaked into store records: {stored_surfaces!r}"
    )

    # Belt-and-suspenders: the ambient drain writes an unembedded pending
    # row (served through recency independent of the embedding); fill the
    # real vector so the ANN-backed recall witness is exercised too.
    store.db.reembed_pending_rows(embedder_for_store(store))

    response = core.dispatch(
        store,
        "memory_recall",
        {
            "cue": "what embedding model did alice decide to use",
            "session_id": "-",
            "budget_tokens": 2000,
        },
    )
    hits = response.get("hits", [])
    surfaces = [h.get("literal_surface") for h in hits]

    assert correction_text in surfaces, (
        f"correction {correction_text!r} did not surface via no-save ambient "
        f"capture; got surfaces={surfaces!r}"
    )
