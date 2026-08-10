"""Hermetic regression tests for the turn-capture ceiling.

Verifies that a transcript with more than 200 turns is captured in full
and that re-running capture on the same transcript inserts no duplicates.
"""
from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)

SESSION_ID = "sess-test"
_N_TURNS = 250  # deliberately above the old 200-turn cap


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-capture-ceiling-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _make_transcript(path: Path, n_turns: int = _N_TURNS) -> Path:
    """Write a JSONL transcript with n_turns alternating user/assistant turns.

    Each turn gets a distinct UUID so the idem key uses source_uuid.
    """
    transcript_path = path / "transcript.jsonl"
    lines = []
    base_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(1, n_turns + 1):
        role = "user" if i % 2 == 1 else "assistant"
        ts = base_ts.replace(second=i % 60, minute=i // 60 % 60, hour=i // 3600 % 24)
        turn = {
            "type": role,
            "uuid": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "sessionId": SESSION_ID,
            "message": {
                "role": role,
                "content": f"Turn {i} — {role} text for ceiling test",
            },
        }
        lines.append(json.dumps(turn))
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript_path


def _count_episodic_records(store) -> int:
    """Return the number of active episodic records in the store."""
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND tier = 'episodic'"
        ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Task 0 tests — these MUST FAIL before the ceiling is raised (RED gate)
# ---------------------------------------------------------------------------

def test_capture_transcript_beyond_200(iai_home, tmp_path):
    """capture_transcript must store all 250 turns, not just the first 200."""
    from iai_mcp.capture import capture_transcript

    transcript = _make_transcript(tmp_path)
    store = _open_store()

    counts = capture_transcript(store, transcript, session_id=SESSION_ID)

    total_captured = counts["inserted"] + counts["reinforced"]
    assert total_captured == _N_TURNS, (
        f"Expected {_N_TURNS} turns captured; got {total_captured}. "
        f"counts={counts!r}. Turns 201+ are being silently dropped — "
        f"this violates the lossless verbatim-recall invariant."
    )

    # Spot-check verbatim: turn 250 (the last one) must be in the store.
    last_turn_text = f"Turn {_N_TURNS} — assistant text for ceiling test"
    db_count = _count_episodic_records(store)
    assert db_count >= _N_TURNS, (
        f"Store holds only {db_count} episodic records; expected at least {_N_TURNS}."
    )

    # Confirm literal_surface is verbatim for a turn past the old cap.
    all_records = store.all_records()
    late_records = [
        r for r in all_records
        if r.literal_surface and last_turn_text in r.literal_surface
    ]
    assert len(late_records) >= 1, (
        f"Turn {_N_TURNS} literal_surface not found in store. "
        f"literal_surface must be verbatim transcript text, never paraphrased."
    )


def test_deferred_capture_beyond_200(iai_home, tmp_path):
    """write_deferred_captures must write all 250 turns to the deferred file."""
    from iai_mcp.capture import write_deferred_captures

    transcript = _make_transcript(tmp_path)
    out_path = write_deferred_captures(
        session_id=SESSION_ID,
        transcript_path=transcript,
        cwd="/tmp/test",
    )

    assert out_path.exists(), f"Deferred capture file not created at {out_path}"
    lines = out_path.read_text(encoding="utf-8").splitlines()

    # First line is the header; the rest are turn events.
    events = [json.loads(ln) for ln in lines[1:] if ln.strip()]
    assert len(events) == _N_TURNS, (
        f"Expected {_N_TURNS} deferred events; got {len(events)}. "
        f"write_deferred_captures is truncating at the old 200-turn cap."
    )


def test_capture_idempotent_after_cap_raise(iai_home, tmp_path):
    """Re-running capture on the same transcript adds zero new records (SHA256 dedup)."""
    from iai_mcp.capture import capture_transcript

    transcript = _make_transcript(tmp_path)
    store = _open_store()

    # First pass — capture all turns.
    counts_first = capture_transcript(store, transcript, session_id=SESSION_ID)
    total_first = counts_first["inserted"] + counts_first["reinforced"]
    assert total_first == _N_TURNS, (
        f"First pass: expected {_N_TURNS} turns; got {total_first}. "
        f"counts={counts_first!r}"
    )

    count_after_first = _count_episodic_records(store)

    # Second pass — must add zero new records.
    counts_second = capture_transcript(store, transcript, session_id=SESSION_ID)
    count_after_second = _count_episodic_records(store)

    assert count_after_second == count_after_first, (
        f"Second capture pass inserted {count_after_second - count_after_first} "
        f"extra records; expected 0. "
        f"The SHA256 idem dedup must prevent duplicates on re-capture. "
        f"counts_second={counts_second!r}"
    )
    assert counts_second.get("reinforced", 0) == _N_TURNS, (
        f"Second pass must reinforce all {_N_TURNS} turns (not re-insert). "
        f"counts_second={counts_second!r}"
    )


def test_capture_turn_concurrent_drains_do_not_duplicate(iai_home, tmp_path):
    """Concurrent capture_turn() calls for the same turn (simulating the daemon
    draining several Stop-hook full-transcript replays on separate
    asyncio.to_thread workers at once) must serialize the dedup
    check-then-insert and produce exactly one record."""
    import threading
    import time as _time

    from iai_mcp import capture as capture_mod

    store = _open_store()
    text = "Duplicate turn raced by concurrent drains for race regression test"
    source_uuid = str(uuid.uuid4())
    ts = "2026-07-01T00:00:00Z"

    n_threads = 6
    state_lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    original_find = store.find_record_by_tag

    def tracking_find(tag):
        nonlocal in_flight, max_in_flight
        with state_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        _time.sleep(0.05)  # widen the race window the bug needed
        try:
            return original_find(tag)
        finally:
            with state_lock:
                in_flight -= 1

    store.find_record_by_tag = tracking_find

    results: list[dict] = []
    results_lock = threading.Lock()

    def worker():
        r = capture_mod.capture_turn(
            store,
            cue="race test",
            text=text,
            tier="episodic",
            session_id=SESSION_ID,
            role="user",
            ts=ts,
            source_uuid=source_uuid,
        )
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == n_threads, f"Not all threads completed: {results!r}"
    assert max_in_flight == 1, (
        f"max_in_flight={max_in_flight}; dedup check-then-insert is not serialized "
        f"across threads -- this is the race that produced duplicate episodic records"
    )

    inserted = [r for r in results if r["status"] == "inserted"]
    reinforced = [r for r in results if r["status"] == "reinforced"]
    assert len(inserted) == 1, f"Expected exactly 1 insert, got {len(inserted)}: {results!r}"
    assert len(reinforced) == n_threads - 1, f"results={results!r}"

    count = _count_episodic_records(store)
    assert count == 1, f"Expected exactly 1 episodic record in the store, found {count}"


# ---------------------------------------------------------------------------
# Additive `provenance_extra` kwarg on capture_turn
# ---------------------------------------------------------------------------

def test_provenance_extra_additive_kwarg(iai_home):
    """capture_turn(..., provenance_extra=D) appends D as a second entry
    after the spine's {ts, cue, session_id, role} entry — ordered, length 2.
    """
    from uuid import UUID

    from iai_mcp.capture import capture_turn

    store = _open_store()
    extra = {"source": "upload", "filename": "a.txt", "chunk_index": 0}
    result = capture_turn(
        store,
        cue="ingest probe x",
        text="some content here long enough to clear the min length gate",
        session_id="s1",
        role="user",
        provenance_extra=extra,
    )
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    prov = rec.provenance
    assert isinstance(prov, list)
    assert len(prov) == 2, f"expected 2 provenance entries, got {prov!r}"
    spine = prov[0]
    assert set(spine.keys()) == {"ts", "cue", "session_id", "role"}
    assert spine["cue"] == "ingest probe x"
    assert spine["session_id"] == "s1"
    assert spine["role"] == "user"
    assert prov[1] == extra


def test_existing_callers_behavior_unchanged(iai_home):
    """capture_turn without provenance_extra writes exactly one provenance
    entry, byte-identical to the pre-extension shape.
    """
    from uuid import UUID

    from iai_mcp.capture import capture_turn

    store = _open_store()
    result = capture_turn(
        store,
        cue="ingest probe y",
        text="other content here long enough to clear the min length gate",
        session_id="s2",
        role="user",
    )
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    prov = rec.provenance
    assert isinstance(prov, list)
    assert len(prov) == 1, (
        f"expected 1 provenance entry (no kwarg), got {prov!r}"
    )
    spine = prov[0]
    assert set(spine.keys()) == {"ts", "cue", "session_id", "role"}, (
        "second provenance slot must be absent when provenance_extra is omitted"
    )


# ---------------------------------------------------------------------------
# capture_turn embeds message content, not the positional cue label
# ---------------------------------------------------------------------------

def _write_deferred_backlog(home: Path, session_id: str, events: list[dict]) -> Path:
    """Write a drainable deferred-capture backlog file (header + events).

    Mirrors the on-disk shape ``write_deferred_captures`` produces: a JSONL file
    whose first line is the version-1 header and whose remaining lines are turn
    events. The filename has the plain ``.jsonl`` suffix (no ``.live`` /
    ``.processing`` marker) so ``drain_deferred_captures`` claims it.
    """
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{session_id}-backlog.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return out_path


def test_drain_writes_new_turn_pending_without_embedding(iai_home, monkeypatch):
    """The in-daemon drain must NOT embed during the drain, and MUST reinforce a
    duplicate.

    A backlog drain is decoupled from embedding: every genuinely-new turn is
    written as a pending (un-embedded) row — the drain is a sequence of cheap
    SQLite writes whose resident-memory cost does not grow with the backlog size.
    The real vector is filled later by the bounded deferred-embed pass. A
    duplicate is detected by its idem-tag before any write, skipped, AND the
    pre-existing record is reinforced once — re-seeing a turn is a
    memory-strengthening signal.

    This test inserts one episodic record (so its idem-tag exists), then drains a
    backlog containing that SAME turn (duplicate) plus one genuinely-new turn, and
    asserts: the embedder is NEVER invoked during the drain, the new turn lands as
    a pending row (recall-findable immediately), and the pre-existing record is
    reinforced exactly once.
    """
    from iai_mcp.capture import capture_turn, drain_deferred_captures
    from iai_mcp.embed import Embedder
    from iai_mcp.store import MemoryStore

    store = _open_store()

    dup_uuid = str(uuid.uuid4())
    dup_ts = "2026-07-02T00:00:00Z"
    dup_text = "Duplicate turn already stored before the drain runs"

    new_uuid = str(uuid.uuid4())
    new_ts = "2026-07-02T00:01:00Z"
    new_text = "Genuinely new turn that has never been embedded or stored"

    # Seed the store with the duplicate turn so its idem-tag exists.
    seed = capture_turn(
        store,
        cue=f"session {SESSION_ID} turn 1",
        text=dup_text,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=dup_ts,
        source_uuid=dup_uuid,
    )
    assert seed["status"] == "inserted", seed
    count_before = _count_episodic_records(store)
    seed_id = seed["record_id"]

    # Spy on the embedder: it must NOT be invoked at all during the drain — the
    # whole point of the two-phase drain is that no embed runs synchronously.
    seen: list[str] = []
    real_embed = Embedder.embed

    def spy_embed(self, text):
        seen.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed", spy_embed)

    # Spy on reinforcement to prove the duplicate strengthens the pre-existing
    # record (and to count how many times it is reinforced).
    reinforced_ids: list[str] = []
    real_reinforce = MemoryStore.reinforce_record

    def spy_reinforce(self, record_id, *args, **kwargs):
        reinforced_ids.append(str(record_id))
        return real_reinforce(self, record_id, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, "reinforce_record", spy_reinforce)

    backlog_events = [
        {
            "text": dup_text,
            "cue": f"session {SESSION_ID} turn 1",
            "tier": "episodic",
            "role": "user",
            "ts": dup_ts,
            "source_uuid": dup_uuid,
        },
        {
            "text": new_text,
            "cue": f"session {SESSION_ID} turn 2",
            "tier": "episodic",
            "role": "user",
            "ts": new_ts,
            "source_uuid": new_uuid,
        },
    ]
    _write_deferred_backlog(iai_home, SESSION_ID, backlog_events)

    counts = drain_deferred_captures(store)

    # The drain embedded nothing — neither the duplicate nor the new turn. The
    # new turn's vector is deferred to the bounded deferred-embed pass.
    assert seen == [], (
        f"the drain must not embed any turn — two-phase capture defers all "
        f"embedding; saw {seen!r}. counts={counts!r}"
    )
    # The duplicate was reinforced (not silently dropped): the memory-strengthening
    # signal survives the drain even with no embed.
    assert counts["events_reinforced"] >= 1, (
        f"duplicate must be reinforced, not silently skipped; counts={counts!r}"
    )
    assert reinforced_ids == [seed_id], (
        f"the pre-existing record must be reinforced exactly once; "
        f"reinforced_ids={reinforced_ids!r}, seed_id={seed_id!r}"
    )
    assert counts["events_inserted"] == 1, (
        f"the new turn must be inserted; counts={counts!r}"
    )

    # The new record is actually in the store as a pending row; the duplicate
    # added nothing.
    count_after = _count_episodic_records(store)
    assert count_after == count_before + 1, (
        f"exactly one new record expected; before={count_before} after={count_after}"
    )
    # The new turn landed as a pending (un-embedded) row, and the seed row from
    # capture_turn stayed fully embedded — so exactly one pending row exists, and
    # it is recall-findable verbatim by its idem tag (encryption-independent).
    with store.db._conn_lock:
        pending_count = store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1 AND tombstoned_at IS NULL"
        ).fetchone()[0]
    assert pending_count == 1, (
        f"exactly one pending row expected after the drain, got {pending_count}"
    )
    from iai_mcp.capture import _idem_tag, _resolve_ts
    new_ts_iso = _resolve_ts(new_ts).isoformat()
    new_tag = _idem_tag(SESSION_ID, "user", new_ts_iso, new_text, source_uuid=new_uuid)
    new_id = store.find_record_by_tag(new_tag)
    assert new_id is not None, (
        "the new pending turn must be findable by its idem tag immediately — "
        "verbatim dedup/recall holds before the embedding lands"
    )


def test_drain_reinforces_repeated_duplicate_at_most_once(iai_home, monkeypatch):
    """A tag repeated many times in one backlog reinforces at most ONCE.

    A crash-rotated backlog can repeat the same turn tens of thousands of times.
    The drain must collapse all of those repeats to a single reinforcement so the
    Hebbian signal is not inflated by the size of the backlog (the ``seen_this_run``
    set protection). The embed must never run for any of the repeats.
    """
    from iai_mcp.capture import capture_turn, drain_deferred_captures
    from iai_mcp.embed import Embedder
    from iai_mcp.store import MemoryStore

    store = _open_store()

    dup_uuid = str(uuid.uuid4())
    dup_ts = "2026-07-03T00:00:00Z"
    dup_text = "A single turn that the crash-backlog repeats many times over"

    seed = capture_turn(
        store,
        cue=f"session {SESSION_ID} turn 1",
        text=dup_text,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=dup_ts,
        source_uuid=dup_uuid,
    )
    assert seed["status"] == "inserted", seed
    seed_id = seed["record_id"]

    # The embedder must never see the duplicate text, no matter how many repeats.
    real_embed = Embedder.embed

    def spy_embed(self, text):
        if text == dup_text:
            raise AssertionError(
                "embedder was invoked for a repeated duplicate — the pre-embed "
                "idem skip failed"
            )
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed", spy_embed)

    reinforced_ids: list[str] = []
    real_reinforce = MemoryStore.reinforce_record

    def spy_reinforce(self, record_id, *args, **kwargs):
        reinforced_ids.append(str(record_id))
        return real_reinforce(self, record_id, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, "reinforce_record", spy_reinforce)

    repeats = 50
    backlog_events = [
        {
            "text": dup_text,
            "cue": f"session {SESSION_ID} turn 1",
            "tier": "episodic",
            "role": "user",
            "ts": dup_ts,
            "source_uuid": dup_uuid,
        }
        for _ in range(repeats)
    ]
    _write_deferred_backlog(iai_home, SESSION_ID, backlog_events)

    counts = drain_deferred_captures(store)

    # The whole repeat-backlog collapses to exactly one reinforcement.
    assert reinforced_ids == [seed_id], (
        f"a {repeats}x-repeated duplicate must reinforce exactly once; "
        f"reinforced_ids={reinforced_ids!r}, seed_id={seed_id!r}"
    )
    assert counts["events_reinforced"] == 1, (
        f"events_reinforced must be exactly 1 for the whole repeat-backlog; "
        f"counts={counts!r}"
    )
    # Every repeat after the first is collapsed by seen_this_run.
    assert counts["events_skipped_existing"] == repeats - 1, (
        f"the {repeats - 1} extra repeats must be collapsed under "
        f"events_skipped_existing; counts={counts!r}"
    )
    assert counts["events_inserted"] == 0, (
        f"nothing genuinely-new in this backlog; counts={counts!r}"
    )


def test_capture_turn_embeds_content_not_cue(iai_home, monkeypatch):
    """capture_turn must embed the message text, never the cue.

    Transcript drains pass a positional cue ("session <id> turn <n>") as a
    provenance label. Embedding the cue instead of the content collapses the
    stored vector space and destroys semantic recall: every record gets a
    near-identical label-embedding. This test pins the contract — the embedder
    must see the real content string.
    """
    from iai_mcp.capture import capture_turn
    from iai_mcp.embed import Embedder

    seen: list[str] = []
    real_embed = Embedder.embed

    def recording_embed(self, text):
        seen.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed", recording_embed)

    store = _open_store()
    cue_label = f"session {SESSION_ID} turn 7"
    content = "real conversational content that must drive the embedding vector"

    result = capture_turn(
        store,
        cue=cue_label,
        text=content,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
    )

    assert result["status"] == "inserted", result
    assert seen, "embedder was never called"
    assert any(s.endswith(content) for s in seen), (
        f"embedder never saw the content string; saw {seen!r}. "
        f"capture_turn must embed text (optionally date-prefixed), not the cue."
    )
    assert not any(cue_label in s for s in seen), (
        f"embedder was handed the cue label {cue_label!r}; this collapses the "
        f"stored vector space and breaks semantic recall."
    )


def test_live_user_capture_records_formality_signal(iai_home):
    """AUTIST-13 (camouflaging_relaxation) has a complete detection pipeline
    in iai_mcp.camouflaging: record_user_formality collects a per-turn
    signal, run_weekly_pass aggregates it and relaxes the register. Before
    this fix, capture_turn never called record_user_formality, so the
    formality_score_weekly event stream that run_weekly_pass depends on
    stayed permanently empty regardless of how much the pipeline was used.
    This exercises the actual capture path, not the collector in isolation
    (already covered by tests/lilli/test_camouflaging_detection.py)."""
    from iai_mcp.capture import capture_turn
    from iai_mcp.events import query_events

    store = _open_store()

    before = query_events(store, kind="formality_score_weekly", limit=50)

    result = capture_turn(
        store,
        cue="formality signal wiring check",
        text=(
            "I would be most grateful if you could kindly assist me with "
            "this matter at your earliest convenience."
        ),
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        live_turn=True,
    )
    assert result["status"] == "inserted", result

    after = query_events(store, kind="formality_score_weekly", limit=50)
    assert len(after) == len(before) + 1, (
        "capture_turn did not record a formality signal for a live user turn"
    )

    # A replayed/bulk-imported turn (live_turn=False) must NOT count toward
    # the trend -- a historical backfill would otherwise skew weeks that
    # never happened live.
    result2 = capture_turn(
        store,
        cue="replay should not record formality",
        text="I would be most grateful if you could kindly assist me again.",
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        live_turn=False,
    )
    assert result2["status"] == "inserted", result2
    after_replay = query_events(store, kind="formality_score_weekly", limit=50)
    assert len(after_replay) == len(after), (
        "a replayed (live_turn=False) turn must not record a formality signal"
    )
