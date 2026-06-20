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


def test_capture_turn_embeds_content_not_cue(iai_home, monkeypatch):
    """Regression: capture_turn() must embed the message CONTENT, not the cue label.

    capture_transcript() passes a positional cue ("session <id> turn <n>") and
    capture_turn() used to embed ``cue or text``, so every episodic capture's
    vector was the embedding of a near-identical positional label instead of the
    message body. That collapsed the stored embedding space (unrelated turns at
    cosine ~0.9+) and broke the similarity graph and semantic recall.
    """
    import iai_mcp.embed as embed_mod
    from iai_mcp import capture as capture_mod

    seen: dict[str, str] = {}

    class _RecordingEmbedder:
        DIM = 384

        def embed(self, text: str) -> list[float]:
            seen["text"] = text
            vec = [0.0] * 384
            vec[0] = 1.0  # valid unit vector; avoids zero-norm edge cases downstream
            return vec

    # capture_turn imports embedder_for_store from iai_mcp.embed at call time.
    monkeypatch.setattr(
        embed_mod, "embedder_for_store", lambda store: _RecordingEmbedder()
    )

    store = _open_store()
    cue = f"session {SESSION_ID} turn 7"
    content = "Le constat d'état des lieux a été réalisé dans l'appartement."
    capture_mod.capture_turn(
        store,
        cue=cue,
        text=content,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
    )

    assert seen.get("text") == content, (
        "capture_turn embedded the wrong string -- it must embed the message "
        f"content, not the cue. Got: {seen.get('text')!r}"
    )
    assert seen["text"] != cue
