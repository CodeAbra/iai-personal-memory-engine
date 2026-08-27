from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


SESSION_ID = "model-attribution-session"
TURN_TEXT = "A direct captured turn keeps its explicit model attribution."


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "model-attribution-test-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_explicit_model_survives_direct_capture_and_replay(iai_home):
    from iai_mcp import session, working_tier
    from iai_mcp.capture import capture_turn

    store = _open_store()
    source_uuid = "abababab-1111-2222-3333-444444444444"
    ts = datetime.now(timezone.utc).isoformat()

    inserted = capture_turn(
        store,
        cue="direct attribution test",
        text=TURN_TEXT,
        session_id=SESSION_ID,
        role="user",
        ts=ts,
        source_uuid=source_uuid,
        model="  gpt-5.6   terra  ",
    )
    replayed = capture_turn(
        store,
        cue="direct attribution replay",
        text=TURN_TEXT,
        session_id=SESSION_ID,
        role="user",
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="different-model",
    )

    assert inserted["status"] == "inserted"
    assert replayed["status"] == "reinforced"
    assert replayed["record_id"] == inserted["record_id"]

    record = store.get(UUID(inserted["record_id"]))
    assert record is not None
    assert record.provenance[0]["model"] == "gpt-5.6 terra"
    assert session._recent_thread_segment(store, max_records=5).count(
        "[model:gpt-5.6 terra]"
    ) == 1

    entry = working_tier.read_task(session_id=SESSION_ID)
    assert entry is not None
    assert working_tier._render_snapshot(entry).count(
        "[model:gpt-5.6 terra]"
    ) == 1

    store.close()
    reopened = _open_store()
    persisted = reopened.get(UUID(inserted["record_id"]))
    assert persisted is not None
    assert persisted.provenance[0]["model"] == "gpt-5.6 terra"


def test_missing_or_malformed_model_has_no_display_label(iai_home):
    from iai_mcp import session
    from iai_mcp.capture import capture_turn

    store = _open_store()
    outcome = capture_turn(
        store,
        cue="missing attribution test",
        text="A turn without a model keeps legacy attribution behavior.",
        session_id="missing-model-session",
        role="user",
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid="cdcdcdcd-1111-2222-3333-444444444444",
    )

    assert outcome["status"] == "inserted"
    record = store.get(UUID(outcome["record_id"]))
    assert record is not None
    assert "model" not in record.provenance[0]
    assert "[model:" not in session._origin_label(record)

    malformed = SimpleNamespace(provenance=[{"model": ["not", "a", "label"]}])
    assert "[model:" not in session._origin_label(malformed)

    sanitized = capture_turn(
        store,
        cue="sanitized attribution test",
        text="A model label must not persist control or display delimiters.",
        session_id="sanitized-model-session",
        role="user",
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid="dededede-1111-2222-3333-444444444444",
        model="model\x00[label]",
    )
    sanitized_record = store.get(UUID(sanitized["record_id"]))
    assert sanitized_record is not None
    assert sanitized_record.provenance[0]["model"] == "modellabel"


def test_deferred_event_preserves_its_own_model_on_replay(iai_home):
    from iai_mcp.capture import (
        capture_turn,
        drain_active_live_captures,
        write_deferred_event,
    )

    store = _open_store()
    session_id = "deferred-model-session"
    model_text = "A deferred event preserves the model from its own envelope."
    missing_text = "A neighboring deferred event has no explicit model value."
    source_uuid = "efefefef-1111-2222-3333-444444444444"

    capture_turn(
        store,
        cue="surrounding model",
        text="A surrounding turn has a distinct explicit model attribution.",
        session_id=session_id,
        role="user",
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid="fefefefe-1111-2222-3333-444444444444",
        model="surrounding-model",
    )
    write_deferred_event(
        session_id,
        "user",
        model_text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="deferred-model",
    )
    drain_active_live_captures(store, exclude_session_id="-")

    write_deferred_event(
        session_id,
        "user",
        model_text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="replacement-model",
    )
    write_deferred_event(
        session_id,
        "user",
        missing_text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid="abababab-9999-2222-3333-444444444444",
    )
    drain_active_live_captures(store, exclude_session_id="-")

    records = store.recent_user_turns(n=20, session_id=session_id)
    attributed = [record for record in records if record.literal_surface == model_text]
    assert len(attributed) == 1
    assert attributed[0].provenance[0]["model"] == "deferred-model"

    unattributed = [record for record in records if record.literal_surface == missing_text]
    assert len(unattributed) == 1
    assert "model" not in unattributed[0].provenance[0]


def test_standard_deferred_drain_preserves_model_through_retry_and_replay(
    iai_home, monkeypatch
):
    from iai_mcp import capture

    store = _open_store()
    session_id = "standard-deferred-model-session"
    source_uuid = "aaaaaaaa-2222-3333-4444-555555555555"
    text = "A standard deferred drain retains its explicit model attribution."

    first_live_path = capture.write_deferred_event(
        session_id,
        "user",
        text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="standard-deferred-model",
    )
    first_live_path.rename(first_live_path.with_name("standard-deferred.jsonl"))

    first_counts = capture.drain_deferred_captures(store)
    assert first_counts["events_inserted"] == 1

    retry_live_path = capture.write_deferred_event(
        "retryable-deferred-model-session",
        "user",
        "A retryable deferred drain retains its explicit model attribution.",
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid="bbbbbbbb-2222-3333-4444-555555555555",
        model="retryable-deferred-model",
    )
    retry_live_path.rename(retry_live_path.with_name("retryable-deferred.jsonl"))

    original_write_pending = capture._drain_write_pending

    def fail_retryable_write_pending(*args, **kwargs):
        if kwargs.get("source_uuid") == "bbbbbbbb-2222-3333-4444-555555555555":
            return {"status": "skipped", "reason": "insert-failed: transient"}
        return original_write_pending(*args, **kwargs)

    monkeypatch.setattr(capture, "_drain_write_pending", fail_retryable_write_pending)
    failed_counts = capture.drain_deferred_captures(store)
    assert failed_counts["events_skipped_insert_failed"] == 1
    assert failed_counts["files_failed"] == 1

    monkeypatch.setattr(capture, "_drain_write_pending", original_write_pending)
    retry_path = next(
        capture.deferred_captures_dir().glob("*.failed-*-attempt-1.jsonl")
    )
    os.utime(retry_path, (0, 0))

    retry_counts = capture.drain_deferred_captures(store)
    assert retry_counts["events_inserted"] == 1

    replay_live_path = capture.write_deferred_event(
        session_id,
        "user",
        text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="replacement-model",
    )
    replay_live_path.rename(replay_live_path.with_name("standard-deferred-replay.jsonl"))

    replay_counts = capture.drain_deferred_captures(store)
    assert replay_counts["events_reinforced"] == 1

    standard_idem_tag = capture._idem_tag(
        session_id,
        "user",
        "-",
        text,
        source_uuid=source_uuid,
    )
    retry_idem_tag = capture._idem_tag(
        "retryable-deferred-model-session",
        "user",
        "-",
        "A retryable deferred drain retains its explicit model attribution.",
        source_uuid="bbbbbbbb-2222-3333-4444-555555555555",
    )
    standard_records = [
        record for record in store.all_records() if standard_idem_tag in record.tags
    ]
    retry_records = [
        record for record in store.all_records() if retry_idem_tag in record.tags
    ]
    assert len(standard_records) == 1
    assert standard_records[0].provenance[0]["model"] == "standard-deferred-model"
    assert len(retry_records) == 1
    assert retry_records[0].provenance[0]["model"] == "retryable-deferred-model"


def test_permanent_failed_recovery_preserves_model_and_first_attribution(iai_home):
    from iai_mcp import capture

    store = _open_store()
    session_id = "permanent-recovery-model-session"
    source_uuid = "cccccccc-2222-3333-4444-555555555555"
    text = "A permanent failed recovery retains its explicit model attribution."

    first_live_path = capture.write_deferred_event(
        session_id,
        "user",
        text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="permanent-recovery-model",
    )
    first_live_path.rename(
        first_live_path.with_name("permanent-recovery.permanent-failed-1.jsonl")
    )

    first_recovery = capture.drain_permanent_failed_files(store)
    assert first_recovery["inserted"] == 1

    replay_live_path = capture.write_deferred_event(
        session_id,
        "user",
        text,
        ts=datetime.now(timezone.utc).isoformat(),
        source_uuid=source_uuid,
        model="replacement-model",
    )
    replay_live_path.rename(
        replay_live_path.with_name(
            "permanent-recovery-replay.permanent-failed-1.jsonl"
        )
    )

    replay_recovery = capture.drain_permanent_failed_files(store)
    assert replay_recovery["inserted"] == 1

    idem_tag = capture._idem_tag(
        session_id,
        "user",
        "-",
        text,
        source_uuid=source_uuid,
    )
    records = [record for record in store.all_records() if idem_tag in record.tags]
    assert len(records) == 1
    assert records[0].provenance[0]["model"] == "permanent-recovery-model"


def test_transcript_and_stop_hook_preserve_explicit_model(iai_home):
    from iai_mcp.capture import capture_transcript, drain_active_live_captures
    from iai_mcp.cli import cmd_capture_turn_deferred

    transcript = iai_home / "model-attribution-transcript.jsonl"
    explicit_text = "A transcript turn carries its own explicit model value."
    unknown_text = "A transcript turn without a model remains unattributed."
    transcript.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "type": "assistant",
                    "uuid": "10101010-1111-2222-3333-444444444444",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": explicit_text,
                        "model": "transcript-model",
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "20202020-1111-2222-3333-444444444444",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": unknown_text},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    store = _open_store()
    captured = capture_transcript(
        store,
        transcript,
        session_id="transcript-model-session",
    )
    assert captured["inserted"] == 2

    rc = cmd_capture_turn_deferred(
        argparse.Namespace(
            session_id="stop-hook-model-session",
            transcript_path=str(transcript),
            max_turns_per_call=10,
        )
    )
    assert rc == 0
    drain_active_live_captures(store, exclude_session_id="-")

    records = store.all_records()
    transcript_records = [
        record
        for record in records
        if record.provenance[0].get("session_id") == "transcript-model-session"
    ]
    stop_hook_records = [
        record
        for record in records
        if record.provenance[0].get("session_id") == "stop-hook-model-session"
    ]
    assert next(
        record for record in transcript_records if record.literal_surface == explicit_text
    ).provenance[0]["model"] == "transcript-model"
    assert "model" not in next(
        record for record in transcript_records if record.literal_surface == unknown_text
    ).provenance[0]
    assert next(
        record for record in stop_hook_records if record.literal_surface == explicit_text
    ).provenance[0]["model"] == "transcript-model"
