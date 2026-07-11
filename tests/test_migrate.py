from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp.types import EMBED_DIM, MemoryRecord, SCHEMA_VERSION_LEGACY


def _v1_record(
    text: str,
    *,
    language: str = "",
    tags: list[str] | None = None,
    dim: int = EMBED_DIM,
) -> MemoryRecord:
    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * dim,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": "2026-04-16T00:00:00Z", "cue": "seed", "session_id": "phase1"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language="en",
        schema_version=SCHEMA_VERSION_LEGACY,
    )
    if language:
        r.language = language
    else:
        r.language = ""
    return r


def test_migrate_v1_to_v2_sets_defaults(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("English legacy record for migration test with enough words")
    store.insert(r)
    result = migrate_v1_to_v2(store)
    assert result["records_migrated"] >= 1

    migrated = store.get(r.id)
    assert migrated is not None
    assert migrated.s5_trust_score == 0.5
    assert migrated.profile_modulation_gain == {}
    from iai_mcp.types import SCHEMA_VERSION_CURRENT
    assert migrated.schema_version == SCHEMA_VERSION_CURRENT
    assert migrated.schema_version >= 2


def test_migrate_v1_to_v2_defaults_language_to_en(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    en = _v1_record("This is a reasonable English sentence with enough words.")
    ru = _v1_record("Это осмысленное предложение с достаточным количеством слов.")
    store.insert(en)
    store.insert(ru)

    migrate_v1_to_v2(store)

    en_mig = store.get(en.id)
    ru_mig = store.get(ru.id)
    assert en_mig.language == "en"
    assert ru_mig.language == "en"


def test_migrate_v1_to_v2_preserves_existing_language_tag(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("legacy row carrying an explicit non-en tag", language="ru")
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert migrated.language == "ru"


def test_migrate_v1_to_v2_idempotent(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for i in range(5):
        store.insert(_v1_record(f"English record number {i} with enough content to detect."))

    first = migrate_v1_to_v2(store)
    assert first["records_migrated"] >= 5

    second = migrate_v1_to_v2(store)
    assert second["records_migrated"] == 0


def test_migrate_dry_run_no_writes(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("Dry run English text with enough words for language detection.")
    store.insert(r)
    before = store.get(r.id)
    assert before.schema_version == 1

    result = migrate_v1_to_v2(store, dry_run=True)
    assert result["records_migrated"] >= 1

    after = store.get(r.id)
    assert after.schema_version == 1


def test_migrate_writes_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    store.insert(_v1_record("English content one for migration event test."))

    migrate_v1_to_v2(store)

    events = query_events(store, kind="migration_v1_to_v2")
    assert len(events) == 1
    assert events[0]["data"]["record_count"] >= 1


def test_migrate_preserves_literal_surface_verbatim(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    verbatim = "SECRET_PHRASE_ABC_XYZ must survive the migration byte-for-byte exactly."
    r = _v1_record(verbatim)
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert migrated.literal_surface == verbatim


def test_migrate_preserves_provenance(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("English content for provenance preservation test through migration.")
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert len(migrated.provenance) == 1
    assert migrated.provenance[0]["cue"] == "seed"
    assert migrated.provenance[0]["session_id"] == "phase1"


def test_migrate_skips_existing_v2_records(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    v2 = _v1_record("Already migrated record with language tag.", language="en")
    v2.schema_version = 2
    store.insert(v2)

    v1 = _v1_record("Legacy v1 record with enough content for detection.")
    store.insert(v1)

    result = migrate_v1_to_v2(store)
    assert result["records_migrated"] == 1

    v2_got = store.get(v2.id)
    assert v2_got.schema_version == 2


def test_migrate_result_carries_model_info(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    store.insert(_v1_record("English content for the migration model info check."))

    result = migrate_v1_to_v2(store)
    assert "previous_model" in result
    assert "new_model" in result
    assert "duration_sec" in result


# ---------------------------------------------------------------------------
# Helpers for timestamp re-derivation tests
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path


def _make_episodic_record(
    text: str,
    session_id: str,
    source_uuid: str,
    collapsed_ts: datetime,
) -> MemoryRecord:
    """Build an episodic MemoryRecord with provenance carrying session_id + source_uuid."""
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id, "source_uuid": source_uuid, "role": "user"}],
        created_at=collapsed_ts,
        updated_at=collapsed_ts,
        tags=[],
        language="en",
        schema_version=5,
    )


def _write_fake_transcript(
    transcript_root: _Path,
    session_id: str,
    entries: list[dict],
) -> _Path:
    """Write a fake Claude transcript JSONL under transcript_root/<hash>/<session_id>.jsonl."""
    # Mimic the real layout: ~/.claude/projects/<hash>/<session_id>.jsonl
    proj_dir = transcript_root / "proj-fake"
    proj_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = proj_dir / f"{session_id}.jsonl"
    with transcript_path.open("w") as f:
        for entry in entries:
            f.write(_json.dumps(entry) + "\n")
    return transcript_path


# ---------------------------------------------------------------------------
# Timestamp re-derivation tests (RED gate — function does not exist yet)
# ---------------------------------------------------------------------------


def test_migrate_rederive_timestamps_updates(tmp_path):
    """Collapsed created_at values get re-derived from the transcript."""
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-test"
    transcript_root = tmp_path / "transcripts"

    # All four records share one collapsed timestamp.
    collapsed_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(4)]
    texts = [f"User turn {i}" for i in range(4)]
    real_ts_strs = [
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:01:00Z",
        "2026-01-01T10:02:00Z",
        "2026-01-01T10:03:00Z",
    ]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(4)
    ]
    for r in records:
        store.insert(r)

    # Fake transcript with matching uuids and distinct timestamps.
    transcript_entries = [
        {"type": "user", "uuid": uuids[i], "timestamp": real_ts_strs[i], "sessionId": session_id}
        for i in range(4)
    ]
    _write_fake_transcript(transcript_root, session_id, transcript_entries)

    result = migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)

    assert result["records_updated"] == 4
    assert result["dry_run"] is False

    fetched_ts = []
    for r in records:
        updated = store.get(r.id)
        assert updated is not None
        fetched_ts.append(updated.created_at)

    # All four timestamps must now be distinct.
    assert len(set(fetched_ts)) == 4


def test_migrate_rederive_timestamps_idempotent(tmp_path):
    """Running the migration twice updates zero records on the second run."""
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-idempotent"
    transcript_root = tmp_path / "transcripts"

    collapsed_ts = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(3)]
    texts = [f"User turn {i}" for i in range(3)]
    real_ts_strs = [
        "2026-02-01T08:00:00Z",
        "2026-02-01T08:01:00Z",
        "2026-02-01T08:02:00Z",
    ]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(3)
    ]
    for r in records:
        store.insert(r)

    transcript_entries = [
        {"type": "user", "uuid": uuids[i], "timestamp": real_ts_strs[i], "sessionId": session_id}
        for i in range(3)
    ]
    _write_fake_transcript(transcript_root, session_id, transcript_entries)

    first = migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)
    assert first["records_updated"] == 3

    second = migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)
    assert second["records_updated"] == 0


def test_migrate_rederive_timestamps_dry_run(tmp_path):
    """dry_run=True reads but never writes; result carries dry_run=True."""
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-dry"
    transcript_root = tmp_path / "transcripts"

    collapsed_ts = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(3)]
    texts = [f"User turn {i}" for i in range(3)]
    real_ts_strs = [
        "2026-03-01T09:00:00Z",
        "2026-03-01T09:01:00Z",
        "2026-03-01T09:02:00Z",
    ]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(3)
    ]
    for r in records:
        store.insert(r)

    transcript_entries = [
        {"type": "user", "uuid": uuids[i], "timestamp": real_ts_strs[i], "sessionId": session_id}
        for i in range(3)
    ]
    _write_fake_transcript(transcript_root, session_id, transcript_entries)

    # Snapshot before.
    before_ts = {r.id: store.get(r.id).created_at for r in records}

    result = migrate_rederive_collapsed_timestamps(
        store, dry_run=True, transcript_root=transcript_root
    )
    assert result["dry_run"] is True

    # Nothing written.
    for r in records:
        after = store.get(r.id)
        assert after.created_at == before_ts[r.id]


def test_migrate_rederive_preserves_literal_surface(tmp_path):
    """Only created_at is mutated; literal_surface and provenance_json are byte-identical."""
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-surface"
    transcript_root = tmp_path / "transcripts"

    collapsed_ts = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(3)]
    texts = [f"Verbatim content that must survive #{i}" for i in range(3)]
    real_ts_strs = [
        "2026-04-01T07:00:00Z",
        "2026-04-01T07:01:00Z",
        "2026-04-01T07:02:00Z",
    ]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(3)
    ]
    for r in records:
        store.insert(r)

    transcript_entries = [
        {"type": "user", "uuid": uuids[i], "timestamp": real_ts_strs[i], "sessionId": session_id}
        for i in range(3)
    ]
    _write_fake_transcript(transcript_root, session_id, transcript_entries)

    # Snapshot literal_surface and provenance before migration.
    before = {r.id: (store.get(r.id).literal_surface, store.get(r.id).provenance) for r in records}

    migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)

    for r in records:
        after = store.get(r.id)
        assert after.literal_surface == before[r.id][0], "literal_surface changed"
        assert after.provenance == before[r.id][1], "provenance changed"
        # created_at must have moved to the transcript timestamp.
        assert after.created_at != collapsed_ts


def test_migrate_rederive_skips_missing_transcript(tmp_path):
    """Records with no matching transcript file are left untouched and counted."""
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-no-transcript"
    transcript_root = tmp_path / "transcripts"
    # No transcript file is written for this session.

    collapsed_ts = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(3)]
    texts = [f"User turn no transcript {i}" for i in range(3)]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(3)
    ]
    for r in records:
        store.insert(r)

    result = migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)

    # No records updated, all skipped because no transcript.
    assert result["records_updated"] == 0
    assert result["skipped_no_transcript"] >= 3

    # created_at must remain exactly the collapsed value — not fabricated to now().
    for r in records:
        after = store.get(r.id)
        assert after.created_at == collapsed_ts


def test_migrate_rederive_writes_event(tmp_path):
    """A migration_rederive_timestamps event is written after a non-dry run."""
    from iai_mcp.events import query_events
    from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "sess-event"
    transcript_root = tmp_path / "transcripts"

    collapsed_ts = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    uuids = [str(uuid4()) for _ in range(3)]
    texts = [f"User turn event {i}" for i in range(3)]
    real_ts_strs = [
        "2026-06-01T06:00:00Z",
        "2026-06-01T06:01:00Z",
        "2026-06-01T06:02:00Z",
    ]

    records = [
        _make_episodic_record(texts[i], session_id, uuids[i], collapsed_ts)
        for i in range(3)
    ]
    for r in records:
        store.insert(r)

    transcript_entries = [
        {"type": "user", "uuid": uuids[i], "timestamp": real_ts_strs[i], "sessionId": session_id}
        for i in range(3)
    ]
    _write_fake_transcript(transcript_root, session_id, transcript_entries)

    migrate_rederive_collapsed_timestamps(store, transcript_root=transcript_root)

    events = query_events(store, kind="migration_rederive_timestamps")
    assert len(events) >= 1
    assert "records_updated" in events[0]["data"]


# ---------------------------------------------------------------------------
# migrate_salvage_torn_permanent_failed
# ---------------------------------------------------------------------------


def test_salvage_torn_permanent_failed_recovers_complete_lines(tmp_path):
    """A torn .permanent-failed-*.jsonl file is salvaged: complete (newline-
    terminated) prefix is re-deferred for normal drain; the torn tail and
    the original file go to .quarantine/."""
    from iai_mcp.migrate import migrate_salvage_torn_permanent_failed

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir()
    torn = deferred / "sess-1-100.permanent-failed-200.jsonl"
    # 3 complete records + 1 torn tail with NUL fill (the "torn write" signature)
    torn.write_bytes(
        b'{"version":1}\n'
        b'{"text":"a"}\n'
        b'{"text":"b"}\n'
        b'{"text":"trunca\x00\x00\x00\x00'
    )

    dry = migrate_salvage_torn_permanent_failed(deferred_dir=deferred, dry_run=True)
    assert dry["files_salvaged"] == 1
    assert dry["records_salvaged"] == 3  # 3 newline-terminated records
    assert dry["bytes_dropped"] > 0
    # dry-run does NOT mutate the deferred-captures dir
    assert torn.exists()
    assert not (deferred / ".quarantine").exists()

    real = migrate_salvage_torn_permanent_failed(deferred_dir=deferred, dry_run=False)
    assert real["files_salvaged"] == 1
    assert real["records_salvaged"] == 3
    assert real["bytes_dropped"] == dry["bytes_dropped"]
    # Original moved to quarantine
    assert not torn.exists()
    quarantined = list((deferred / ".quarantine").iterdir())
    assert len(quarantined) == 1
    # Salvaged prefix sits in the deferred dir under a .salvaged- name
    salvaged_files = [
        p for p in deferred.iterdir()
        if p.is_file() and ".salvaged-" in p.name
    ]
    assert len(salvaged_files) == 1
    # And only the complete lines survived
    salvaged = salvaged_files[0].read_bytes()
    assert salvaged.endswith(b"\n")
    assert b'"trunca' not in salvaged

    # Idempotent: a second run is a no-op
    again = migrate_salvage_torn_permanent_failed(deferred_dir=deferred, dry_run=False)
    assert again["files_salvaged"] == 0
    assert again["records_salvaged"] == 0


def test_salvage_no_op_when_no_torn_files(tmp_path):
    """An empty or torn-free deferred dir returns zero counts cleanly."""
    from iai_mcp.migrate import migrate_salvage_torn_permanent_failed

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir()

    r = migrate_salvage_torn_permanent_failed(deferred_dir=deferred, dry_run=False)
    assert r == {
        "files_salvaged": 0,
        "records_salvaged": 0,
        "bytes_dropped": 0,
        "dry_run": False,
    }


def test_write_deferred_captures_uses_atomic_rename_with_pid(tmp_path, monkeypatch):
    """write_deferred_captures writes to a sibling .jsonl.tmp and os.replaces
    it to the final name, which includes os.getpid() for parallel-worker
    collision safety. The final file is only visible AFTER fsync+replace."""
    import json
    from pathlib import Path

    from iai_mcp import capture as capture_mod

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "hello there"},
            "uuid": "u-1",
            "timestamp": "2026-07-01T00:00:00Z",
        }) + "\n"
    )

    out_path = capture_mod.write_deferred_captures(
        "sess-X", transcript, cwd="/tmp",
    )
    assert out_path.exists()
    assert out_path.suffix == ".jsonl"
    # Filename includes the pid as the last numeric component
    import os as _os
    parts = out_path.stem.split("-")
    assert parts[-1] == str(_os.getpid()), (
        f"final filename {out_path.name!r} should end with pid {_os.getpid()}"
    )
    # No leftover .tmp sibling
    deferred = tmp_path / ".iai-mcp" / ".deferred-captures"
    tmp_siblings = list(deferred.glob("*.tmp"))
    assert tmp_siblings == [], f"unexpected .tmp leftover: {tmp_siblings}"
