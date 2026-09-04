from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path



def _deferred_dir(home: Path) -> Path:
    d = home / ".iai-mcp" / ".deferred-captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_live_file(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    version: int = 1,
) -> Path:
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": version,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ts(offset_secs: float = 0.0) -> str:
    dt = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_secs)
    return dt.isoformat()


def test_immediate_recall_no_drain(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import write_deferred_event, read_pending_live_events

    session = "sess-immediate-recall"
    write_deferred_event(session, "user", "hello world from live layer", source_uuid="u1")

    events = read_pending_live_events()
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events!r}"
    ev = events[0]
    assert ev["text"] == "hello world from live layer"
    assert ev["role"] == "user"
    assert ev["session_id"] == session
    assert ev["source_uuid"] == "u1"
    assert isinstance(ev["ts"], datetime), f"ts must be datetime, got {type(ev['ts'])}"
    assert ev["ts"].tzinfo is not None, "ts must be tz-aware"


def test_processing_marker_file_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "proc-session-0001"
    deferred = _deferred_dir(tmp_path)

    name = f"{session}.live-1700000000.processing-99999.jsonl"
    path = deferred / name
    header = json.dumps({
        "version": 1,
        "deferred_at": _ts(),
        "session_id": session,
        "cwd": "/tmp",
    })
    ev_line = json.dumps({
        "text": "processing window marker turn",
        "role": "user",
        "tier": "episodic",
        "ts": _ts(1),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "processing window marker turn" in texts, (
        f"processing-window file not read; got events: {events!r}"
    )


def test_tail_returns_newest_past_500(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "tail-600-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    lines = [header]
    for i in range(600):
        ev = {
            "text": "OLDEST" if i == 0 else (f"line-{i}" if i < 599 else "NEWEST"),
            "role": "user",
            "tier": "episodic",
            "ts": _ts(i),
        }
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "NEWEST" in texts, f"NEWEST turn not returned; sample: {texts[:3]}"
    assert "OLDEST" not in texts, "OLDEST turn (line 1) must be dropped by deque tail"


def test_large_cyrillic_file_newest_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "big-cyrillic-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })

    long_cyrillic = "Русский текст для теста UTF-8 в больших файлах захвата памяти. " * 30  # non-English fixture data: ru round-trip under test

    lines = [header]
    for i in range(400):
        ev = {
            "text": "САМОЕ НОВОЕ" if i == 399 else f"строка-{i} {long_cyrillic}",  # non-English fixture data: ru round-trip under test
            "role": "user",
            "tier": "episodic",
            "ts": _ts(i),
        }
        lines.append(json.dumps(ev, ensure_ascii=False))
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    assert path.stat().st_size > 1_000_000, (
        f"file must exceed 1 MB; got {path.stat().st_size} bytes"
    )

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "САМОЕ НОВОЕ" in texts, (  # non-English fixture data: ru round-trip under test
        f"newest Cyrillic turn not returned; got {len(events)} events"
    )


def test_partial_last_line_no_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "partial-line-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    ev_complete = json.dumps({
        "text": "complete event line", "role": "user", "tier": "episodic", "ts": _ts()
    })
    ev_partial = '{"text": "incomplete'

    path.write_text(header + "\n" + ev_complete + "\n" + ev_partial, encoding="utf-8")

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "complete event line" in texts, "complete event should be returned"
    assert not any("incomplete" in t for t in texts), "partial line must not appear"


def test_version2_header_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "v2-forward-compat-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": "should not appear", "role": "user", "tier": "episodic", "ts": _ts()}],
        version=2,
    )

    events = read_pending_live_events()
    assert len(events) == 0, f"version:2 file must be skipped; got {events!r}"


def test_bounded_scan_ignores_noise_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)

    for i in range(100):
        (deferred / f"some-session.failed-{i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    for i in range(100):
        (deferred / f"some-session.permanent-failed-{i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    for i in range(100):
        (deferred / f"some-session.live-{1700000000 + i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    (deferred / ".quarantine").mkdir(exist_ok=True)

    for i in range(3):
        _write_live_file(
            deferred, f"real-session-{i}",
            [{"text": f"real turn {i}", "role": "user", "tier": "episodic", "ts": _ts(i)}],
        )

    parse_count = 0
    real_json_loads = json.loads

    def counting_loads(s, **kw):
        nonlocal parse_count
        parse_count += 1
        return real_json_loads(s, **kw)

    monkeypatch.setattr("iai_mcp.capture.json.loads", counting_loads)

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    for i in range(3):
        assert f"real turn {i}" in texts, f"real turn {i} missing"

    assert parse_count <= 100, (
        f"allowlist should exclude noise files before parse; got {parse_count} json.loads calls"
    )


def test_mtime_desc_caps_to_20(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)

    base_mtime = time.time() - 3000
    for i in range(30):
        sid = f"mtime-session-{i:02d}"
        marker = "NEWEST_MARKER" if i == 29 else ("OLD_MARKER" if i == 0 else f"mid-{i}")
        _write_live_file(
            deferred, sid,
            [{"text": marker, "role": "user", "tier": "episodic", "ts": _ts(i)}],
        )
        p = deferred / f"{sid}.live.jsonl"
        mtime = base_mtime + i * 100
        os.utime(str(p), (mtime, mtime))

    events = read_pending_live_events()
    texts = [e["text"] for e in events]

    assert "NEWEST_MARKER" in texts, "newest file (i=29) must be in results"
    assert "OLD_MARKER" not in texts, "oldest file (i=0) must be excluded by the ≤20 cap"
    assert len(events) <= 20, f"must read ≤20 files; got {len(events)}"


def test_session_aware_select_not_starved_by_other_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)
    base_mtime = time.time() - 10000

    X = "target-session-x"
    _write_live_file(
        deferred, X,
        [{"text": "X_MARKER_TURN", "role": "user", "tier": "episodic", "ts": _ts(0)}],
    )
    x_path = deferred / f"{X}.live.jsonl"
    os.utime(str(x_path), (base_mtime, base_mtime))

    for i in range(25):
        sid = f"other-session-{i:02d}"
        _write_live_file(
            deferred, sid,
            [{"text": f"other-{i}", "role": "user", "tier": "episodic", "ts": _ts(i + 1)}],
        )
        p = deferred / f"{sid}.live.jsonl"
        newer_mtime = base_mtime + (i + 1) * 100
        os.utime(str(p), (newer_mtime, newer_mtime))

    events = read_pending_live_events(session_id=X)
    texts = [e["text"] for e in events]
    assert "X_MARKER_TURN" in texts, (
        f"session X was starved by 25 newer sessions; got texts: {texts!r}"
    )

    assert len(events) <= 20, f"≤20 cap must hold; got {len(events)}"


def test_session_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import write_deferred_event, read_pending_live_events

    write_deferred_event("session-A-filter", "user", "turn from session A for filter test")
    write_deferred_event("session-B-filter", "user", "turn from session B for filter test")

    a_events = read_pending_live_events(session_id="session-A-filter")
    assert all(e["session_id"] == "session-A-filter" for e in a_events), (
        f"filtered result contains non-A events: {a_events!r}"
    )
    assert len(a_events) == 1, f"expected 1 A event; got {len(a_events)}"

    all_events = read_pending_live_events()
    session_ids = {e["session_id"] for e in all_events}
    assert "session-A-filter" in session_ids
    assert "session-B-filter" in session_ids


def test_sorted_ts_desc(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "sort-ts-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [
            {"text": "oldest", "role": "user", "tier": "episodic", "ts": _ts(0)},
            {"text": "middle", "role": "user", "tier": "episodic", "ts": _ts(1)},
            {"text": "newest", "role": "user", "tier": "episodic", "ts": _ts(2)},
        ],
    )

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert texts[0] == "newest", f"newest should be first; got {texts}"
    assert texts[-1] == "oldest", f"oldest should be last; got {texts}"


def test_read_time_idem_tag_matches_drain_time(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import _idem_tag, _resolve_ts, read_pending_live_events

    session = "idem-ts-norm-session"
    ts_raw = "2026-05-31T12:00:00.000000+00:00"
    text = "idem tag normalization test turn content here"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": text, "role": "user", "tier": "episodic", "ts": ts_raw}],
    )

    events = read_pending_live_events()
    assert len(events) == 1, f"expected 1 event; got {events!r}"
    ev = events[0]

    read_tag = _idem_tag(session, "user", ev["ts_iso"], text, source_uuid=None)

    now = _resolve_ts(ts_raw)
    ts_iso_drain = now.isoformat()
    drain_tag = _idem_tag(session, "user", ts_iso_drain, text, source_uuid=None)

    assert read_tag == drain_tag, (
        f"read-side tag {read_tag!r} != drain-side tag {drain_tag!r}; "
        f"ev['ts_iso']={ev['ts_iso']!r}, drain ts_iso={ts_iso_drain!r}"
    )


def test_header_session_id_overrides_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)
    filename_stem = "wrong-session-name"
    header_session_id = "correct-header-session"

    path = deferred / f"{filename_stem}.live.jsonl"
    header = json.dumps({
        "version": 1,
        "deferred_at": _ts(),
        "session_id": header_session_id,
        "cwd": "/tmp",
    })
    ev_line = json.dumps({
        "text": "header-session-turn marker text xyzw",
        "role": "user",
        "tier": "episodic",
        "ts": _ts(1),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    events_header = read_pending_live_events(session_id=header_session_id)
    texts_header = [e["text"] for e in events_header]
    assert "header-session-turn marker text xyzw" in texts_header, (
        f"header-session filter should return event; got {events_header!r}"
    )

    events_stem = read_pending_live_events(session_id=filename_stem)
    texts_stem = [e["text"] for e in events_stem]
    assert "header-session-turn marker text xyzw" not in texts_stem, (
        f"filename-stem filter must NOT return header-session event; got {events_stem!r}"
    )


def test_live_epoch_file_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "epoch-file-session"
    deferred = _deferred_dir(tmp_path)

    path = deferred / f"{session}.live-1700000000.jsonl"
    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    ev_line = json.dumps({
        "text": "should not appear from epoch file",
        "role": "user", "tier": "episodic", "ts": _ts(),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    events = read_pending_live_events(session_id=session)
    texts = [e["text"] for e in events]
    assert "should not appear from epoch file" not in texts, (
        f"live-epoch file must be excluded by allowlist; got {events!r}"
    )


def test_assistant_role_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "assistant-role-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": "assistant response text here", "role": "assistant",
          "tier": "episodic", "ts": _ts()}],
    )

    events = read_pending_live_events()
    roles = [e["role"] for e in events]
    assert "assistant" in roles, (
        f"assistant-role event must be returned (helper does not filter role); "
        f"got roles: {roles!r}"
    )


def test_no_heavy_import(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    events = read_pending_live_events()
    assert events == [], f"empty dir must return []; got {events!r}"

    deferred = _deferred_dir(tmp_path)
    events = read_pending_live_events()
    assert events == [], f"empty deferred-captures dir must return []; got {events!r}"
