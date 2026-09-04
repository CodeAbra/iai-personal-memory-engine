"""Named things must be recallable by name: anchors at capture and by
backfill, flowing into the AAAK E: field the rank overlap reads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai_mcp.entity_anchors import entity_tags, extract_entities
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")
    yield


def test_extractor_catches_the_incident_shapes() -> None:
    text = (
        # non-English fixture data: multilingual behavior under test — keep
        "Работаем над Zephyrbot: правим `parse_kit` в render_skill.py, "
        # non-English fixture data: multilingual behavior under test — keep
        "деплой через @shipbot, дальше SkyPilot и text_rewriter."
    )
    got = extract_entities(text)
    for expected in (
        "zephyrbot", "parse_kit", "render_skill.py", "shipbot",
        "skypilot", "text_rewriter",
    ):
        assert expected in got, f"{expected!r} missing from {got}"


def test_extractor_never_emits_aaak_structural_chars() -> None:
    got = extract_entities(
        "Path `src/engine/render_kit.py` and url `https://alice.dev/x` "
        "plus a:b and c,d tokens."
    )
    assert "render_kit.py" in got, got
    for tok in got:
        assert "/" not in tok and ":" not in tok and "," not in tok, got


def test_extractor_survives_pathological_input() -> None:
    import signal

    # 60 underscore-separated groups ending in a separator that forces the
    # trailing lookahead to fail — the old ambiguous pattern was exponential.
    # The alarm turns a backtracking regression into a failure, not a hang.
    def _hang(_signum, _frame):
        raise TimeoutError("extractor backtracked exponentially")

    old = signal.signal(signal.SIGALRM, _hang)
    signal.alarm(20)
    try:
        text = ("a" + "_b" * 60 + "-") * 20 + " " + "_" * 500 + "-"
        got = extract_entities(text)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert isinstance(got, list)


def test_extractor_skips_sentence_starts_and_furniture() -> None:
    got = extract_entities(
        # non-English fixture data: multilingual behavior under test — keep
        "Это просто предложение. Когда мы начали, The plan was ready. "
        # non-English fixture data: multilingual behavior under test — keep
        "Внутри упоминается Лилли и проект Hippo."
    )
    # non-English fixture data: multilingual behavior under test — keep
    assert "лилли" in got and "hippo" in got
    # non-English fixture data: multilingual behavior under test — keep
    for absent in ("это", "когда", "the"):
        assert absent not in got


def test_extractor_bounds_and_purity() -> None:
    text = " ".join(f"Word{i}word" for i in range(40)) + " 12345 a-b"
    got = extract_entities(text)
    assert len(got) <= 10
    assert "12345" not in got
    assert all(any(c.isalpha() for c in e) for e in got)


def test_capture_writes_anchors_and_aaak(tmp_path):
    from iai_mcp.capture import capture_turn

    store = MemoryStore(path=tmp_path / "store")
    result = capture_turn(
        store,
        cue="",
        text="Shipped Zephyrbot deep-readme extraction with `parse_kit`.",
        tier="episodic",
        session_id="s-anchor",
        role="assistant",
        live_turn=True,
    )
    assert result["status"] == "inserted", result
    from uuid import UUID

    rec = store.get(UUID(result["record_id"]))
    assert any(t == "entity:zephyrbot" for t in rec.tags), rec.tags
    assert "E:" in rec.aaak_index and "zephyrbot" in rec.aaak_index, (
        rec.aaak_index
    )


def test_backfill_anchors_old_records(tmp_path):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors

    store = MemoryStore(path=tmp_path / "store")
    rid = UUID(capture_turn(
        store, cue="", text="Working on Zephyrbot card generation today.",
        tier="episodic", session_id="s-old", role="user", live_turn=True,
    )["record_id"])
    # Simulate a pre-anchor record: strip the anchors capture just wrote.
    rec = store.get(rid)
    store.remove_tags(rid, [t for t in rec.tags if t.startswith("entity:")])
    store.set_aaak_index(rid, "")

    dry = backfill_entity_anchors(store, apply=False, store_path=tmp_path / "store")
    assert dry["mode"] == "dry-run"
    assert dry["records_to_anchor"] >= 1
    assert dry["records_written"] == 0

    # The runtime-graph cache key covers record/edge counts only — a
    # payload-only sweep must drop it or a rebooting daemon serves
    # pre-backfill payloads without the anchors.
    stale_cache = tmp_path / "store" / "runtime_graph_cache.json"
    stale_cache.write_text("{}")

    applied = backfill_entity_anchors(store, apply=True, store_path=tmp_path / "store")
    assert applied["records_written"] >= 1
    assert applied["aaak_refreshed"] >= 1
    assert applied["graph_cache_invalidated"] is True
    assert not stale_cache.exists()
    journal = Path(applied["journal"])
    assert journal.exists()
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert any(e["record_id"] == str(rid) for e in entries)

    refreshed = store.get(rid)
    assert any(t == "entity:zephyrbot" for t in refreshed.tags)
    assert "zephyrbot" in refreshed.aaak_index

    again = backfill_entity_anchors(store, apply=False, store_path=tmp_path / "store")
    assert again["records_to_anchor"] == 0, "backfill must be idempotent"


def test_extractor_denies_stopwords_and_machine_tokens() -> None:
    got = extract_entities(
        # non-English fixture data: multilingual behavior under test — keep
        "Знаешь Что? Из-за Чего-то сломалось, Пока непонятно. "
        "Quoted <task-notification> and tool-use-id tokens, path "
        "/Users/alice/Desktop stuff, Request from Thanks."
    )
    for junk in (
        # non-English fixture data: multilingual behavior under test — keep
        "что", "из-за", "чего-то", "чего", "пока", "task-notification",
        "tool-use-id", "users", "desktop", "request", "thanks",
    ):
        assert junk not in got, f"{junk!r} must never anchor: {got}"


def test_extractor_rejects_code_fragments() -> None:
    got = extract_entities(
        'Call `req.get("method")=="brain_view` then `$files` and '
        "`0.6*cos+0.4*centrality` but keep `parse_kit` working."
    )
    # Whole code fragments and formulas must never anchor; clean
    # identifier tokens inside them are fine.
    assert "parse_kit" in got
    for tok in got:
        assert not any(c in tok for c in "()\"'=$*+«»<>"), got


def test_refresh_removes_junk_anchors(tmp_path):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors

    store = MemoryStore(path=tmp_path / "store")
    rid = UUID(capture_turn(
        store, cue="", text="Deploying Zephyrbot to the fleet today.",
        tier="episodic", session_id="s-junk", role="user", live_turn=True,
    )["record_id"])
    # Simulate anchors written by the older, junk-happy extractor.
    # non-English fixture data: multilingual behavior under test — keep
    store.add_tags(rid, ["entity:что", "entity:users"])

    dry = backfill_entity_anchors(
        store, apply=False, store_path=tmp_path / "store", refresh=True
    )
    assert dry["anchors_to_remove"] >= 2

    applied = backfill_entity_anchors(
        store, apply=True, store_path=tmp_path / "store", refresh=True
    )
    assert applied["records_written"] >= 1
    refreshed = store.get(rid)
    # non-English fixture data: multilingual behavior under test — keep
    assert "entity:что" not in refreshed.tags
    assert "entity:users" not in refreshed.tags
    assert "entity:zephyrbot" in refreshed.tags
    # non-English fixture data: multilingual behavior under test — keep
    assert "что" not in refreshed.aaak_index
    assert "zephyrbot" in refreshed.aaak_index

    again = backfill_entity_anchors(
        store, apply=False, store_path=tmp_path / "store", refresh=True
    )
    assert again["records_to_anchor"] == 0, "refresh must be idempotent"


def test_entity_tags_never_enter_schema_induction() -> None:
    from dataclasses import dataclass, field
    from uuid import UUID, uuid4

    from iai_mcp.lilli.cycle.schema import _tag_cooccurrence

    @dataclass
    class _Rec:
        id: UUID = field(default_factory=uuid4)
        tags: list = field(default_factory=list)

    # 10 anchors per record over a shared vocabulary: without the entity:
    # exclusion this all-pairs dict goes quadratic in the anchor
    # vocabulary and floods schema induction with per-record identity.
    recs = [
        _Rec(tags=["capture", "role:user"]
             + [f"entity:name{i % 7}x{j}" for j in range(10)])
        for i in range(30)
    ]
    pairs = _tag_cooccurrence(recs)
    for key in pairs:
        assert not any(t.startswith("entity:") for t in key), (
            f"entity anchors leaked into schema co-occurrence: {set(key)}"
        )
    assert len(pairs) == 1  # capture+role:user only


def test_backfill_surfaces_apply_errors(tmp_path, monkeypatch):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors

    store = MemoryStore(path=tmp_path / "store")
    rid = UUID(capture_turn(
        store, cue="", text="Working on Zephyrbot rollout tonight.",
        tier="episodic", session_id="s-err", role="user", live_turn=True,
    )["record_id"])
    rec = store.get(rid)
    store.remove_tags(rid, [t for t in rec.tags if t.startswith("entity:")])
    store.set_aaak_index(rid, "")

    def _boom(*_a, **_k):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(store, "add_tags", _boom)
    applied = backfill_entity_anchors(store, apply=True, store_path=tmp_path / "store")
    assert applied["errors"], "a failed write must land in the errors list"
    assert "simulated write failure" in applied["errors"][0]
    assert applied["records_written"] == 0


def test_backfill_refreshes_index_of_tagged_records(tmp_path):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors

    store = MemoryStore(path=tmp_path / "store")
    rid = UUID(capture_turn(
        store, cue="", text="Debugging Zephyrbot spool rotation.",
        tier="episodic", session_id="s-refresh", role="user", live_turn=True,
    )["record_id"])
    # Entity tags present but the stored index is empty: the backfill must
    # refresh the index without proposing duplicate tags.
    store.set_aaak_index(rid, "")

    applied = backfill_entity_anchors(store, apply=True, store_path=tmp_path / "store")
    assert applied["anchors_proposed"] == 0
    assert applied["aaak_refreshed"] >= 1
    assert "zephyrbot" in store.get(rid).aaak_index


def test_community_stamp_refreshes_stored_aaak_index(tmp_path):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.sleep import _persist_community_assignment
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    rids = [
        UUID(capture_turn(
            store, cue="", text=f"Notes on Zephyrbot progress, number {i}.",
            tier="episodic", session_id="s-stamp", role="user", live_turn=True,
        )["record_id"])
        for i in range(3)
    ]
    flush_record_buffer(store)
    records_by_id = {rid: store.get(rid) for rid in rids}
    for rec in records_by_id.values():
        assert "/R:unknown/" in rec.aaak_index

    stamped = _persist_community_assignment(store, [rids], records_by_id)
    assert stamped == 3

    room = str(min(rids))[:8]
    for rid in rids:
        refreshed = store.get(rid)
        assert f"/R:{room}/" in refreshed.aaak_index, (
            "community stamping must refresh the stored index; got "
            f"{refreshed.aaak_index}"
        )
        assert "zephyrbot" in refreshed.aaak_index


def test_session_display_regenerates_stale_room_and_truncates(tmp_path, monkeypatch):
    from uuid import UUID

    from iai_mcp.capture import capture_turn
    from iai_mcp.session import _rich_club_segment
    from iai_mcp.sleep import _persist_community_assignment
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    long_names = " ".join(f"Zephyrmodule{i}core" for i in range(10))
    rids = [
        UUID(capture_turn(
            store, cue="", text=f"Anchors {long_names} in build {i}.",
            tier="episodic", session_id="s-disp", role="user", live_turn=True,
        )["record_id"])
        for i in range(2)
    ]
    flush_record_buffer(store)
    _persist_community_assignment(
        store, [rids], {rid: store.get(rid) for rid in rids}
    )
    # Freeze a stale mint-time index: the display must ignore it and show
    # the live room.
    store.set_aaak_index(rids[0], "W:E/R:unknown/E:-/T:capture")
    room = str(min(rids))[:8]

    # Legacy verbose label (kill-switch OFF): room hash is directly
    # observable, so the live-vs-stale regeneration is checked verbatim.
    monkeypatch.setenv("IAI_MCP_RICH_CLUB_COMPACT_LABEL", "0")
    legacy_segment = _rich_club_segment(store, rids)
    assert f"/R:{room}/" in legacy_segment
    assert "/R:unknown/" not in legacy_segment
    for line in legacy_segment.splitlines():
        head = line.split(": ", 1)[0]
        assert len(head.split(" (")[0]) <= 89, line

    # Compact label (default ON): the room hash is dropped from the display
    # entirely -- assert it never leaks, and the frozen stale index string
    # is never echoed verbatim (generation still regenerates fresh).
    monkeypatch.delenv("IAI_MCP_RICH_CLUB_COMPACT_LABEL", raising=False)
    compact_segment = _rich_club_segment(store, rids)
    assert "R:" not in compact_segment
    assert "W:E/R:unknown/E:-/T:capture" not in compact_segment
    for line in compact_segment.splitlines():
        head = line.split(": ", 1)[0]
        assert len(head.split(" (")[0]) <= 89, line
