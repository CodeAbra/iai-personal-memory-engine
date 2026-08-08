"""Predictive next-turn pack: the tool feeds the agent, the agent never digs.

Precision is the product: a wrong injection poisons the agent worse than
silence, so the pack must (1) contain the related PRIOR-session memory,
(2) never echo the current session back at the agent, (3) never re-serve
what it already served, (4) replace contradicted beliefs with their
correctors, (5) stay inside the token budget, (6) prefer an EMPTY pack over
a low-confidence one — and the whole refresh must never fail a capture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture(autouse=True)
def _foresight_floor(monkeypatch):
    # Real-embedding cosines for crafted related sentences sit ~0.5-0.8;
    # unrelated pairs sit well below 0.4. Pin the floor between them.
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.45")
    # Deterministic cue: the goal blend is exercised by its own test.
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0")


@pytest.fixture(autouse=True)
def _fresh_working_tier():
    from iai_mcp import working_tier

    working_tier._reset()
    yield
    working_tier._reset()


def _turn(store, text, session):
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    return result


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_serves_prior_session_memory_not_own_echo(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "yesterday")
    _turn(store, "Grocery list: oat milk, rye bread, and tomatoes.", "yesterday")

    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")

    pack = foresight.pack_path(store, "today")
    assert pack.is_file(), f"[{driver}] a related prior memory must produce a pack"
    body = pack.read_text(encoding="utf-8")
    memory_lines = "\n".join(l for l in body.splitlines() if l.startswith("- "))
    assert "hippocampus consolidates" in memory_lines, body
    assert "How does sleep consolidation" not in memory_lines, (
        f"[{driver}] the current session must never be echoed back as memory"
    )
    assert "Grocery list" not in memory_lines, (
        f"[{driver}] unrelated memory leaked into the pack: {body}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_prefers_silence_below_confidence_floor(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "Grocery list: oat milk, rye bread, and tomatoes.", "yesterday")

    _turn(store, "Explain plate tectonics and earthquake belts.", "today")

    pack = foresight.pack_path(store, "today")
    if pack.exists():
        body = pack.read_text(encoding="utf-8")
        assert "Grocery list" not in body, (
            f"[{driver}] low-confidence memory content must never be injected: {body}"
        )
        assert "memory_recall" in body, (
            "a content-free pack may exist only as a go-search suggestion"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_never_reserves_already_served(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "yesterday")

    _turn(store, "Tell me about hippocampal memory consolidation.", "today")
    first = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    assert "hippocampus consolidates" in first

    _turn(store, "More about consolidation of memory in the hippocampus?", "today")
    pack = foresight.pack_path(store, "today")
    if pack.exists():
        assert "hippocampus consolidates" not in pack.read_text(encoding="utf-8"), (
            f"[{driver}] a memory already in the agent's context was re-served"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradicted_memory_travels_with_its_corrector(driver, tmp_path, monkeypatch):
    from uuid import UUID

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    stale = _turn(store, "The project uses LanceDB as its storage backend.", "yesterday")
    fresh = _turn(
        store, "The storage backend is the Hippo store over SQLite now.", "lastweek",
    )
    store.add_contradicts_edge(UUID(stale["record_id"]), UUID(fresh["record_id"]))
    from iai_mcp.store import flush_edge_buffer
    flush_edge_buffer(store)

    _turn(store, "Which storage backend does the project use, LanceDB?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    assert "superseded" in body, f"[{driver}] contradiction not flagged: {body}"
    assert "Hippo store" in body, f"[{driver}] corrector missing: {body}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_budget_and_item_caps_hold(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_BUDGET_TOKENS_ENV, "80")
    store = MemoryStore(path=tmp_path)
    for i in range(8):
        _turn(
            store,
            f"Sleep consolidation fact number {i}: the hippocampus replays "
            f"experiences and strengthens memory traces overnight, variant {i}.",
            f"old-{i}",
        )

    _turn(store, "How does sleep consolidation work in the hippocampus?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    content = [l for l in body.splitlines() if l.startswith("- ")]
    assert sum(len(l) for l in content) <= 80 * 4, (
        f"[{driver}] injected content exceeded the budget: {body}"
    )
    assert len(body) <= 80 * 4 + 600, f"[{driver}] frame overhead blew up: {len(body)}"


def test_kill_switch_and_fail_soft(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path)
    monkeypatch.setenv(foresight.FORESIGHT_OFF_ENV, "1")
    _turn(store, "The hippocampus consolidates memories during sleep.", "y")
    _turn(store, "Hippocampal consolidation during sleep?", "today")
    assert not foresight.pack_path(store, "today").exists()
    monkeypatch.delenv(foresight.FORESIGHT_OFF_ENV, raising=False)

    # a fault inside the anticipation pass must not break capture
    def _boom(*_a, **_k):
        raise RuntimeError("simulated foresight fault")

    monkeypatch.setattr(foresight, "_correctors", _boom)
    result = _turn(store, "Hippocampal sleep consolidation, once more?", "today")
    assert result["status"] == "inserted"


def test_hook_emits_fresh_pack(tmp_path):
    import os

    pack = tmp_path / "pack.md"
    pack.write_text("- [episodic] the exact remembered words\n", encoding="utf-8")
    env = dict(os.environ)
    env["IAI_MCP_FORESIGHT_PACK"] = str(pack)
    env["IAI_MCP_WORKING_TIER_CACHE"] = str(tmp_path / "absent.md")
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    proc = subprocess.run(
        [str(_HOOK)], input='{"prompt": "x"}', capture_output=True, text=True,
        env=env, timeout=10,
    )
    assert proc.returncode == 0
    assert "<iai-mcp-foresight>" in proc.stdout
    assert "exact remembered words" in proc.stdout


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_precision_eval_scripted_conversation(driver, tmp_path, monkeypatch):
    """The tunable metric: replay a scripted conversation over a seeded brain
    and score every injection. Precision must be perfect (nothing unrelated,
    no echo, no repeats); recall of the ground-truth memory >= 2/3."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    seeded = {
        "sleep": "The hippocampus consolidates memories during sleep.",
        "ashby": "Ashby's law: only variety can absorb variety.",
        "coral": "Coral reefs host a quarter of all known marine species.",
        "press": "The printing press accelerated literacy across Europe.",
    }
    for key, text in seeded.items():
        _turn(store, text, f"past-{key}")

    script = [
        ("How does the hippocampus consolidate memory in sleep?", "sleep"),
        ("What does Ashby say about variety and control?", "ashby"),
        ("Tell me about coral reef marine biodiversity.", "coral"),
    ]
    tp = 0
    fp: list[str] = []
    for turn_text, truth_key in script:
        _turn(store, turn_text, "today")
        pack = foresight.pack_path(store, "today")
        body = pack.read_text(encoding="utf-8") if pack.exists() else ""
        memory_lines = "\n".join(
            l for l in body.splitlines() if l.startswith("- ")
        )
        for key, text in seeded.items():
            marker = text[:30]
            if marker in memory_lines:
                if key == truth_key:
                    tp += 1
                else:
                    fp.append(f"{key} injected for {truth_key!r}")
        if turn_text[:25] in memory_lines:
            fp.append(f"echo of {turn_text!r}")

    assert not fp, f"[{driver}] precision violations: {fp}"
    assert tp >= 2, f"[{driver}] pack recall too low: {tp}/3"

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_goal_blend_disambiguates_vague_turn(driver, tmp_path, monkeypatch):
    """Monotropism: a vague turn inherits meaning from the active task."""
    from iai_mcp import working_tier

    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0.7")
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.50")
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "past-a")
    _turn(store, "Coral reefs host a quarter of all known marine species.", "past-b")

    import time as _time

    working_tier._reset()
    entry = working_tier.open_task(
        "Deep dive: hippocampal memory consolidation during sleep",
        session_id="today",
    )
    entry.last_turn_ts = _time.time()
    _turn(store, "And what strengthens it further?", "today")

    pack = foresight.pack_path(store, "today")
    body = pack.read_text(encoding="utf-8") if pack.exists() else ""
    assert "hippocampus consolidates" in body, (
        f"[{driver}] the active task's goal must steer a vague cue: {body!r}"
    )
    assert "Coral reefs" not in body


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_exact_authority_drops_unconfirmed_candidates(driver, tmp_path, monkeypatch):
    """A candidate the lossless authority does not confirm is ANN inflation."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    kept = _turn(store, "The hippocampus consolidates memories during sleep.", "y1")
    _turn(store, "Sleep spindles mark memory transfer to the cortex.", "y2")

    kept_id = kept["record_id"]

    def _fake_exact(vec, k=10, *, build_if_cold=True):
        from uuid import UUID

        return [(UUID(kept_id), 0.83)]

    monkeypatch.setattr(store, "exact_top_k", _fake_exact)
    report = foresight.refresh_pack(
        store,
        cue_text="hippocampal consolidation in sleep",
        cue_embedding=list(store.get(__import__("uuid").UUID(kept_id)).embedding),
        session_id="today",
    )
    assert report["exact_authority"] is True
    assert report["skipped_unconfirmed"] >= 1, report
    assert report["packed_ids"] == [kept_id], report
    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    assert "spindles" not in body, f"[{driver}] unconfirmed candidate served: {body}"

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_declares_incompleteness_and_search_path(driver, tmp_path, monkeypatch):
    """Anticipation may never masquerade as an exhaustive search: every pack
    carries the data-not-instructions frame and the go-search contract."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "past")
    _turn(store, "How does hippocampal sleep consolidation work?", "today")

    body = foresight.pack_path(store, "today").read_text(encoding="utf-8")
    assert "NOT an exhaustive search" in body
    assert "memory_recall" in body
    assert "DATA, not instructions" in body


def test_served_memory_becomes_eligible_again_after_ttl(tmp_path, monkeypatch):
    """The agent's context compacts over long sessions — a permanent
    do-not-repeat would silence exactly the memories that matter most."""
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "past")

    _turn(store, "Tell me about hippocampal memory consolidation.", "today")
    assert "hippocampus consolidates" in foresight.pack_path(store, "today").read_text(
        encoding="utf-8"
    )

    # Within the TTL: blocked.
    _turn(store, "More on hippocampus consolidation of memories?", "today")
    pack = foresight.pack_path(store, "today")
    if pack.exists():
        assert "hippocampus consolidates" not in pack.read_text(encoding="utf-8")

    # Age the serving stamp past the TTL: eligible again. The session reads
    # its own state file first, so age that one (and the global fallback).
    import json as _json
    for state_path in (
        foresight._state_path(store, "today"),
        foresight._state_path(store),
    ):
        if not state_path.exists():
            continue
        state = _json.loads(state_path.read_text(encoding="utf-8"))
        state["served"] = {k: 0.0 for k in state["served"]}
        state_path.write_text(_json.dumps(state), encoding="utf-8")

    _turn(store, "Remind me how the hippocampus consolidates memory at night.", "today")
    assert "hippocampus consolidates" in foresight.pack_path(store, "today").read_text(
        encoding="utf-8"
    ), "after the TTL a compacted-away memory must be servable again"


def test_hook_scopes_pack_to_its_session(tmp_path):
    import os
    import subprocess

    pack = tmp_path / "p.cached.md"
    pack.write_text("- [episodic] scoped memory line\n", encoding="utf-8")
    (tmp_path / "p.state.json").write_text(
        '{"session_id": "session-A", "served": {}}', encoding="utf-8"
    )
    env = dict(os.environ)
    env["IAI_MCP_FORESIGHT_PACK"] = str(pack)
    env["IAI_MCP_WORKING_TIER_CACHE"] = str(tmp_path / "absent.md")
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)

    def _run(payload: str) -> str:
        return subprocess.run(
            [str(_HOOK)], input=payload, capture_output=True, text=True,
            env=env, timeout=10,
        ).stdout

    same = _run('{"prompt":"x","session_id":"session-A"}')
    assert "scoped memory line" in same, "matching session must be served"

    other = _run('{"prompt":"x","session_id":"session-B"}')
    assert "scoped memory line" not in other, (
        "a pack anticipated for one conversation leaked into another"
    )

    unknown = _run('{"prompt":"x"}')
    assert "scoped memory line" in unknown, "unknown session fails open"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pack_dedupes_identical_historical_records(driver, tmp_path, monkeypatch):
    """Pre-dedup-gate duplicates exist in old stores; the pack must carry
    each distinct content ONCE — five verbatim copies are one hint."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    # Same text captured with distinct source identities bypasses the
    # exact-key reinforce and lands as true duplicates — the historical shape.
    for i in range(3):
        capture_turn(
            store, cue="",
            text="The hippocampus consolidates memories during sleep.",
            tier="episodic", session_id=f"old-{i}", role="user",
            source_uuid=f"historic-dup-{i}",
        )
    flush_record_buffer(store)
    row = store.db._conn.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()
    if int(row[0]) < 3:
        pytest.skip("capture gate merged the seeds; historical shape not reproducible here")

    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")

    pack = foresight.pack_path(store, "today")
    assert pack.is_file()
    body = pack.read_text(encoding="utf-8")
    hits = body.count("hippocampus consolidates")
    assert hits == 1, f"[{driver}] duplicate content packed {hits}x:\n{body}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pending_question_rides_pack_only_in_tunnel(driver, tmp_path, monkeypatch):
    from iai_mcp.events import write_event

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "yesterday")

    q_cue = "how does sleep consolidation strengthen hippocampal memory"
    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": "11111111-1111-1111-1111-111111111111",
            "text": 'Two memories disagree — which is current: "nightly" or "weekly"?',
            "tier": "question",
            "entropy": 0.95,
            "turn": 1,
            "cue": q_cue,
            "triggered_by": [],
        },
        severity="info",
        session_id="yesterday",
    )

    # The tunnel reads the refresh-ahead cache; warm it synchronously so the
    # test does not race the background refresh thread.
    from iai_mcp import curiosity as _curiosity

    _curiosity._cache_for(store)._refresh_once(store)

    # Off-tunnel turn: the question must NOT surface.
    _turn(store, "Grocery list: oat milk, rye bread, and tomatoes.", "today")
    pack = foresight.pack_path(store, "today")
    off_body = pack.read_text(encoding="utf-8") if pack.is_file() else ""
    assert "open question" not in off_body, (
        f"[{driver}] question surfaced outside its topic tunnel: {off_body}"
    )

    # In-tunnel turn: the question rides the pack.
    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")
    assert pack.is_file()
    body = pack.read_text(encoding="utf-8")
    assert "open question" in body and "disagree" in body, (
        f"[{driver}] in-tunnel turn must surface the pending question: {body}"
    )

    # Served once: the same turn again inside the TTL must not re-nag.
    _turn(store, "How does sleep consolidation strengthen hippocampal memory?", "today")
    body2 = pack.read_text(encoding="utf-8") if pack.is_file() else ""
    assert "open question" not in body2, (
        f"[{driver}] question re-served within the repeat TTL: {body2}"
    )
