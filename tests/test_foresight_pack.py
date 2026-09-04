"""Predictive next-turn pack: the tool feeds the agent, the agent never digs.

Precision is the product: a wrong injection poisons the agent worse than
silence, so the pack must (1) contain the related PRIOR-session memory,
(2) never echo the current session back at the agent, (3) never re-serve
what it already served, (4) replace contradicted beliefs with their
correctors, (5) stay inside the token budget, (6) prefer an EMPTY pack over
a low-confidence one — and the whole refresh must never fail a capture.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn, write_deferred_event
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
    # The assistant-tail lane defaults on and reads the real live spool
    # (Path.home()/.iai-mcp/.deferred-captures) when enabled -- every test in
    # this file not specifically exercising that lane must stay off it, both
    # so the suite never touches ambient user state and so window/cap
    # arithmetic assertions written before this lane existed keep holding.
    # The lane's own tests re-enable it explicitly with HOME redirected.
    # (raw env-var string, not foresight.FORESIGHT_ASSISTANT_TAIL_OFF_ENV --
    # this fixture must keep working during the RED phase, before that
    # constant exists.)
    monkeypatch.setenv("IAI_MCP_FORESIGHT_ASSISTANT_TAIL_OFF", "1")


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


# --- Multi-cue drown fixture -------------------------------------------------
# A rule buried in a long multi-topic prompt, reproducing the real-store
# finding in 248-RESEARCH.md: the whole-prompt cue's aggregate vector is
# dominated by the prompt's dominant register, missing a rare-token rule that
# a short derived cue finds directly. The Russian literals below ARE the
# value under test (a real drown case, not narrative prose) — non-English
# fixture DATA, no automated public-scrub marker exists in this codebase
# (see 248-01-SUMMARY.md); this comment is the exemption.
DROWN_TARGET = "Прогони через хуманайзер каждый пост перед публикацией."
DROWN_DERIVED_CUE = "хуианайзер"
DROWN_LONG_PROMPT = (
    "Объясни идею карпатого и как она применима на практике, наш "
    "проект тут хороший пример, я вчера выложил видео про новый "
    "алгоритм поиска, подписчиков в телеграме стало больше, кстати "
    "хуианайзер не забудь, и подготовь короткое объяснение обычным "
    "языком без сложных терминов для широкой аудитории"
)
DROWN_CROWD = [
    "Подготовь короткое объяснение на понятном языке без сложных слов.",
    "Наш проект хороший пример того, как объяснять сложные идеи просто.",
    "Объясни карпатого простыми словами для широкой аудитории без терминов.",
    "Подписчики в телеграме любят короткие понятные объяснения без терминов.",
    "Расскажи про карпатого на практике, обычным языком, без жаргона.",
    "Пиши разговорным тоном, как будто объясняешь другу на кухне.",
    "Готовь объяснение для широкой аудитории, коротко и без сложных слов.",
]
DROWN_FILLER = [
    "The hippocampus consolidates memories during sleep.",
    "Ashby's law: only variety can absorb variety.",
    "Coral reefs host a quarter of all known marine species.",
    "The printing press accelerated literacy across Europe.",
    "Sleep spindles mark memory transfer to the cortex.",
    "Grocery list: oat milk, rye bread, and tomatoes.",
    "The mitochondria is the powerhouse of the cell.",
    "Rivers carve canyons over geological timescales.",
    "The stock market closed slightly higher today.",
    "Bees communicate location through a waggle dance.",
    "The bridge was completed two years ahead of schedule.",
    "Volcanic ash can alter regional climate for years.",
    "The novel was translated into fourteen languages.",
    "Quantum entanglement puzzled physicists for decades.",
    "The marathon route passes through five neighborhoods.",
]

# Calibrated empirically against the real embedder this plan (see
# 248-01-SUMMARY.md): cos(long_prompt, target) ~= 0.7634,
# cos(derived_cue, target) ~= 0.8152. DROWN_FLOOR sits strictly between them;
# DROWN_WINDOW is the locked IAI_MCP_FORESIGHT_CUE_WINDOW default (<=16).
DROWN_FLOOR = 0.78
DROWN_WINDOW = 12

# Frozen single-cue baseline (captured this plan, while the multi-cue knobs
# are inert): the kill switch's contract is "reproduces this content, in
# this order, byte-for-byte" — NOT "matches whatever the default run
# produces" (after the multi-cue reorder the default run contains the target while the
# kill-switch run still misses it). Content-only (stamp/age/cos stripped),
# since the age label ticks with wall-clock time.
_FROZEN_SINGLE_CUE_LINES: tuple[str, ...] = (
    "Подготовь короткое объяснение на понятном языке без сложных слов.",
    "Наш проект хороший пример того, как объяснять сложные идеи просто.",
    "Объясни карпатого простыми словами для широкой аудитории без терминов.",
    "Подписчики в телеграме любят короткие понятные объяснения без терминов.",
    "Расскажи про карпатого на практике, обычным языком, без жаргона.",
)

_LINE_META_RE = re.compile(r"^- \[[^\]]*\] ")


def _pack_snippets(body: str) -> list[str]:
    """Content-only projection of a pack's memory lines: strips the
    ``[tier · stamp · cos X.XX]`` metadata, which is not stable across runs
    (age labels tick, cosines round)."""
    return [
        _LINE_META_RE.sub("", line)
        for line in body.splitlines()
        if line.startswith("- [")
    ]


def _seed_drown_fixture(store) -> None:
    _turn(store, DROWN_TARGET, "past-target")
    for i, txt in enumerate(DROWN_CROWD):
        _turn(store, txt, f"past-crowd-{i}")
    for i, txt in enumerate(DROWN_FILLER):
        _turn(store, txt, f"past-filler-{i}")


def _drown_cue_vec(store) -> "list[float]":
    from iai_mcp.embed import embed_query, embedder_for_store

    return embed_query(embedder_for_store(store), DROWN_LONG_PROMPT)


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


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_bank_fallback_unaffected_by_multi_cue_off(driver, tmp_path, monkeypatch):
    """memory_bank.append_recent_record runs upstream of candidate selection
    (capture.py:657-663) — the degraded-recall bank must carry the same
    content whether the multi-cue kill switch is set or not."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import memory_bank

    store = MemoryStore(path=tmp_path)

    def _bank_text_for(record_id):
        for obj in memory_bank.read_recent_records():
            if obj["id"] == record_id:
                return obj["text"]
        return None

    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    off = _turn(store, "Bank fallback content stays the same either way.", "bank-default")
    assert _bank_text_for(off["record_id"]) == (
        "Bank fallback content stays the same either way."
    ), f"[{driver}] bank-recent content missing/altered with the kill switch unset"

    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    on = _turn(store, "Bank fallback content stays the same either way, again.", "bank-off")
    assert _bank_text_for(on["record_id"]) == (
        "Bank fallback content stays the same either way, again."
    ), f"[{driver}] bank-recent content missing/altered with the kill switch set"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_derive_short_cues_determinism(driver, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.foresight import _derive_short_cues

    prompt = (
        "Explain carpathy on a real example, our project works well here, "
        "remember the humanizer skill for the telegram audience post."
    )
    first = _derive_short_cues(prompt, max_n=3)
    second = _derive_short_cues(prompt, max_n=3)
    assert first == second, f"[{driver}] cue derivation must be deterministic: {first!r} vs {second!r}"
    assert isinstance(first, list)
    assert all(isinstance(c, str) for c in first)
    assert len(first) <= 3


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_calibration_floor_and_window_lock_drown_fixture(driver, tmp_path, monkeypatch):
    """Empirically locks DROWN_FLOOR (F) and DROWN_WINDOW (W) against the real
    embedder: F sits strictly between the whole-prompt and derived-cue
    cosines to the target, a real single-cue pack MISSes at F (non-empty, no
    target), and W surfaces the target for the derived cue."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    # Pin single-cue mode explicitly — multi-cue derivation is active by
    # default (IAI_MCP_FORESIGHT_CUE_RESERVE default=1), and this test
    # documents the single-cue MISS baseline.
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)

    from iai_mcp.embed import embed_query, embedder_for_store

    embedder = embedder_for_store(store)
    long_vec = embed_query(embedder, DROWN_LONG_PROMPT)
    target_vec = embed_query(embedder, DROWN_TARGET)
    deriv_vec = embed_query(embedder, DROWN_DERIVED_CUE)

    c_long = _cosine(long_vec, target_vec)
    c_deriv = _cosine(deriv_vec, target_vec)
    print(f"[{driver}] c_long={c_long:.4f} c_deriv={c_deriv:.4f} F={DROWN_FLOOR} W={DROWN_WINDOW}")

    assert c_deriv > c_long, (
        f"[{driver}] drown condition requires the derived cue to rank the "
        f"target closer than the whole prompt does: c_long={c_long:.4f} "
        f"c_deriv={c_deriv:.4f}"
    )
    assert c_long < DROWN_FLOOR < c_deriv, (
        f"[{driver}] the calibrated floor must sit strictly between c_long "
        f"and c_deriv: {c_long:.4f} < {DROWN_FLOOR} < {c_deriv:.4f}"
    )

    report = foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=long_vec,
        session_id="today-calib",
    )
    assert report["packed"] > 0, (
        f"[{driver}] at least one distractor must clear the floor for the "
        f"long prompt — a real MISS, not silence: {report}"
    )
    body = foresight.pack_path(store, "today-calib").read_text(encoding="utf-8")
    assert DROWN_TARGET not in body, f"[{driver}] fixture MISS check failed: {body}"

    window_hits = store.query_similar(deriv_vec, k=DROWN_WINDOW)
    hit_surfaces = [rec.literal_surface for rec, _cos in window_hits]
    assert DROWN_TARGET in hit_surfaces, (
        f"[{driver}] the per-cue window (k={DROWN_WINDOW}) must surface the "
        f"drowned target for the derived cue: {hit_surfaces!r}"
    )


# --- SC-1: single-cue MISS is real, and the two xfail(strict) RED cases -----

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc1_single_cue_misses_drowned_target(driver, tmp_path, monkeypatch):
    """The documented MISS: at the calibrated floor, the single-cue pack is
    non-empty (real distractor content, not silence) yet never contains the
    drowned rule."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    # Pin single-cue mode explicitly — multi-cue derivation is active by
    # default (IAI_MCP_FORESIGHT_CUE_RESERVE default=1), and this test
    # documents the single-cue MISS baseline.
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    report = foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-sc1-single",
    )
    assert report["packed"] > 0, (
        f"[{driver}] single-cue pack must be non-empty at the floor: {report}"
    )
    body = foresight.pack_path(store, "today-sc1-single").read_text(encoding="utf-8")
    assert DROWN_TARGET not in body, f"[{driver}] documented single-cue MISS: {body}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc1_multi_cue_surfaces_drowned_target(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.setenv("IAI_MCP_FORESIGHT_CUE_RESERVE", "1")
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-sc1-multi",
    )
    body = foresight.pack_path(store, "today-sc1-multi").read_text(encoding="utf-8")
    assert DROWN_TARGET in body, (
        f"[{driver}] multi-cue pack must surface the drowned rule: {body}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc1_default_reserve_surfaces_drowned_target(driver, tmp_path, monkeypatch):
    """Load-bearing default-path guard: with NO IAI_MCP_FORESIGHT_CUE_RESERVE
    override at all (relying on the coded default, not an explicit "1"), the
    drowned rule still surfaces. A future edit that silently flips the
    default back to inert cannot pass a fully green suite without this."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.delenv("IAI_MCP_FORESIGHT_CUE_RESERVE", raising=False)
    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-sc1-default",
    )
    body = foresight.pack_path(store, "today-sc1-default").read_text(encoding="utf-8")
    assert DROWN_TARGET in body, (
        f"[{driver}] the default (unset) config must surface the drowned "
        f"rule — multi-cue derivation must be active by default: {body}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_full_pack_reserve_preserves_primary_order(driver, tmp_path, monkeypatch):
    """The owner-locked SC-2 ordering reading: with every slot already filled
    by the primary cue, the reserved cue must land the drowned rule in the
    reserved slot WITHOUT reordering the primary's first max_items-1 hits."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)
    max_items = foresight.FORESIGHT_MAX_ITEMS_DEFAULT

    # Pin single-cue mode explicitly — multi-cue derivation is active by
    # default (IAI_MCP_FORESIGHT_CUE_RESERVE default=1), so a bare delenv of
    # CUE_RESERVE would leave this baseline leg running multi-cue too.
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    single_report = foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-b-single",
    )
    assert single_report["packed"] == max_items, (
        f"[{driver}] fixture must fill every slot with zero humanizer "
        f"content: {single_report}"
    )
    single_lines = _pack_snippets(
        foresight.pack_path(store, "today-b-single").read_text(encoding="utf-8")
    )

    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    monkeypatch.setenv("IAI_MCP_FORESIGHT_CUE_RESERVE", "1")
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-b-multi",
    )
    multi_lines = _pack_snippets(
        foresight.pack_path(store, "today-b-multi").read_text(encoding="utf-8")
    )

    assert multi_lines[: max_items - 1] == single_lines[: max_items - 1], (
        f"[{driver}] primary cue's first {max_items - 1} slots must retain "
        f"order: {multi_lines!r} vs {single_lines!r}"
    )
    assert len(multi_lines) >= max_items and DROWN_TARGET in multi_lines[max_items - 1], (
        f"[{driver}] the reserved slot must carry the derived humanizer hit: "
        f"{multi_lines!r}"
    )


# --- Five green-now invariant guards: hold today, must hold after the multi-cue reorder --

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_kill_switch_reproduces_frozen_single_cue_baseline(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-kill",
    )
    body = foresight.pack_path(store, "today-kill").read_text(encoding="utf-8")
    snippets = tuple(_pack_snippets(body))
    assert snippets == _FROZEN_SINGLE_CUE_LINES, (
        f"[{driver}] kill switch must reproduce the frozen single-cue "
        f"baseline: {snippets!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc2_superset_at_reserve_zero(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    # Pin single-cue mode explicitly — multi-cue derivation is active by
    # default (IAI_MCP_FORESIGHT_CUE_RESERVE default=1). The "multi" leg
    # below uses the CUE_RESERVE=0 opt-out knob instead of the kill switch,
    # so this test validates the two opt-out paths converge.
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-d-single",
    )
    single_lines = _pack_snippets(
        foresight.pack_path(store, "today-d-single").read_text(encoding="utf-8")
    )

    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    monkeypatch.setenv("IAI_MCP_FORESIGHT_CUE_RESERVE", "0")
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-d-multi",
    )
    multi_lines = _pack_snippets(
        foresight.pack_path(store, "today-d-multi").read_text(encoding="utf-8")
    )

    assert multi_lines[: len(single_lines)] == single_lines, (
        f"[{driver}] reserve=0 must keep every single-cue line, in order: "
        f"{multi_lines!r} vs {single_lines!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc3_budget_and_item_caps_hold_on_drown_fixture(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    report = foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-e",
    )
    max_items = foresight.FORESIGHT_MAX_ITEMS_DEFAULT
    budget_chars = int(foresight.FORESIGHT_BUDGET_TOKENS_DEFAULT * 4)
    assert report["packed"] <= max_items, f"[{driver}] item cap breached: {report}"

    body = foresight.pack_path(store, "today-e").read_text(encoding="utf-8")
    content = [l for l in body.splitlines() if l.startswith("- ")]
    assert sum(len(l) for l in content) <= budget_chars, (
        f"[{driver}] injected content exceeded the budget: {body}"
    )
    assert len(body) < 6144, (
        f"[{driver}] published pack exceeds the hook's read cap: {len(body)}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_negative_control_no_noise_when_no_rule_is_drowned(driver, tmp_path, monkeypatch):
    """Top regression risk RESEARCH.md names: a prompt with NO drowned rule
    must pack identically whether the multi-cue kill switch is set or not —
    the derivation lane must never inject topically-irrelevant noise."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    for i, txt in enumerate(DROWN_CROWD):
        _turn(store, txt, f"neg-crowd-{i}")
    for i, txt in enumerate(DROWN_FILLER):
        _turn(store, txt, f"neg-filler-{i}")
    vec = _drown_cue_vec(store)

    monkeypatch.delenv("IAI_MCP_FORESIGHT_CUE_RESERVE", raising=False)
    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="neg-default",
    )
    default_lines = _pack_snippets(
        foresight.pack_path(store, "neg-default").read_text(encoding="utf-8")
    )

    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="neg-off",
    )
    off_lines = _pack_snippets(
        foresight.pack_path(store, "neg-off").read_text(encoding="utf-8")
    )

    assert default_lines == off_lines, (
        f"[{driver}] a prompt with no drowned rule must pack identically "
        f"regardless of the multi-cue kill switch: {default_lines!r} vs "
        f"{off_lines!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_negative_control_no_noise_at_production_floor(driver, tmp_path, monkeypatch):
    """Same top regression risk as the floor-calibrated negative control
    above, but pinned at the actual PRODUCTION min_cos (0.60), not the
    stricter DROWN_FLOOR (0.78) every other no-noise guard runs at. The
    derived-cue admission floor (IAI_MCP_FORESIGHT_CUE_MIN_COS) must keep
    the multi-cue lane clean even when the primary floor is loose."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.60")
    store = MemoryStore(path=tmp_path)
    for i, txt in enumerate(DROWN_CROWD):
        _turn(store, txt, f"neg60-crowd-{i}")
    for i, txt in enumerate(DROWN_FILLER):
        _turn(store, txt, f"neg60-filler-{i}")
    vec = _drown_cue_vec(store)

    monkeypatch.delenv("IAI_MCP_FORESIGHT_CUE_RESERVE", raising=False)
    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="neg60-default",
    )
    default_lines = _pack_snippets(
        foresight.pack_path(store, "neg60-default").read_text(encoding="utf-8")
    )

    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="neg60-off",
    )
    off_lines = _pack_snippets(
        foresight.pack_path(store, "neg60-off").read_text(encoding="utf-8")
    )

    assert default_lines == off_lines, (
        f"[{driver}] at the production floor, a prompt with no drowned rule "
        f"must still pack identically regardless of the multi-cue kill "
        f"switch: {default_lines!r} vs {off_lines!r}"
    )


# --- MEDIUM-1: an off-topic outlier must never evict the real rule ---------

_M1_RULE_TARGET = "Занеси тензодатчик на склад и подпиши серийный номер прибора."
_M1_OUTLIER_RECORD = "Сосед купил новый холодильник в прошлые выходные."
_M1_LONG_PROMPT = (
    "Объясни идею карпатого и как она применима на практике, наш "
    "проект тут хороший пример, я вчера выложил видео про новый "
    "алгоритм поиска, подписчиков в телеграме стало больше, кстати "
    "тензодатчик не забудь, да и холодильник тоже, и подготовь короткое "
    "объяснение обычным языком без сложных терминов для широкой аудитории"
)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_outlier_never_evicts_the_real_drowned_rule(driver, tmp_path, monkeypatch):
    """MEDIUM-1: "most distant from the mean cue" selects which short cues
    get queried, not which hit is relevant. `холодильник` (an unrelated
    errand, "the fridge arrived") out-distances the genuine buried rule
    `тензодатчик` for this prompt and has a STRONGER match to its own
    trigger word than the rule has to its own — a first-cue-to-clear-the-
    floor pick would let it win the single reserved slot. The real rule
    must still surface, ranked by coherence with the whole conversation."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    rule_result = _turn(store, _M1_RULE_TARGET, "past-rule")
    outlier_result = _turn(store, _M1_OUTLIER_RECORD, "past-outlier")
    for i, txt in enumerate(DROWN_CROWD):
        _turn(store, txt, f"m1-crowd-{i}")
    for i, txt in enumerate(DROWN_FILLER):
        _turn(store, txt, f"m1-filler-{i}")

    from uuid import UUID

    from iai_mcp.embed import embed_query, embedder_for_store

    vec = embed_query(embedder_for_store(store), _M1_LONG_PROMPT)

    # Precondition: this fixture must be adversarial BY CONSTRUCTION, not by
    # luck of the current embedder — lock the three numbers that make it so,
    # or a future embedder drift silently turns this into a no-op test.
    embedder = embedder_for_store(store)
    prefilter_cap = min(
        foresight.FORESIGHT_CUE_PREFILTER_CEILING,
        max(foresight.FORESIGHT_CUE_CAP_DEFAULT, foresight.FORESIGHT_CUE_CAP_DEFAULT * 2),
    )
    pool = foresight._derive_short_cues(_M1_LONG_PROMPT, prefilter_cap, store=store)
    by_tok = dict(zip(pool, embedder.embed_batch(pool, input_type="query")))
    assert "холодильник" in by_tok and "тензодатчик" in by_tok, (
        f"[{driver}] both tokens must survive derivation: {pool!r}"
    )
    outlier_dist = foresight._cos(by_tok["холодильник"], vec)
    rule_dist = foresight._cos(by_tok["тензодатчик"], vec)
    assert outlier_dist < rule_dist, (
        f"[{driver}] the outlier must out-distance (rank ahead of) the real "
        f"rule for the derived-cue queries to try it first: "
        f"outlier={outlier_dist:.4f} rule={rule_dist:.4f}"
    )

    rule_rec = store.get(UUID(rule_result["record_id"]))
    outlier_rec = store.get(UUID(outlier_result["record_id"]))
    outlier_self_match = foresight._cos(by_tok["холодильник"], outlier_rec.embedding)
    rule_self_match = foresight._cos(by_tok["тензодатчик"], rule_rec.embedding)
    assert outlier_self_match > rule_self_match, (
        f"[{driver}] the outlier's own match must be STRONGER than the "
        f"rule's — else ranking by derived-cue cosine alone would already "
        f"pick the rule and this test would prove nothing new: "
        f"outlier={outlier_self_match:.4f} rule={rule_self_match:.4f}"
    )
    rule_primary_cos = foresight._cos(rule_rec.embedding, vec)
    outlier_primary_cos = foresight._cos(outlier_rec.embedding, vec)
    assert rule_primary_cos > outlier_primary_cos, (
        f"[{driver}] the rule must cohere with the whole conversation more "
        f"than the outlier does — this is the signal the fix rides on: "
        f"rule={rule_primary_cos:.4f} outlier={outlier_primary_cos:.4f}"
    )

    foresight.refresh_pack(
        store, cue_text=_M1_LONG_PROMPT, cue_embedding=vec,
        session_id="today-m1",
    )
    body = foresight.pack_path(store, "today-m1").read_text(encoding="utf-8")
    assert _M1_RULE_TARGET in body, (
        f"[{driver}] the genuinely drowned rule must surface: {body}"
    )
    assert _M1_OUTLIER_RECORD not in body, (
        f"[{driver}] an off-topic outlier must not evict the real rule from "
        f"the single reserved slot: {body}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cue_budget_exceeded_degrades_to_single_cue(driver, tmp_path, monkeypatch):
    """MEDIUM-2: a wall-clock budget of ~0 must make the derived lane a
    no-op — the pack falls back to the primary tail, byte-identical to the
    frozen single-cue baseline. Proves the degrade branch is reachable, not
    just present."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.setenv("IAI_MCP_FORESIGHT_CUE_BUDGET_SEC", "0.0")
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-budget0",
    )
    body = foresight.pack_path(store, "today-budget0").read_text(encoding="utf-8")
    snippets = tuple(_pack_snippets(body))
    assert DROWN_TARGET not in body, (
        f"[{driver}] a zero derived-cue budget must degrade to single-cue, "
        f"never surfacing the derived-only target: {body}"
    )
    assert snippets == _FROZEN_SINGLE_CUE_LINES, (
        f"[{driver}] a zero derived-cue budget must reproduce the frozen "
        f"single-cue baseline: {snippets!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_tail_backfill_failure_still_publishes_primary_lines(driver, tmp_path, monkeypatch):
    """LOW-2: the reserve's unused-slot tail backfill is the only
    _pack_candidates call with a nonzero `start` (it resumes the primary's
    own retained candidate list). A fault there must not discard the
    primary lines already assembled — the pack still publishes what was
    built, short only the backfilled slot."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    for i, txt in enumerate(DROWN_CROWD):
        _turn(store, txt, f"tb-crowd-{i}")
    for i, txt in enumerate(DROWN_FILLER):
        _turn(store, txt, f"tb-filler-{i}")
    vec = _drown_cue_vec(store)

    real_pack_candidates = foresight._pack_candidates

    def _boom_on_tail(*args, **kwargs):
        if kwargs.get("start", 0) > 0:
            raise RuntimeError("simulated tail backfill fault")
        return real_pack_candidates(*args, **kwargs)

    monkeypatch.setattr(foresight, "_pack_candidates", _boom_on_tail)
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-tailfault",
    )
    pack = foresight.pack_path(store, "today-tailfault")
    assert pack.is_file(), f"[{driver}] a tail-backfill fault must not discard the pack"
    lines = _pack_snippets(pack.read_text(encoding="utf-8"))
    assert len(lines) == 4, (
        f"[{driver}] the primary's 4 slots must still publish when the tail "
        f"backfill faults: {lines!r}"
    )


def test_latin_lane_probes_are_capped():
    """MEDIUM-3: the Latin rarity lane must never issue more than
    _CUE_LATIN_PROBE_CAP store probes on the synchronous capture path, even
    when a long English prompt has thousands of distinct 4+char tokens and
    none of them ever clear the warm-lexical gate. Pure function: no store
    driver involved, so this runs once, not per-driver."""
    calls = {"n": 0}

    class _FakeStore:
        def lexical_query_warm(self, tok, k=1, min_idf=2.0):
            calls["n"] += 1
            return []

    words = " ".join(f"wordtoken{i}" for i in range(2000))
    result = foresight._derive_short_cues(words, max_n=3, store=_FakeStore())

    assert result == [], f"no candidate should clear the gate: {result!r}"
    assert calls["n"] <= foresight._CUE_LATIN_PROBE_CAP, (
        f"Latin lane probed the store {calls['n']} times, over the cap of "
        f"{foresight._CUE_LATIN_PROBE_CAP}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_curiosity_tunnel_question_retained_on_drown_fixture(driver, tmp_path, monkeypatch):
    """The derived-cue budget spend must never evict the in-tunnel pending
    question — the tunnel line is a per-turn guard independent of which cue
    filled the memory slots."""
    from iai_mcp import curiosity as _curiosity
    from iai_mcp.events import write_event

    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)

    question_text = "Стоит ли объяснять карпатого простыми словами для видео в телеграм?"
    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": "22222222-2222-2222-2222-222222222222",
            "text": question_text,
            "tier": "question",
            "entropy": 0.9,
            "turn": 1,
            "cue": question_text,
            "triggered_by": [],
        },
        severity="info",
        session_id="past-question",
    )
    _curiosity._cache_for(store)._refresh_once(store)

    vec = _drown_cue_vec(store)
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
        session_id="today-g",
    )
    body = foresight.pack_path(store, "today-g").read_text(encoding="utf-8")
    assert "open question" in body, (
        f"[{driver}] the in-tunnel pending question must ride the pack: {body}"
    )


@pytest.mark.perf
def test_multicue_added_latency_report(tmp_path, monkeypatch):
    """Report only: no fixed ratio ceiling. The embedder is sequential
    (embed.py supports_batch=False, ~N x one encode per derived cue), so a
    ratio target is not a sound gate — this prints the measured single-cue
    vs multi-cue delta with its N (cue_cap) x W (cue_window) basis. The only
    assertion is a very generous sanity bound plus a non-vacuity check: the
    reserve must actually have fired (the drowned target must appear) or the
    reported delta would be a fabricated number for a lane that never ran."""
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.delenv("IAI_MCP_FORESIGHT_CUE_RESERVE", raising=False)
    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    vec = _drown_cue_vec(store)

    cue_cap = int(os.environ.get(
        foresight.FORESIGHT_CUE_CAP_ENV, foresight.FORESIGHT_CUE_CAP_DEFAULT
    ))
    cue_window = int(os.environ.get(
        foresight.FORESIGHT_CUE_WINDOW_ENV, foresight.FORESIGHT_CUE_WINDOW_DEFAULT
    ))

    # Warmup: excludes one-time model-load/first-encode cost from either leg.
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec, session_id="perf-warmup",
    )

    n_iters = 5
    single_samples: list[float] = []
    multi_samples: list[float] = []
    last_multi_body = ""
    for i in range(n_iters):
        monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
        t0 = time.perf_counter()
        foresight.refresh_pack(
            store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
            session_id=f"perf-single-{i}",
        )
        single_samples.append(time.perf_counter() - t0)
        monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)

        t0 = time.perf_counter()
        foresight.refresh_pack(
            store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
            session_id=f"perf-multi-{i}",
        )
        multi_samples.append(time.perf_counter() - t0)
        last_multi_body = foresight.pack_path(store, f"perf-multi-{i}").read_text(
            encoding="utf-8"
        )

    single_samples.sort()
    multi_samples.sort()
    single_median = single_samples[len(single_samples) // 2]
    multi_median = multi_samples[len(multi_samples) // 2]
    delta_ms = (multi_median - single_median) * 1000

    # Non-vacuity: the reserve must have actually surfaced the drowned target
    # via a derived cue this run, or the timing above measured a lane that
    # never engaged.
    assert DROWN_TARGET in last_multi_body, (
        "multi-cue reserve never fired on the drown fixture — the latency "
        f"delta below would be a fabricated number: {last_multi_body!r}"
    )

    print(
        f"\nmulti-cue added latency: N={cue_cap} derived cues, W={cue_window} "
        f"ANN window per cue — single={single_median * 1000:.1f}ms "
        f"multi={multi_median * 1000:.1f}ms delta={delta_ms:.1f}ms "
        f"(n={n_iters} iters, median)"
    )
    # Sanity bound only — no fixed ratio ceiling.
    assert multi_median < single_median + 3.0, (
        f"multi-cue path took {multi_median * 1000:.1f}ms vs single "
        f"{single_median * 1000:.1f}ms — far outside a sane synchronous-"
        "capture bound"
    )


@pytest.mark.perf
def test_three_lane_added_latency_report(tmp_path, monkeypatch):
    """Report only: measures where the assistant-tail lane's added cost
    actually lands on the CAPTURE path -- primary alone vs
    primary+derived+assistant-tail. Non-vacuous: both the derived lane's
    drowned target AND the assistant-tail lane's counter-evidence must have
    fired this run, or the delta below is fabricated for lanes that never
    ran. This is the hermetic sanity check for the live per-turn
    capture-latency ceiling -- a finding landing near or over that bound
    here is a signal for the live gate, not a gate here."""
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(DROWN_FLOOR))
    monkeypatch.delenv("IAI_MCP_FORESIGHT_CUE_RESERVE", raising=False)
    monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(_TAIL_OFF_ENV, raising=False)
    store = MemoryStore(path=tmp_path)
    _seed_drown_fixture(store)
    _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-perf")
    vec = _drown_cue_vec(store)

    # Warmup: excludes one-time model-load/first-encode cost from either leg.
    write_deferred_event("perf3-warmup", "assistant", _TAIL_WRONG_CLAIM)
    foresight.refresh_pack(
        store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec, session_id="perf3-warmup",
    )

    n_iters = 5
    primary_samples: list[float] = []
    full_samples: list[float] = []
    last_full_body = ""
    for i in range(n_iters):
        monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
        monkeypatch.setenv(_TAIL_OFF_ENV, "1")
        t0 = time.perf_counter()
        foresight.refresh_pack(
            store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec,
            session_id=f"perf3-primary-{i}",
        )
        primary_samples.append(time.perf_counter() - t0)
        monkeypatch.delenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", raising=False)
        monkeypatch.delenv(_TAIL_OFF_ENV, raising=False)

        session = f"perf3-full-{i}"
        write_deferred_event(session, "assistant", _TAIL_WRONG_CLAIM)
        t0 = time.perf_counter()
        foresight.refresh_pack(
            store, cue_text=DROWN_LONG_PROMPT, cue_embedding=vec, session_id=session,
        )
        full_samples.append(time.perf_counter() - t0)
        last_full_body = foresight.pack_path(store, session).read_text(encoding="utf-8")

    primary_samples.sort()
    full_samples.sort()
    primary_median = primary_samples[len(primary_samples) // 2]
    full_median = full_samples[len(full_samples) // 2]
    delta_ms = (full_median - primary_median) * 1000

    assert DROWN_TARGET in last_full_body, (
        "derived lane never fired on the drown fixture this run — the "
        f"latency delta below would be fabricated: {last_full_body!r}"
    )
    assert _TAIL_COUNTER_EVIDENCE in last_full_body, (
        "assistant-tail lane never fired this run — the latency delta "
        f"below would be fabricated: {last_full_body!r}"
    )

    print(
        f"\n3-lane added capture-path latency: primary={primary_median * 1000:.1f}ms "
        f"primary+derived+tail={full_median * 1000:.1f}ms delta={delta_ms:.1f}ms "
        f"(n={n_iters} iters, median)"
    )
    if full_median * 1000 >= 250:
        print(
            f"FINDING: hermetic full-lane capture latency "
            f"{full_median * 1000:.1f}ms is at or over the live "
            "per-turn capture-latency 250ms ceiling before any "
            "daemon/socket overhead is added."
        )
    # Sanity bound only — no fixed ratio ceiling; the live 250ms ceiling is
    # the live gate's, not this hermetic report's.
    assert full_median < primary_median + 5.0, (
        f"3-lane path took {full_median * 1000:.1f}ms vs primary-only "
        f"{primary_median * 1000:.1f}ms — far outside a sane synchronous-"
        "capture bound"
    )


# --- Assistant-tail counter-evidence lane ------------------------------------
#
# Memory verifies the INPUT, never the OUTPUT: every lane above cues on the
# user's own message. This lane cues on the tail of the LAST ASSISTANT reply
# so a claim the assistant introduced pulls its own counter-evidence into the
# very next pack -- a separate bounded recall on its own vector, merged
# post-hoc into its own reserved slot, never a mean-pool with the primary cue.

_TAIL_RESERVE_ENV = "IAI_MCP_FORESIGHT_ASSISTANT_TAIL_RESERVE"
_TAIL_MIN_COS_ENV = "IAI_MCP_FORESIGHT_ASSISTANT_TAIL_MIN_COS"
_TAIL_MAX_AGE_ENV = "IAI_MCP_FORESIGHT_ASSISTANT_TAIL_MAX_AGE_SEC"
_TAIL_BUDGET_SEC_ENV = "IAI_MCP_FORESIGHT_ASSISTANT_TAIL_BUDGET_SEC"
_TAIL_OFF_ENV = "IAI_MCP_FORESIGHT_ASSISTANT_TAIL_OFF"


def _enable_tail(monkeypatch, tmp_path) -> None:
    # write_deferred_event / read_pending_live_events resolve their spool
    # root from Path.home() regardless of the store's own tmp_path -- every
    # test that turns the lane on must redirect HOME, or it reads and (via
    # the daemon, in production) drains the operator's real live spool.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(_TAIL_OFF_ENV, raising=False)


_TAIL_COUNTER_EVIDENCE = (
    "The analytics dashboard shipped last month and went fully live with "
    "chart support for every team."
)
_TAIL_WRONG_CLAIM = (
    "I will queue the analytics dashboard as new work since it has not "
    "been built yet."
)
_TAIL_UNRELATED_CUE = (
    "Can you check why the login page CSS has extra spacing around the "
    "submit button?"
)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_tail_surfaces_counter_evidence(driver, tmp_path, monkeypatch):
    """Assistant-tail lane core: the assistant wrongly claims the dashboard
    still needs to be built; the store already holds the refutation. The
    very next pack for this session must surface it, even though the
    user's own message is on an unrelated topic."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    store = MemoryStore(path=tmp_path)
    _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-1")

    session = "ct-session-core"
    write_deferred_event(session, "assistant", _TAIL_WRONG_CLAIM)

    from iai_mcp.embed import embed_query, embedder_for_store

    vec = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec, session_id=session,
    )
    pack = foresight.pack_path(store, session)
    body = pack.read_text(encoding="utf-8") if pack.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE in body, (
        f"[{driver}] the assistant-tail lane must surface the dashboard "
        f"counter-evidence for the very next pack: {body!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_distant_counter_evidence_not_evicted(driver, tmp_path, monkeypatch):
    """Distant-counter-evidence gate: the counter-evidence is topically
    distant from the CURRENT user cue (would be evicted if the derived
    lane's own re-score `_cos(rec.embedding, cue_vec)` were reused) but
    coherent with what the assistant just said. Locks both cosines
    numerically before the behavioral assertion so embedder drift cannot
    silently neuter this test."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    tail_floor = 0.5
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, str(tail_floor))
    store = MemoryStore(path=tmp_path)
    result = _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-2")

    from uuid import UUID

    from iai_mcp.embed import embed_query, embedder_for_store

    embedder = embedder_for_store(store)
    cue_vec = embed_query(embedder, _TAIL_UNRELATED_CUE)
    cue_cap = foresight.FORESIGHT_CUE_CAP_DEFAULT
    tail_cues = foresight._derive_short_cues(_TAIL_WRONG_CLAIM, cue_cap, store=store)
    tail_cue_text = " ".join(tail_cues) if tail_cues else _TAIL_WRONG_CLAIM[:512]
    tail_vec = embed_query(embedder, tail_cue_text)

    ce_rec = store.get(UUID(result["record_id"]))
    ce_cos_cue = foresight._cos(ce_rec.embedding, cue_vec)
    ce_cos_tail = foresight._cos(ce_rec.embedding, tail_vec)
    primary_floor = float(os.environ[foresight.FORESIGHT_MIN_COS_ENV])
    assert ce_cos_cue < primary_floor, (
        f"[{driver}] fixture precondition: counter-evidence must be too "
        f"distant from the PRIMARY cue to clear its floor: "
        f"cos={ce_cos_cue:.4f} floor={primary_floor}"
    )
    assert ce_cos_tail >= tail_floor, (
        f"[{driver}] fixture precondition: counter-evidence must clear the "
        f"TAIL floor against the assistant's own vector: "
        f"cos={ce_cos_tail:.4f} floor={tail_floor}"
    )

    session = "ct-session-distant"
    write_deferred_event(session, "assistant", _TAIL_WRONG_CLAIM)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=cue_vec, session_id=session,
    )
    pack = foresight.pack_path(store, session)
    body = pack.read_text(encoding="utf-8") if pack.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE in body, (
        f"[{driver}] a topically-distant-from-cue counter-evidence must "
        f"still survive via the tail lane's own scoring basis: {body!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_blended_cue_called_once_for_primary_only(driver, tmp_path, monkeypatch):
    """No-mean-pool gate: _blended_cue is the mean-pool the assistant-tail
    lane must never be folded into. It must run exactly once per
    refresh_pack call (the primary cue only), even when the assistant-tail
    lane fires."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    store = MemoryStore(path=tmp_path)
    _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-3")

    session = "ct-session-blend"
    write_deferred_event(session, "assistant", _TAIL_WRONG_CLAIM)

    calls = {"n": 0}
    real_blended = foresight._blended_cue

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_blended(*args, **kwargs)

    monkeypatch.setattr(foresight, "_blended_cue", _spy)

    from iai_mcp.embed import embed_query, embedder_for_store

    vec = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec, session_id=session,
    )
    pack = foresight.pack_path(store, session)
    body = pack.read_text(encoding="utf-8") if pack.exists() else ""
    # Tied to the lane actually firing: a call count of 1 is true whether or
    # not the lane exists, so this alone cannot prove the invariant holds
    # WITH the lane active -- the surfaced-content check makes that concrete.
    assert _TAIL_COUNTER_EVIDENCE in body, (
        f"[{driver}] the assistant-tail lane must have fired for this call "
        f"count to mean anything: {body!r}"
    )
    assert calls["n"] == 1, (
        f"[{driver}] _blended_cue must run exactly once (primary cue only), "
        f"even with the assistant-tail lane active: {calls['n']}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_tail_session_scoped_and_fresh(driver, tmp_path, monkeypatch):
    """Session A must never see session B's assistant-tail counter-evidence,
    and an assistant reply older than the age horizon is treated as absent
    -- the cross-session leakage mitigation."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    store = MemoryStore(path=tmp_path)
    _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-4")

    from iai_mcp.embed import embed_query, embedder_for_store

    session_a = "ct-session-a"
    session_b = "ct-session-b"
    write_deferred_event(session_b, "assistant", _TAIL_WRONG_CLAIM)

    vec_a = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec_a, session_id=session_a,
    )
    pack_a = foresight.pack_path(store, session_a)
    body_a = pack_a.read_text(encoding="utf-8") if pack_a.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE not in body_a, (
        f"[{driver}] session A must never see session B's assistant-tail "
        f"counter-evidence: {body_a!r}"
    )

    monkeypatch.setenv(_TAIL_MAX_AGE_ENV, "5")
    session_c = "ct-session-c"
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(seconds=3600)
    ).isoformat()
    write_deferred_event(session_c, "assistant", _TAIL_WRONG_CLAIM, ts=stale_ts)
    vec_c = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec_c, session_id=session_c,
    )
    pack_c = foresight.pack_path(store, session_c)
    body_c = pack_c.read_text(encoding="utf-8") if pack_c.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE not in body_c, (
        f"[{driver}] an assistant reply past the age horizon must be "
        f"treated as absent: {body_c!r}"
    )

    # Positive control, same store, same mechanism, correctly scoped and
    # fresh: without this leg, "must not leak" is trivially true whenever
    # the lane does not exist at all, and this test could never fail before
    # the lane is implemented.
    monkeypatch.delenv(_TAIL_MAX_AGE_ENV, raising=False)
    session_d = "ct-session-d"
    write_deferred_event(session_d, "assistant", _TAIL_WRONG_CLAIM)
    vec_d = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec_d, session_id=session_d,
    )
    pack_d = foresight.pack_path(store, session_d)
    body_d = pack_d.read_text(encoding="utf-8") if pack_d.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE in body_d, (
        f"[{driver}] a correctly-scoped, fresh assistant reply must surface "
        f"via the tail lane -- proves the negative legs above are a real "
        f"scope/staleness gate, not an unimplemented lane: {body_d!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_tail_dual_transport(driver, tmp_path, monkeypatch):
    """Dual-transport gate: the lane must surface the same counter-evidence
    whether refresh_pack is reached via the direct live capture path or via
    refresh_from_anchor (the backlog-drain-stashed anchor consumed on the
    next wake sequence) -- same session, same spool, same lane."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    store = MemoryStore(path=tmp_path)
    _turn(store, _TAIL_COUNTER_EVIDENCE, "past-dashboard-build-5")

    from iai_mcp.embed import embed_query, embedder_for_store

    session_direct = "ct-session-direct"
    write_deferred_event(session_direct, "assistant", _TAIL_WRONG_CLAIM)
    vec = embed_query(embedder_for_store(store), _TAIL_UNRELATED_CUE)
    foresight.refresh_pack(
        store, cue_text=_TAIL_UNRELATED_CUE, cue_embedding=vec,
        session_id=session_direct,
    )
    pack_direct = foresight.pack_path(store, session_direct)
    body_direct = pack_direct.read_text(encoding="utf-8") if pack_direct.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE in body_direct, (
        f"[{driver}] the direct live capture path must surface the "
        f"counter-evidence: {body_direct!r}"
    )

    session_drain = "ct-session-drain"
    write_deferred_event(session_drain, "assistant", _TAIL_WRONG_CLAIM)
    embedder = embedder_for_store(store)
    store._foresight_anchor = (time.time(), _TAIL_UNRELATED_CUE, session_drain)
    ok = foresight.refresh_from_anchor(store, embedder)
    assert ok, f"[{driver}] refresh_from_anchor must consume the stashed anchor"
    pack_drain = foresight.pack_path(store, session_drain)
    body_drain = pack_drain.read_text(encoding="utf-8") if pack_drain.exists() else ""
    assert _TAIL_COUNTER_EVIDENCE in body_drain, (
        f"[{driver}] the drain-originated capture path must surface the "
        f"same counter-evidence: {body_drain!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_tail_off_is_byte_identical(driver, tmp_path, monkeypatch):
    """The A/B baseline plan 04 relies on this: with the lane off, the
    PRIMARY lane's own store.query_similar call must use k=20 (not 24) and
    its own _pack_candidates call must use slot_limit=4 -- byte-identical to
    today for ANY max_items env value. With the lane on (default reserve),
    k widens to 24 while the primary cap stays 4."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    cue_text = "Please review the quarterly budget summary before the meeting tomorrow."

    from iai_mcp.embed import embed_query, embedder_for_store

    vec = embed_query(embedder_for_store(store), cue_text)

    real_query_similar = store.query_similar
    real_pack_candidates = foresight._pack_candidates
    calls = {"ks": [], "slot_limits": []}

    def _spy_query_similar(cue, *, k):
        calls["ks"].append(k)
        return real_query_similar(cue, k=k)

    def _spy_pack_candidates(*args, **kwargs):
        calls["slot_limits"].append(kwargs.get("slot_limit"))
        return real_pack_candidates(*args, **kwargs)

    monkeypatch.setattr(store, "query_similar", _spy_query_similar)
    monkeypatch.setattr(foresight, "_pack_candidates", _spy_pack_candidates)

    monkeypatch.setenv(_TAIL_OFF_ENV, "1")
    foresight.refresh_pack(
        store, cue_text=cue_text, cue_embedding=vec, session_id="offswitch-off",
    )
    assert calls["ks"], f"[{driver}] store.query_similar was never called"
    assert calls["ks"][0] == 20, (
        f"[{driver}] primary ANN window must stay 20 with the lane off: "
        f"{calls['ks']!r}"
    )
    assert calls["slot_limits"][0] == 4, (
        f"[{driver}] primary slot cap must stay 4 with the lane off: "
        f"{calls['slot_limits']!r}"
    )

    calls["ks"].clear()
    calls["slot_limits"].clear()
    monkeypatch.delenv(_TAIL_OFF_ENV, raising=False)
    foresight.refresh_pack(
        store, cue_text=cue_text, cue_embedding=vec, session_id="offswitch-on",
    )
    assert calls["ks"][0] == 24, (
        f"[{driver}] primary ANN window must widen to 24 with the lane on: "
        f"{calls['ks']!r}"
    )
    assert calls["slot_limits"][0] == 4, (
        f"[{driver}] primary slot cap must stay 4 with the lane on too: "
        f"{calls['slot_limits']!r}"
    )


_BUDGET_STALE_PAIRS = [
    (
        "The break room coffee machine only had drip coffee before the upgrade.",
        "The break room coffee machine now also makes espresso and cold brew.",
    ),
    (
        "The parking garage had no reserved spots for visitors previously.",
        "The parking garage now reserves five spots for visitors near the "
        "entrance.",
    ),
    (
        "The office gym used to close at six in the evening.",
        "The office gym now stays open until ten in the evening on weekdays.",
    ),
    (
        "The mail room only accepted packages before noon in the past.",
        "The mail room now accepts packages any time before six in the "
        "evening.",
    ),
    (
        "The rooftop lounge was closed during the winter months before.",
        "The rooftop lounge now stays open year round with heated seating.",
    ),
]
# Software/analytics content, deliberately off-topic from the office-
# amenities theme above: the primary lane must never surface this on its
# own -- only the assistant-tail lane's own vector (from the wrong claim
# text, near-identical to the stale side) can find it.
_BUDGET_TAIL_STALE = (
    "The analytics dashboard shipped last month and went fully live with "
    "chart support for every team."
)
_BUDGET_TAIL_CORRECTOR = (
    "The analytics dashboard was pulled back to beta after a data-accuracy "
    "bug was found in the chart totals."
)
_BUDGET_TAIL_WRONG_CLAIM = (
    "As discussed, the analytics dashboard shipped last month and is fully "
    "live with chart support for every team."
)
_BUDGET_CUE = "What changed recently around the office building and its amenities?"
# Calibrated against the real embedder (this plan): the office-amenities
# pairs score ~0.49-0.62 against _BUDGET_CUE; the dashboard pair scores
# ~0.38-0.49 -- a floor strictly between the two keeps dashboard content out
# of the primary lane's own reach entirely.
_BUDGET_PRIMARY_FLOOR = 0.55


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_six_item_pack_within_budget(driver, tmp_path, monkeypatch):
    """With the lane on, superseded (stale+corrector, two snippets each)
    lines from the primary lane's now-widened five-slot cap (derived lane
    off) plus the tail lane's own slot must never blow the published pack
    past its budget -- either every line fits, or the existing
    drop-over-budget guard removes the last one whole, never truncated or
    corrupted."""
    _select_driver(driver, monkeypatch)
    _enable_tail(monkeypatch, tmp_path)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, str(_BUDGET_PRIMARY_FLOOR))
    monkeypatch.setenv(_TAIL_MIN_COS_ENV, "0.5")
    monkeypatch.setenv("IAI_MCP_FORESIGHT_MULTI_CUE_OFF", "1")
    store = MemoryStore(path=tmp_path)

    from uuid import UUID

    from iai_mcp.store import flush_edge_buffer

    for i, (stale, fresh) in enumerate(_BUDGET_STALE_PAIRS):
        stale_r = _turn(store, stale, f"budget-stale-{i}")
        fresh_r = _turn(store, fresh, f"budget-fresh-{i}")
        store.add_contradicts_edge(
            UUID(stale_r["record_id"]), UUID(fresh_r["record_id"]),
        )
    tail_stale_r = _turn(store, _BUDGET_TAIL_STALE, "budget-tail-stale")
    tail_fresh_r = _turn(store, _BUDGET_TAIL_CORRECTOR, "budget-tail-fresh")
    store.add_contradicts_edge(
        UUID(tail_stale_r["record_id"]), UUID(tail_fresh_r["record_id"]),
    )
    flush_edge_buffer(store)

    session = "ct-session-budget"
    write_deferred_event(session, "assistant", _BUDGET_TAIL_WRONG_CLAIM)

    from iai_mcp.embed import embed_query, embedder_for_store

    vec = embed_query(embedder_for_store(store), _BUDGET_CUE)
    report = foresight.refresh_pack(
        store, cue_text=_BUDGET_CUE, cue_embedding=vec, session_id=session,
    )
    assert report["packed"] <= 6, f"[{driver}] item cap breached: {report}"

    pack = foresight.pack_path(store, session)
    body = pack.read_text(encoding="utf-8") if pack.exists() else ""
    budget_chars = int(foresight.FORESIGHT_BUDGET_TOKENS_DEFAULT * 4)
    content_lines = [l for l in body.splitlines() if l.startswith("- ")]
    assert sum(len(l) for l in content_lines) <= budget_chars, (
        f"[{driver}] injected content exceeded the budget: {body!r}"
    )
    for l in content_lines:
        assert l.startswith("- ⚠ superseded belief:") or l.startswith("- ["), (
            f"[{driver}] malformed/corrupted pack line: {l!r}"
        )
    assert _BUDGET_TAIL_CORRECTOR in body, (
        f"[{driver}] the assistant-tail lane's superseded corrector must "
        f"still surface within budget: {body!r}"
    )
