"""Recall-reliability fixture + dual-path eval harness for the "remind me
what's right" ("vspomni") failure class.

Scores ANSWER-level arbitration-sufficiency, not record-in-hits: a case
passes only when the rendered recall block (a) contains the current-truth
content AND (b) carries a staleness/supersession signal strong enough to
outrank a competing unhedged stale assertion the agent already holds in
context. Presence of the current-truth record alone is not a pass -- that
is exactly the failure mode a naive "hit rate" metric cannot see.

Two harnesses exercise the SAME cue against a read-only COPY of the
operator's own Hippo store (never the live store, per
``bench/recall_accuracy_real.py``'s discipline, reused here directly):

  * OFFLINE PACK PATH  -- ``foresight.refresh_pack`` (the per-turn
    anticipation pack a later human turn would receive via the
    ``<iai-mcp-foresight>`` hook block).
  * LIVE SOCKET PATH   -- a real isolated daemon subprocess on the store
    copy, queried the SAME way ``emit_socket_recall`` in
    ``src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh`` queries the
    live daemon (cue, limit 3, ``<iai-mcp-recall>`` render).

The labelled fixture (``~/.iai-mcp/eval-fixtures/labelled_vspomni_cues.json``)
lives OUTSIDE the repo tree and carries real incident content; it is
generated as a DRAFT on first run (marked ``locked: false``) and is never
committed. The committed sample at ``bench/fixtures/vspomni_sample.json``
carries synthetic alice-style content only, for the gate-wired unit test.

Run: .venv/bin/python bench/vspomni_fixture_eval.py --before
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import (  # noqa: E402
    _copy_real_store,
    _open_copy_store_shared,
)

_DEFAULT_FIXTURE_PATH = Path(
    "~/.iai-mcp/eval-fixtures/labelled_vspomni_cues.json"
).expanduser()

_MAX_HITS = 3
_MAX_CHARS = 400
_SESSION_ID_HARNESS = "harness-vspomni"
_SUN_PATH_MAX_BYTES = 104
_SOCKET_READ_LIMIT_BYTES = 16 * 1024 * 1024  # a rich memory_recall response can
# exceed asyncio's 64KB StreamReader default (bench/isolated_daemon_boot_proof.py
# convention, reused here).

# A staleness signal is a superseded sibling's own valid_to metadata, or an
# explicit dated-supersession line in the rendered text. A bare creation
# date on the current-truth record alone does NOT qualify -- that is
# metadata, not a "this replaces that" framing, and is exactly the
# distinction the incident recursed on (a confident-looking hint with no
# supersession framing lost arbitration to an unhedged stale claim).
_STALENESS_RE = re.compile(
    r"⚠|superseded|supersedes prior version dated|valid.to\s*[:=]",
    re.IGNORECASE,
)

_RU_TRIGGER_RE = re.compile(
    r"\b(вспомни(?:ть)?|напомни(?:ть)?)\b[,:]?\s*", re.IGNORECASE,
)


def fixture_path() -> Path:
    """Resolve the labelled fixture path: env override, else the local default."""
    env_val = os.environ.get("IAI_MCP_VSPOMNI_FIXTURE_PATH")
    if env_val:
        return Path(env_val).expanduser()
    return _DEFAULT_FIXTURE_PATH


def sample_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "vspomni_sample.json"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


# ---------------------------------------------------------------------------
# Scoring: pure logic over hit dicts (record_id, literal_surface, valid_to,
# valid_from, epistemic_status -- the shape core/_serializers.py:_hit_to_json
# already serializes) plus the case's competing_stale_assertion.
# ---------------------------------------------------------------------------


def score_case(
    hits: "list[dict]",
    case: dict,
    *,
    max_hits: int = _MAX_HITS,
    max_chars: int = _MAX_CHARS,
) -> dict:
    """ANSWER-level arbitration-sufficiency.

    Scored against ``hits[:max_hits]``, each ``literal_surface`` truncated
    to ``[:max_chars]`` -- the same render budget every consumer (hook,
    pack) applies; content past that boundary can never contribute, by
    construction (a multi-row table cut mid-row fails here, not later).
    A case passes only when BOTH hold: the current-truth answer content is
    present, AND a staleness/supersession signal is present. Neither alone
    is sufficient.
    """
    answer_norm = _norm(case.get("current_truth_answer", ""))
    content_present = False
    staleness_present = False
    rendered_lines: "list[str]" = []
    for h in (hits or [])[:max_hits]:
        surface = (h.get("literal_surface") or h.get("text") or "")[:max_chars]
        if not surface:
            continue
        rendered_lines.append(f"- {surface}")
        if answer_norm and answer_norm in _norm(surface):
            content_present = True
        if h.get("valid_to"):
            staleness_present = True
        if _STALENESS_RE.search(surface):
            staleness_present = True
    passed = content_present and staleness_present
    return {
        "case_id": case.get("id"),
        "content_present": content_present,
        "staleness_present": staleness_present,
        "passed": passed,
        "rendered_block": "\n".join(rendered_lines),
    }


def render_socket_block_raw(hits: "list[dict]") -> str:
    """Mirror ``emit_socket_recall`` in
    ``iai-mcp-per-turn-recall.sh`` EXACTLY: no metadata marker, matching
    what the hook actually renders today. Used to capture faithful
    evidence of the CURRENT (unfixed) production behavior."""
    lines = []
    for h in (hits or [])[:3]:
        text = (h.get("literal_surface") or h.get("text") or "")[:400]
        if text:
            lines.append(f"- {text}")
    if not lines:
        return ""
    return "<iai-mcp-recall>\n" + "\n".join(lines) + "\n</iai-mcp-recall>"


# ---------------------------------------------------------------------------
# OFFLINE PACK PATH.
# ---------------------------------------------------------------------------


def run_offline_pack_pass(store: Any, case: dict) -> dict:
    """foresight.refresh_pack against the store copy; capture the rendered
    pack block for evidence, and score the same cue's memory_recall hits
    (the canonical hit shape, with valid_to, that both harnesses share)."""
    from iai_mcp import core, foresight
    from iai_mcp.embed import embed_query, embedder_for_store

    embedder = embedder_for_store(store)
    cue_vec = list(embed_query(embedder, case["trigger_text"][:512]))

    session_id = case.get("session_id") or _SESSION_ID_HARNESS
    report = foresight.refresh_pack(
        store, cue_text=case["trigger_text"], cue_embedding=cue_vec, session_id=session_id,
    )
    pack_file = foresight.pack_path(store, session_id)
    pack_text_raw = pack_file.read_text(encoding="utf-8") if pack_file.exists() else ""
    rendered_pack_block = (
        "<iai-mcp-foresight>\n" + pack_text_raw[:6144] + "\n</iai-mcp-foresight>"
        if pack_text_raw
        else ""
    )

    response = core.dispatch(
        store,
        "memory_recall",
        {
            "cue": case["trigger_text"],
            "session_id": f"{session_id}-scoring",
            "budget_tokens": 2000,
        },
    )
    hits = response.get("hits", [])
    score = score_case(hits, case)

    return {
        "path": "offline_pack",
        "refresh_pack_report": {
            k: report.get(k)
            for k in ("packed", "superseded", "exact_authority", "written", "packed_ids")
        },
        "rendered_pack_block": rendered_pack_block,
        "hits_used_for_scoring": [h.get("record_id") for h in hits[:_MAX_HITS]],
        "score": score,
    }


# ---------------------------------------------------------------------------
# LIVE SOCKET PATH -- isolated daemon subprocess on the store copy, never
# the live store. Spawn/socket helpers mirror
# bench/isolated_daemon_boot_proof.py's shape, trimmed to spawn+recall+
# teardown (this harness does not need the full 7-stage proof).
# ---------------------------------------------------------------------------


def _spawn_daemon_for_copy(copy_root: Path) -> "tuple[subprocess.Popen, Path, Path]":
    scratch_home = copy_root.parent / "scratch-home"
    scratch_home.mkdir(parents=True, exist_ok=True)
    (scratch_home / ".iai-mcp").mkdir(parents=True, exist_ok=True)
    socket_dir = Path(tempfile.mkdtemp(prefix="iai-vspomni-sock-"))
    socket_path = socket_dir / ".daemon.sock"
    assert len(str(socket_path).encode("utf-8")) < _SUN_PATH_MAX_BYTES, (
        f"socket path too long for sun_path ({socket_path})"
    )

    env = os.environ.copy()
    env["HOME"] = str(scratch_home)
    env["IAI_MCP_STORE"] = str(copy_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(socket_path)
    env["LILLI_STORAGE_DRIVER"] = "lilli"
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    proc = subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc, socket_path, socket_dir


def _await_socket(socket_path: Path, timeout: float = 30.0) -> float:
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return time.monotonic() - t0
        time.sleep(0.05)
    raise TimeoutError(f"socket did not appear at {socket_path} within {timeout}s")


def _recall_over_socket(socket_path: Path, cue: str, *, hit_limit: int = 3, timeout: float = 30.0) -> dict:
    async def _runner() -> dict:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path), limit=_SOCKET_READ_LIMIT_BYTES),
            timeout=timeout,
        )
        try:
            req = {
                "jsonrpc": "2.0", "id": 1, "method": "memory_recall",
                "params": {"cue": cue, "limit": hit_limit, "session_id": _SESSION_ID_HARNESS},
            }
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise ConnectionError("daemon closed the connection with no response")
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return asyncio.run(_runner())


def _teardown_daemon(proc: subprocess.Popen, socket_dir: Path) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15.0)
    shutil.rmtree(socket_dir, ignore_errors=True)


def run_live_socket_pass(copy_root: Path, case: dict) -> dict:
    """Boot an isolated daemon on ``copy_root`` (never the live store),
    send ``memory_recall`` with cue=trigger_text limit 3, capture the
    rendered ``<iai-mcp-recall>`` block exactly as the hook renders it, and
    score the same hits for arbitration-sufficiency."""
    proc, socket_path, socket_dir = _spawn_daemon_for_copy(copy_root)
    try:
        socket_up_sec = _await_socket(socket_path)
        t0 = time.perf_counter()
        resp = _recall_over_socket(socket_path, case["trigger_text"], hit_limit=_MAX_HITS)
        recall_ms = (time.perf_counter() - t0) * 1000.0
        hits = (resp.get("result") or {}).get("hits") or []
        rendered_block = render_socket_block_raw(hits)
        score = score_case(hits, case)
        return {
            "path": "live_socket",
            "socket_up_sec": socket_up_sec,
            "recall_ms": recall_ms,
            "rendered_recall_block": rendered_block,
            "hits_used_for_scoring": [h.get("record_id") for h in hits[:_MAX_HITS]],
            "score": score,
        }
    finally:
        _teardown_daemon(proc, socket_dir)


def measure_per_turn_overhead(copy_root: Path, cue: str, *, warm_n: int = 5) -> dict:
    """Time the ``emit_socket_recall`` round trip (isolated daemon warm)
    against the store copy, as its OWN number -- NOT compared to
    ``_WARM_SLA_SEC`` (that constant bounds a different call site, the
    ``memory_recall`` MCP tool)."""
    proc, socket_path, socket_dir = _spawn_daemon_for_copy(copy_root)
    try:
        socket_up_sec = _await_socket(socket_path)
        _recall_over_socket(socket_path, cue, hit_limit=3)  # cold call primes the connection
        warm_ms: "list[float]" = []
        for _ in range(warm_n):
            t0 = time.perf_counter()
            _recall_over_socket(socket_path, cue, hit_limit=3)
            warm_ms.append((time.perf_counter() - t0) * 1000.0)
        return {
            "socket_up_sec": socket_up_sec,
            "warm_round_trip_ms": warm_ms,
            "warm_round_trip_mean_ms": (sum(warm_ms) / len(warm_ms)) if warm_ms else None,
        }
    finally:
        _teardown_daemon(proc, socket_dir)


# ---------------------------------------------------------------------------
# Baseline breakout: lexical/substring/BM25 vs embedding/ANN contribution
# to surfacing the current-truth record(s).
# ---------------------------------------------------------------------------


def strip_trigger_phrase(text: str) -> str:
    """Strip the RU 'vspomni/napomni' trigger phrasing from a trigger
    turn, leaving the topical remainder as the stripped cue -- the input
    the breakout mode hands to both the lexical lane and the (EN-only)
    embedder."""
    return _RU_TRIGGER_RE.sub("", text).strip()


def attribute_breakout(
    relevant: "set[str]",
    lexical_ids: "set[str]",
    embedding_ids: "set[str]",
    *,
    case_id: "str | None" = None,
    stripped_cue: str = "",
) -> dict:
    """Pure attribution logic: does substring/BM25 alone carry the
    current-truth record(s), only the embedding/ANN lane, both, or
    neither. No store/embedder access -- testable with synthetic sets."""
    lexical_hit = bool(relevant & lexical_ids)
    embedding_hit = bool(relevant & embedding_ids)
    if lexical_hit and embedding_hit:
        channel = "both"
    elif lexical_hit:
        channel = "lexical"
    elif embedding_hit:
        channel = "embedding"
    else:
        channel = "neither"
    return {
        "case_id": case_id,
        "stripped_cue": stripped_cue,
        "lexical_hit": lexical_hit,
        "embedding_hit": embedding_hit,
        "channel": channel,
    }


def breakout_case(store: Any, case: dict, *, k: int = 8) -> dict:
    """Real-store breakout for one case: mines the lexical/BM25 lane and
    the embedding/ANN lane independently against the stripped cue, then
    attributes contribution via ``attribute_breakout``."""
    from iai_mcp.embed import embed_query, embedder_for_store

    stripped_cue = strip_trigger_phrase(case["trigger_text"])
    relevant = {str(r) for r in (case.get("current_truth_record_ids") or [])}

    lexical_ids: "set[str]" = set()
    try:
        lex_hits = store.lexical_query_warm(stripped_cue, k=k)
        lexical_ids = {str(rid) for rid, _score in lex_hits}
    except Exception:  # noqa: BLE001 -- lexical lane best-effort
        lexical_ids = set()

    embedder = embedder_for_store(store)
    cue_vec = list(embed_query(embedder, stripped_cue[:512]))
    try:
        ann_hits = store.exact_top_k(cue_vec, k=k, build_if_cold=True)
        embedding_ids = {str(rid) for rid, _cos in ann_hits}
    except Exception:  # noqa: BLE001 -- ANN lane best-effort
        embedding_ids = set()

    return attribute_breakout(
        relevant, lexical_ids, embedding_ids,
        case_id=case.get("id"), stripped_cue=stripped_cue,
    )


# ---------------------------------------------------------------------------
# Fixture cases -- LOCKED 2026-08-23, owner-confirmed; do not substitute.
# Two cases only: gsd_routing (the arbitration/tracer case) and
# push_protocol. A third class (humanizer) was considered and dropped by
# owner decision -- its verbatim trigger turn and current rule were not
# available at lock time.
# ---------------------------------------------------------------------------


def locked_gsd_routing_case() -> dict:
    """Case 1 -- the arbitration/tracer case: the correct record IS
    present in the rendered block, yet the naive answer is wrong until a
    dated supersession signal outranks the competing stale assertion."""
    return {
        "id": "gsd_routing_case_1",
        "class": "gsd_routing",
        "trigger_text": "мне кажется это не правильный роутинг… вспомни",
        "session_id": None,
        "current_truth_record_ids": [
            "958d0de8-dc0b-4f9d-aebf-b8e0644be144",
            "8060f113-7e42-4d0e-9c6d-05257e2c0c72",
        ],
        "superseded_record_ids": [],
        "current_truth_answer": (
            "Balanced routing profile: plan=opus; research, plan-check, "
            "execute, code-review, review-fix, verify=sonnet. Planning is "
            "the only Opus stage; everything else Sonnet."
        ),
        "competing_stale_assertion": (
            "Old pinned line: \"code-review=Opus, verify=Opus\" "
            "(superseded 2026-08-23)."
        ),
        "locked": True,
        "notes": (
            "trigger_text is a verbatim quote from RECALL-INVESTIGATION-"
            "2026-08-23.md's pain-quantification table (08-23, 'эта сессия'); "
            "the source session_id for that quote is not recorded there. "
            "current_truth_record_ids are candidates discovered via a rich "
            "manual-cue probe against a store copy (cos 0.90-0.93), not "
            "owner-confirmed ids."
        ),
    }


def locked_push_protocol_case() -> dict:
    """Case 2 -- push protocol. Not the designated arbitration case (case
    1 is); no competing_stale_assertion was supplied at lock time."""
    return {
        "id": "push_protocol_case_1",
        "class": "push_protocol",
        "trigger_text": "мы не делаем пуш в мейн вспомни правильный протокол",
        "session_id": None,
        "current_truth_record_ids": [],
        "superseded_record_ids": [],
        "current_truth_answer": (
            "Private repo: push allowed on explicit 'пуш'. PUBLIC repo: "
            "READ-ONLY -- never push (manager's lane, only on explicit "
            "owner 'go')."
        ),
        "competing_stale_assertion": None,
        "locked": True,
        "notes": (
            "current_truth_record_ids not yet discovered against a store "
            "copy -- unlike case 1, no probe has been run for this case."
        ),
    }


def build_locked_fixture() -> dict:
    return {
        "schema_version": 1,
        "cases": [locked_gsd_routing_case(), locked_push_protocol_case()],
    }


def _validate_fixture_dict(raw: dict) -> "list[dict]":
    if "cases" not in raw or not isinstance(raw["cases"], list):
        raise ValueError("fixture missing 'cases' array")
    cases = raw["cases"]
    for entry in cases:
        for required_key in ("id", "class", "trigger_text", "current_truth_answer"):
            if required_key not in entry:
                raise ValueError(f"fixture case missing required key {required_key!r}: {entry}")
    return cases


def load_or_draft_fixture(*, regenerate: bool = False) -> dict:
    fpath = fixture_path()
    if regenerate or not fpath.exists():
        fixture = build_locked_fixture()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[vspomni_fixture_eval] wrote fixture ({len(fixture['cases'])} cases) to {fpath}", flush=True)
    else:
        fixture = json.loads(fpath.read_text(encoding="utf-8"))
    _validate_fixture_dict(fixture)
    return fixture


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def run_before_measurement(*, regenerate: bool = False) -> dict:
    fixture = load_or_draft_fixture(regenerate=regenerate)
    case = fixture["cases"][0]  # the routing case -- the one wired end-to-end

    with tempfile.TemporaryDirectory(prefix="iai-mcp-vspomni-") as td:
        copy_root = _copy_real_store(Path(td) / "copy")
        store = _open_copy_store_shared(copy_root)
        try:
            offline_result = run_offline_pack_pass(store, case)
        finally:
            store.close()

        live_result = run_live_socket_pass(copy_root, case)
        overhead = measure_per_turn_overhead(copy_root, case["trigger_text"])

    return {
        "case": case,
        "offline_pack": offline_result,
        "live_socket": live_result,
        "per_turn_overhead": overhead,
    }


def run_breakout(*, regenerate: bool = False) -> dict:
    fixture = load_or_draft_fixture(regenerate=regenerate)
    case = fixture["cases"][0]

    with tempfile.TemporaryDirectory(prefix="iai-mcp-vspomni-breakout-") as td:
        copy_root = _copy_real_store(Path(td) / "copy")
        store = _open_copy_store_shared(copy_root)
        try:
            try:
                store.lexical_search("_vspomni_eval_warmup_", k=1)
            except Exception as exc:  # noqa: BLE001 -- warm-up is best-effort
                print(f"[vspomni_fixture_eval] lexical warm-up skipped: {exc}", flush=True)
            breakout = breakout_case(store, case)
        finally:
            store.close()

    return {"case_id": case["id"], "breakout": breakout}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-path (offline pack + live socket) eval harness for the vspomni recall-reliability fixture.",
    )
    parser.add_argument("--regenerate", action="store_true", help="rewrite the local DRAFT fixture")
    parser.add_argument("--before", action="store_true", help="run the BEFORE measurement (both harnesses)")
    parser.add_argument("--breakout", action="store_true", help="run the lexical/embedding breakout")
    args = parser.parse_args()

    if args.before or not args.breakout:
        result = run_before_measurement(regenerate=args.regenerate)
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    if args.breakout:
        result = run_breakout(regenerate=args.regenerate)
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
