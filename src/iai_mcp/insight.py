from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from iai_mcp.claude_cli import (
    BudgetTracker,
    invoke_claude_once,
    verify_credentials_subscription,
)
from iai_mcp.daemon_state import load_state
from iai_mcp.events import query_events, write_event
from iai_mcp.schema import induce_schemas_tier0
from iai_mcp.tz import load_user_tz
from iai_mcp.types import MemoryRecord

INSIGHT_PROMPT_TEMPLATE: str = (
    "Here are 3 locally-found patterns from today + 1 surprising episode. "
    "What is the unifying insight? Reply in 1-2 sentences.\n\n"
    "Patterns:\n{patterns}\n\n"
    "Surprise:\n{surprise}"
)

PROMPT_ESTIMATE_TOKENS: int = 500

_SURPRISE_KINDS: frozenset[str] = frozenset({
    "art_gate_high_novelty",
    "contradiction_detected",
    "s4_contradiction",
    "s5_drift",
})


def _gather_patterns(store) -> list[str]:
    try:
        schemas = induce_schemas_tier0(store) or []
    except Exception:  # noqa: BLE001 -- pattern extraction must never crash insight
        schemas = []

    def _conf(s: Any) -> float:
        val = getattr(s, "confidence", None)
        if val is None and isinstance(s, dict):
            val = s.get("confidence")
        try:
            return float(val or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _text(s: Any) -> str:
        for attr in ("pattern", "description", "summary"):
            val = getattr(s, attr, None)
            if val:
                return str(val)
            if isinstance(s, dict) and s.get(attr):
                return str(s[attr])
        return str(s)

    schemas_sorted = sorted(schemas, key=_conf, reverse=True)
    top3 = schemas_sorted[:3]
    if not top3:
        return ["[no patterns yet]"]
    return [_text(s) for s in top3]


def _gather_surprise(store) -> str:
    try:
        since = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        candidates = query_events(store, since=since, limit=1000) or []
    except Exception:  # noqa: BLE001 -- event query must never crash insight
        candidates = []

    for event in candidates:
        if event.get("kind") in _SURPRISE_KINDS:
            data = event.get("data") or event
            return str(data)[:500]
    return "[no surprise yet]"


async def generate_overnight_insight(store, session_id: str) -> dict:
    creds = verify_credentials_subscription()
    if not creds.get("ok"):
        return {
            "ok": False,
            "reason": "credentials_check_failed",
            "text": None,
            "details": creds,
        }

    state = await asyncio.to_thread(load_state)
    tracker = BudgetTracker(state)

    try:
        tz = load_user_tz()
    except Exception:  # noqa: BLE001 -- tz lookup never crashes the call path
        tz = timezone.utc

    now = datetime.now(timezone.utc)
    tracker.reset_if_new_day(now, tz)

    if tracker.claude_disabled_after_billing_event():
        return {"ok": False, "reason": "claude_disabled_c3", "text": None}

    if not tracker.can_spend(PROMPT_ESTIMATE_TOKENS):
        return {"ok": False, "reason": "budget_exceeded", "text": None}

    patterns = _gather_patterns(store)
    surprise = _gather_surprise(store)
    prompt = INSIGHT_PROMPT_TEMPLATE.format(
        patterns="\n".join(f"- {p}" for p in patterns),
        surprise=surprise,
    )

    result = await invoke_claude_once(prompt, model="haiku")

    tokens_in = int(result.get("tokens_in", 0) or 0)
    tokens_out = int(result.get("tokens_out", 0) or 0)
    if tokens_in + tokens_out > 0:
        tracker.record(tokens_in, tokens_out, now)

    if not result.get("ok"):
        return {
            "ok": False,
            "reason": result.get("reason", "claude_call_failed"),
            "text": None,
            "details": {k: v for k, v in result.items() if k != "data"},
        }

    data = result.get("data") or {}
    insight_text = str(data.get("result", "")).strip()
    if not insight_text:
        return {"ok": False, "reason": "empty_insight", "text": None}

    # A real embedding, never zeros: a zero-embedded insight is semantically
    # unfindable (cosine ~0 to every cue) yet occupies an ANN slot. An
    # insight that cannot be embedded is not stored.
    try:
        from iai_mcp.embed import embedder_for_store
        insight_embedding = list(
            await asyncio.to_thread(embedder_for_store(store).embed, insight_text)
        )
    except Exception as exc:  # noqa: BLE001 -- skip beats junk
        return {
            "ok": False,
            "reason": f"insight_embed_failed: {type(exc).__name__}",
            "text": insight_text,
        }
    record = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=insight_text,
        aaak_index="",
        embedding=insight_embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{
            "ts": now.isoformat(),
            "cue": "overnight_insight",
            "session_id": session_id,
        }],
        created_at=now,
        updated_at=now,
        tags=["overnight_insight"],
        language="en",
    )
    try:
        object.__setattr__(record, "tag", "overnight_insight")
    except Exception:  # noqa: BLE001 -- attribute attach is best-effort
        pass

    try:
        await asyncio.to_thread(store.insert, record)
    except Exception as exc:  # noqa: BLE001 -- store errors must not crash daemon
        try:
            write_event(
                store,
                "overnight_insight_store_error",
                {"error": str(exc)[:500]},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write failure is non-fatal
            pass
        return {
            "ok": False,
            "reason": "store_insert_failed",
            "text": insight_text,
            "error": str(exc)[:500],
        }

    try:
        write_event(
            store,
            "overnight_insight_generated",
            {
                "session_id": session_id,
                "text_len": len(insight_text),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
        )
    except Exception:  # noqa: BLE001 -- event emission failure is non-fatal
        pass

    return {
        "ok": True,
        "text": insight_text,
        "reason": None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
