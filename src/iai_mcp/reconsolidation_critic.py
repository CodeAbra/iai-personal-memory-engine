from __future__ import annotations

import json
import logging
import uuid
from typing import Iterable

logger = logging.getLogger(__name__)


MAX_RECORDS_PER_CALL: int = 100


_SURFACE_TRUNC: int = 300

_BATCH_PROMPT_HEADER: str = (
    "Score each memory record for prediction error. Output ONLY a JSON "
    'array, no prose. Format: [{"id":"<uuid>","err":<float 0.0-1.0>}, ...]. '
    "0.0 = memory still accurate; 1.0 = memory contradicted by current "
    "state. Records:\n"
)


def _build_batch_prompt(items: list[tuple[uuid.UUID, str]]) -> str:
    lines = [
        json.dumps({
            "id": str(rid),
            "surface": (surface or "")[:_SURFACE_TRUNC],
        })
        for rid, surface in items
    ]
    return _BATCH_PROMPT_HEADER + "\n".join(lines)


def _parse_batch_response(raw: str, expected_ids: set[uuid.UUID]) -> dict[uuid.UUID, float]:
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {}
    if not isinstance(parsed, list):
        return {}

    out: dict[uuid.UUID, float] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        rid_str = item.get("id")
        err_raw = item.get("err")
        if rid_str is None or err_raw is None:
            continue
        try:
            rid = uuid.UUID(str(rid_str))
        except (TypeError, ValueError):
            continue
        if rid not in expected_ids:
            continue
        try:
            err = float(err_raw)
        except (TypeError, ValueError):
            continue
        out[rid] = max(0.0, min(1.0, err))
    return out


def evaluate_batch_reconsolidation(
    items: Iterable[tuple[uuid.UUID, str]],
    *,
    llm_enabled: bool = True,
    max_records: int = MAX_RECORDS_PER_CALL,
) -> dict[uuid.UUID, float]:
    if not llm_enabled:
        return {}

    pool = list(items)
    if not pool:
        return {}
    if len(pool) > max_records:
        pool = pool[-max_records:]
    expected_ids = {rid for rid, _ in pool}

    try:
        from iai_mcp.claude_cli import verify_credentials_subscription
        from iai_mcp.reflection_provider import (
            configured_reflection_provider,
            invoke_reflection_sync,
        )
    except ImportError:
        return {}

    try:
        provider = configured_reflection_provider()
    except ValueError as exc:
        logger.info("reconsolidation critic: %s", exc)
        return {}

    # The credential check gates the Anthropic subscription only.
    if provider == "claude":
        creds = verify_credentials_subscription()
        if not creds.get("ok"):
            return {}

    prompt = _build_batch_prompt(pool)

    try:
        result = invoke_reflection_sync(prompt, model="haiku", provider=provider)
    except Exception as exc:  # noqa: BLE001 -- critic must never raise into REM
        logger.debug("reconsolidation critic subprocess raised: %s", exc)
        return {}

    if not result.get("ok"):
        logger.info(
            "reconsolidation critic call failed: reason=%s",
            result.get("reason"),
        )
        return {}

    raw_text = ""
    data = result.get("data") or {}
    if isinstance(data, dict):
        raw_text = str(data.get("result") or data.get("text") or "")

    scores = _parse_batch_response(raw_text, expected_ids)
    if not scores and raw_text.strip():
        # Distinguish "provider returned prose we cannot parse" from
        # "nothing was labile" — a silent {} hides a broken provider.
        logger.warning(
            "reconsolidation critic output unparseable (provider=%s, %d chars)",
            provider, len(raw_text),
        )
    return scores


PROMPT_TEMPLATE: str = (
    "Given this stored memory `{literal_surface}` and the current state of "
    "repository `{current_summary}`, return prediction_error in 0.0-1.0 where "
    "1.0 means the memory is now wrong/contradicted."
)


def call_critic(
    literal_surface: str,
    current_summary: str = "",
    store=None,  # noqa: ARG001 -- legacy signature, retained for back-compat
    *,
    llm_enabled: bool = True,
    has_api_key: bool = False,  # noqa: ARG001 -- legacy, ignored
    estimated_usd: float = 0.001,  # noqa: ARG001 -- legacy, ignored
) -> float:
    if not llm_enabled:
        return 0.0
    synthetic_rid = uuid.uuid4()
    result = evaluate_batch_reconsolidation(
        [(synthetic_rid, literal_surface)],
        llm_enabled=llm_enabled,
        max_records=1,
    )
    return result.get(synthetic_rid, 0.0)
