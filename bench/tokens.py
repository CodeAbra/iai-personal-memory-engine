from __future__ import annotations

import json
import os
import sys
from typing import Callable

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import SessionStartPayload, assemble_session_start
from iai_mcp.store import MemoryStore

STEADY_LIMIT = 3000
FRESH_LIMIT = 8000


def _anthropic_count_tokens(text: str) -> int:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": text}],
    )
    return int(resp.input_tokens)


def _tiktoken_count(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _char4_count(text: str) -> int:
    return max(1, len(text) // 4)


def _payload_to_prompt(payload: SessionStartPayload) -> str:
    parts: list[str] = []
    if payload.l0:
        parts.append(f"# L0 identity\n{payload.l0}")
    if payload.l1:
        parts.append(f"# L1 critical facts\n{payload.l1}")
    for segment in payload.l2:
        parts.append(f"# L2 community\n{segment}")
    if payload.rich_club:
        parts.append(f"# Global rich-club\n{payload.rich_club}")
    compact = getattr(payload, "compact_handle", "")
    wake_depth = getattr(payload, "wake_depth", "minimal")
    if wake_depth == "minimal":
        if compact:
            parts.append(compact)
    else:
        lazy = [
            s for s in (
                getattr(payload, "identity_pointer", ""),
                getattr(payload, "brain_handle", ""),
                getattr(payload, "topic_cluster_hint", ""),
            ) if s
        ]
        if lazy:
            parts.append(" ".join(lazy))
    return "\n\n".join(parts)


def _fresh_prompt(payload: SessionStartPayload) -> str:
    prompt = _payload_to_prompt(payload)
    tail = "dynamic tail content " * 125
    return f"{prompt}\n\n{tail}" if prompt else tail


def run_token_bench(
    store: MemoryStore | None = None,
    n_runs: int = 3,
    count_tokens_fn: Callable[[str], int] | None = None,
    wake_depth: str = "minimal",
) -> dict:
    s = store if store is not None else MemoryStore()
    records_count = s.db.open_table("records").count_rows()
    if records_count > 0:
        _graph, assignment, rc = build_runtime_graph(s)
        payload = assemble_session_start(
            s, assignment, rc, profile_state={"wake_depth": wake_depth},
        )
    else:
        from iai_mcp.handle import encode_compact_handle
        from uuid import uuid4

        _compact = encode_compact_handle("", str(uuid4())[:8], "none", 0)
        payload = SessionStartPayload(
            l0="",
            l1="",
            l2=[],
            rich_club="",
            total_cached_tokens=max(1, len(_compact) // 4),
            total_dynamic_tokens=1000,
            compact_handle=_compact,
            wake_depth=wake_depth,
        )

    counter: Callable[[str], int]
    mode: str
    if count_tokens_fn is not None:
        counter = count_tokens_fn
        mode = "injected"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        counter = _anthropic_count_tokens
        mode = "anthropic-count-tokens"
    else:
        try:
            import tiktoken  # noqa: F401
            counter = _tiktoken_count
            mode = "tiktoken-cl100k-proxy"
        except ImportError:
            counter = _char4_count
            mode = "heuristic-char4"

    warm_prompt = _payload_to_prompt(payload) or "."
    fresh_prompt = _fresh_prompt(payload)
    fresh = int(counter(fresh_prompt))
    warm = [int(counter(warm_prompt)) for _ in range(n_runs)]

    fresh_ok = fresh <= FRESH_LIMIT
    steady_ok = all(w <= STEADY_LIMIT for w in warm)

    return {
        "fresh": fresh,
        "warm": warm,
        "steady_ok": steady_ok,
        "fresh_ok": fresh_ok,
        "mode": mode,
        "limits": {"steady": STEADY_LIMIT, "fresh": FRESH_LIMIT},
        "payload_cached_tokens": payload.total_cached_tokens,
        "payload_dynamic_tokens": payload.total_dynamic_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="bench.tokens",
        description=(
            "Session-start token bench. Also included: "
            "--wake-depth for measuring the lazy <=30-tok payload vs lazy "
            "eager dump vs the deep variant."
        ),
    )
    parser.add_argument(
        "--wake-depth",
        choices=("minimal", "standard", "deep"),
        default="minimal",
        help="Session-start payload mode (default: minimal per D5-02).",
    )
    args = parser.parse_args(argv)
    result = run_token_bench(wake_depth=args.wake_depth)
    print(json.dumps(result))
    return 0 if (result["steady_ok"] and result["fresh_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
