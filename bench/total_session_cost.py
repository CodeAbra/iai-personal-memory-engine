from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from bench.tokens import (
    _anthropic_count_tokens,
    _char4_count,
    _tiktoken_count,
)


SCRIPT_NAME = "session-cost-v1"

_SCRIPT: list[dict] = [
    {
        "kind": "recall",
        "input": "Tell me the decisions we made about the storage architecture",
    },
    {
        "kind": "chat",
        "input": "Let me iterate on this function; no recall needed here",
    },
    {
        "kind": "recall",
        "input": "What did I say about bench discipline?",
    },
    {
        "kind": "recall_cross_community",
        "input": "What is the connection between the formality knob and the autistic kernel?",
    },
    {
        "kind": "save",
        "input": "Decision locked: use cachetools TTLCache for the session cache LRU",
    },
    {
        "kind": "introspect",
        "input": "profile_get_set operation=get knob=wake_depth",
    },
    {
        "kind": "chat",
        "input": "Continuing this refactor; still no recall",
    },
    {
        "kind": "recall",
        "input": "alice said something about pressplay cross-validation",
    },
    {
        "kind": "reinforce",
        "input": "memory_reinforce the last 3 hits",
    },
    {
        "kind": "introspect",
        "input": "events_query kind=first_turn_recall limit=5",
    },
]


_POST_TOK15_TOOL_DESCRIPTIONS = "\n".join([
    "Recall verbatim memories matching cue. Returns hits + anti_hits.",
    "Structural recall over role->filler bindings. Returns hits.",
    "Boost Hebbian edges among co-retrieved record ids.",
    "Mark a record contradicted; new fact stored as new record.",
    "Trigger memory consolidation.",
    "Read or write a profile knob (15 sealed). operation: get|set.",
    "List pending curiosity questions. Optional session_id filter.",
    "List induced schemas. Optional domain + confidence_min filters.",
    "Query user-visible events by kind, since, severity, limit.",
    "Topology snapshot: N, C, L, sigma, community_count, regime.",
    "Camouflaging detection status; window_size weekly points.",
])

_RESULT_BODIES: dict[str, str] = {
    "recall": (
        "hits=[{record_id, literal_surface, score}] "
        "anti_hits=[{record_id, reason}] "
        "activation_trace=[community_gate, spread, rank] "
        "budget_used=200"
    ),
    "save": "ok=true id=<uuid>",
    "introspect": '{"value": "minimal"}',
    "reinforce": "ok=true edges_boosted=3",
    "chat": "",
    "recall_cross_community": (
        "hits=[{record_id, literal_surface, score, community_id}] "
        "anti_hits=[] activation_trace=[cross_community_spread] "
        "budget_used=350"
    ),
}


def _select_counter(
    count_tokens_fn: Callable[[str], int] | None = None,
) -> tuple[Callable[[str], int], str]:
    if count_tokens_fn is not None:
        return count_tokens_fn, "injected"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _anthropic_count_tokens, "anthropic-count-tokens"
    try:
        import tiktoken  # noqa: F401
        return _tiktoken_count, "tiktoken-cl100k-proxy"
    except ImportError:
        return _char4_count, "heuristic-char4"


def _session_start_overhead_tokens(wake_depth: str) -> int:
    if wake_depth == "minimal":
        return 24
    if wake_depth == "standard":
        return 1388
    return 2000


def _simulate_turn(
    turn: dict,
    counter: Callable[[str], int],
) -> int:
    parts: list[str] = [
        _POST_TOK15_TOOL_DESCRIPTIONS,
        turn["input"],
        _RESULT_BODIES.get(turn["kind"], ""),
    ]
    return int(counter("\n".join(p for p in parts if p)))


def run_total_session_cost(
    *,
    wake_depth: str = "minimal",
    refs: "dict[str, int] | None" = None,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> dict:
    """Measure the fixed 10-turn script's token cost. ``refs`` maps
    operator-supplied reference labels to token totals; the gate fails when
    our total exceeds any reference. References are numbers the operator
    brings — this bench never executes third-party tools."""
    counter, mode = _select_counter(count_tokens_fn)

    per_turn: list[int] = []
    for i, turn in enumerate(_SCRIPT):
        t = _simulate_turn(turn, counter)
        if i == 0:
            t += _session_start_overhead_tokens(wake_depth)
        per_turn.append(int(t))

    total = int(sum(per_turn))

    refs_out: dict[str, int] = {
        str(name): int(value) for name, value in (refs or {}).items()
    }
    passed = all(total <= value for value in refs_out.values())

    return {
        "adapter": "iai-mcp",
        "wake_depth": wake_depth,
        "total_tokens": total,
        "per_turn": per_turn,
        "mode": mode,
        "refs": refs_out,
        "passed": passed,
        "script_name": SCRIPT_NAME,
    }


def _parse_ref(spec: str) -> tuple[str, int]:
    name, sep, value = spec.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(
            f"--ref expects NAME=TOKENS, got {spec!r}"
        )
    try:
        return name, int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--ref expects an integer token total, got {value!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.total_session_cost",
        description=(
            "Total session cost bench. Fixed 10-turn representative script; "
            "measures token cost at wake_depth minimal|standard|deep and "
            "optionally gates against operator-supplied reference totals."
        ),
    )
    parser.add_argument(
        "--wake-depth",
        choices=("minimal", "standard", "deep"),
        default="minimal",
        help="session-start payload size (default minimal)",
    )
    parser.add_argument(
        "--ref",
        dest="refs",
        type=_parse_ref,
        action="append",
        default=[],
        metavar="NAME=TOKENS",
        help=(
            "reference total for the comparative gate; repeatable "
            "(e.g. --ref external=12000)"
        ),
    )
    args = parser.parse_args(argv)

    result = run_total_session_cost(
        wake_depth=args.wake_depth,
        refs=dict(args.refs),
    )
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
