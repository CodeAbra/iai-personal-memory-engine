"""Per-component token breakdown of the session-start payload.

Two modes:
- Mode A (cheap, zero store access): parses the already-rendered
  ``.session-start-payload.cached.md`` cache file the SessionStart hook
  serves on cache-hit.
- Mode B (controllable recompose): opens a read-only COPY of the operator's
  real store (never the original) and recomposes the payload through the
  exact production assembly path (``build_runtime_graph`` ->
  ``_compose_session_start_payload`` -> ``format_payload_as_markdown``),
  matching the daemon precache's hardcoded ``wake_depth="standard"``.

Every report carries sizes/token-counts/percentages only -- never stored
memory text.
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import open_eval_copy_store  # noqa: E402
from iai_mcp.session import (  # noqa: E402
    RICH_CLUB_BUDGET_TOKENS,
    SESSION_START_CACHE_MAX_CHARS,
    _approx_tokens,
    _compose_session_start_payload,
    format_payload_as_markdown,
)

# Mirrors daemon._write_session_start_cache's SESSION_START_CACHE_PATH,
# reconstructed here to avoid pulling the full daemon module into a bench
# script.
_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"

_HEADER_RICH_CLUB = "Key memories"


@functools.lru_cache(maxsize=1)
def _tiktoken_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _measure_text(text: str, encoder) -> dict:
    return {
        "chars": len(text),
        "chars4_tokens": _approx_tokens(text),
        "tiktoken_tokens": len(encoder.encode(text)) if text else 0,
    }


def _pct(measure: dict, total: dict) -> dict:
    out = dict(measure)
    out["pct_chars"] = (
        round(100 * measure["chars"] / total["chars"], 2) if total["chars"] else 0.0
    )
    out["pct_tiktoken_tokens"] = (
        round(100 * measure["tiktoken_tokens"] / total["tiktoken_tokens"], 2)
        if total["tiktoken_tokens"]
        else 0.0
    )
    return out


def _pct_map(measures: dict, total: dict) -> dict:
    return {k: _pct(v, total) for k, v in measures.items()}


def _split_rendered_blocks(rendered: str) -> list[str]:
    return [b for b in rendered.split("\n\n") if b]


def _block_header(block: str) -> "str | None":
    first_line, _, _ = block.partition("\n")
    if first_line.startswith("## "):
        return first_line[3:]
    return None


def _component_set(rendered: str) -> set:
    headers = set()
    for block in _split_rendered_blocks(rendered):
        header = _block_header(block)
        if header is not None:
            headers.add(header)
    return headers


def _split_index_content(rich_club_text: str, encoder) -> dict:
    """Split each rich_club line on its first ": " -- the fixed
    ``f"{aaak}{age_part}: {cleaned[:60]}"`` structure -- into the derived
    index+age side vs the stored-content side."""
    lines = [ln for ln in rich_club_text.split("\n") if ln]
    index_parts = []
    content_parts = []
    for line in lines:
        idx, sep, content = line.partition(": ")
        if sep:
            index_parts.append(idx)
            content_parts.append(content)
        else:
            index_parts.append(line)
    return {
        "line_count": len(lines),
        "index_age": _measure_text("\n".join(index_parts), encoder),
        "content": _measure_text("\n".join(content_parts), encoder),
    }


def _rich_club_saturation(rich_club_text: str) -> dict:
    """Re-derive the composition loop's running chars/4 total from the
    rendered lines -- the same cost accounting the loop itself used
    (session.py's budget loop, mirrored here from its own output, not
    re-fetched from the store)."""
    lines = [ln for ln in rich_club_text.split("\n") if ln]
    running = sum(_approx_tokens(line) + 1 for line in lines)
    return {
        "line_count": len(lines),
        "running_chars4_tokens": running,
        "budget_tokens": RICH_CLUB_BUDGET_TOKENS,
        "saturated": running >= 0.95 * RICH_CLUB_BUDGET_TOKENS,
    }


def mode_a_report(cache_path: Path = _CACHE_PATH) -> dict:
    if not cache_path.exists():
        return {"skipped": True, "reason": f"cache file not found at {cache_path}"}

    rendered = cache_path.read_text(encoding="utf-8")
    encoder = _tiktoken_encoder()
    total = _measure_text(rendered, encoder)

    blocks_by_header: dict = {}
    scaffolding_blocks: list = []
    for block in _split_rendered_blocks(rendered):
        header = _block_header(block)
        if header is None:
            scaffolding_blocks.append(block)
            continue
        body = block.split("\n", 1)[1] if "\n" in block else ""
        blocks_by_header.setdefault(header, []).append(body)

    components = {}
    for header, bodies in blocks_by_header.items():
        measure = _measure_text("\n".join(bodies), encoder)
        measure["block_count"] = len(bodies)
        components[header] = measure

    rich_club_bodies = blocks_by_header.get(_HEADER_RICH_CLUB, [])
    rich_club_text = "\n".join(rich_club_bodies)
    rich_club_split = None
    if rich_club_bodies:
        split = _split_index_content(rich_club_text, encoder)
        rich_club_split = {
            "line_count": split["line_count"],
            "index_age": _pct(split["index_age"], total),
            "content": _pct(split["content"], total),
        }

    scaffolding_measure = _measure_text("\n\n".join(scaffolding_blocks), encoder)

    return {
        "skipped": False,
        "mode": "A",
        "source": str(cache_path),
        "total": total,
        "cache_max_chars": SESSION_START_CACHE_MAX_CHARS,
        "truncated": total["chars"] > SESSION_START_CACHE_MAX_CHARS,
        "components": _pct_map(components, total),
        "rich_club_split": rich_club_split,
        "rich_club_saturation": _rich_club_saturation(rich_club_text),
        "scaffolding": _pct(scaffolding_measure, total),
        "component_set": sorted(blocks_by_header.keys()),
    }


def compose_and_measure(store, assignment, rich_club, *, session_id: str = "bench") -> dict:
    """Recompose the session-start payload against the given store/assignment/
    rich_club through the exact production assembly path and measure every
    component. ``store`` may be a real-store read-only copy or a synthetic
    test store -- this function never opens a store itself."""
    payload = _compose_session_start_payload(
        store,
        assignment,
        rich_club,
        session_id=session_id,
        profile_state={"wake_depth": "standard"},
    )
    rendered = format_payload_as_markdown(payload)
    encoder = _tiktoken_encoder()
    total = _measure_text(rendered, encoder)

    raw_fields = {
        "l0": payload.l0,
        "l1": payload.l1,
        "recent_thread": payload.recent_thread,
        "rich_club": payload.rich_club,
    }
    components = {name: _measure_text(text, encoder) for name, text in raw_fields.items()}

    l2_entries = [_measure_text(seg, encoder) for seg in payload.l2]
    components["l2"] = {
        "chars": sum(e["chars"] for e in l2_entries),
        "chars4_tokens": sum(e["chars4_tokens"] for e in l2_entries),
        "tiktoken_tokens": sum(e["tiktoken_tokens"] for e in l2_entries),
        "entry_count": len(l2_entries),
    }

    sum_raw_chars = sum(c["chars"] for c in components.values())
    sum_raw_chars4 = sum(c["chars4_tokens"] for c in components.values())
    sum_raw_tiktoken = sum(c["tiktoken_tokens"] for c in components.values())

    scaffolding_measure = {
        "chars": max(0, total["chars"] - sum_raw_chars),
        "chars4_tokens": max(0, total["chars4_tokens"] - sum_raw_chars4),
        "tiktoken_tokens": max(0, total["tiktoken_tokens"] - sum_raw_tiktoken),
    }

    rich_club_split = None
    if payload.rich_club:
        split = _split_index_content(payload.rich_club, encoder)
        rich_club_split = {
            "line_count": split["line_count"],
            "index_age": _pct(split["index_age"], total),
            "content": _pct(split["content"], total),
        }

    return {
        "mode": "B",
        "wake_depth": payload.wake_depth,
        "session_id": session_id,
        "total": total,
        "cache_max_chars": SESSION_START_CACHE_MAX_CHARS,
        "truncated": total["chars"] > SESSION_START_CACHE_MAX_CHARS,
        "components": _pct_map(components, total),
        "l2_entries": [_pct(e, total) for e in l2_entries],
        "scaffolding": _pct(scaffolding_measure, total),
        "rich_club_split": rich_club_split,
        "rich_club_saturation": _rich_club_saturation(payload.rich_club),
        "component_set": sorted(_component_set(rendered)),
    }


def mode_b_from_real_store(*, driver: "str | None" = None) -> dict:
    from iai_mcp.retrieve import build_runtime_graph

    resolved_driver = driver if driver is not None else os.environ.get("LILLI_STORAGE_DRIVER", "stdlib")
    with open_eval_copy_store(driver=driver) as store:
        _graph, assignment, rich_club = build_runtime_graph(store)
        report = compose_and_measure(store, assignment, rich_club, session_id="bench")
    report["driver"] = resolved_driver
    return report


def mode_agreement(mode_a: dict, mode_b: dict) -> dict:
    if mode_a.get("skipped"):
        return {"skipped": True, "reason": mode_a.get("reason")}
    set_a = set(mode_a["component_set"])
    set_b = set(mode_b["component_set"])
    return {
        "skipped": False,
        "mode_a_set": sorted(set_a),
        "mode_b_set": sorted(set_b),
        "agree": set_a == set_b,
    }


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-component token breakdown of the session-start payload."
    )
    parser.add_argument("--mode", choices=["a", "b", "both"], default="both")
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default="stdlib")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="write the combined baseline report JSON (incl. fixture-coverage verdict) to this path",
    )
    args = parser.parse_args(argv)

    result: dict = {}
    mode_a = mode_a_report() if args.mode in ("a", "both") else None
    mode_b = mode_b_from_real_store(driver=args.driver) if args.mode in ("b", "both") else None
    if mode_a is not None:
        result["mode_a"] = mode_a
    if mode_b is not None:
        result["mode_b"] = mode_b
    if mode_a is not None and mode_b is not None:
        result["mode_agreement"] = mode_agreement(mode_a, mode_b)

    if args.baseline is not None:
        combined = dict(result)
        try:
            from bench.token_recall_guard import fixture_coverage_verdict

            combined["fixture_coverage"] = fixture_coverage_verdict(driver=args.driver)
        except Exception as exc:  # noqa: BLE001 -- an optional cross-check must not block the baseline write
            combined["fixture_coverage"] = {"skipped": True, "reason": f"error: {exc}"}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
