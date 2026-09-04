"""Read-only characterizer of `memory_recall` `tool_result.content` shapes.

Walks transcript `.jsonl` files, pairs each `memory_recall` tool_use with its
tool_result, and classifies the result's `content` into a stable shape key.
Opens no store, imports no embedder, writes nothing back to any transcript --
the histogram this emits is the evidence gate for the parser's coverage set.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

RECALL_TOOL_NAME_SUBSTR = "memory_recall"


def classify_tool_result_shape(content: Any) -> str:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            return "str_error"
        if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
            return "str_hits_json"
        return "str_other_json"
    if isinstance(content, list):
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            text = content[0].get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
                    return "list_text_other"
                if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
                    return "list_text_hits_json"
            return "list_text_other"
        return "list_other"
    if isinstance(content, dict):
        if isinstance(content.get("hits"), list):
            return "dict_hits"
        return "dict_other"
    return f"other_{type(content).__name__}"


def _iter_jsonl_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".jsonl"):
                yield Path(dirpath) / name


def _recall_tool_use_ids(objs: "list[dict]") -> set:
    ids = set()
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and RECALL_TOOL_NAME_SUBSTR in str(block.get("name") or "")
            ):
                tool_use_id = block.get("id")
                if tool_use_id:
                    ids.add(tool_use_id)
    return ids


def shape_histogram(paths: "list[Path]") -> "Counter[str]":
    histogram: "Counter[str]" = Counter()
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        objs = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        recall_ids = _recall_tool_use_ids(objs)
        if not recall_ids:
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") in recall_ids
                ):
                    histogram[classify_tool_result_shape(block.get("content"))] += 1
    return histogram


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only shape census of memory_recall tool_result.content."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--projects-dir", type=Path, help="Real transcript corpus root")
    group.add_argument("--fixture-dir", type=Path, help="Deterministic fixture root")
    parser.add_argument("--emit-json", type=Path, default=None, help="Write histogram JSON to this path")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    root = args.projects_dir or args.fixture_dir
    paths = list(_iter_jsonl_files(root))
    histogram = shape_histogram(paths)
    payload = dict(sorted(histogram.items()))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.emit_json is not None:
        args.emit_json.parent.mkdir(parents=True, exist_ok=True)
        args.emit_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
