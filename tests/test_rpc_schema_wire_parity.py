"""Every field a tool's outputSchema declares must actually be served.

A field that lives in the schema but never in the response is a silent
contract breach no host can detect. This suite parses the outputSchema
of every hot tool out of tools.ts and dispatches a real call for each,
asserting both directions:

  * every declared top-level property is present in the response;
  * every served top-level key is either declared or an underscore-
    prefixed private field.

Scope: top-level property NAMES on the happy path — value types and
nested shapes are not checked here.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

TOOLS_TS = Path(__file__).resolve().parents[1] / "mcp-wrapper" / "src" / "tools.ts"

_TOOL_NAME_LINE = re.compile(r'^  (?P<name>\w+): \{', re.M)


def _balance_braces(text: str, start_idx: int) -> int:
    depth = 0
    in_str: str | None = None
    i = start_idx
    while i < len(text):
        ch = text[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        else:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    raise AssertionError(f"unbalanced braces starting at {start_idx}")


def _top_level_keys(obj_text: str) -> set[str]:
    """Keys at depth 1 of a TS object literal `{...}`."""
    keys: set[str] = set()
    depth = 0
    in_str: str | None = None
    i = 0
    while i < len(obj_text):
        ch = obj_text[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 1:
                m = re.match(r"(\w+):", obj_text[i:])
                if m and (i == 0 or obj_text[i - 1] in "{,\n "):
                    keys.add(m.group(1))
                    i += len(m.group(0))
                    continue
        else:
            if ch == "\\" and i + 1 < len(obj_text):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    return keys


def declared_output_fields() -> dict[str, set[str]]:
    text = TOOLS_TS.read_text()
    out: dict[str, set[str]] = {}
    for m in _TOOL_NAME_LINE.finditer(text):
        block = text[m.end() - 1:_balance_braces(text, m.end() - 1)]
        os_at = block.find("outputSchema: {")
        if os_at < 0:
            continue
        schema = block[os_at + len("outputSchema: "):]
        schema = schema[:_balance_braces(schema, 0)]
        props_at = schema.find("properties: {")
        if props_at < 0:
            # An open object declares no field contract to hold the wire to.
            continue
        props = schema[props_at + len("properties: "):]
        props = props[:_balance_braces(props, 0)]
        out[m.group("name")] = _top_level_keys(props)
    return out


def _independent_property_bearing_outputschema_count(text: str) -> int:
    """Count tool entries whose outputSchema declares a properties block.

    Trivially independent of declared_output_fields(): a plain indentation
    window scan, no brace balancing, no regex. Exists to catch a silently
    broken sophisticated parser that returns a near-empty result.
    """
    count = 0
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith("outputSchema:"):
            indent = len(lines[i]) - len(stripped)
            j = i + 1
            has_props = False
            while j < n:
                inner = lines[j]
                inner_stripped = inner.lstrip()
                if inner_stripped and (len(inner) - len(inner_stripped)) <= indent:
                    break
                if inner_stripped.startswith("properties:"):
                    has_props = True
                    break
                j += 1
            if has_props:
                count += 1
        i += 1
    return count


def _seed_record(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False,
        provenance=[{"session_id": "parity-seed"}],
        created_at=now, updated_at=now, tags=[], language="en",
    )


@pytest.fixture(scope="module")
def seeded_store(tmp_path_factory) -> tuple[MemoryStore, list[MemoryRecord]]:
    import os

    os.environ.setdefault("IAI_MCP_CRYPTO_PASSPHRASE", "iai-mcp-test-passphrase")
    store = MemoryStore(path=tmp_path_factory.mktemp("parity") / "store")
    rng = np.random.default_rng(7)
    recs = []
    for i in range(6):
        v = rng.normal(size=EMBED_DIM)
        rec = _seed_record((v / np.linalg.norm(v)).tolist(), f"parity record {i}")
        store.insert(rec)
        recs.append(rec)
    return store, recs


def _params_for(tool: str, store: MemoryStore, recs: list[MemoryRecord]) -> dict:
    cue_vec = list(recs[0].embedding)
    table = {
        "memory_recall": {
            "cue": "parity record", "session_id": "parity",
            "cue_embedding": cue_vec,
        },
        "memory_recall_structural": {
            "structure_query": {"TIER": "episodic"},
            "budget_tokens": 2000,
        },
        "memory_temporal_recall": {
            "cue": "parity record",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "limit": 5,
        },
        "memory_search": {"query": "parity record", "k": 3},
        "memory_reinforce": {"ids": [str(recs[0].id)]},
        "memory_contradict": {
            "id": str(recs[1].id),
            "new_fact": "the parity record is superseded by this statement",
            "cue_embedding": cue_vec,
        },
        "memory_capture": {
            "text": "a fresh parity capture about hosting rates",
            "session_id": "parity",
        },
        "memory_consolidate": {"session_id": "parity"},
        "curiosity_pending": {"session_id": "parity"},
        "schema_list": {},
        "events_query": {"kind": "s4_contradiction", "limit": 5},
        "topology": {},
        "episodes_recent": {"n": 3},
        "claim_check": {"cue": "parity record", "session_id": "parity"},
    }
    return table[tool]


def test_every_declared_output_field_is_served(seeded_store) -> None:
    from iai_mcp import core as _core

    store, recs = seeded_store
    declared = declared_output_fields()
    independent = _independent_property_bearing_outputschema_count(TOOLS_TS.read_text())
    assert independent > 0, (
        "the independent outputSchema scan found no property-bearing tools — "
        "one of the two parsers is broken"
    )
    assert len(declared) == independent, (
        "the outputSchema parser and an independent scan disagree on how many "
        "tools declare a properties block — one of them is broken:\n"
        f"  parser: {len(declared)} {sorted(declared)}\n"
        f"  independent scan: {independent}"
    )

    problems: list[str] = []
    for tool, fields in sorted(declared.items()):
        resp = _core.dispatch(store, tool, _params_for(tool, store, recs))
        assert isinstance(resp, dict), f"{tool}: non-dict response {resp!r}"
        missing = fields - set(resp)
        undeclared = {
            k for k in resp
            if k not in fields and not k.startswith("_")
        }
        if missing:
            problems.append(f"{tool}: declared but not served: {sorted(missing)}")
        if undeclared:
            problems.append(f"{tool}: served but not declared: {sorted(undeclared)}")

    assert not problems, (
        "outputSchema and the wire disagree — a declared field that is never "
        "sent is a silent contract breach:\n  " + "\n  ".join(problems)
    )
