from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore

TRICKY_STRINGS = [
    "simple ascii text",
    "unicode: こんにちは 你好世界",
    'quotes: "double" and \'single\'',
    "newlines:\nline2\n\tindented",
    "long: " + "x" * 8000,
    "empty-ish:   ",
    "code: def f(x):\n    return x ** 2\n",
    "json-like: {\"key\": [1, 2, 3]}",
    "path: /home/alice/.iai-mcp/store/records.lance",
]

def _record(text: str):
    return _make(text=text)

@pytest.mark.parametrize("text", TRICKY_STRINGS, ids=[s[:30] for s in TRICKY_STRINGS])
def test_literal_surface_roundtrip_exact(tmp_path, text):
    store = MemoryStore(str(tmp_path))
    rec = _record(text)
    rec_id = rec.id
    store.insert(rec)
    got = store.get(rec_id)
    assert got is not None, f"Record {rec_id} not found after insert"
    assert got.literal_surface == text, (
        f"literal_surface mutated.\n"
        f"  Expected: {text!r}\n"
        f"  Got:      {got.literal_surface!r}"
    )

def test_literal_surface_survives_multiple_inserts(tmp_path):
    store = MemoryStore(str(tmp_path))
    originals = {}
    for text in TRICKY_STRINGS:
        rec = _record(text)
        originals[rec.id] = text
        store.insert(rec)

    for rid, expected in originals.items():
        got = store.get(rid)
        assert got.literal_surface == expected, f"literal_surface mutated on record {rid}"

def test_literal_surface_not_trimmed(tmp_path):
    store = MemoryStore(str(tmp_path))
    text = "  leading and trailing spaces  "
    rec = _record(text)
    store.insert(rec)
    got = store.get(rec.id)
    assert got.literal_surface == text, "Whitespace was trimmed"
