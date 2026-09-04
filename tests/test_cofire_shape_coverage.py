"""Coverage gate: shapes observed on a real transcript corpus must be known
shapes, and str_hits_json (the dominant real-traffic shape) is genuinely
extracted, not fail-safe-skipped.

`observed_shapes.json` is a static snapshot from a one-time bench probe run
over the corpus available at authoring time -- it will not detect a genuinely
new production shape until the probe is rerun and the fixture regenerated.
Periodic regeneration against a live corpus sample is a manual follow-up, not
covered by this gate.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.cofire_shape_probe import classify_tool_result_shape as probe_classify
from iai_mcp.cofire import (
    HANDLED_SHAPES,
    KNOWN_SHAPES,
    _parse_hits,
    _parse_list_hits,
    _parse_str_hits_json,
)
from iai_mcp.cofire import classify_tool_result_shape as cofire_classify

OBSERVED_SHAPES_PATH = Path(__file__).parent / "fixtures" / "cofire" / "observed_shapes.json"


def _raise_recursion_error(*_args, **_kwargs):
    raise RecursionError("maximum recursion depth exceeded")


# One representative content value per KNOWN_SHAPES key, so the two
# independent classifiers (bench/cofire_shape_probe.py's read-only probe and
# cofire.py's runtime copy) stay pinned in agreement.
_SHAPE_SAMPLES: "dict[str, object]" = {
    "str_error": "not json at all",
    "str_other_json": json.dumps({"no_hits_key": True}),
    "str_hits_json": json.dumps({"hits": [{"record_id": "x"}]}),
    "list_text_hits_json": [{"type": "text", "text": json.dumps({"hits": [{"record_id": "x"}]})}],
    "list_text_other": [{"type": "text", "text": "not json"}],
    "list_other": [1, 2, 3],
    "dict_hits": {"hits": [{"record_id": "x"}]},
    "dict_other": {"no_hits_key": True},
}


def test_probe_and_cofire_classifiers_agree_on_every_known_shape():
    assert set(_SHAPE_SAMPLES) == KNOWN_SHAPES, "sample table must cover every KNOWN_SHAPES key"
    for shape, content in _SHAPE_SAMPLES.items():
        assert probe_classify(content) == shape == cofire_classify(content), (
            f"classifier drift on {shape!r} sample"
        )


def test_observed_shapes_are_all_known():
    observed = json.loads(OBSERVED_SHAPES_PATH.read_text(encoding="utf-8"))
    unknown = set(observed) - KNOWN_SHAPES
    assert not unknown, f"unclassified tool_result shape(s): {sorted(unknown)}"


def test_str_hits_json_is_handled():
    assert "str_hits_json" in KNOWN_SHAPES
    assert "str_hits_json" in HANDLED_SHAPES


def test_str_hits_json_classified_and_extracted():
    content = json.dumps({"hits": [{"record_id": "x", "literal_surface": "y"}]})
    assert cofire_classify(content) == "str_hits_json"
    assert _parse_hits(content) == (["x"], ["y"])
    # The list-form parser stays str_hits_json-blind: only the shape-dispatched
    # _parse_hits routes a string to the str_hits_json extractor.
    assert _parse_list_hits(content) == (None, None)


def test_list_text_hits_json_classified_and_extracted():
    content = [{"type": "text", "text": json.dumps({"hits": [
        {"record_id": "x", "literal_surface": "y"},
    ]})}]
    assert cofire_classify(content) == "list_text_hits_json"
    hit_ids, hit_surfaces = _parse_list_hits(content)
    assert hit_ids == ["x"]
    assert hit_surfaces == ["y"]


def test_deeply_nested_json_never_raises_in_either_classifier_copy(monkeypatch):
    monkeypatch.setattr(json, "loads", _raise_recursion_error)
    assert cofire_classify("content") == "str_error"
    assert probe_classify("content") == "str_error"

    nested_list_text = [{"type": "text", "text": "content"}]
    assert cofire_classify(nested_list_text) == "list_text_other"
    assert probe_classify(nested_list_text) == "list_text_other"


def test_deeply_nested_json_never_raises_in_hit_extraction(monkeypatch):
    monkeypatch.setattr(json, "loads", _raise_recursion_error)
    assert _parse_str_hits_json("content") == (None, None)
    assert _parse_hits("content") == (None, None)
    assert _parse_list_hits([{"type": "text", "text": "content"}]) == (None, None)
