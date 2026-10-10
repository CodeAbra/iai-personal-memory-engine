"""save() must decide the size-cap shed BEFORE encoding node_payload.

Encoding every node (embedding + surface) then discarding the result over
cap is a transient full-corpus materialization on the cache-miss path. These
tests pin: over-cap never touches per-node embeddings, under-cap output is
unchanged field-for-field, and payload_record_count keeps counting only
dict-valued entries.
"""
from __future__ import annotations

import json as real_json
import pathlib
from collections import Counter
from types import SimpleNamespace

import pytest

import iai_mcp.runtime_graph_cache as rgc
from iai_mcp.community import CommunityAssignment


def _make_fake_store(tmp_path):
    return SimpleNamespace(root=tmp_path)


def _decrypt_cache_for_inspection(store, path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("iai:enc:v1:"):
        return real_json.loads(raw)
    from iai_mcp.crypto import decrypt_field

    plaintext = decrypt_field(raw, rgc._cache_encryption_key(store), rgc._CACHE_AAD)
    return real_json.loads(plaintext)


def _empty_assignment() -> CommunityAssignment:
    return CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="flat",
        top_communities=[],
        mid_regions={},
    )


class _TrackingDict(dict):
    """Dict subclass recording every key passed to .get() into a shared Counter."""

    def __init__(self, data, counter: Counter):
        super().__init__(data)
        self._counter = counter

    def get(self, key, default=None):
        self._counter[key] += 1
        return super().get(key, default)


def _make_tracking_node_payload(count: int, counter: Counter, embed_dim: int = 384):
    payload = {}
    for i in range(count):
        payload[f"u{i}"] = _TrackingDict(
            {
                "embedding": [0.123456789] * embed_dim,
                "surface": "hello world",
                "centrality": 0.1 + 0.001 * i,
                "tier": "episodic",
                "pinned": False,
                "tags": ["t1", "t2"],
                "language": "en",
                "aaak_index": "",
                "created_at": "",
                "stability": 0.5,
            },
            counter,
        )
    return payload


def test_over_cap_never_encodes_embeddings(tmp_path, monkeypatch):
    """Over-cap: the cheap centrality pass runs for every node, but no node's
    embedding is ever read -- the giant encoded dict is never built."""
    monkeypatch.setattr(rgc, "MAX_CACHE_BYTES", 50_000)
    store = _make_fake_store(tmp_path)
    n = 20
    counter: Counter = Counter()
    node_payload = _make_tracking_node_payload(n, counter)

    ok = rgc.save(store, _empty_assignment(), [], node_payload=node_payload, max_degree=4)

    assert ok is True
    assert counter["embedding"] == 0, (
        f"embedding read {counter['embedding']} times -- over-cap path must "
        "never encode any node's embedding"
    )
    assert counter["centrality"] == n, (
        "the cheap centrality pass must run for every node"
    )

    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    written = _decrypt_cache_for_inspection(store, cache_path)
    assert written["node_payload"] == {}
    assert len(written["centrality"]) == n
    for i in range(n):
        assert written["centrality"][f"u{i}"] == pytest.approx(0.1 + 0.001 * i)
    assert written["payload_record_count"] == n


def test_under_cap_encodes_fully_and_identically(tmp_path):
    """Under-cap output is byte-identical (field-for-field) to the encode shape."""
    store = _make_fake_store(tmp_path)
    node_payload = {
        "u0": {
            "embedding": [0.1, 0.2, 0.3],
            "surface": "first surface",
            "centrality": 0.5,
            "tier": "semantic",
            "pinned": True,
            "tags": ["a", None, "b"],
            "language": "ru",
            "aaak_index": "idx-0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "stability": 0.75,
        },
        "u1": {
            "embedding": [0.4, 0.5],
            "surface": "second surface",
            "centrality": None,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": None,
            "aaak_index": None,
            "created_at": None,
            "stability": None,
        },
        "u2": {
            "embedding": None,
            "surface": "",
            "centrality": 0.0,
            "tier": "episodic",
            "pinned": False,
            "tags": None,
            "language": "en",
            "aaak_index": "",
            "created_at": "",
            "stability": 0.0,
        },
    }

    ok = rgc.save(store, _empty_assignment(), [], node_payload=node_payload, max_degree=4)
    assert ok is True

    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    written = _decrypt_cache_for_inspection(store, cache_path)
    encoded = written["node_payload"]
    assert set(encoded.keys()) == {"u0", "u1", "u2"}

    for k, v in node_payload.items():
        expected_emb = [float(x) for x in (v.get("embedding") or [])]
        expected_surface = str(v.get("surface", ""))
        expected_centrality = float(v.get("centrality") or 0.0)
        expected_tier = str(v.get("tier", "episodic"))
        expected_pinned = bool(v.get("pinned", False))
        expected_tags = [str(t) for t in (v.get("tags") or []) if t is not None]
        expected_language = str(v.get("language", "en") or "en")
        expected_aaak = str(v.get("aaak_index", "") or "")
        expected_created_at = str(v.get("created_at", "") or "")
        expected_stability = float(v.get("stability", 0.5) or 0.5)

        node = encoded[k]
        assert node["embedding"] == expected_emb
        assert node["surface"] == expected_surface
        assert node["centrality"] == pytest.approx(expected_centrality)
        assert node["tier"] == expected_tier
        assert node["pinned"] == expected_pinned
        assert node["tags"] == expected_tags
        assert node["language"] == expected_language
        assert node["aaak_index"] == expected_aaak
        assert node["created_at"] == expected_created_at
        assert node["stability"] == pytest.approx(expected_stability)

    assert written["centrality"] == {
        "u0": pytest.approx(0.5),
        "u1": pytest.approx(0.0),
        "u2": pytest.approx(0.0),
    }
    assert written["payload_record_count"] == 3


def test_payload_record_count_counts_dict_entries_only(tmp_path):
    """A non-dict node_payload value is skipped by the encode filter, so
    payload_record_count must count only the dict-valued entries."""
    store = _make_fake_store(tmp_path)
    node_payload = {
        "u0": {
            "embedding": [0.1],
            "surface": "a",
            "centrality": 0.2,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": "en",
        },
        "u1": {
            "embedding": [0.3],
            "surface": "b",
            "centrality": 0.4,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": "en",
        },
        "not_a_record": "this value is not a dict and must be skipped",
    }

    ok = rgc.save(store, _empty_assignment(), [], node_payload=node_payload, max_degree=4)
    assert ok is True

    cache_path = pathlib.Path(tmp_path) / rgc.CACHE_FILENAME
    written = _decrypt_cache_for_inspection(store, cache_path)
    assert set(written["node_payload"].keys()) == {"u0", "u1"}
    assert written["payload_record_count"] == 2
    assert len(written["centrality"]) == 2
