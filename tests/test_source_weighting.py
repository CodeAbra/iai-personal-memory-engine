from types import SimpleNamespace

import pytest

from iai_mcp.source_weighting import (
    MAX_SOURCE_WEIGHT_FACTOR,
    source_weighted_score,
)


def _record(*, tier: str = "episodic", tags: list[str] | None = None):
    return SimpleNamespace(tier=tier, tags=tags or [])


def test_boost_applies_once_to_curated_docs_and_semantic_digests(monkeypatch):
    monkeypatch.setenv("IAI_MCP_SOURCE_WEIGHT_FACTOR", "1.1")

    assert source_weighted_score(0.8, _record(tags=["doc:rules.md"])) == pytest.approx(0.88)
    assert source_weighted_score(0.8, _record(tier="semantic")) == pytest.approx(0.88)
    assert source_weighted_score(
        0.8, _record(tier="semantic", tags=["doc:rules.md"])
    ) == pytest.approx(0.88)
    assert source_weighted_score(0.8, _record()) == pytest.approx(0.8)


def test_factor_one_disables_weighting_without_order_drift(monkeypatch):
    monkeypatch.setenv("IAI_MCP_SOURCE_WEIGHT_FACTOR", "1.0")
    scored = [
        (0.90, _record()),
        (0.85, _record(tags=["doc:rules.md"])),
        (0.80, _record(tier="semantic")),
    ]

    weighted = [source_weighted_score(score, record) for score, record in scored]

    assert weighted == [0.90, 0.85, 0.80]


@pytest.mark.parametrize("tag", ["schema", "pattern:tags:capture+role:user"])
def test_semantic_inference_records_are_not_boosted(monkeypatch, tag):
    monkeypatch.setenv("IAI_MCP_SOURCE_WEIGHT_FACTOR", "1.1")

    assert source_weighted_score(
        0.8, _record(tier="semantic", tags=[tag])
    ) == pytest.approx(0.8)


def test_semantic_order_is_preserved_within_each_source_nature(monkeypatch):
    monkeypatch.setenv("IAI_MCP_SOURCE_WEIGHT_FACTOR", "1.1")
    episodic = _record()
    curated = _record(tags=["doc:rules.md"])

    assert source_weighted_score(0.9, episodic) > source_weighted_score(0.8, episodic)
    assert source_weighted_score(0.9, curated) > source_weighted_score(0.8, curated)
    assert source_weighted_score(0.8, curated) > source_weighted_score(0.8, episodic)


def test_source_marker_cannot_overrule_a_non_comparable_semantic_gap(monkeypatch):
    monkeypatch.setenv("IAI_MCP_SOURCE_WEIGHT_FACTOR", "999")

    curated = source_weighted_score(0.70, _record(tags=["doc:rules.md"]))
    episodic = source_weighted_score(0.99, _record())

    assert curated == pytest.approx(0.70 * MAX_SOURCE_WEIGHT_FACTOR)
    assert episodic > curated
