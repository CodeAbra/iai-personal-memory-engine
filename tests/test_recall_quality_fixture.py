"""Fast, pure integrity checks for the recall-quality fixture.

No store, no embedder, no dispatch: every check here reads either the
committed JSON fixture or a bench module's SOURCE TEXT directly. The
heavier real-rebuild determinism check and the full corpus-build/dispatch
pipeline are opt-in (``@pytest.mark.bench``), never collected by default.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bench.fhrr_verdict_membership import (
    CAUSE_DISCOVERY_EXCLUDED,
    CAUSE_LEXICAL_COLD_OR_IDF_GATED,
    GoldMembership,
    classify_miss_cause,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = _REPO_ROOT / "bench" / "results" / "v28" / "recall_quality_fixture.json"
_MODULE_PATHS = sorted((_REPO_ROOT / "bench").glob("recall_quality_*.py"))

REQUIRED_CUE_KEYS = ("cue_id", "cue_text", "gold_record_id", "cluster_id", "quote", "split")
_ALLOWED_EXTRA_CUE_KEYS = {"gold_literal_surface"}

_FORBIDDEN_IMPORT_NAMES = (
    "recall_accuracy_real",
    "_copy_real_store",
    "_operator_home",
    "proc_corpus_census",
)
_FORBIDDEN_INTENT_KEYS = {
    "intended_cause", "expected_cause", "intent", "cause", "target_cause",
}
_CUE_VAR_NAMES = {"cue", "c"}
"""Variable names bound to a fixture-declared cue dict below and in
bench/recall_quality_verdict.py -- add any new alias here or the bare
``"cause"`` .get(...) guard silently narrows."""
_UNAMBIGUOUS_INTENT_KEYS = _FORBIDDEN_INTENT_KEYS - {"cause"}


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _validate_cue_schema(cue: dict) -> None:
    """Local mirror of ``bench.recall_quality_fixture.validate_cue_schema``
    (and, further back, ``_validate_verdict_fixture_dict``'s fail-loud
    shape) -- reimplemented here so this test module never imports the
    bench fixture/verdict modules at collection time."""
    for key in REQUIRED_CUE_KEYS:
        if key not in cue:
            raise ValueError(f"fixture cue missing required key {key!r}: {cue}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_fixture_file_exists() -> None:
    assert _FIXTURE_PATH.exists(), f"committed fixture missing: {_FIXTURE_PATH}"


def test_every_cue_carries_required_keys() -> None:
    payload = _load_fixture()
    assert payload["cues"], "fixture must carry at least one cue"
    for cue in payload["cues"]:
        _validate_cue_schema(cue)
        extra = set(cue.keys()) - set(REQUIRED_CUE_KEYS) - _ALLOWED_EXTRA_CUE_KEYS
        assert not extra, f"cue {cue.get('cue_id')} carries unexpected keys: {extra}"


def test_malformed_cue_raises_loud() -> None:
    malformed = {"cue_id": "x", "cue_text": "y"}
    with pytest.raises(ValueError):
        _validate_cue_schema(malformed)


# ---------------------------------------------------------------------------
# Quotation-verbatim
# ---------------------------------------------------------------------------


def test_committed_quote_and_surface_fields_agree() -> None:
    """Cheap default-lane sanity check: the committed JSON's own ``quote``
    and ``gold_literal_surface`` fields are internally consistent. This
    does NOT prove ``gold_literal_surface`` matches what the gold record
    built by ``build_full_fixture_corpus`` actually contains -- see
    ``test_quote_is_verbatim_in_freshly_rebuilt_gold_record`` below for
    that check."""
    payload = _load_fixture()
    for cue in payload["cues"]:
        quote = cue["quote"]
        surface = cue["gold_literal_surface"]
        assert quote in surface, (
            f"cue {cue['cue_id']!r} quote {quote!r} is not a verbatim "
            f"substring of gold literal_surface {surface!r}"
        )


@pytest.mark.bench
def test_quote_is_verbatim_in_freshly_rebuilt_gold_record() -> None:
    """The genuine fixture-to-store fidelity check: rebuild the corpus from
    the committed fixture's own seed and assert every cue's ``quote`` is a
    verbatim substring of the FRESHLY BUILT gold record's literal_surface
    -- not the committed JSON's own (possibly stale or hand-edited)
    ``gold_literal_surface`` field."""
    from bench.recall_quality_fixture import build_full_fixture_corpus

    payload = _load_fixture()
    corpus, _rationale = build_full_fixture_corpus(seed=payload["seed"])
    try:
        for cue in payload["cues"]:
            surface = corpus.gold_literal_surface[cue["gold_record_id"]]
            assert cue["quote"] in surface, (
                f"cue {cue['cue_id']!r} quote {cue['quote']!r} is not a "
                f"verbatim substring of the freshly rebuilt gold record's "
                f"literal_surface {surface!r}"
            )
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# Cluster-level split disjointness
# ---------------------------------------------------------------------------


def test_no_cluster_straddles_dev_and_holdout() -> None:
    payload = _load_fixture()
    split_by_cluster: "dict[str, set[str]]" = {}
    for cue in payload["cues"]:
        split_by_cluster.setdefault(cue["cluster_id"], set()).add(cue["split"])
    straddling = {cid: splits for cid, splits in split_by_cluster.items() if len(splits) > 1}
    assert not straddling, f"clusters straddling dev/holdout: {straddling}"


def test_both_splits_are_non_empty() -> None:
    payload = _load_fixture()
    splits = {cue["split"] for cue in payload["cues"]}
    assert splits == {"dev", "holdout"}, f"expected both dev and holdout, got {splits}"


# ---------------------------------------------------------------------------
# All four causes provoked (structural: distinct cue_id prefixes)
# ---------------------------------------------------------------------------


def test_all_four_causes_represented_by_cue_id_prefix() -> None:
    from bench.fhrr_verdict_membership import ALL_CAUSES

    payload = _load_fixture()
    prefixes = {cue["cue_id"].rsplit("_", 1)[0] for cue in payload["cues"]}
    for cause in ALL_CAUSES:
        assert cause in prefixes, f"cause {cause!r} has no cue in the committed fixture"


# ---------------------------------------------------------------------------
# Determinism (metadata-level, default lane) -- see the bench-marked
# rebuild-and-compare variant below for the real check.
# ---------------------------------------------------------------------------


def test_metadata_internally_consistent() -> None:
    payload = _load_fixture()
    assert payload["cue_count"] == len(payload["cues"])
    assert payload["dev_count"] + payload["holdout_count"] == payload["cue_count"]
    assert payload["dev_count"] == sum(1 for c in payload["cues"] if c["split"] == "dev")
    assert payload["holdout_count"] == sum(1 for c in payload["cues"] if c["split"] == "holdout")
    assert isinstance(payload["seed"], int)
    assert isinstance(payload["n_rationale"], str) and payload["n_rationale"]


@pytest.mark.bench
def test_bench_rebuild_is_byte_identical_under_same_seed(tmp_path) -> None:
    from bench.recall_quality_fixture import write_fixture

    payload_a = write_fixture(tmp_path / "a.json", seed=0)
    payload_b = write_fixture(tmp_path / "b.json", seed=0)
    assert payload_a["cues"] == payload_b["cues"]


def test_reload_guard_catches_cue_gold_pairing_swap(tmp_path, monkeypatch) -> None:
    """A rebuild that preserves both the cue_id set and the gold_record_id
    set, but reassigns which cue owns which gold record, must still be
    rejected -- comparing the two sets independently would miss it."""
    import bench.recall_quality_fixture as fixture_mod
    from bench.recall_quality_verdict import run_full_fixture

    loaded_cues = [
        {"cue_id": "cue-a", "gold_record_id": "gold-1", "split": "dev", "cluster_id": "k0"},
        {"cue_id": "cue-b", "gold_record_id": "gold-2", "split": "dev", "cluster_id": "k0"},
    ]
    fixture_path = tmp_path / "swapped.json"
    fixture_path.write_text(json.dumps({"seed": 0, "cues": loaded_cues}), encoding="utf-8")

    class _StubCorpus:
        def __init__(self, cues: "list[dict]") -> None:
            self.cues = cues

        def close(self) -> None:
            pass

    swapped_cues = [
        {"cue_id": "cue-a", "gold_record_id": "gold-2"},
        {"cue_id": "cue-b", "gold_record_id": "gold-1"},
    ]

    def _fake_build_full_fixture_corpus(*, seed):
        return _StubCorpus(swapped_cues), {}

    monkeypatch.setattr(
        fixture_mod, "build_full_fixture_corpus", _fake_build_full_fixture_corpus
    )

    with pytest.raises(RuntimeError):
        run_full_fixture(fixture_path, tmp_path / "out.json")


# ---------------------------------------------------------------------------
# Import fence: bench/recall_quality_*.py must never reference the
# real-store copy path, directly or transitively (proc_corpus_census
# module-level-imports recall_accuracy_real, so it is fenced too).
# ---------------------------------------------------------------------------


def _source_and_tree(path: Path) -> "tuple[str, ast.Module]":
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _imported_names(tree: ast.Module) -> "set[str]":
    names: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            for alias in node.names:
                names.add(alias.name)
    return names


def test_import_fence_against_real_store_path() -> None:
    assert _MODULE_PATHS, "expected bench/recall_quality_*.py modules to exist"
    for path in _MODULE_PATHS:
        assert path.name != "test_recall_quality_fixture.py"  # glob scope, defensive
        source, tree = _source_and_tree(path)
        for forbidden in _FORBIDDEN_IMPORT_NAMES:
            assert forbidden not in source, (
                f"{path.name} references forbidden real-store symbol {forbidden!r}"
            )
        imported = _imported_names(tree)
        hit = {name for name in imported if any(f in name for f in _FORBIDDEN_IMPORT_NAMES)}
        assert not hit, f"{path.name} imports forbidden real-store module(s): {hit}"


# ---------------------------------------------------------------------------
# Non-circularity of cause attribution
# ---------------------------------------------------------------------------


def test_ladder_flip_test_lexical_signal_changes_cause() -> None:
    """Changing ONLY the observed lexical-lane signal changes the
    classified cause -- proof the classification is a function of
    OBSERVED lane state, not a fixture-declared intent."""
    membership = GoldMembership(
        gold_id="flip-test-gold",
        in_post_cap_pool=False,
        capped_out=False,
        spread_edge_missing=False,
    )
    rank_context = {"in_budget_topk": False}

    cause_gated = classify_miss_cause(
        membership, rank_context,
        {"lexical_only_rescue": True, "lexical_contributed": False},
    )
    cause_served = classify_miss_cause(
        membership, rank_context,
        {"lexical_only_rescue": True, "lexical_contributed": True},
    )

    assert cause_gated == CAUSE_LEXICAL_COLD_OR_IDF_GATED
    assert cause_served == CAUSE_DISCOVERY_EXCLUDED
    assert cause_gated != cause_served


def test_committed_fixture_carries_no_intended_cause_field() -> None:
    payload = _load_fixture()
    for cue in payload["cues"]:
        hit = _FORBIDDEN_INTENT_KEYS & set(cue.keys())
        assert not hit, f"cue {cue['cue_id']} carries a fixture-declared intent key: {hit}"


def test_runner_classification_call_site_reads_no_intent_key() -> None:
    """AST-walk bench/recall_quality_verdict.py: no subscript access or
    ``.get(...)`` call anywhere in the module uses a string key that names
    an 'intended cause' -- the classification path may only consume
    capture_discovery_membership / classify_miss_cause output. Bare
    ``"cause"`` is only flagged on a cue-dict receiver, since
    ``row.get("cause")`` in ``_cause_distribution`` legitimately reads the
    runner's own computed output."""
    verdict_path = _REPO_ROOT / "bench" / "recall_quality_verdict.py"
    _source, tree = _source_and_tree(verdict_path)

    hits: "list[str]" = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key_node = node.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if key_node.value.lower() in _FORBIDDEN_INTENT_KEYS:
                    hits.append(key_node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = node.args[0].value.lower()
            receiver = node.func.value
            receiver_is_cue = isinstance(receiver, ast.Name) and receiver.id in _CUE_VAR_NAMES
            if key in _UNAMBIGUOUS_INTENT_KEYS or (key == "cause" and receiver_is_cue):
                hits.append(node.args[0].value)
    assert not hits, f"runner reads a fixture-declared intent key: {hits}"
