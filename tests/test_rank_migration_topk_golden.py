"""Golden top-k over the synthetic production-shaped cue set.

Records today's ranker's top-k (k=10) for every cue in the synthetic corpus
(``tests/_synthetic_cue_corpus``) and asserts the current ranker still
matches it, tie-tolerantly, using the same comparator the recall-scoring
differential harness already ships (``tests/test_exact_authority_index.py``).

Record identity is a POSITION in the corpus build order, never the raw
per-run UUID: ``build_corpus_records`` mints a fresh ``uuid4()`` per record on
every call, so a golden keyed on raw ids could never replay against a later
run's freshly-built corpus. The topic/sentence iteration order is a static
list, so positional identity is stable across runs and across processes.

A missing golden file is a hard failure, not a silent re-record: set
``IAI_MCP_RECORD_RANK_GOLDEN=1`` to intentionally (re-)record
``tests/data/rank_migration_topk_golden.json``, otherwise the test always
only asserts. Without that gate, a missing file on some future (e.g.
post-retirement) tree would regenerate the golden from whatever ranker is
current and pass trivially -- exactly the failure mode this file exists to
catch.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    build_corpus_records,
    build_cue_set,
    flatten_cues,
    insert_corpus,
)
from tests.test_exact_authority_index import (  # noqa: E402
    _TOP_K_TIE_TOL,
    _assert_top_k_tie_tolerant,
)

import iai_mcp.pipeline as _pm  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402

_TOP_K = 10
_SEED = 0
_GOLDEN_PATH = Path(__file__).parent / "data" / "rank_migration_topk_golden.json"
_RECORD_ENV_VAR = "IAI_MCP_RECORD_RANK_GOLDEN"


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _freeze_age_penalty(monkeypatch: pytest.MonkeyPatch, records: list) -> None:
    """Freeze the one clock-reading score term to an anchor derived from the
    corpus's OWN timestamps, never `datetime.now()` at test-setup time: the
    gap between corpus-build time and freeze time is itself wall-clock
    dependent (embedder cache warmth, host load), and at this comparator's
    1e-6 tolerance a multi-second gap is large enough to flip a near-tied
    score pair -- observed directly this session (`expected=0.750904
    got=0.750903`, diff `1.0000000000287557e-06`, over tol by one ULP-scale
    step). Anchoring to the corpus's own max `created_at` makes every age
    delta a pure function of the corpus in hand, reproducible independent of
    how long the build took."""
    frozen_now = max(r.created_at for r in records)

    def _frozen(created_at: datetime) -> float:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        days = (frozen_now - created_at).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        return min(1.0, days / _pm.AGE_HALF_LIFE_DAYS)

    monkeypatch.setattr(_pm, "_age_penalty", _frozen)


def _build_store_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "tuple[MemoryStore, object, object, list, Embedder, dict[str, int], list]":
    _monkeypatch_env(monkeypatch, tmp_path)
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    # Positional label -- stable across runs; the raw uuid4() id is not.
    id_to_label = {str(rec.id): idx for idx, rec in enumerate(records)}

    store = MemoryStore(path=tmp_path / "store")
    insert_corpus(store, records)

    from iai_mcp.retrieve import build_runtime_graph

    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club, embedder, id_to_label, records


def _compute_actual_topk(
    store: MemoryStore, graph, assignment, rich_club, embedder: Embedder,
    id_to_label: "dict[str, int]", cues: list,
) -> "dict[str, list]":
    actual: "dict[str, list]" = {}
    for idx, cue in enumerate(cues):
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue.text, session_id="rank-migration-topk-golden",
            budget_tokens=1500, mode=cue.mode,
        )
        top_k = []
        for h in response.hits[:_TOP_K]:
            label = id_to_label.get(str(h.record_id))
            if label is None:
                raise AssertionError(
                    f"hit record_id {h.record_id!r} is not one of this run's "
                    f"corpus records -- id_to_label is incomplete or the "
                    f"recall path returned an id outside the synthetic corpus"
                )
            top_k.append([label, float(h.score)])
        actual[str(idx)] = {
            "cue_text": cue.text,
            "band": cue.band,
            "mode": cue.mode,
            "top_k": top_k,
        }
    return actual


def _load_or_record_golden(actual: "dict[str, list]") -> "dict[str, list]":
    if _GOLDEN_PATH.exists():
        with _GOLDEN_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    # Absence must never silently re-record against whatever tree happens to
    # be checked out -- a missing file on the post-retirement tree would
    # otherwise regenerate the golden from the CANDIDATE ranker and pass
    # trivially, defeating the whole point of a frozen before-number. Opt in
    # explicitly to (re-)record.
    if os.environ.get(_RECORD_ENV_VAR) != "1":
        raise AssertionError(
            f"golden file absent at {_GOLDEN_PATH}; set {_RECORD_ENV_VAR}=1 "
            f"to intentionally (re-)record it"
        )
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _GOLDEN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(actual, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return actual


def _assert_no_owner_store_content(golden: "dict[str, list]") -> None:
    """The committed golden is over the synthetic corpus only: every label is
    a small non-negative int (a corpus position), never a real UUID/string
    surface, so the file cannot leak owner store content. Folded into the
    single test below (not a standalone test) so it never depends on
    execution order against the record-or-load step."""
    assert golden, "golden file must not be empty"
    for entry in golden.values():
        assert isinstance(entry["cue_text"], str)
        for label, score in entry["top_k"]:
            assert isinstance(label, int) and label >= 0
            assert isinstance(score, (int, float))
            # A UUID string would never parse as int -- structural guard that
            # a future edit cannot silently swap positional labels for ids.
            with pytest.raises(ValueError):
                UUID(str(label))


def test_current_ranker_matches_recorded_golden_topk(tmp_path, monkeypatch):
    store, graph, assignment, rich_club, embedder, id_to_label, records = _build_store_and_labels(
        tmp_path, monkeypatch,
    )
    _freeze_age_penalty(monkeypatch, records)
    cues = flatten_cues(build_cue_set(seed=_SEED))

    actual = _compute_actual_topk(store, graph, assignment, rich_club, embedder, id_to_label, cues)
    golden = _load_or_record_golden(actual)
    _assert_no_owner_store_content(golden)

    assert set(golden.keys()) == set(actual.keys()), (
        "golden cue-index set diverged from the current cue set -- "
        "the corpus/cue builders changed shape since the golden was recorded"
    )
    for key, expected_entry in golden.items():
        got_entry = actual[key]
        assert got_entry["cue_text"] == expected_entry["cue_text"], (
            f"cue text at index {key} diverged from the recorded golden"
        )
        expected_pairs = [(label, score) for label, score in expected_entry["top_k"]]
        got_pairs = [(label, score) for label, score in got_entry["top_k"]]
        _assert_top_k_tie_tolerant(
            expected_pairs, got_pairs, k=_TOP_K, cue_seed=int(key), tol=_TOP_K_TIE_TOL,
        )
