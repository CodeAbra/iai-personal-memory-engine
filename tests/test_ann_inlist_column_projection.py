"""Byte-identical rank-view differential for the ann_inlist column projection.

``_ann_knn_fetch_core`` (``hippo/_table.py``) runs the shared ``vec_label
IN(...)`` row fetch for every ANN output shape. It has no decode-mode signal
of its own: the same fetch backs both ``query_similar(decode="rank")`` (the
lazy ``RankCandidateView`` tier) and ``query_similar(decode="full")`` /
``query_similar_temporal``'s vec+as_of branch (the eager ``MemoryRecord``
tier), which decide their decode mode entirely in Python after the fetch
returns. Narrowing the projection is therefore only safe when gated
per-query-instance on the caller's actual decode mode -- never unconditional.

This file pins:
  - the gate exists (``HippoQuery.select_rank_view``) and is a no-op unless
    invoked;
  - narrowing drops exactly the two AES columns the rank-view decode never
    reads (``profile_modulation_gain_json``, ``provenance_json``) and keeps
    every other column, including ``vec_label``/``tombstoned_at``/
    ``embedding_pending`` that query_similar's own post-filter reads
    directly off the raw row;
  - matched rows, the vec_label -> _distance mapping, and every decoded
    ``RankCandidateView`` field are byte-identical narrowed vs unnarrowed,
    on both storage drivers;
  - ``query_similar`` invokes the gate exactly once for ``decode="rank"``
    and never for ``decode="full"``;
  - ``query_similar_temporal``'s vec+as_of branch (the one full-decode ANN
    caller in the codebase) never invokes the gate and keeps every column
    a full ``MemoryRecord`` decode needs;
  - a non-vacuity control: removing a column the rank-view decode actually
    reads changes the decoded output, proving the byte-identity comparison
    above is a genuine field-level pin, not a "no exception raised" check.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_query_similar_fast_decode import _nudge, _unit_vec  # noqa: E402

from iai_mcp.hippo._table import HippoQuery
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]

#: Fields RankCandidateView actually carries from the fetched row (excludes
#: provenance/profile_modulation_gain -- _from_row_rank_view never populates
#: them from the row; they stay at their dataclass defaults regardless of
#: projection width, so comparing them would prove nothing about this lever).
_RANK_VIEW_FIELDS = (
    "id", "embedding", "literal_surface", "aaak_index", "created_at",
    "stability", "tier", "tags", "language", "community_id",
    "structure_hv", "salience_level", "valence",
)

_EXCLUDED_COLUMNS = ("profile_modulation_gain_json", "provenance_json")

#: Columns query_similar's own post-fetch filter reads straight off the raw
#: row (tombstoned_at/embedding_pending) or that _from_row_rank_view requires
#: unconditionally (id) -- must survive narrowing.
_MUST_SURVIVE_COLUMNS = (
    "id", "vec_label", "tombstoned_at", "embedding_pending",
    "literal_surface", "structure_hv",
)


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _make_rec(vec: list[float], text: str, idx: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index=f"aaak-{idx}",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.37, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False,
        provenance=[{"session_id": f"s{idx}"}],
        created_at=now, updated_at=now, tags=[f"tag{idx}"], language="en",
        profile_modulation_gain={"curiosity": 0.5 + idx * 0.001},
    )


def _build_rank_corpus(
    store: MemoryStore, n: int = 200, seed_base: int = 100,
) -> tuple[list[float], list]:
    base = _unit_vec(seed_base)
    ids = []
    for i in range(n):
        vec = _nudge(base, seed_base + i + 1, strength=0.01 + 0.001 * i)
        rec = _make_rec(vec, f"record body {i} carries the word lantern", i)
        store.insert(rec)
        ids.append(rec.id)
    return base, ids


# ---------------------------------------------------------------------------
# Mechanism + byte-identity differential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_rank_view_projection_byte_identical_and_narrows_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base, _ids = _build_rank_corpus(store, n=200)
        tbl = store.db.open_table("records")

        full_rows = (
            tbl.search(list(base)).distance_type("cosine").limit(200).to_row_dicts()
        )
        assert full_rows, "fixture must produce ANN matches"
        for col in _EXCLUDED_COLUMNS:
            assert col in full_rows[0], (
                f"baseline SELECT * must still carry {col} -- ground truth for "
                "this differential must not itself be narrowed"
            )

        assert hasattr(HippoQuery, "select_rank_view"), (
            "HippoQuery.select_rank_view is the rank-tier projection gate "
            "narrowing the ann_inlist fetch -- not yet implemented"
        )

        narrow_rows = (
            tbl.search(list(base)).distance_type("cosine").limit(200)
            .select_rank_view().to_row_dicts()
        )

        assert len(narrow_rows) == len(full_rows)
        full_by_label = {r["vec_label"]: r for r in full_rows}
        narrow_by_label = {r["vec_label"]: r for r in narrow_rows}
        assert set(full_by_label) == set(narrow_by_label), (
            "narrowing the column projection must never change the matched "
            "row set"
        )

        for col in _EXCLUDED_COLUMNS:
            assert all(col not in row for row in narrow_rows), (
                f"{col} must be dropped from the rank-tier projection"
            )
        for col in _MUST_SURVIVE_COLUMNS:
            assert all(col in row for row in narrow_rows), (
                f"{col} must survive the rank-tier projection"
            )

        for label, full_row in full_by_label.items():
            narrow_row = narrow_by_label[label]
            assert full_row["_distance"] == narrow_row["_distance"], (
                f"vec_label->_distance mapping diverged for {label}"
            )
            full_view = store._from_row_rank_view(dict(full_row))
            narrow_view = store._from_row_rank_view(dict(narrow_row))
            for field_name in _RANK_VIEW_FIELDS:
                fv = getattr(full_view, field_name)
                nv = getattr(narrow_view, field_name)
                if field_name == "embedding":
                    assert np.allclose(
                        np.array(fv, dtype=np.float32), np.array(nv, dtype=np.float32),
                    ), f"embedding diverged for vec_label={label}"
                else:
                    assert fv == nv, (
                        f"{field_name} diverged for vec_label={label}: "
                        f"full={fv!r} narrow={nv!r}"
                    )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Production wiring: decode-mode gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_query_similar_gates_projection_on_decode_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base, _ids = _build_rank_corpus(store, n=20, seed_base=400)

        assert hasattr(HippoQuery, "select_rank_view")
        calls = {"n": 0}
        _orig = HippoQuery.select_rank_view

        def _wrapped(self: HippoQuery) -> HippoQuery:
            calls["n"] += 1
            return _orig(self)

        monkeypatch.setattr(HippoQuery, "select_rank_view", _wrapped)

        store.query_similar(base, k=5, decode="full")
        assert calls["n"] == 0, "decode='full' must never narrow the projection"

        store.query_similar(base, k=5, decode="rank")
        assert calls["n"] == 1, "decode='rank' must narrow exactly once per call"
    finally:
        store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_query_similar_temporal_full_decode_caller_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base, _ids = _build_rank_corpus(store, n=20, seed_base=300)

        assert hasattr(HippoQuery, "select_rank_view")
        calls = {"n": 0}
        _orig = HippoQuery.select_rank_view

        def _wrapped(self: HippoQuery) -> HippoQuery:
            calls["n"] += 1
            return _orig(self)

        monkeypatch.setattr(HippoQuery, "select_rank_view", _wrapped)

        as_of = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
        hits = store.query_similar_temporal(vec=base, as_of=as_of, k=5)

        assert hits, "fixture must produce hits for the vec+as_of branch"
        assert calls["n"] == 0, (
            "query_similar_temporal's vec+as_of branch is the codebase's one "
            "full-decode ANN caller and must never receive the rank-tier "
            "projection"
        )
        rec, _score = hits[0]
        assert rec.provenance, "full-decode output must retain provenance_json"
        assert rec.profile_modulation_gain, (
            "full-decode output must retain profile_modulation_gain_json"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Non-vacuity control
# ---------------------------------------------------------------------------


def test_dropped_needed_column_control_changes_decoded_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column the rank-view decode DOES need (literal_surface,
    structure_hv) must change the decoded output when absent from the row --
    proving the byte-identity assertions above are a real field-level pin,
    not a vacuous "no exception raised" check."""
    _set_driver(monkeypatch, "stdlib")
    store = MemoryStore(path=tmp_path / "store")
    try:
        base, _ids = _build_rank_corpus(store, n=5, seed_base=200)
        tbl = store.db.open_table("records")
        rows = tbl.search(list(base)).distance_type("cosine").limit(5).to_row_dicts()
        assert rows

        ground_truth = store._from_row_rank_view(dict(rows[0]))

        missing_literal = dict(rows[0])
        del missing_literal["literal_surface"]
        broken_view = store._from_row_rank_view(missing_literal)
        assert broken_view.literal_surface != ground_truth.literal_surface, (
            "dropping literal_surface must change the decoded field -- if it "
            "doesn't, the differential above cannot be trusted to catch a "
            "wrongly-narrowed projection"
        )

        missing_structure = dict(rows[0])
        del missing_structure["structure_hv"]
        broken_view2 = store._from_row_rank_view(missing_structure)
        assert broken_view2.structure_hv != ground_truth.structure_hv, (
            "dropping structure_hv must change the decoded field"
        )
    finally:
        store.close()
