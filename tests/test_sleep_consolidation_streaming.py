from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from iai_mcp.sleep import CLUSTER_MIN_SIZE, _tier0_schema_surfacing
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

def _make(
    text: str = "hello world",
    tier: str = "episodic",
    tags: list[str] | None = None,
    detail: int = 2,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags if tags is not None else [],
        language=language,
    )

@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")

def _populate_mixed_16(store: MemoryStore) -> None:
    for _ in range(4):
        store.insert(_make(text="alpha", tags=["tag-a"]))
    for _ in range(5):
        store.insert(_make(text="beta", tags=["tag-b"]))
    for _ in range(7):
        store.insert(_make(text="gamma", tags=["raw:noise", "domain:misc"]))

def test_tier0_schema_surfacing_uses_iter_record_columns_not_all_records(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate_mixed_16(store)

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    _tier0_schema_surfacing(store)

    assert spy_all.call_count == 0, (
        f"_tier0_schema_surfacing must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count == 1, (
        f"_tier0_schema_surfacing must call store.iter_record_columns() exactly "
        f"once; got {spy_iter.call_count} call(s)"
    )

    args, kwargs = spy_iter.call_args
    if args:
        cols = args[0]
    else:
        cols = kwargs.get("columns")
    assert cols == ["tags_json", "tombstoned_at"], (
        "projection must stay metadata-only — tags for the counts, "
        "tombstoned_at for the liveness skip; neither column is encrypted, "
        f"so the zero-AES-GCM cost holds. got columns={cols!r}"
    )

def test_tier0_schema_surfacing_zero_decrypt_calls(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate_mixed_16(store)

    decrypt_spy = MagicMock(wraps=store._decrypt_for_record)
    monkeypatch.setattr(store, "_decrypt_for_record", decrypt_spy)

    _tier0_schema_surfacing(store)

    assert decrypt_spy.call_count == 0, (
        f"_tier0_schema_surfacing must NOT trigger ANY _decrypt_for_record "
        f"calls (-16210 AES-GCM operations on the 8105-record "
        f"production store); got {decrypt_spy.call_count} call(s)"
    )

def test_tier0_schema_surfacing_filters_raw_and_domain_tags(
    store: MemoryStore,
) -> None:
    for _ in range(5):
        store.insert(_make(text="r1", tags=["raw:literal", "tag-real"]))
    for _ in range(5):
        store.insert(_make(text="r2", tags=["domain:foo", "tag-real-2"]))

    candidates = _tier0_schema_surfacing(store)
    patterns = sorted(c["pattern"] for c in candidates)

    assert "tag:tag-real" in patterns
    assert "tag:tag-real-2" in patterns
    assert "tag:raw:literal" not in patterns
    assert "tag:domain:foo" not in patterns

    by_pattern = {c["pattern"]: c for c in candidates}
    assert by_pattern["tag:tag-real"]["evidence_count"] == 5
    assert by_pattern["tag:tag-real"]["confidence"] == pytest.approx(0.5)
    assert by_pattern["tag:tag-real-2"]["evidence_count"] == 5
    assert by_pattern["tag:tag-real-2"]["confidence"] == pytest.approx(0.5)

def test_tier0_schema_surfacing_floor_count_3(store: MemoryStore) -> None:
    for _ in range(3):
        store.insert(_make(text="a", tags=["tag-a"]))
    for _ in range(3):
        store.insert(_make(text="b", tags=["tag-b"]))

    candidates = _tier0_schema_surfacing(store)
    assert len(candidates) == 2

    expected = sorted(
        [
            {"pattern": "tag:tag-a", "confidence": 0.3, "evidence_count": 3},
            {"pattern": "tag:tag-b", "confidence": 0.3, "evidence_count": 3},
        ],
        key=lambda d: d["pattern"],
    )
    actual = sorted(candidates, key=lambda d: d["pattern"])
    for e, a in zip(expected, actual, strict=True):
        assert a["pattern"] == e["pattern"]
        assert a["evidence_count"] == e["evidence_count"]
        assert a["confidence"] == pytest.approx(e["confidence"])

def test_tier0_schema_surfacing_below_cluster_min_size_returns_empty(
    store: MemoryStore,
) -> None:
    for _ in range(CLUSTER_MIN_SIZE - 1):
        store.insert(_make(text="below-floor", tags=["any-tag"]))

    candidates = _tier0_schema_surfacing(store)
    assert candidates == [], (
        f"expected [] when record count ({CLUSTER_MIN_SIZE - 1}) is below "
        f"CLUSTER_MIN_SIZE ({CLUSTER_MIN_SIZE}); got {candidates!r}"
    )

def test_tier0_schema_surfacing_byte_identical_to_pre_w3(
    store: MemoryStore,
) -> None:
    for _ in range(5):
        store.insert(_make(text="a", tags=["a"]))
    for _ in range(5):
        store.insert(_make(text="b", tags=["b"]))
    for _ in range(4):
        store.insert(_make(text="c", tags=["c"]))
    for _ in range(3):
        store.insert(_make(text="ar", tags=["a", "raw:noise"]))
    for _ in range(3):
        store.insert(_make(text="bd", tags=["b", "domain:x"]))

    records = store.all_records()
    tag_counts: dict[str, int] = {}
    for r in records:
        for t in r.tags or []:
            if t.startswith("raw:") or t.startswith("domain:"):
                continue
            tag_counts[t] = tag_counts.get(t, 0) + 1
    expected = [
        {
            "pattern": f"tag:{tag}",
            "confidence": min(1.0, count / 10.0),
            "evidence_count": count,
        }
        for tag, count in tag_counts.items()
        if count >= 3
    ]

    actual = _tier0_schema_surfacing(store)

    expected_sorted = sorted(expected, key=lambda d: d["pattern"])
    actual_sorted = sorted(actual, key=lambda d: d["pattern"])

    assert len(actual_sorted) == len(expected_sorted)
    for e, a in zip(expected_sorted, actual_sorted, strict=True):
        assert a["pattern"] == e["pattern"]
        assert a["evidence_count"] == e["evidence_count"]
        assert a["confidence"] == pytest.approx(e["confidence"])

    assert {c["pattern"] for c in actual} == {"tag:a", "tag:b", "tag:c"}
    by_pattern = {c["pattern"]: c["evidence_count"] for c in actual}
    assert by_pattern["tag:a"] == 8
    assert by_pattern["tag:b"] == 8
    assert by_pattern["tag:c"] == 4

def test_tier0_schema_surfacing_handles_malformed_tags_json_gracefully(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(5):
        store.insert(_make(text="g", tags=["tag-good"]))

    real_iter = store.iter_record_columns

    def iter_with_malformed_tail(columns, **kwargs):  # noqa: ANN001 — match arg shape
        yield from real_iter(columns, **kwargs)
        yield {"tags_json": "not valid json {{{"}

    monkeypatch.setattr(store, "iter_record_columns", iter_with_malformed_tail)

    candidates = _tier0_schema_surfacing(store)

    by_pattern = {c["pattern"]: c for c in candidates}
    assert "tag:tag-good" in by_pattern, (
        f"valid records' tag must still surface despite malformed-row tail; "
        f"got candidates={candidates!r}"
    )
    assert by_pattern["tag:tag-good"]["evidence_count"] == 5
    assert by_pattern["tag:tag-good"]["confidence"] == pytest.approx(0.5)

@pytest.fixture
def _patch_schema_embedder(monkeypatch: pytest.MonkeyPatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):  # noqa: ANN001
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):  # noqa: ANN001
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def _populate_for_heavy(store: MemoryStore) -> list[MemoryRecord]:
    from iai_mcp.types import EMBED_DIM as _EMBED_DIM
    from datetime import datetime as _dt, timezone as _tz
    from uuid import uuid4 as _uuid

    inserted: list[MemoryRecord] = []
    for i in range(10):
        r = MemoryRecord(
            id=_uuid(),
            tier="episodic",
            literal_surface=f"meeting-rec-{i}",
            aaak_index="",
            embedding=[1.0] + [0.0] * (_EMBED_DIM - 1),
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=_dt.now(_tz.utc),
            updated_at=_dt.now(_tz.utc),
            tags=["meeting", "notes"],
            language="en",
        )
        store.insert(r)
        inserted.append(r)
    return inserted

def test_run_heavy_consolidation_calls_all_records_at_most_once(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    _patch_schema_embedder,
) -> None:
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    spy = MagicMock(wraps=store.all_records)
    monkeypatch.setattr(store, "all_records", spy)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    run_heavy_consolidation(
        store, session_id="s-w4-inv", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    assert spy.call_count <= 1, (
        f"run_heavy_consolidation must call store.all_records() "
        f"AT MOST ONCE per invocation; got {spy.call_count} call(s)."
    )

def test_run_heavy_consolidation_iter_record_columns_called_at_least_once(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    _patch_schema_embedder,
) -> None:
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    spy = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "iter_record_columns", spy)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    run_heavy_consolidation(
        store, session_id="s-w4-iter", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    assert spy.call_count >= 1, (
        f"run_heavy_consolidation must call store.iter_record_columns() at "
        f"least once per invocation (_tier0_schema_surfacing path + "
        f"streaming schema.py paths); got {spy.call_count} call(s)."
    )

def test_run_heavy_consolidation_returns_expected_keys(
    store: MemoryStore,
    _patch_schema_embedder,
) -> None:
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    result = run_heavy_consolidation(
        store, session_id="s-w4-keys", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    expected_keys = {
        "mode",
        "tier",
        "summaries_created",
        "decay_result",
        "schema_candidates",
        "schemas_induced",
    }
    assert set(result.keys()) == expected_keys, (
        f"run_heavy_consolidation public contract: expected keys "
        f"{sorted(expected_keys)}; got {sorted(result.keys())}"
    )
    assert result["mode"] == "heavy"
    assert result["tier"] in ("tier0", "tier1")
