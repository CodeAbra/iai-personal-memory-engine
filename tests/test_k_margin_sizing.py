"""Bucket-B promotion-distance measurement.

Measures how far outside a bare Bucket-A-only ranking a candidate can sit
before a Bucket-B adjustment (profile modulation, tier boost, salience
boost, temporal-match boost, the historical_verbatim anchor rewrite, and
the post-hoc stale-downweight / supersede-cap passes that run over the
same unbounded candidate list) promotes it into the served top-k. The
resulting distribution derives ``k_margin``, the over-fetch window a
Bucket-A-only candidate producer must return so no Bucket-B-promotable
row is truncated away before Python ever sees it.

Two-arm design, calling `_recall_core` directly (not `recall_for_response`,
which caps its output at `_POST_RANK_MAX_HITS=50` -- too shallow to see a
promotion event whose bare rank sits well below that). `cue_intent` is
computed once per cue and passed IDENTICALLY to both arms, so neither arm
can diverge in which candidates it constructs (community gate, spread,
escalation widen never read a Bucket-B lever); only score VALUES differ:

- "bare" arm: every Bucket-B mechanism neutralized in place (env-var
  boost overrides, a patched no-op profile-modulation function, and an
  empty contradicts-outgoing map for T17), same store/graph/pool/cue_intent.
- "full" arm: today's unmodified scoring, plus the same post-hoc
  stale-downweight / supersede-cap / sort pass `recall_for_response` applies.

Both arms share one combined store: the production-shaped 3-band corpus
(`build_corpus_records`) plus the dedicated Bucket-B evidence fixture
(`build_bucket_b_evidence_fixture`), so promotion events are measured
against a realistic-scale background pool, not the fixture's own
2-record isolation.
"""
from __future__ import annotations

import builtins
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    CueSpec,
    apply_term_discrimination_edges,
    build_bucket_b_evidence_fixture,
    build_corpus_records,
    build_cue_set,
    flatten_cues,
    insert_corpus,
)

import iai_mcp.pipeline as _pm  # noqa: E402
import iai_mcp.profile as _profile_mod  # noqa: E402
from iai_mcp.core import set_community_names  # noqa: E402
from iai_mcp.cue_router import _classify_cue  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.pipeline import _recall_core  # noqa: E402
from iai_mcp.retrieve import (  # noqa: E402
    apply_stale_downweight,
    apply_supersede_cap,
    build_temporal_validity_maps,
    derive_temporal_validity,
    sort_served_hits,
)
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

_SEED = 0
_K = 10
_CORPUS_AGE_FLOOR_DAYS = 7
_CORPUS_AGE_JITTER_HOURS = 3 * 24


def _freeze_age_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen_now = datetime.now(timezone.utc)

    def _frozen(created_at: datetime) -> float:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        days = (frozen_now - created_at).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        return min(1.0, days / _pm.AGE_HALF_LIFE_DAYS)

    monkeypatch.setattr(_pm, "_age_penalty", _frozen)


def _salience_shadow(forced_id: object):
    """Module-level `getattr` shadow (installed as `iai_mcp.pipeline.getattr`):
    forces `salience_level` to "critical" for one record id, every other
    read passes through to the real builtin. `salience_level` is not a
    `SimpleRecordView` field on the graph-hydrated hot path, so T15 needs
    this injection to be reachable at all -- same technique the differential
    harness already uses for the same reason."""
    def _getattr(obj, attr, default=None):
        if attr == "salience_level" and builtins.getattr(obj, "id", None) == forced_id:
            return "critical"
        return builtins.getattr(obj, attr, default)

    return _getattr


def _noop_profile_modulation(*_args, **_kwargs) -> dict:
    return {}


def _extra_filler_records(embedder: Embedder, n: int, base_ts: datetime, seed: int) -> "list[MemoryRecord]":
    """Generic distractor records for the scale-sensitivity check -- distinct
    embedded sentences, never touching the shared fixture corpus so the
    differential harness's own fixture stays untouched by this plan."""
    rng = np.random.default_rng(seed)
    records: list[MemoryRecord] = []
    for i in range(n):
        text = f"Filler note number {i} about an unrelated household errand."
        vec = np.asarray(embedder.embed(text), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        ts = base_ts + timedelta(hours=int(rng.integers(0, _CORPUS_AGE_JITTER_HOURS)))
        records.append(
            MemoryRecord(
                id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
                embedding=vec.tolist(), community_id=None, centrality=0.0, detail_level=2,
                pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
                never_decay=False, never_merge=False, provenance=[],
                created_at=ts, updated_at=ts, tags=[], language="en",
            ),
        )
    return records


def _build_combined_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, store_name: str, extra_records=None):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / store_name
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    background = build_corpus_records(seed=_SEED, embedder=embedder)
    fixture = build_bucket_b_evidence_fixture(seed=_SEED, embedder=embedder)
    all_records = background + fixture.records + list(extra_records or [])

    store = MemoryStore(path=store_root)
    insert_corpus(store, all_records)
    apply_term_discrimination_edges(store, fixture)

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)

    # Alternating monotropism_depth per community -- the only T8 sub-term
    # capable of varying per candidate (interest_boost/dunn_quadrant are
    # uniform multipliers, order-preserving by construction); this makes T8
    # a real per-cue promotion driver instead of the score-only evidence the
    # small isolated fixture demonstrates. Values stay inside the knob's own
    # legal range (`monotropism_depth`'s schema is `float_range:0.0..1.0`,
    # `lilli/profile/knobs.py:25`) so the measurement reflects a value a real
    # caller could actually set, not an out-of-schema exaggeration.
    names: dict[str, str] = {}
    depth: dict[str, float] = {}
    distinct = sorted({str(c) for c in assignment.node_to_community.values()})
    for i, cid in enumerate(distinct):
        name = f"k_margin_probe_community_{i}"
        names[cid] = name
        depth[name] = 1.0 if i % 2 == 0 else 0.0
    set_community_names(names)
    profile_state = {"monotropism_depth": depth}

    tv_outgoing, tv_ts = build_temporal_validity_maps(store)
    return store, graph, assignment, rich_club, embedder, fixture, profile_state, tv_outgoing, tv_ts


def _bare_bucket_a_ranking(
    store, graph, assignment, rich_club, embedder, cue: str, mode: str, cue_intent, profile_state: dict,
) -> "list[str]":
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("IAI_MCP_TIER_BOOST", "1.0")
        mp.setenv("IAI_MCP_SALIENCE_BOOST", "0.0")
        mp.setenv("IAI_MCP_TEMPORAL_BOOST", "1.0")
        mp.setattr(_profile_mod, "profile_modulation_for_record", _noop_profile_modulation)
        graph._records_view_cache = None
        core = _recall_core(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="k-margin-bucket-a",
            profile_state=profile_state, mode=mode,
            cue_intent=cue_intent, contradicts_outgoing={},
        )
    return [str(h.record_id) for h in core.scored_hits]


def _full_ranking(
    store, graph, assignment, rich_club, embedder, cue: str, mode: str, cue_intent,
    profile_state: dict, tv_outgoing: dict, tv_ts: dict,
) -> "list[str]":
    graph._records_view_cache = None
    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id="k-margin-full",
        profile_state=profile_state, mode=mode,
        cue_intent=cue_intent, contradicts_outgoing=tv_outgoing,
    )
    derive_temporal_validity(store, core.scored_hits, outgoing=tv_outgoing, ts_by_id=tv_ts)
    apply_stale_downweight(core.scored_hits, cue_intent=cue_intent)
    apply_supersede_cap(core.scored_hits, tv_outgoing, cue_intent=cue_intent)
    sort_served_hits(core.scored_hits)
    return [str(h.record_id) for h in core.scored_hits]


def _harvest_promotion_events(bare_ranking: "list[str]", full_ranking: "list[str]", k: int) -> "list[int]":
    bare_rank = {rid: idx + 1 for idx, rid in enumerate(bare_ranking)}
    events: list[int] = []
    for rid in full_ranking[:k]:
        r = bare_rank[rid]
        if r > k:
            events.append(r - k)
    return events


def _percentile(sorted_values: "list[int]", pct: float) -> int:
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


def test_measure_bucket_b_promotion_distance(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    (store, graph, assignment, rich_club, embedder, fixture, profile_state,
     tv_outgoing, tv_ts) = _build_combined_store(tmp_path, monkeypatch, store_name="k-margin-sizing")
    monkeypatch.setattr(_pm, "getattr", _salience_shadow(fixture.ids["t15_critical"]), raising=False)

    cues = flatten_cues(build_cue_set(seed=_SEED))
    for term, text in fixture.probe_cue.items():
        cues.append(CueSpec(text=text, band=f"bucket_b_probe_{term}", mode="concept"))

    all_events: list[int] = []
    per_cue: list[tuple[str, list[int]]] = []
    for cue in cues:
        _, cue_intent, _ = _classify_cue(cue.text)
        bare = _bare_bucket_a_ranking(
            store, graph, assignment, rich_club, embedder, cue.text, cue.mode, cue_intent, profile_state,
        )
        full = _full_ranking(
            store, graph, assignment, rich_club, embedder, cue.text, cue.mode, cue_intent,
            profile_state, tv_outgoing, tv_ts,
        )
        assert set(bare) == set(full), (
            f"candidate-scope mismatch between the bucket-A-only and full arms for cue={cue.text!r} -- "
            f"bare-only={set(bare) - set(full)} full-only={set(full) - set(bare)}; "
            "a Bucket-B neutralization lever changed candidate membership, invalidating the distance measurement"
        )
        events = _harvest_promotion_events(bare, full, _K)
        if events:
            per_cue.append((cue.text, events))
        all_events.extend(events)

    assert all_events, (
        "no Bucket-B promotion events observed across the whole cue sweep -- "
        "the measurement is vacuous, check the neutralization levers actually differ from the full arm"
    )

    all_events.sort()
    p95 = _percentile(all_events, 0.95)
    p99 = _percentile(all_events, 0.99)
    max_distance = all_events[-1]
    k_margin = max_distance

    print(f"\n  promotion-distance events: n={len(all_events)} (across {len(cues)} cues, k={_K})")
    print(f"  max={max_distance} p99={p99} p95={p95}")
    print("  derivation rule: k_margin = max observed promotion distance (no additional rounding)")
    print(f"  derived k_margin = {k_margin}")
    print("  by-cue events (cue: [distances]):")
    for cue_text, events in per_cue:
        print(f"    {cue_text!r}: {events}")


def test_promotion_distance_scale_sensitivity(tmp_path, monkeypatch):
    """Cheap second-scale check: does the max promotion distance grow with
    pool size? Only the five dedicated Bucket-B probe cues are swept here
    (not the full 3-band cue set) to keep the extra embedding cost bounded
    -- this is a sensitivity signal for the FFI constant, not a second full
    distribution."""
    _freeze_age_penalty(monkeypatch)
    base_ts = datetime.now(timezone.utc) - timedelta(days=_CORPUS_AGE_FLOOR_DAYS + 1)
    embedder = Embedder()
    filler_n = 90
    filler = _extra_filler_records(embedder, n=filler_n, base_ts=base_ts, seed=_SEED)
    (store, graph, assignment, rich_club, embedder, fixture, profile_state,
     tv_outgoing, tv_ts) = _build_combined_store(
        tmp_path, monkeypatch, store_name="k-margin-sizing-scaled", extra_records=filler,
    )
    monkeypatch.setattr(_pm, "getattr", _salience_shadow(fixture.ids["t15_critical"]), raising=False)

    events: list[int] = []
    for term, text in fixture.probe_cue.items():
        _, cue_intent, _ = _classify_cue(text)
        bare = _bare_bucket_a_ranking(
            store, graph, assignment, rich_club, embedder, text, "concept", cue_intent, profile_state,
        )
        full = _full_ranking(
            store, graph, assignment, rich_club, embedder, text, "concept", cue_intent,
            profile_state, tv_outgoing, tv_ts,
        )
        assert set(bare) == set(full), f"candidate-scope mismatch at scaled pool for {term}"
        events.extend(_harvest_promotion_events(bare, full, _K))

    print(f"\n  scaled pool (+{filler_n} filler records) promotion-distance events: {sorted(events)}")
    if events:
        print(f"  scaled max={max(events)}")
    else:
        print("  scaled max=0 (no promotion events at this scale)")
