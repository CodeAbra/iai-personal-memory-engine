from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

class _ControlledEmbedder:
    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self.fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self.fixed:
            return list(self.fixed[text])
        import hashlib
        import random
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _unit_vector_with_cosine(cue_vec: list[float], target_cos: float) -> list[float]:
    cue = np.asarray(cue_vec, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue))
    if cue_norm == 0.0:
        raise ValueError("cue_vec must be non-zero")
    cue = cue / cue_norm

    probe = np.zeros(EMBED_DIM, dtype=np.float32)
    probe[1] = 1.0
    if abs(float(np.dot(cue, probe))) > 0.999:
        probe = np.zeros(EMBED_DIM, dtype=np.float32)
        probe[0] = 1.0
    orth = probe - float(np.dot(cue, probe)) * cue
    orth = orth / float(np.linalg.norm(orth))

    alpha = float(target_cos)
    beta = float(math.sqrt(max(0.0, 1.0 - alpha * alpha)))
    v = alpha * cue + beta * orth
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32).tolist()

def _make_episodic(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )

def _make_schema_hub_with_pattern(vec: list[float], text: str, pattern: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["schema", "draft", f"pattern:{pattern}"],
        language="en",
    )

def _make_proc_chunk(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="procedural",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.7,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["chunk", "source:cofire"],
        language="en",
    )

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

HUB_DEGREE = 8
CONCEPT_CUE = "concept question about the project structure overall"

SCHEMA_PATTERNS = [
    "tags:capture+role:user",
    "tags:capture+role:assistant",
    "tags:auto+schema",
    "tags:auto+pattern:capture",
    "tags:domain:project+role:agent",
]

def _seed_10_verbatim_plus_5_schema_hubs(tmp_path, hub_cos: float = 0.65):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    cue_vec = embedder.embed(CONCEPT_CUE)
    embedder.set_fixed(CONCEPT_CUE, cue_vec)

    verbatim_ids: list = []
    cos_values = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    for i, c in enumerate(cos_values):
        v = _unit_vector_with_cosine(cue_vec, c)
        rec = _make_episodic(v, f"verbatim text content variant {i} cosine {c}")
        store.insert(rec)
        verbatim_ids.append(rec.id)

    hub_records: list = []
    edge_pairs: list = []
    distractor_idx = 0
    for h, pattern in enumerate(SCHEMA_PATTERNS):
        hub_vec = _unit_vector_with_cosine(cue_vec, hub_cos)
        hub_rec = _make_schema_hub_with_pattern(
            hub_vec, f"schema hub {h} with pattern {pattern}", pattern=pattern,
        )
        store.insert(hub_rec)
        hub_records.append(hub_rec)
        for _ in range(HUB_DEGREE):
            d_vec = embedder.embed(f"r6-distractor-{distractor_idx}")
            d_rec = _make_episodic(d_vec, f"r6 distractor junk {distractor_idx}")
            store.insert(d_rec)
            edge_pairs.append((hub_rec.id, d_rec.id))
            distractor_idx += 1

    store.boost_edges(edge_pairs, edge_type="schema_instance_of", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        verbatim_ids, hub_records, CONCEPT_CUE,
    )

def test_concept_mode_excludes_schemas_from_hits(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids, hub_records, cue_text) = _seed_10_verbatim_plus_5_schema_hubs(tmp_path)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r6_exclude", mode="concept",
    )
    assert resp.cue_mode == "concept", f"expected cue_mode='concept', got {resp.cue_mode!r}"

    hub_id_set = {h.id for h in hub_records}
    for h in resp.hits:
        assert h.record_id not in hub_id_set, (
            f"concept mode must EXCLUDE schemas from hits[]; "
            f"schema {h.record_id} appeared at position "
            f"{[hh.record_id for hh in resp.hits].index(h.record_id)}"
        )
        rec = store.get(h.record_id)
        assert rec is not None, f"unknown record id {h.record_id} in hits"
        is_schema = (
            rec.tier == "semantic"
            and any(t.startswith("pattern:") for t in (rec.tags or []))
        )
        assert not is_schema, (
            f"hit {h.record_id} is a schema record (tier={rec.tier}, "
            f"tags={rec.tags}) but appeared in hits[]"
        )

def test_concept_mode_patterns_observed_capped_at_three(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids, hub_records, cue_text) = _seed_10_verbatim_plus_5_schema_hubs(
        tmp_path, hub_cos=0.95,
    )

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r6_cap", mode="concept",
    )
    assert resp.cue_mode == "concept"
    assert len(resp.patterns_observed) <= 3, (
        f"patterns_observed must be capped at 3 entries; got {len(resp.patterns_observed)}: "
        f"{resp.patterns_observed}"
    )

def test_concept_mode_patterns_observed_evidence_count_matches_edges(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids, hub_records, cue_text) = _seed_10_verbatim_plus_5_schema_hubs(
        tmp_path, hub_cos=0.95,
    )

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r6_evidence", mode="concept",
    )

    edges_df = store.db.open_table("edges").to_pandas()
    assert resp.patterns_observed, (
        "expected at least one pattern_observed entry on this fixture"
    )
    for entry in resp.patterns_observed:
        schema_id = entry["schema_id"]
        true_count = int(
            ((edges_df["edge_type"] == "schema_instance_of")
             & ((edges_df["dst"] == schema_id) | (edges_df["src"] == schema_id))).sum()
        )
        dst_only_count = int(
            ((edges_df["edge_type"] == "schema_instance_of")
             & (edges_df["dst"] == schema_id)).sum()
        )
        assert entry["evidence_count"] in (true_count, dst_only_count), (
            f"evidence_count for schema {schema_id} = {entry['evidence_count']}, "
            f"expected one of (OR-count {true_count}, dst-only {dst_only_count}). "
            f"HUB_DEGREE seeded = {HUB_DEGREE}"
        )

def test_concept_mode_patterns_observed_pattern_field_matches_tag(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids, hub_records, cue_text) = _seed_10_verbatim_plus_5_schema_hubs(
        tmp_path, hub_cos=0.95,
    )

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r6_pattern_field", mode="concept",
    )

    expected_patterns: dict[str, str] = {}
    for hub in hub_records:
        for t in hub.tags:
            if t.startswith("pattern:"):
                expected_patterns[str(hub.id)] = t.split(":", 1)[1]
                break

    assert resp.patterns_observed
    for entry in resp.patterns_observed:
        sid = entry["schema_id"]
        assert sid in expected_patterns, (
            f"unexpected schema_id {sid} in patterns_observed; "
            f"seeded hubs: {sorted(expected_patterns.keys())}"
        )
        assert entry["pattern"] == expected_patterns[sid], (
            f"pattern field mismatch for schema {sid}: "
            f"expected {expected_patterns[sid]!r}, got {entry['pattern']!r}"
        )

def test_concept_mode_drops_procedural_chunks_silently(tmp_path):
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph

    (store_base, embedder_base, graph_base, assignment_base, rich_club_base,
     _verbatim_ids_base, _hub_records_base, cue_text_base
     ) = _seed_10_verbatim_plus_5_schema_hubs(tmp_path / "baseline")
    resp_baseline = recall_for_response(
        store=store_base, graph=graph_base, assignment=assignment_base,
        rich_club=rich_club_base, embedder=embedder_base, cue=cue_text_base,
        session_id="r287_baseline", mode="concept",
    )

    (store, embedder, graph, assignment, rich_club,
     _verbatim_ids, hub_records, cue_text) = _seed_10_verbatim_plus_5_schema_hubs(tmp_path / "with_chunk")

    cue_vec = embedder.fixed[cue_text]
    proc_vec = _unit_vector_with_cosine(cue_vec, 0.75)
    proc_rec = _make_proc_chunk(proc_vec, "procedural chunk cofire pair label")
    store.insert(proc_rec)

    graph, assignment, rich_club = build_runtime_graph(store)

    candidate_ids = {r.id for r, _ in store.query_similar(cue_vec, k=32)}
    assert proc_rec.id in candidate_ids, (
        "test setup: procedural chunk must be a real ranking contender, "
        "not orthogonal noise excluded by construction"
    )

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r287_drop", mode="concept",
    )
    assert resp.cue_mode == "concept"

    hit_ids = {h.record_id for h in resp.hits}
    assert proc_rec.id not in hit_ids, (
        f"procedural chunk must be dropped from resp.hits under concept mode; "
        f"hits={hit_ids}"
    )

    for entry in resp.patterns_observed:
        assert entry["schema_id"] != str(proc_rec.id), (
            "procedural chunks get no patterns_observed summary entry (silent drop); "
            f"found one for {proc_rec.id}"
        )
    assert len(resp.patterns_observed) == len(resp_baseline.patterns_observed), (
        "the procedural chunk must not add or remove any patterns_observed "
        f"entry vs the no-chunk baseline: got {len(resp.patterns_observed)}, "
        f"expected {len(resp_baseline.patterns_observed)}"
    )

    hub_id_set = {h.id for h in hub_records}
    for h in resp.hits:
        assert h.record_id not in hub_id_set, (
            "the pre-existing schema pattern:-tag strip must still exclude "
            f"schema hits alongside the new procedural drop; leaked {h.record_id}"
        )
