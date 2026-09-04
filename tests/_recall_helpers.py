from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from test_store import _make  # noqa: E402

EMBED_DIM = 384
RNG_SEED = 20260601

UUID_HUB = UUID(int=2)
UUID_SEED = UUID(int=3)
UUID_INTER = UUID(int=4)
UUID_TWO_HOP = UUID(int=5)

UUID_TWO_HOP_SURFACE = "User reference gold doc 5"


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _deterministic_vec(seed: int = 12345) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_gold_record(i: int, vec: list[float]):
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
        aaak_index="",
        embedding=vec,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _populate_store(store, cue_vec: list[float], n_filler: int = 300) -> None:
    from iai_mcp.types import MemoryRecord as _MR

    rng = np.random.default_rng(RNG_SEED)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    for i in range(n_filler):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User filler record {i}", vec=v.tolist()))

    store.insert(_make_gold_record(1, cue_arr.tolist()))

    hub_rng = np.random.default_rng(44444)
    hub_vec = hub_rng.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])

    seed3_rec = _MR(
        id=UUID(int=3), tier="episodic",
        literal_surface="User reference gold doc 3",
        aaak_index="", embedding=cue_arr.tolist(), community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
        provenance=[], created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), tags=[], language="en",
    )
    store.insert(seed3_rec)

    inter_rng = np.random.default_rng(55555)
    inter_noise = inter_rng.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_arr) * cue_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))

    two_hop_rng = np.random.default_rng(66666)
    noise = two_hop_rng.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_arr) * cue_arr
    noise /= np.linalg.norm(noise)
    target_cos = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cos**2)))
    two_hop_vec = target_cos * cue_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))

    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])

    rng_boost = np.random.default_rng(77001)
    for boost_i in range(50):
        raw_v = rng_boost.random(EMBED_DIM).astype(np.float32)
        raw_v -= np.dot(raw_v, cue_arr) * cue_arr
        raw_v /= np.linalg.norm(raw_v)
        boost_rec = _MR(
            id=UUID(int=100 + boost_i), tier="episodic",
            literal_surface=f"User boost helper {boost_i}",
            aaak_index="", embedding=raw_v.tolist(), community_id=None,
            centrality=0.0, detail_level=2, pinned=False, stability=0.0,
            difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
            provenance=[], created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc), tags=[], language="en",
        )
        store.insert(boost_rec)
        store.boost_edges(
            [(UUID(int=5), UUID(int=100 + boost_i))], edge_type="hebbian", delta=[2.0]
        )


def _prime_structural_cache(store) -> None:
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc)


# ANN top-200 ranking on near-tied cosines carries legitimate run-to-run
# noise; a drop within this absolute band is not treated as a regression.
RECALL_AT_200_EPSILON = 0.02


def diff_recall_quality_baseline_entry(
    fresh: dict, committed: dict, epsilon: float = RECALL_AT_200_EPSILON,
) -> list[str]:
    """Diff one recall-quality baseline entry (the ``n1k``/``n10k`` shape:
    ``reference_cues``, ``recall_at_200``, ``two_hop_gold_reachable_via_2hop``,
    ``two_hop_gold_outside_ann_top200``, ``small_k_ef_blast_radius``) against
    the previously committed values. Returns a list of violation strings --
    empty means no regression. Never mutates either argument."""
    violations: list[str] = []

    committed_cues_by_label = {
        c["cue_label"]: c for c in committed.get("reference_cues", [])
    }
    fresh_cues_by_label = {
        c.get("cue_label"): c for c in fresh.get("reference_cues", [])
    }
    for label, committed_cue in committed_cues_by_label.items():
        fresh_cue = fresh_cues_by_label.get(label)
        if fresh_cue is None:
            violations.append(f"{label}: cue missing from fresh recompute")
            continue
        if committed_cue.get("must_hit"):
            for field in ("recall_at_5", "recall_at_10"):
                fresh_val = fresh_cue.get(field)
                committed_val = committed_cue.get(field)
                if fresh_val != committed_val:
                    violations.append(
                        f"{label}.{field}: committed={committed_val} "
                        f"fresh={fresh_val} (exact match required for a "
                        "must-hit cue)"
                    )
        fresh_anti = fresh_cue.get("anti_hit_surfaced")
        committed_anti = committed_cue.get("anti_hit_surfaced")
        if fresh_anti != committed_anti:
            violations.append(
                f"{label}.anti_hit_surfaced: committed={committed_anti} "
                f"fresh={fresh_anti}"
            )

    committed_r200 = committed.get("recall_at_200", {})
    fresh_r200 = fresh.get("recall_at_200", {})
    for label, committed_val in committed_r200.items():
        fresh_val = fresh_r200.get(label)
        if fresh_val is None:
            violations.append(f"recall_at_200.{label}: missing from fresh recompute")
            continue
        if fresh_val < committed_val - epsilon:
            violations.append(
                f"recall_at_200.{label}: committed={committed_val} "
                f"fresh={fresh_val} dropped more than epsilon={epsilon}"
            )

    for field in ("two_hop_gold_reachable_via_2hop", "two_hop_gold_outside_ann_top200"):
        if field not in committed:
            continue
        fresh_val = fresh.get(field)
        committed_val = committed.get(field)
        if fresh_val != committed_val:
            violations.append(f"{field}: committed={committed_val} fresh={fresh_val}")

    committed_blast = committed.get("small_k_ef_blast_radius", {})
    committed_dir = committed_blast.get("change_direction")
    if committed_dir is not None:
        fresh_blast = fresh.get("small_k_ef_blast_radius", {})
        fresh_dir = fresh_blast.get("change_direction")
        if fresh_dir != committed_dir:
            violations.append(
                "small_k_ef_blast_radius.change_direction: "
                f"committed={committed_dir} fresh={fresh_dir}"
            )

    return violations
