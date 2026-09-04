"""Constructibility gate for the eval-copy recall baseline.

A freshly-opened eval-copy store comes up with a deterministically-cold
lexical index and a cold exact-cosine index; the recall-path entry points
degrade a cold index to an empty result with zero side effects rather than
rebuilding it. These tests prove the explicit warm-up fixes the cold
degrade, prove the graph-cache generation-parity gate raises when it
should, and prove the discovery-stage 2-hop spread mechanism actually
fires (and is cue-order independent) on an isolated, synthetic seeded
store -- never the live store, never real memory content.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

_ANCHOR_ACTIVE = range(0, 20)
_TARGET_ACTIVE = range(180, 200)


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture(autouse=True)
def _reset_graph_cache_generation_epoch():
    """The graph-cache module generation counter, dirty counter, and
    community-detection child-process pool are all process-global state
    shared across every store opened in this pytest process -- reset them
    before each test so one test's structural state can never leak into the
    next and make a failure uninterpretable."""
    from iai_mcp import runtime_graph_cache as rgc

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    yield


def _dense_vec(active_idx, dim: int = EMBED_DIM, mag: float = 0.9, base: float = 0.02) -> list[float]:
    vec = [base] * dim
    for i in active_idx:
        vec[i] = mag
    return vec


def _make_record(text: str, vec: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _flush(store) -> None:
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    flush_record_buffer(store)
    flush_edge_buffer(store)


# ---------------------------------------------------------------------------
# Warm-up constructibility proofs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_warm_lexical_nonempty_after_warmup(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from bench.recall_accuracy_real import warm_eval_copy_store
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    rec = _make_record(
        "alice distinctive lexical token zzqpx for a warm-baseline check",
        _dense_vec(_ANCHOR_ACTIVE),
    )
    store.insert(rec)
    _flush(store)

    warm_eval_copy_store(store)

    hits = store.lexical_query_warm("zzqpx", k=5)
    assert hits, (
        "lexical_query_warm returned empty after warm_eval_copy_store — the "
        "warm-up must leave the lexical index generation-current"
    )
    assert any(str(rec.id) == rid for rid, _score in hits), (
        f"seeded record {rec.id} did not appear in the warm lexical result {hits!r}"
    )
    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_unwarmed_lexical_empty(driver, tmp_path, monkeypatch) -> None:
    """Regression guard: an UNWARMED copy must still degrade to an empty
    lexical result, not merely happen to return one. If the production
    entry point's cold-degrade contract ever regresses to an implicit
    rebuild, this goes red -- proving the explicit warm-up above is
    load-bearing, not decorative."""
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    rec = _make_record(
        "alice distinctive lexical token zzqpx for a warm-baseline check",
        _dense_vec(_ANCHOR_ACTIVE),
    )
    store.insert(rec)
    _flush(store)

    hits = store.lexical_query_warm("zzqpx", k=5)
    assert hits == [], (
        f"expected an empty result from an unwarmed lexical index, got {hits!r} "
        "— a fresh store's index must never rebuild implicitly on this path"
    )
    store.close()


# ---------------------------------------------------------------------------
# Graph-cache generation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graphcache_generation_parity(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.store import MemoryStore
    from bench.recall_accuracy_real import assert_graphcache_generation_parity

    store = MemoryStore(path=tmp_path / "store")
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _dense_vec(_ANCHOR_ACTIVE)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={comm_id: [node_id]},
    )

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    ok = rgc.save_with_generation(store, assignment, [node_id])
    assert ok, "test setup: save_with_generation must succeed"

    # Positive control: the module generation was just stamped from THIS
    # store's own snapshot write, so the copy is generation-matched.
    assert_graphcache_generation_parity(store)

    # Negative control: skew the process-wide module generation ABOVE what
    # the on-disk snapshot stamps. This is not self-healing — the loader
    # only ever raises its bootstrap floor, never lowers it — so this
    # deterministically reproduces the cross-store generation-leak risk a
    # process that opens more than one store can hit.
    with rgc._GEN_LOCK:
        rgc._current_generation = 99
    with pytest.raises(RuntimeError, match="generation mismatch"):
        assert_graphcache_generation_parity(store)

    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graphcache_generation_parity_fresh_process_single_store(driver, tmp_path, monkeypatch) -> None:
    """Regression guard for the standalone-consult-overlay ordering bug: a
    snapshot written by a DIFFERENT process (module generation counter still
    at 0, never bootstrapped in THIS process) must still pass. This is
    exactly the eval harness's real call pattern -- `open_eval_copy_store`
    opens one fresh store per process and never itself calls
    `save_with_generation` -- which the positive control above does not
    exercise: it bootstraps the module generation as a side effect of
    writing the snapshot in the SAME process, so it cannot see a bug that
    only fires when the read happens before any bootstrap in-process."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.store import MemoryStore
    from bench.recall_accuracy_real import assert_graphcache_generation_parity

    store = MemoryStore(path=tmp_path / "store")
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _dense_vec(_ANCHOR_ACTIVE)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={comm_id: [node_id]},
    )

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    ok = rgc.save_with_generation(store, assignment, [node_id])
    assert ok, "test setup: save_with_generation must succeed"

    # Simulate a fresh process opening this copy: the snapshot on disk is
    # real and current (just stamped above), but nothing in THIS process
    # has bootstrapped the module-global generation counter yet.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0

    assert_graphcache_generation_parity(store)  # must not raise

    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graphcache_generation_parity_raises_on_cold_degrade(driver, tmp_path, monkeypatch) -> None:
    """Negative control distinct from the epoch-skew case above: a copy with
    NO snapshot at all (never built, or a fully torn/deleted file) must
    still raise -- proving the reordered check is not tautologically green
    on a genuinely cold copy."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.store import MemoryStore
    from bench.recall_accuracy_real import assert_graphcache_generation_parity

    store = MemoryStore(path=tmp_path / "store")
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    with pytest.raises(RuntimeError, match="generation mismatch"):
        assert_graphcache_generation_parity(store)

    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graphcache_generation_parity_exempts_fuse_tripped(driver, tmp_path, monkeypatch) -> None:
    """The fuse_tripped exemption must survive the reorder: a
    generation-matched snapshot whose age/dirty-count freshness fuse has
    tripped still decodes the SAME copied file via the `normal` fallback
    tier, so it must NOT raise -- an operator copy whose daemon simply
    hasn't rebuilt in the fuse window is not a broken copy. Without this
    test, deleting the exemption would still pass every other test in this
    file (they only ever exercise `overlay` or `cold_degrade`)."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.store import MemoryStore
    from bench.recall_accuracy_real import assert_graphcache_generation_parity

    store = MemoryStore(path=tmp_path / "store")
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _dense_vec(_ANCHOR_ACTIVE)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={comm_id: [node_id]},
    )

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    ok = rgc.save_with_generation(store, assignment, [node_id])
    assert ok, "test setup: save_with_generation must succeed"

    # Trip the dirty-count fuse (fixed floor of 50 at this corpus scale) --
    # the generation stays matched, so this is the fuse path, not epoch skew.
    for _ in range(rgc._FUSE_DIRTY_THRESHOLD + 1):
        rgc.increment_dirty_counter()

    assert_graphcache_generation_parity(store)  # must not raise

    _assignment, _rich_club, _max_degree, structural_source, _node_degrees = (
        rgc.load_recall_structural(store)
    )
    assert structural_source == "normal", (
        f"expected the fuse_tripped bypass to fall to the 'normal' decode "
        f"tier, got {structural_source!r}"
    )

    store.close()


# ---------------------------------------------------------------------------
# _copy_real_store torn graph-cache detection
# ---------------------------------------------------------------------------


def _write_minimal_real_store(root, *, graph_cache_bytes: "bytes | None") -> None:
    """Write a minimal stand-in for a real operator store root: just enough
    for `_copy_real_store` to find `hippo/brain.sqlite3` and (optionally) a
    `runtime_graph_cache.json` sibling. `_copy_real_store` never opens
    either file itself (`shutil.copy2` only), so byte-plausible stand-ins
    are sufficient -- no real store construction needed."""
    hippo_dir = root / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    (hippo_dir / "brain.sqlite3").write_bytes(b"stand-in, never opened by _copy_real_store")
    if graph_cache_bytes is not None:
        (root / "runtime_graph_cache.json").write_bytes(graph_cache_bytes)


def test_copy_real_store_raises_on_empty_graph_cache(tmp_path, monkeypatch) -> None:
    """A 0-byte graph-cache copy (the observed race with the live daemon's
    atomic snapshot rewrite) must raise loud, not silently degrade to a
    structurally-cold eval copy."""
    from bench import recall_accuracy_real as bar

    fake_real_home = tmp_path / "operator_home"
    _write_minimal_real_store(fake_real_home / ".iai-mcp", graph_cache_bytes=b"")
    monkeypatch.setattr(bar, "_operator_home", lambda: fake_real_home)

    with pytest.raises(RuntimeError, match="graph-cache"):
        bar._copy_real_store(tmp_path / "copy")


def test_copy_real_store_raises_on_unencrypted_graph_cache(tmp_path, monkeypatch) -> None:
    """A non-empty but non-ciphertext copy (torn mid-write short of clean
    0 bytes) must also raise loud rather than pass a corrupt file through."""
    from bench import recall_accuracy_real as bar

    fake_real_home = tmp_path / "operator_home"
    _write_minimal_real_store(
        fake_real_home / ".iai-mcp", graph_cache_bytes=b'{"not": "encrypted"}'
    )
    monkeypatch.setattr(bar, "_operator_home", lambda: fake_real_home)

    with pytest.raises(RuntimeError, match="graph-cache"):
        bar._copy_real_store(tmp_path / "copy")


def test_copy_real_store_accepts_valid_encrypted_graph_cache(tmp_path, monkeypatch) -> None:
    """A genuinely-encrypted, non-empty graph-cache copy must pass through
    unmodified -- the verification is a torn/absent detector, not a decrypt
    gate (decrypt happens later, in the eval copy's own store context)."""
    from iai_mcp.crypto import CIPHERTEXT_PREFIX
    from bench import recall_accuracy_real as bar

    fake_real_home = tmp_path / "operator_home"
    payload = (CIPHERTEXT_PREFIX + "deadbeef").encode("utf-8")
    _write_minimal_real_store(fake_real_home / ".iai-mcp", graph_cache_bytes=payload)
    monkeypatch.setattr(bar, "_operator_home", lambda: fake_real_home)

    dest = bar._copy_real_store(tmp_path / "copy")
    assert (dest / "runtime_graph_cache.json").read_bytes() == payload


def test_copy_real_store_no_graph_cache_source_is_not_an_error(tmp_path, monkeypatch) -> None:
    """A store that has genuinely never built a graph-cache snapshot (no
    source file at all) is a legitimate cold-start state, not a race --
    `_copy_real_store` must not raise for it."""
    from bench import recall_accuracy_real as bar

    fake_real_home = tmp_path / "operator_home"
    _write_minimal_real_store(fake_real_home / ".iai-mcp", graph_cache_bytes=None)
    monkeypatch.setattr(bar, "_operator_home", lambda: fake_real_home)

    dest = bar._copy_real_store(tmp_path / "copy")
    assert not (dest / "runtime_graph_cache.json").exists()


# ---------------------------------------------------------------------------
# 2-hop spread discovery + cue-order independence
# ---------------------------------------------------------------------------


def _build_spread_fixture(store, *, with_edge: bool) -> dict:
    """Seed a corpus where ``target`` is cosine-excluded from the
    (test-shrunk) ANN candidate window and reachable only through a
    hebbian edge from ``anchor`` -- so serving it in the response proves
    the 2-hop discovery path, not plain cosine rank."""
    anchor = _make_record(
        "alice project anchor note about topic zero", _dense_vec(_ANCHOR_ACTIVE)
    )
    target = _make_record(
        "alice project target note about topic ninety", _dense_vec(_TARGET_ACTIVE)
    )
    distractors = [
        _make_record(
            f"alice distractor note number {i}",
            # Partial overlap with the anchor/cue window: cosine-ranked
            # between the anchor and the fully-disjoint target.
            _dense_vec(range(2 * i, 2 * i + 20)),
        )
        for i in range(1, 6)
    ]

    store.insert(anchor)
    store.insert(target)
    for d in distractors:
        store.insert(d)
    if with_edge:
        store.boost_edges([(anchor.id, target.id)], delta=0.9, edge_type="hebbian")
    _flush(store)

    return {"anchor": anchor, "target": target, "distractors": distractors}


def _dispatch_spread_cue(store, monkeypatch) -> list:
    """Dispatch the anchor cue through the exact production entry point,
    with the ANN candidate window shrunk (test-only monkeypatch, no
    production-code change) so a small seeded corpus can still exercise a
    genuine cosine-rank exclusion, and exact-authority off so it cannot
    smuggle the low-cosine target in independently of the graph edge."""
    from iai_mcp import core

    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    monkeypatch.setattr("iai_mcp.pipeline.K_CANDIDATES", 2)

    from bench.recall_accuracy_real import warm_eval_copy_store

    warm_eval_copy_store(store)

    resp = core.dispatch(
        store,
        "memory_recall",
        {
            "cue": "alice project anchor note about topic zero",
            "cue_embedding": _dense_vec(_ANCHOR_ACTIVE),
            "session_id": "gate-spread",
            "budget_tokens": 5000,
        },
    )
    return [str(h["record_id"]) for h in resp.get("hits", [])]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_two_hop_only_record_discovered(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    store_without = MemoryStore(path=tmp_path / "without")
    fixture_without = _build_spread_fixture(store_without, with_edge=False)
    hits_without = _dispatch_spread_cue(store_without, monkeypatch)
    store_without.close()

    assert str(fixture_without["anchor"].id) in hits_without, (
        "the anchor itself must be discovered by cosine — a sanity check "
        "that the shrunk candidate window still admits the seed"
    )
    assert str(fixture_without["target"].id) not in hits_without, (
        "the target must be cosine-excluded without the edge — otherwise "
        "the 2-hop check below is vacuous (the record would be discovered "
        "regardless of the graph mechanism)"
    )

    store_with = MemoryStore(path=tmp_path / "with")
    fixture_with = _build_spread_fixture(store_with, with_edge=True)
    hits_with = _dispatch_spread_cue(store_with, monkeypatch)
    store_with.close()

    assert str(fixture_with["target"].id) in hits_with, (
        "the target must be discovered once a hebbian edge connects it to "
        "the ANN-discovered anchor — proving the 2-hop spread path fires"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cue_order_independence(driver, tmp_path, monkeypatch) -> None:
    """Dispatch two cues, forward against one fresh copy and reversed
    against a separate fresh copy of the identical seed content, and assert
    identical per-cue hit sets. The corpus includes the spread-only fixture
    above so this check has a live graph-dependent mechanism to be
    sensitive to, not just plain cosine rank."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import core
    from iai_mcp.store import MemoryStore
    from bench.recall_accuracy_real import warm_eval_copy_store

    def _build(root) -> tuple:
        store = MemoryStore(path=root)
        fixture = _build_spread_fixture(store, with_edge=True)
        return store, fixture

    def _dispatch_all(store, order: list[dict]) -> dict[str, set]:
        monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
        monkeypatch.setattr("iai_mcp.pipeline.K_CANDIDATES", 2)
        warm_eval_copy_store(store)
        outcomes: dict[str, set] = {}
        for cue in order:
            resp = core.dispatch(
                store,
                "memory_recall",
                {
                    "cue": cue["text"],
                    "cue_embedding": cue["vec"],
                    "session_id": "gate-order",
                    "budget_tokens": 5000,
                },
            )
            outcomes[cue["name"]] = {str(h["record_id"]) for h in resp.get("hits", [])}
        return outcomes

    from iai_mcp import runtime_graph_cache as rgc

    # Forward direction.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    store_fwd, fixture_fwd = _build(tmp_path / "forward")
    cues = [
        {
            "name": "anchor_cue",
            "text": "alice project anchor note about topic zero",
            "vec": _dense_vec(_ANCHOR_ACTIVE),
        },
        {
            "name": "distractor_cue",
            "text": "alice distractor note number 3",
            "vec": _dense_vec(range(6, 26)),
        },
    ]
    outcomes_fwd = _dispatch_all(store_fwd, cues)
    store_fwd.close()

    # Reversed direction, on a SEPARATE fresh copy of identical content.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    store_rev, fixture_rev = _build(tmp_path / "reversed")
    outcomes_rev = _dispatch_all(store_rev, list(reversed(cues)))
    store_rev.close()

    def _relabel(outcome: dict[str, set], fixture: dict) -> dict[str, set]:
        """Map each direction's own record ids back to fixture roles
        (anchor/target/distractor-N) so the two independently-constructed
        copies (different uuid4() ids) compare by ROLE, not by raw id."""
        role_by_id = {str(fixture["anchor"].id): "anchor", str(fixture["target"].id): "target"}
        role_by_id.update(
            {str(d.id): f"distractor_{i}" for i, d in enumerate(fixture["distractors"], start=1)}
        )
        return {
            cue_name: {role_by_id.get(rid, rid) for rid in ids}
            for cue_name, ids in outcome.items()
        }

    relabelled_fwd = _relabel(outcomes_fwd, fixture_fwd)
    relabelled_rev = _relabel(outcomes_rev, fixture_rev)

    for cue in cues:
        name = cue["name"]
        assert relabelled_fwd[name] == relabelled_rev[name], (
            f"cue {name!r} served a different outcome depending on fixture "
            f"dispatch order: forward={relabelled_fwd[name]!r} "
            f"reversed={relabelled_rev[name]!r} — the eval harness's "
            "single-store-per-run design is order-sensitive and needs a "
            "fresh store per cue"
        )
    assert "target" in relabelled_fwd["anchor_cue"], (
        "the order-independence check must exercise a live spread-dependent "
        "record (the fixture's target), not just plain-cosine cues"
    )
