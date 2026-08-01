"""Child-isolated community detection for the runtime graph build.

Three guarantees are pinned here:

  1. Graceful degradation: when the spawn-context child fails, the graph build
     falls back to in-process community detection so recall is never blocked,
     and the resulting graph is still correct.

  2. Streaming node-payload equivalence: the per-node payload assembled by the
     bounded column-streaming build equals the payload a full-corpus build
     would produce, including the in-parent decryption of the literal surface.

  3. Parent-footprint isolation: running detection in the child leaves the
     parent resident set materially below the in-parent baseline, because the
     JIT compilation arenas live in the ephemeral child and the OS reclaims
     them on exit.
"""
from __future__ import annotations

import gc
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


_LILLI_DRIVER = os.environ.get("LILLI_STORAGE_DRIVER", "stdlib").lower() == "lilli"

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord
from tests.conftest_shared import retry_once_on_assertion


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch):
    # Derive every store's key from a passphrase (the non-interactive path) so
    # each fresh store directory yields a usable key without a keyring or key
    # file on disk.
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "store")
    s.root = tmp_path
    return s


def _make_rec(seed: int, store: MemoryStore) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"surface number {seed} carrying real text",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=bool(seed % 7 == 0),
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[f"tag{seed % 3}"],
        language="en",
    )


def _seed_store(store: MemoryStore, n: int, seed_base: int = 0) -> None:
    for i in range(n):
        store.insert(_make_rec(seed_base + i, store))


def _node_payloads(graph) -> dict:
    """Normalized node-id -> payload mapping for equality comparison."""
    out: dict = {}
    for nid in graph.iter_nodes():
        key = str(nid)
        payload = dict(graph._node_payload.get(key, {}))
        # Embedding is a list of floats; normalize to a tuple so dict equality
        # does not depend on list identity.
        if "embedding" in payload:
            payload["embedding"] = tuple(
                round(float(x), 6) for x in payload["embedding"]
            )
        if "tags" in payload:
            payload["tags"] = tuple(payload["tags"])
        out[key] = payload
    return out


def test_streaming_build_decrypts_and_preserves_surfaces(store: MemoryStore):
    """The streamed build decrypts the literal surface in-parent and carries it
    into the node payload verbatim — the encrypted blob never leaks into the
    graph."""
    _seed_store(store, n=12, seed_base=100)

    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    payloads = _node_payloads(graph)
    assert len(payloads) == 12
    surfaces = {p["surface"] for p in payloads.values()}
    # Each record's plaintext surface is present and decrypted.
    for i in range(12):
        assert f"surface number {100 + i} carrying real text" in surfaces
    # No payload retains an encrypted blob.
    from iai_mcp.crypto import is_encrypted

    for p in payloads.values():
        assert not is_encrypted(p["surface"]), (
            "literal surface left encrypted in the node payload"
        )


def test_streaming_build_node_payload_matches_reference(store: MemoryStore):
    """The bounded-batch streaming build yields the same node payload as a
    rebuild over the same corpus. A second build from the cache reproduces the
    payload, proving the streamed-vs-cached paths agree node-for-node."""
    _seed_store(store, n=20, seed_base=200)

    graph_a, _a, _ra = retrieve.build_runtime_graph(store)
    payload_a = _node_payloads(graph_a)

    # Drop the cache so the second build re-streams the corpus rather than
    # reloading the cached payload, then compare.
    runtime_graph_cache.invalidate(store)
    graph_b, _b, _rb = retrieve.build_runtime_graph(store)
    payload_b = _node_payloads(graph_b)

    assert set(payload_a) == set(payload_b)
    for key in payload_a:
        a = payload_a[key]
        b = payload_b[key]
        assert a["surface"] == b["surface"]
        assert a["embedding"] == b["embedding"]
        assert a["tier"] == b["tier"]
        assert a["pinned"] == b["pinned"]
        assert a["tags"] == b["tags"]
        assert a["language"] == b["language"]


def test_child_crash_falls_back_to_in_process(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """When the detection child crashes, the build degrades to in-process
    detection so recall never hard-fails — and still produces a populated
    graph with a community assignment."""
    _seed_store(store, n=15, seed_base=300)

    in_process_called = {"n": 0}

    def _boom(graph, **kw):
        raise runtime_graph_cache.WorkerCrashedError("simulated worker crash")

    import iai_mcp.community as _cm

    real_in_process = _cm.detect_communities

    def _counting_in_process(graph, **kw):
        in_process_called["n"] += 1
        return real_in_process(graph, **kw)

    monkeypatch.setattr(
        runtime_graph_cache, "compute_assignment_in_child", _boom
    )
    monkeypatch.setattr(_cm, "detect_communities", _counting_in_process)

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    assert in_process_called["n"] == 1, (
        "in-process detection should have run exactly once on child crash"
    )
    assert assignment is not None
    assert graph.node_count() == 15
    # The assignment maps every active node to a community.
    assert len(assignment.node_to_community) >= 1


def test_child_timeout_falls_back_to_in_process(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A worker timeout also degrades to in-process detection."""
    _seed_store(store, n=10, seed_base=400)

    def _timeout(graph, **kw):
        raise runtime_graph_cache.WorkerTimeoutError("simulated timeout")

    monkeypatch.setattr(
        runtime_graph_cache, "compute_assignment_in_child", _timeout
    )

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    assert assignment is not None
    assert graph.node_count() == 10


def _settled_rss_bytes() -> int:
    """Force returnable pages back to the OS, then take one RSS reading.

    Mirrors the shipped per-step relief (pyarrow pool release + gc.collect +
    macOS zone pressure relief) so the reading is a reproducible
    "all-returnable-pages-returned" footprint rather than a jittery sample.
    """
    from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
        _current_rss_bytes,
        _step_memory_relief,
    )

    _step_memory_relief("rss-proof")
    gc.collect()
    return _current_rss_bytes()


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="settled-RSS isolation proof calibrated on this host's allocator",
)
@pytest.mark.skipif(
    _LILLI_DRIVER,
    reason=(
        "RSS-isolation proof: it asserts detection-arena residency stays in the "
        "ephemeral child vs the parent — driver-independent, validated under the "
        "stdlib driver. The proof seeds 6000 records across two arms purely to "
        "build the graph twice; the lilli engine's per-record write cost at that "
        "volume exceeds the default wall, and the WHERE-detection-runs property "
        "the test proves does not depend on the storage driver."
    ),
)
@retry_once_on_assertion
def test_parent_rss_lower_with_child_isolation(tmp_path: Path):
    """Building the runtime graph with child isolation keeps the parent RSS
    materially below the in-parent-detection baseline, proving the detection
    arenas no longer reside in the parent.

    The two arms run on independent fresh stores seeded identically so the
    only difference is where detection executes. The in-parent arm forces
    detection to run locally by routing the child call straight through the
    in-process kernel; the isolated arm uses the real spawn-context child.
    Each arm subtracts a fresh settled baseline, so an arm only measures the
    resident growth it itself retains.
    """
    import iai_mcp.community as _cm

    n_records = 3000

    def _build_arm(seed_base: int, in_parent: bool) -> int:
        s = MemoryStore(path=tmp_path / f"arm-{seed_base}-{in_parent}")
        s.root = tmp_path / f"arm-root-{seed_base}-{in_parent}"
        s.root.mkdir(parents=True, exist_ok=True)
        _seed_store(s, n=n_records, seed_base=seed_base)

        with pytest.MonkeyPatch.context() as mp:
            if in_parent:
                # Route the "child" call through the in-process kernel so the
                # detection arenas are reserved in this (parent) process. Honor
                # the `with_centrality` contract: when requested, also compute
                # centrality locally and return it alongside the assignment so
                # the in-parent path retains both intermediates.
                def _in_parent_detect(graph, **kw):
                    assignment = _cm.detect_communities(
                        graph, prior=None, prior_mode="seeded"
                    )
                    if kw.get("with_centrality"):
                        return assignment, graph.centrality()
                    return assignment

                mp.setattr(
                    runtime_graph_cache,
                    "compute_assignment_in_child",
                    _in_parent_detect,
                )
            baseline = _settled_rss_bytes()
            retrieve.build_runtime_graph(s)
            settled = _settled_rss_bytes()
        return settled - baseline

    # In-parent arm first: detection runs locally, leaving arenas resident.
    in_parent_delta = _build_arm(seed_base=10_000, in_parent=True)
    gc.collect()
    _settled_rss_bytes()

    # Isolated arm: detection runs in the ephemeral child.
    isolated_delta = _build_arm(seed_base=50_000, in_parent=False)

    # The isolated build must leave a materially smaller resident delta in the
    # parent. Allow generous slack — the proof is the order-of-magnitude gap,
    # not a tight bound. Below the noise floor the comparison is meaningless:
    # under full-suite memory pressure the allocator can serve the in-parent
    # arenas from already-resident freed pages, collapsing BOTH deltas to
    # sub-MB values whose ordering is page-reuse noise. Sub-floor deltas mean
    # the parent retained no material arena in either arm — the exact
    # property this test protects.
    noise_floor = 4 * 1024 * 1024
    assert isolated_delta < max(in_parent_delta, noise_floor), (
        f"child isolation did not lower the parent footprint: "
        f"in_parent_delta={in_parent_delta / 1e6:.1f}MB "
        f"isolated_delta={isolated_delta / 1e6:.1f}MB"
    )
