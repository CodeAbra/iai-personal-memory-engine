from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make  # noqa: E402

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
    s = MemoryStore(path=tmp_path / "lancedb")
    s.root = tmp_path
    return s

def _read_decrypted_cache(store: MemoryStore, path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.startswith("iai:enc:v1:"):
        return json.loads(raw_text)
    from iai_mcp.crypto import decrypt_field
    plaintext = decrypt_field(
        raw_text,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    return json.loads(plaintext)

def _write_encrypted_cache(store: MemoryStore, path: Path, data: dict) -> None:
    from iai_mcp.crypto import encrypt_field
    plaintext = json.dumps(data)
    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    path.write_text(ciphertext, encoding="ascii")

def _make_assignment(n_communities: int = 2) -> CommunityAssignment:
    comms = [uuid4() for _ in range(n_communities)]
    nodes = [uuid4() for _ in range(n_communities * 3)]
    return CommunityAssignment(
        node_to_community={
            nodes[i]: comms[i // 3] for i in range(len(nodes))
        },
        community_centroids={c: [0.1, 0.2, 0.3] for c in comms},
        modularity=0.42,
        backend="leiden-networkx",
        top_communities=comms,
        mid_regions={c: nodes[i * 3:(i + 1) * 3] for i, c in enumerate(comms)},
    )

def test_save_creates_json_file(store):
    assignment = _make_assignment()
    rich_club = [uuid4() for _ in range(5)]
    ok = runtime_graph_cache.save(store, assignment, rich_club)
    assert ok is True
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("iai:enc:v1:"), (
        "cache must be v3 ciphertext on disk"
    )
    data = _read_decrypted_cache(store, path)
    assert data["cache_version"] == runtime_graph_cache.CACHE_VERSION
    assert "assignment" in data
    assert "rich_club" in data
    assert "key" in data

def test_rank_columns_survive_cache_round_trip(store):
    # Every rank column the payload carries must survive save->load with a
    # NON-EMPTY value: the serializer whitelists keys, and a key missing
    # there silently kills the matching rank term (aaak overlap, age
    # decay, stability) for every cache-served recall until a rebuild.
    assignment = _make_assignment()
    nid = uuid4()
    payload = {
        str(nid): {
            "embedding": [0.1] * 4,
            "surface": "notes on zephyrbot rollout",
            "centrality": 0.3,
            "tier": "episodic",
            "pinned": False,
            "tags": ["capture", "entity:zephyrbot"],
            "language": "en",
            "aaak_index": "W:E/R:abc12345/E:zephyrbot/T:capture",
            "created_at": "2026-07-01T00:00:00+00:00",
            "stability": 0.83,
        }
    }
    assert runtime_graph_cache.save(
        store, assignment, [nid], node_payload=payload
    )
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    node_payload = loaded[2]
    got = node_payload[str(nid)]
    assert got["aaak_index"] == "W:E/R:abc12345/E:zephyrbot/T:capture"
    assert got["created_at"] == "2026-07-01T00:00:00+00:00"
    assert got["stability"] == pytest.approx(0.83)


def test_try_load_round_trip_on_unchanged_store(store):
    assignment = _make_assignment()
    rich_club = [uuid4() for _ in range(3)]
    runtime_graph_cache.save(store, assignment, rich_club)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    loaded_assignment, loaded_rich_club, _node_payload, _max_degree, _node_degrees = loaded
    assert loaded_assignment.backend == assignment.backend
    assert loaded_assignment.modularity == pytest.approx(assignment.modularity)
    assert set(loaded_assignment.top_communities) == set(assignment.top_communities)
    assert set(loaded_rich_club) == set(rich_club)

def test_key_mismatch_invalidates_cache(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    data = _read_decrypted_cache(store, path)
    data["key"][0] = 999
    _write_encrypted_cache(store, path, data)

    assert runtime_graph_cache.try_load(store) is None

def test_cache_version_mismatch_triggers_rebuild(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    data = _read_decrypted_cache(store, path)
    data["cache_version"] = "old-format-v0"
    _write_encrypted_cache(store, path, data)

    assert runtime_graph_cache.try_load(store) is None

def test_corrupt_json_returns_none(store):
    path = runtime_graph_cache._cache_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json at all")
    assert runtime_graph_cache.try_load(store) is None

def test_aes_gcm_tag_failure_returns_none_not_raises(store):
    from cryptography.exceptions import InvalidTag

    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("iai:enc:v1:")

    from iai_mcp.crypto import decrypt_field

    prefix = "iai:enc:v1:"
    body = raw[len(prefix):]
    mid = len(body) // 2
    flipped = "A" if body[mid] != "A" else "B"
    tampered = prefix + body[:mid] + flipped + body[mid + 1:]
    path.write_text(tampered, encoding="ascii")

    with pytest.raises(InvalidTag):
        decrypt_field(tampered, store._key(), runtime_graph_cache._CACHE_AAD)

    assert runtime_graph_cache._load_and_decrypt_cache(store) is None
    assert runtime_graph_cache.try_load(store) is None

def test_absent_cache_returns_none(store):
    path = runtime_graph_cache._cache_path(store)
    assert not path.exists()
    assert runtime_graph_cache.try_load(store) is None

def test_build_runtime_graph_uses_cache_on_second_call(store):
    with mock.patch.object(
        runtime_graph_cache,
        "compute_assignment_in_child",
        wraps=runtime_graph_cache.compute_assignment_in_child,
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 1

    with mock.patch.object(
        runtime_graph_cache, "compute_assignment_in_child"
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 0

def test_build_runtime_graph_invalidates_on_record_added(store, tmp_path):
    from datetime import datetime, timezone
    from iai_mcp.types import MemoryRecord

    def _make_rec(seed: int) -> MemoryRecord:
        import numpy as np
        rng = np.random.default_rng(seed)
        vec = rng.random(store.embed_dim).astype(np.float32)
        vec = (vec / np.linalg.norm(vec)).tolist()
        now = datetime.now(timezone.utc)
        return MemoryRecord(
            id=uuid4(), tier="episodic", literal_surface=f"r{seed}",
            aaak_index="", embedding=vec, community_id=None, centrality=0.0,
            detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
            last_reviewed=None, never_decay=False, never_merge=False,
            provenance=[], created_at=now, updated_at=now, tags=[], language="en",
        )

    window = runtime_graph_cache._STALENESS_WINDOW

    base = window + 2
    for i in range(base):
        store.insert(_make_rec(i))

    retrieve.build_runtime_graph(store)
    assert runtime_graph_cache._cache_path(store).exists()

    store.insert(_make_rec(seed=base + 100))
    with mock.patch.object(
        runtime_graph_cache, "compute_assignment_in_child"
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        assert detect_spy.call_count == 0, (
            "community detection fired on a single-record insert within the "
            "staleness window — the windowed key should absorb single writes."
        )

    for i in range(window):
        store.insert(_make_rec(seed=base + 200 + i))
    with mock.patch.object(
        runtime_graph_cache,
        "compute_assignment_in_child",
        wraps=runtime_graph_cache.compute_assignment_in_child,
    ) as detect_spy:
        retrieve.build_runtime_graph(store)
        # Stale-while-revalidate: the window-crossing writes invalidate the
        # warm bundle, the awake call serves the stale bundle immediately,
        # and the rebuild (with detection) runs on the single-flight
        # refresher thread — join it before asserting.
        refresher = getattr(store, "_graph_refresh_thread", None)
        if refresher is not None:
            refresher.join(timeout=120)
        assert detect_spy.call_count == 1, (
            "community detection should have fired after a window-crossing "
            "insert."
        )

def test_save_is_atomic_leaves_old_file_on_error(store, monkeypatch):
    original_assignment = _make_assignment()
    runtime_graph_cache.save(store, original_assignment, [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    original_text = path.read_text()

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace",
        mock.Mock(side_effect=OSError("rename failed")),
    )
    ok = runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    assert ok is False
    assert path.read_text() == original_text
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_path.exists()

def test_invalidate_removes_cache_file(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    runtime_graph_cache.invalidate(store)
    assert not path.exists()

    runtime_graph_cache.invalidate(store)

def test_embed_dim_change_invalidates(store):
    runtime_graph_cache.save(store, _make_assignment(), [uuid4()])
    assert runtime_graph_cache.try_load(store) is not None

    store._embed_dim = 1024
    assert runtime_graph_cache.try_load(store) is None

def test_save_drops_oversize_community_centroids(store):
    big_centroids = {uuid4(): [0.123456] * 1024 for _ in range(2000)}
    big_node_to_community = {uuid4(): uuid4() for _ in range(50)}
    assignment = CommunityAssignment(
        node_to_community=big_node_to_community,
        community_centroids=big_centroids,
        modularity=0.37,
        backend="leiden-networkx",
        top_communities=list(big_centroids.keys())[:5],
        mid_regions={c: [] for c in list(big_centroids.keys())[:5]},
    )
    rich_club = [uuid4() for _ in range(10)]

    node_payload = {
        str(uuid4()): {
            "embedding": [0.0] * 384,
            "surface": "probe",
            "centrality": 0.1,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": "en",
        }
    }

    ok = runtime_graph_cache.save(store, assignment, rich_club, node_payload=node_payload)
    assert ok is True

    path = runtime_graph_cache._cache_path(store)
    assert path.exists()

    size = path.stat().st_size
    assert size <= runtime_graph_cache.MAX_CACHE_BYTES, (
        f"cache file {size} bytes exceeds cap "
        f"{runtime_graph_cache.MAX_CACHE_BYTES}"
    )

    data = _read_decrypted_cache(store, path)

    assert data["node_payload"] == {}
    assert data["assignment"]["community_centroids"] == {}

    assert data["assignment"]["modularity"] == pytest.approx(0.37)
    assert data["assignment"]["backend"] == "leiden-networkx"
    assert len(data["assignment"]["node_to_community"]) == len(big_node_to_community)
    assert len(data["assignment"]["top_communities"]) == 5
    assert len(data["rich_club"]) == len(rich_club)

def test_save_small_payload_survives_unchanged(store):
    assignment = _make_assignment(n_communities=2)
    rich_club = [uuid4() for _ in range(3)]
    node_payload = {
        str(uuid4()): {
            "embedding": [0.1] * 384,
            "surface": "hello",
            "centrality": 0.2,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t"],
            "language": "en",
        }
        for _ in range(5)
    }

    ok = runtime_graph_cache.save(store, assignment, rich_club, node_payload=node_payload)
    assert ok is True

    path = runtime_graph_cache._cache_path(store)
    data = _read_decrypted_cache(store, path)

    assert path.stat().st_size < runtime_graph_cache.MAX_CACHE_BYTES

    assert data["node_payload"] != {}
    assert len(data["node_payload"]) == 5
    assert data["assignment"]["community_centroids"] != {}
    assert len(data["assignment"]["community_centroids"]) == 2
    assert data["assignment"]["mid_regions"] != {}
    assert len(data["assignment"]["mid_regions"]) == 2

def test_save_writes_ciphertext_no_plaintext_surface(store):
    canary = "PLAINTEXT_CANARY_4d7f_07_9_W3"
    rid = uuid4()
    node_payload = {
        str(rid): {
            "embedding": [0.1] * 384,
            "surface": canary,
            "centrality": 0.5,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t"],
            "language": "en",
        },
    }
    ok = runtime_graph_cache.save(store, _make_assignment(), [uuid4()],
                                  node_payload=node_payload, max_degree=3)
    assert ok is True
    path = runtime_graph_cache._cache_path(store)
    raw_bytes = path.read_bytes()

    assert canary.encode("utf-8") not in raw_bytes, (
        "plaintext surface canary leaked into the on-disk sidecar"
    )
    assert raw_bytes.startswith(b"iai:enc:v1:"), (
        f"expected v3 ciphertext envelope; got prefix {raw_bytes[:32]!r}"
    )

def test_save_then_try_load_preserves_surface_byte_for_byte(store):
    rid = uuid4()
    # non-English fixture data: multilingual behavior under test — keep
    surface = "user сказал важное — please remember this 重要"
    node_payload = {
        str(rid): {
            "embedding": [0.42] * 384,
            "surface": surface,
            "centrality": 0.42,
            "tier": "episodic",
            "pinned": True,
            "tags": ["t1", "t2"],
            "language": "ru",
        },
    }
    runtime_graph_cache.save(store, _make_assignment(), [rid],
                             node_payload=node_payload, max_degree=7)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _, _, payload, max_deg, _nd = loaded
    assert payload is not None
    assert payload[str(rid)]["surface"] == surface
    assert payload[str(rid)]["centrality"] == pytest.approx(0.42)
    assert payload[str(rid)]["tags"] == ["t1", "t2"]
    assert payload[str(rid)]["language"] == "ru"
    assert max_deg == 7

def test_v2_plaintext_lazy_migrates_to_v3(store):
    path = runtime_graph_cache._cache_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)

    rid = uuid4()
    legacy_data = {
        "cache_version": runtime_graph_cache.LEGACY_CACHE_VERSION_PLAINTEXT,
        "key": list(runtime_graph_cache._cache_key(store)),
        "assignment": runtime_graph_cache._encode_assignment(_make_assignment()),
        "rich_club": [str(uuid4())],
        "node_payload": {
            str(rid): {
                "embedding": [0.0] * 384,
                "surface": "legacy_plain_canary",
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            }
        },
        "max_degree": 1,
        "saved_at": "2026-04-29T00:00:00Z",
    }
    # The cache version is the LAST key component whatever the key arity —
    # mirror the loader's expected-legacy-key derivation exactly.
    legacy_data["key"][-1] = runtime_graph_cache.LEGACY_CACHE_VERSION_PLAINTEXT
    path.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _, _, payload, _, _nd = loaded
    assert payload is not None
    assert payload[str(rid)]["surface"] == "legacy_plain_canary"

    raw_after = path.read_bytes()
    assert raw_after.startswith(b"iai:enc:v1:"), (
        "v2 plaintext file must be lazily migrated to v3 ciphertext"
    )
    assert b"legacy_plain_canary" not in raw_after, (
        "post-migration on-disk bytes must not contain the legacy plaintext"
    )


# --- coldness-aware rebuild gate -----------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_home(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    yield


@pytest.fixture
def _reset_rgc_state():
    from iai_mcp import runtime_graph_cache as rgc

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    rgc._persistent_graph = None
    yield


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _thick_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore as _MS, flush_record_buffer

    s = _MS(str(tmp_path / "thick_store"))
    for i in range(15):
        s.insert(_make(text=f"User record {i}", vec=_norm_vec(i + 100)))
    flush_record_buffer(s)
    return s


def test_rebuild_warms_cold_cache_even_when_clean(tmp_path, _reset_rgc_state):
    rgc = runtime_graph_cache
    store = _thick_store(tmp_path)

    rgc._rebuild_and_save_rgc(store, force=True)
    rgc.invalidate(store)

    assert rgc.load_recall_structural(store)[3] in ("cold_degrade", "last_good")
    assert rgc.get_dirty_counter() == 0

    result = rgc._rebuild_and_save_rgc(store)
    assert result["rebuilt"] is True
    assert rgc.load_recall_structural(store)[3] in ("overlay", "normal")


def test_rebuild_skips_warm_and_below_dirty_threshold(tmp_path, _reset_rgc_state):
    rgc = runtime_graph_cache
    store = _thick_store(tmp_path)

    rgc._rebuild_and_save_rgc(store)
    assert rgc.load_recall_structural(store)[3] in ("overlay", "normal")

    gen_before = rgc.get_current_generation()
    dirty_before = rgc.get_dirty_counter()

    result = rgc._rebuild_and_save_rgc(store)
    assert result["rebuilt"] is False
    assert result.get("skipped")
    assert rgc.get_current_generation() == gen_before
    assert rgc.get_dirty_counter() == dirty_before


def test_rebuild_fires_above_dirty_threshold(tmp_path, _reset_rgc_state):
    rgc = runtime_graph_cache
    store = _thick_store(tmp_path)

    rgc._rebuild_and_save_rgc(store)
    assert rgc.load_recall_structural(store)[3] in ("overlay", "normal")

    for _ in range(rgc._FUSE_DIRTY_THRESHOLD + 1):
        rgc.increment_dirty_counter()
    assert rgc.get_dirty_counter() > rgc._FUSE_DIRTY_THRESHOLD

    result = rgc._rebuild_and_save_rgc(store)
    assert result["rebuilt"] is True
    assert rgc.get_dirty_counter() == 0


def test_rebuild_force_overrides_warm_clean_gate(tmp_path, _reset_rgc_state):
    rgc = runtime_graph_cache
    store = _thick_store(tmp_path)

    rgc._rebuild_and_save_rgc(store)
    assert rgc.load_recall_structural(store)[3] in ("overlay", "normal")
    assert rgc.get_dirty_counter() <= rgc._FUSE_DIRTY_THRESHOLD

    result = rgc._rebuild_and_save_rgc(store, force=True)
    assert result["rebuilt"] is True


def _structural_snapshot(store) -> tuple:
    rgc = runtime_graph_cache
    assignment, rich_club, _max_degree, src, _node_degrees = rgc.load_recall_structural(store)
    ntc = getattr(assignment, "node_to_community", {})
    # Community UUIDs are minted per detection; compare the partition (grouping)
    # plus the rich-club node set — the observable recall surface.
    groups: dict = {}
    for nid, cid in ntc.items():
        groups.setdefault(cid, set()).add(str(nid))
    partition = frozenset(frozenset(m) for m in groups.values())
    return partition, frozenset(str(x) for x in rich_club), src


def test_reused_instance_recall_parity(tmp_path, _reset_rgc_state):
    rgc = runtime_graph_cache
    store = _thick_store(tmp_path)

    # Reused-instance path: build through the persistent holder, snapshot recall.
    rgc._rebuild_and_save_rgc(store, force=True)
    reused_snapshot = _structural_snapshot(store)

    # Fresh path: drop the persistent instance so the next rebuild allocates a
    # new MemoryGraph, then snapshot recall over the same store + generation line.
    rgc._persistent_graph = None
    rgc._rebuild_and_save_rgc(store, force=True)
    fresh_snapshot = _structural_snapshot(store)

    assert reused_snapshot[0] == fresh_snapshot[0]  # partition identical
    assert reused_snapshot[1] == fresh_snapshot[1]  # rich-club identical
    assert reused_snapshot[2] in ("overlay", "normal")


def test_hippea_cascade_not_on_fragmentation_path():
    import iai_mcp.daemon._watchdog as _watchdog

    source = inspect.getsource(_watchdog)
    assert "_rebuild_and_save_rgc" not in source, (
        "the hippea cascade must not route through the fragmenting refresh builder"
    )
    assert "build_runtime_graph" in source, (
        "the hippea cascade must use the interval-gated recall builder"
    )
