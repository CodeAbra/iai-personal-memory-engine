from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import runtime_graph_cache
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore


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


class _CrashAfterTmpWrite(OSError):
    pass


def test_crash_after_tmp_write_leaves_previous_snapshot_intact(store, monkeypatch):
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_sidecar = cache_path.with_suffix(cache_path.suffix + ".tmp")

    assignment_v1 = _make_assignment(n_communities=2)
    rich_club_v1 = [uuid4() for _ in range(3)]
    assert runtime_graph_cache.save(store, assignment_v1, rich_club_v1) is True

    v1_bytes = cache_path.read_bytes()
    v1_loaded = runtime_graph_cache.try_load(store)
    assert v1_loaded is not None
    v1_assignment, v1_rich, _v1_payload, _v1_maxdeg, _v1_node_degrees = v1_loaded
    assert v1_assignment.modularity == pytest.approx(0.42)
    assert set(v1_rich) == set(rich_club_v1)

    real_replace = os.replace

    def _crash_only_cache(src, dst, *a, **k):
        if str(dst) == str(cache_path):
            raise _CrashAfterTmpWrite("simulated crash")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace", _crash_only_cache
    )

    assignment_v2 = _make_assignment(n_communities=5)
    rich_club_v2 = [uuid4() for _ in range(7)]
    assert runtime_graph_cache.save(store, assignment_v2, rich_club_v2) is False

    assert cache_path.read_bytes() == v1_bytes, (
        "v1 cache must remain intact after a v2 save crashes mid-rename"
    )
    after_crash = runtime_graph_cache.try_load(store)
    assert after_crash is not None
    after_assignment, after_rich, _ap, _am, _after_node_degrees = after_crash
    assert after_assignment.modularity == pytest.approx(0.42)
    assert set(after_rich) == set(rich_club_v1)
    assert not tmp_sidecar.exists(), (
        ".tmp sidecar must be unlinked by the OSError-handler cleanup path"
    )


def test_save_calls_fsync_before_replace(store, monkeypatch):
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    real_fsync = os.fsync
    real_replace = os.replace

    events: list[str] = []
    cache_fsync_count = 0
    cache_replace_count = 0

    def _scoped_fsync(fd: int) -> None:
        nonlocal cache_fsync_count
        try:
            same = os.fstat(fd).st_ino == os.stat(tmp_path).st_ino
        except OSError:
            same = False
        if same:
            cache_fsync_count += 1
            events.append("fsync")
        return real_fsync(fd)

    def _scoped_replace(src, dst, *a, **k):
        nonlocal cache_replace_count
        if str(dst) == str(cache_path):
            cache_replace_count += 1
            events.append("replace")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.fsync", _scoped_fsync
    )
    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.replace", _scoped_replace
    )

    assignment = _make_assignment(n_communities=3)
    rich_club = [uuid4() for _ in range(4)]
    assert runtime_graph_cache.save(store, assignment, rich_club) is True

    assert cache_fsync_count >= 1, (
        "os.fsync must be invoked on the cache tmp file before os.replace"
    )
    assert cache_replace_count == 1, (
        f"save() must os.replace the cache file exactly once; "
        f"got {cache_replace_count}"
    )

    assert events.index("fsync") < events.index("replace"), (
        f"fsync must precede replace for the cache file; order was {events!r}"
    )

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    loaded_assignment, loaded_rich, _np, _md, _loaded_node_degrees = loaded
    assert loaded_assignment.modularity == pytest.approx(0.42)
    assert set(loaded_rich) == set(rich_club)


def test_save_handles_fsync_failure_gracefully(store, monkeypatch):
    cache_path = runtime_graph_cache._cache_path(store)
    tmp_sidecar = cache_path.with_suffix(cache_path.suffix + ".tmp")

    assert runtime_graph_cache.save(
        store, _make_assignment(n_communities=2), [uuid4()]
    ) is True
    v1_bytes = cache_path.read_bytes()

    real_fsync = os.fsync

    def _raise_disk_full(fd: int) -> None:
        try:
            same = os.fstat(fd).st_ino == os.stat(tmp_sidecar).st_ino
        except OSError:
            same = False
        if same:
            raise OSError("disk full")
        return real_fsync(fd)

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.os.fsync", _raise_disk_full
    )

    assert runtime_graph_cache.save(
        store, _make_assignment(n_communities=5), [uuid4() for _ in range(3)]
    ) is False

    assert cache_path.read_bytes() == v1_bytes, (
        "v1 cache must remain intact when fsync fails on the v2 save"
    )
    assert not tmp_sidecar.exists(), (
        ".tmp sidecar must be unlinked when fsync raises mid-save"
    )
