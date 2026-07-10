from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest


pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX paths; hermetic fixture uses HOME monkeypatching",
)


_EMBED_DIM = 384
_SEED_RECORDS = 300
_PROBE_TURNS = 30


def _zero_embedding() -> list[float]:
    return [0.0] * _EMBED_DIM


def _make_record(text: str, tags: list[str] | None = None):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_zero_embedding(),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags or ["role:user"],
        language="en",
    )


def _old_find_record_by_tag(self, tag: str) -> UUID | None:
    tag_json_literal = json.dumps(tag)
    for row in self.iter_record_columns(["id", "tags_json"]):
        tags_raw = row.get("tags_json") or "[]"
        if tag_json_literal not in tags_raw:
            continue
        try:
            tags = json.loads(tags_raw)
        except (ValueError, TypeError):
            continue
        if tag in tags:
            raw_id = row.get("id")
            if raw_id is None:
                continue
            try:
                return UUID(str(raw_id))
            except (ValueError, AttributeError):
                continue
    return None


@pytest.fixture
def rss_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-rss-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    import keyring.core
    keyring.core._keyring_backend = None

    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = _EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return _zero_embedding()

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [_zero_embedding() for _ in texts]

    monkeypatch.setattr(embed_mod, "embedder_for_store", lambda store: _FakeEmbedder())

    yield tmp_path

    keyring.core._keyring_backend = None


def _open_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_store(store, n: int) -> None:
    for i in range(n):
        r = _make_record(
            text=f"seeded record {i:05d} with enough text to represent a real turn",
            tags=["role:user", f"seed:{i}"],
        )
        store.insert(r)


def test_old_full_scan_streams_corpus_each_probe(
    rss_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Baseline contrast: the retired implementation pays one whole-corpus
    stream on EVERY probe. The streaming substrate has changed transports over
    time (DataFrame -> to_batches -> keyset pages), so the count instruments
    the stable seam — the per-probe corpus stream — not any one transport
    beneath it."""
    from iai_mcp.store import MemoryStore

    monkeypatch.setattr(MemoryStore, "find_record_by_tag", _old_find_record_by_tag)

    stream_call_count: list[int] = [0]
    original_iter = MemoryStore.iter_record_columns

    def counting_iter(self, *args, **kwargs):
        stream_call_count[0] += 1
        return original_iter(self, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, "iter_record_columns", counting_iter)

    store = _open_store(rss_env)
    try:
        _seed_store(store, _SEED_RECORDS)

        stream_call_count[0] = 0

        for i in range(_PROBE_TURNS):
            store.find_record_by_tag(f"nonexistent-idem-tag-{i}")
    finally:
        store.close()

    call_count = stream_call_count[0]
    print(f"\n[RED] Old full-scan: corpus streamed {call_count} times for {_PROBE_TURNS} probes")

    assert call_count >= _PROBE_TURNS, (
        f"Expected old full-scan to stream the corpus at least {_PROBE_TURNS}"
        f" times ({_PROBE_TURNS} probes); got {call_count}."
        " The old implementation must pay a full corpus stream per call."
    )


def test_new_sql_does_not_call_to_batches(rss_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp import hippo as hippo_mod

    to_batches_call_count: list[int] = [0]
    original_to_batches = hippo_mod.HippoQuery.to_batches

    def counting_to_batches(self, batch_size=1024):
        to_batches_call_count[0] += 1
        return original_to_batches(self, batch_size)

    monkeypatch.setattr(hippo_mod.HippoQuery, "to_batches", counting_to_batches)

    store = _open_store(rss_env)
    try:
        _seed_store(store, _SEED_RECORDS)
        to_batches_call_count[0] = 0

        for i in range(_PROBE_TURNS):
            store.find_record_by_tag(f"nonexistent-idem-tag-{i}")
    finally:
        store.close()

    call_count = to_batches_call_count[0]
    print(f"\n[GREEN] New SQL: to_batches called {call_count} times for {_PROBE_TURNS} probes")

    assert call_count == 0, (
        f"New SQL find_record_by_tag must NOT call to_batches; got {call_count} calls."
        " The fix is not effective — the Arrow materialization path is still active."
    )


def test_find_record_by_tag_correctness(rss_env: Path) -> None:
    from iai_mcp.capture import _idem_tag

    store = _open_store(rss_env)
    try:
        target_tag = "role:user"
        absent_tag = "definitely-not-present-zyxwv"
        idem = _idem_tag("test-sess", "user", "2026-06-05T12:00:00+00:00", "hello world")

        r1 = _make_record("record with target tag", tags=[target_tag, "extra:tag"])
        store.insert(r1)

        r2 = _make_record("record with idem tag", tags=[idem, "role:user"])
        store.insert(r2)

        r3 = _make_record("record with different tags only", tags=["role:assistant"])
        store.insert(r3)

        found = store.find_record_by_tag(target_tag)
        assert found == r1.id, (
            f"Expected r1.id={r1.id}, got {found}"
        )

        found_idem = store.find_record_by_tag(idem)
        assert found_idem == r2.id, (
            f"Expected r2.id={r2.id}, got {found_idem}"
        )

        not_found = store.find_record_by_tag(absent_tag)
        assert not_found is None, (
            f"Expected None for absent tag, got {not_found}"
        )

    finally:
        store.close()


def test_find_record_by_tag_empty_store(rss_env: Path) -> None:
    store = _open_store(rss_env)
    try:
        result = store.find_record_by_tag("idem:anything")
        assert result is None, f"Expected None on empty store, got {result}"
    finally:
        store.close()


def test_find_record_by_tag_no_tombstone_filter(rss_env: Path) -> None:
    store = _open_store(rss_env)
    try:
        tag = "idem:tombstone-parity-test"
        r = _make_record("tombstone parity record", tags=[tag])
        store.insert(r)

        found = store.find_record_by_tag(tag)
        assert found == r.id, f"Expected {r.id}, got {found}"

        found_again = store.find_record_by_tag(tag)
        assert found_again == r.id, "tag lookup must be stable across multiple calls"

    finally:
        store.close()
