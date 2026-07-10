"""Two-phase deferred-capture drain: the resident-memory bound proof.

The in-daemon backlog drain is decoupled from embedding. Phase one writes every
genuinely-new turn as a pending (un-embedded) row — a sequence of cheap SQLite
writes whose resident-memory cost does not grow with the backlog size, so a large
backlog can no longer climb the resident set through one long synchronous embed
run. Phase two fills the real vectors in a bounded, resident-set-gated pass.

These tests pin that contract:
  (a) the drain writes N pending rows and never calls the embedder;
  (b) each pending row is recall-findable by its verbatim text immediately,
      before any embedding lands (the recency pending-union path);
  (c) the duplicate dedup/reinforce behaviour survives the decoupling — a
      re-seen turn reinforces the pre-existing record and does not double-insert;
  (d) the deferred-embed pass embeds the pending rows in bounded windows and
      yields early when the resident set crosses an injected soft cap, leaving
      the remaining rows pending for the next cycle.

Revert-proof: a synchronous drain that embeds during the run would trip the
embedder spy in (a).
"""
from __future__ import annotations

import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make  # noqa: E402

from iai_mcp.types import EMBED_DIM  # noqa: E402

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename; deferred-drain is POSIX-only here",
)

SESSION_ID = "two-phase-session"


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-two-phase-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no-such.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def _make_event(text: str, *, role: str = "user", ts: str, source_uuid: str) -> dict:
    return {
        "text": text,
        "cue": f"session {SESSION_ID} turn",
        "tier": "episodic",
        "role": role,
        "ts": ts,
        "source_uuid": source_uuid,
    }


def _write_backlog(home: Path, events: list[dict]) -> Path:
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{SESSION_ID}-backlog.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": SESSION_ID,
        "cwd": "/tmp/test",
    }
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return out_path


def _count_pending(store) -> int:
    with store.db._conn_lock:
        return store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1 AND tombstoned_at IS NULL"
        ).fetchone()[0]


# ---------------------------------------------------------------------------
# (a) the drain writes N pending rows and never embeds
# ---------------------------------------------------------------------------


def test_drain_writes_pending_rows_without_embedding(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.embed import Embedder

    store = _open_store()

    n = 12
    events = [
        _make_event(
            f"Genuinely new backlog turn number {i} with enough text to store",
            ts=f"2026-07-04T00:{i:02d}:00Z",
            source_uuid=str(uuid.uuid4()),
        )
        for i in range(n)
    ]
    _write_backlog(iai_home, events)

    seen: list[str] = []
    real_embed = Embedder.embed

    def spy_embed(self, text):
        seen.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed", spy_embed)

    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == n, counts
    assert seen == [], (
        f"the drain must not embed any turn — all embedding is deferred; saw "
        f"{len(seen)} embed calls. counts={counts!r}"
    )
    assert _count_pending(store) == n, (
        f"all {n} genuinely-new turns must land as pending rows; "
        f"pending={_count_pending(store)}"
    )

    # Every pending row carries the zero-vector placeholder, never a fabricated
    # real embedding.
    with store.db._conn_lock:
        blobs = store.db._conn.execute(
            "SELECT embedding FROM records WHERE COALESCE(embedding_pending,0)=1"
        ).fetchall()
    for (blob,) in blobs:
        vec = np.frombuffer(blob, dtype=np.float32)
        assert len(vec) == EMBED_DIM
        assert np.all(vec == 0.0), "pending row must hold a zero-vector placeholder"


# ---------------------------------------------------------------------------
# (b) pending rows are recall-findable by verbatim text immediately
# ---------------------------------------------------------------------------


def test_pending_rows_recall_findable_before_embedding(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.embed import Embedder

    store = _open_store()

    # A handful of normal embedded filler rows so the corpus is non-trivial.
    rng = np.random.default_rng(7)
    for i in range(8):
        v = rng.random(EMBED_DIM).astype(np.float32)
        store.insert(_make(text=f"embedded filler row {i}", vec=(v / np.linalg.norm(v)).tolist()))

    probe_text = "User pending probe turn that must recall verbatim before embed"
    probe_uuid = str(uuid.uuid4())
    probe_ts = "2026-07-04T01:00:00Z"
    _write_backlog(iai_home, [_make_event(probe_text, ts=probe_ts, source_uuid=probe_uuid)])

    # The embedder must not be touched by the drain — recall of the pending row
    # is embedding-independent.
    seen: list[str] = []
    real_embed = Embedder.embed
    monkeypatch.setattr(
        Embedder, "embed",
        lambda self, text: (seen.append(text), real_embed(self, text))[1],
    )

    counts = drain_deferred_captures(store)
    assert counts["events_inserted"] == 1, counts
    assert seen == [], f"drain must not embed; saw {seen!r}"
    assert _count_pending(store) == 1, _count_pending(store)

    # The pending row is recall-findable verbatim via the recency pending-union
    # path the recall pipeline uses — embedding-independent.
    markers = store.recent_pending_markers(n=50)
    surfaces = [m.literal_surface for m in markers]
    assert any(probe_text in (s or "") for s in surfaces), (
        f"pending turn not surfaced by recent_pending_markers; surfaces={surfaces!r}"
    )

    # And it is findable by its idem tag the instant it is written.
    from iai_mcp.capture import _idem_tag, _resolve_ts
    tag = _idem_tag(
        SESSION_ID, "user", _resolve_ts(probe_ts).isoformat(), probe_text,
        source_uuid=probe_uuid,
    )
    assert store.find_record_by_tag(tag) is not None, (
        "pending turn must be dedup-findable by idem tag immediately"
    )


# ---------------------------------------------------------------------------
# (c) duplicate dedup/reinforce survives the decoupling
# ---------------------------------------------------------------------------


def test_drain_dedup_reinforces_duplicate_without_double_insert(iai_home):
    from iai_mcp.capture import capture_turn, drain_deferred_captures

    store = _open_store()

    dup_text = "A turn already stored that the backlog repeats as a duplicate"
    dup_uuid = str(uuid.uuid4())
    dup_ts = "2026-07-04T02:00:00Z"

    seed = capture_turn(
        store,
        cue="seed",
        text=dup_text,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=dup_ts,
        source_uuid=dup_uuid,
    )
    assert seed["status"] == "inserted", seed

    with store.db._conn_lock:
        active_before = store.db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()[0]

    new_text = "A genuinely-new turn alongside the duplicate in the backlog file"
    new_uuid = str(uuid.uuid4())
    new_ts = "2026-07-04T02:01:00Z"

    _write_backlog(
        iai_home,
        [
            _make_event(dup_text, ts=dup_ts, source_uuid=dup_uuid),
            _make_event(new_text, ts=new_ts, source_uuid=new_uuid),
        ],
    )

    counts = drain_deferred_captures(store)

    assert counts["events_reinforced"] >= 1, counts
    assert counts["events_inserted"] == 1, counts

    with store.db._conn_lock:
        active_after = store.db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()[0]
    assert active_after == active_before + 1, (
        f"the duplicate must not double-insert; before={active_before} "
        f"after={active_after}"
    )
    # The duplicate's pre-existing record stays embedded (not flipped to pending),
    # and exactly the one new turn is pending.
    assert _count_pending(store) == 1, _count_pending(store)


# ---------------------------------------------------------------------------
# (d) deferred-embed pass is bounded: windowed + RSS soft cap stops early
# ---------------------------------------------------------------------------


class _SpyEmbedder:
    """Counts embed calls and returns a deterministic unit vector per text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        h = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(h)
        v = rng.random(EMBED_DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()


def test_deferred_embed_pass_is_windowed(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    store = _open_store()

    n = 9
    events = [
        _make_event(
            f"pending turn for the bounded embed pass number {i} long enough",
            ts=f"2026-07-04T03:{i:02d}:00Z",
            source_uuid=str(uuid.uuid4()),
        )
        for i in range(n)
    ]
    _write_backlog(iai_home, events)
    drain_deferred_captures(store)
    assert _count_pending(store) == n

    spy = _SpyEmbedder()
    # Window of 3, no RSS cap (cap=0 disables): every pending row is embedded.
    embedded = store.db.reembed_pending_rows(spy, batch_size=3, rss_soft_cap_bytes=0)

    assert embedded == n, embedded
    assert len(spy.calls) == n, spy.calls
    assert _count_pending(store) == 0, "all pending rows must be embedded"


def test_deferred_embed_pass_honors_rss_soft_cap(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.hippo import HippoDB

    store = _open_store()

    n = 9
    events = [
        _make_event(
            f"pending turn for the rss-capped embed pass number {i} long enough",
            ts=f"2026-07-04T04:{i:02d}:00Z",
            source_uuid=str(uuid.uuid4()),
        )
        for i in range(n)
    ]
    _write_backlog(iai_home, events)
    drain_deferred_captures(store)
    assert _count_pending(store) == n

    # Window of 3 → three windows. The soft cap is growth-relative: the pass
    # samples its resident set once at start and the effective ceiling is
    # max(requested cap, start + growth budget) — an absolute cap below the
    # process baseline must not starve the pass into doing nothing. The
    # injected reader returns the baseline on the pass-start sample, then a
    # value past the budget on every per-window check (which runs before the
    # second and third windows, never the first), so exactly one window
    # (3 rows) embeds and the remaining six stay pending.
    from iai_mcp.hippo._db import REEMBED_RSS_GROWTH_BUDGET_BYTES

    _rss_calls = {"n": 0}

    def _growing_rss() -> int:
        _rss_calls["n"] += 1
        if _rss_calls["n"] == 1:
            return 10_000
        return 10_000 + REEMBED_RSS_GROWTH_BUDGET_BYTES + 1

    monkeypatch.setattr(
        HippoDB, "_reembed_rss_bytes",
        staticmethod(_growing_rss),
    )

    spy = _SpyEmbedder()
    embedded = store.db.reembed_pending_rows(spy, batch_size=3, rss_soft_cap_bytes=1000)

    assert embedded == 3, f"only the first window must embed before the cap; got {embedded}"
    assert len(spy.calls) == 3, spy.calls
    assert _count_pending(store) == n - 3, (
        f"the six rows beyond the soft cap must stay pending; "
        f"pending={_count_pending(store)}"
    )

    # A second pass (cap not tripped) drains the remainder — nothing is lost.
    spy2 = _SpyEmbedder()
    embedded2 = store.db.reembed_pending_rows(spy2, batch_size=3, rss_soft_cap_bytes=0)
    assert embedded2 == n - 3, embedded2
    assert _count_pending(store) == 0, _count_pending(store)
