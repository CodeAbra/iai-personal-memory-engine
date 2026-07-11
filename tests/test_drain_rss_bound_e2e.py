"""Hermetic end-to-end proof that the post-boot backlog drain is resident-set bounded.

The failure this pins: a daemon restart drained a large backlog of genuinely-new
turns synchronously — every turn was embedded inline — and the resident set
climbed past the hard cap while the embedder, JIT cache, and columnar pages were
held resident through one long run. The fix is two-phase: phase one writes every
genuinely-new turn as a pending (un-embedded) row, so the drain is a sequence of
cheap SQLite writes whose cost does not grow with the backlog size; phase two
fills the real vectors in resident-set-gated windows; and the watchdog fast-kills
on the first over-cap sample as the last line of defence.

This module rebuilds the failure scenario on a throwaway store and proves the
bound holds, with no live daemon and no real embedder memory behaviour on the
critical assertions:

  1. Drain-is-bounded — a LARGE backlog (many files, thousands of events, sized
     so the OLD synchronous path would embed them all) drains with ZERO embedder
     calls, every event lands as a pending row, and the measured peak resident-set
     delta during the drain stays under a generous ceiling (a structural proof:
     no embedder is materialised, so there is no multi-GiB climb).

  2. Deferred-embed-is-bounded — the deferred-embed pass over those pending rows,
     driven with a small window and an injected over-cap resident-set reader,
     embeds one bounded window and STOPS EARLY, leaving the rest pending for the
     next cycle. Even phase two cannot run away.

  3. Watchdog backstop — an over-hard-cap sample returns ("kill", "leak") on the
     first tick, bypassing grace and debounce; the integrated tick issues a
     SIGKILL (mocked — the test process is never killed).

Determinism: the resident-set readers are injected, the embedder is a spy, and
the peak-RSS assertion carries a generous ceiling so it certifies "no runaway"
without depending on real allocator timing. The load-bearing proof is structural
— the drain calls no embedder and writes only pending rows.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make  # noqa: E402

from iai_mcp.types import EMBED_DIM  # noqa: E402

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename; deferred-drain is POSIX-only here",
)

SESSION_PREFIX = "rss-bound-e2e"

# Backlog scale. The real failure was ~160 files / multi-thousand events drained
# synchronously to 4.7 GiB. We mirror that order of magnitude: many files, well
# over a thousand genuinely-new events — sized so the OLD synchronous path would
# have embedded every one of them. The per-run drain cap is 5000 events, so the
# whole backlog drains in one pass.
N_FILES = 40
EVENTS_PER_FILE = 40
TOTAL_EVENTS = N_FILES * EVENTS_PER_FILE  # 1600

# Peak resident-set delta ceiling for the drain. Pure SQLite writes plus jsonl
# parsing for 1600 short rows cost on the order of a few MB; the embedder, JIT,
# and columnar transients that drove the 4.7 GiB climb are never materialised.
# The ceiling is deliberately generous (far below a single embedder load, which
# alone is hundreds of MB) so the bound is certified without flakiness.
RSS_DELTA_CEILING_BYTES = 350 * 1024 * 1024  # 350 MiB


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """Throwaway HOME + store + crypto, fully isolated from the live daemon.

    Nothing here touches ~/.iai-mcp or the launchd-managed daemon: HOME is
    redirected to a tmp dir, the store path is a tmp dir, and the daemon socket
    points at a non-existent path so no socket call can reach a live process.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-rss-bound-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no-such.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")
    # Keep the drain's own soft cap out of the way for the bounded-drain proof —
    # we want the whole backlog to drain in one pass and prove it stays bounded
    # without the early-stop rail doing the work.
    monkeypatch.delenv("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES", raising=False)
    monkeypatch.delenv("IAI_MCP_DISABLE_INDAEMON_DRAIN", raising=False)
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def _make_event(text: str, *, role: str, ts: str, source_uuid: str) -> dict:
    return {
        "text": text,
        "cue": "session turn",
        "tier": "episodic",
        "role": role,
        "ts": ts,
        "source_uuid": source_uuid,
    }


def _write_backlog_files(home: Path, n_files: int, events_per_file: int) -> int:
    """Write a multi-file backlog of genuinely-new events; return total count.

    Each event is globally unique (a fresh uuid + a distinct timestamp + distinct
    text), so none dedup against another — the drain must treat every one as a
    new turn and write a pending row for it, exactly as the real backlog did.
    """
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    base = datetime(2026, 7, 4, 0, 0, 0, tzinfo=timezone.utc)
    for f in range(n_files):
        session_id = f"{SESSION_PREFIX}-{f:03d}"
        out_path = deferred_dir / f"{session_id}-backlog.jsonl"
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": "/tmp/test",
        }
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for e in range(events_per_file):
                # Unique, monotone-ish timestamp per event across all files.
                ordinal = f * events_per_file + e
                seconds = base.timestamp() + ordinal * 60
                ts = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
                text = (
                    f"Genuinely new backlog turn {ordinal:05d} in session {session_id} "
                    f"with enough verbatim text to be stored as an episodic record"
                )
                ev = _make_event(text, role="user", ts=ts, source_uuid=str(uuid.uuid4()))
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                total += 1
    return total


def _count_pending(store) -> int:
    with store.db._conn_lock:
        return store.db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1 AND tombstoned_at IS NULL"
        ).fetchone()[0]


def _count_active(store) -> int:
    with store.db._conn_lock:
        return store.db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Drain-is-bounded — the core proof
# ---------------------------------------------------------------------------


def test_large_backlog_drain_is_rss_bounded_and_embeds_nothing(iai_home, monkeypatch):
    """The definitive bound: a large backlog drains writes-only-pending.

    Proves (a) the embedder is never called during the drain, (b) every one of
    the ~1600 genuinely-new events lands as a pending row, and (c) the measured
    peak resident-set delta during the drain stays under a generous ceiling —
    nowhere near the multi-GiB climb the old synchronous path produced.
    """
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.embed import Embedder

    store = _open_store()

    written = _write_backlog_files(iai_home, N_FILES, EVENTS_PER_FILE)
    assert written == TOTAL_EVENTS, written

    # Spy: any embed call during the drain is a revert of the two-phase fix.
    embed_calls: list[str] = []
    real_embed = Embedder.embed

    def spy_embed(self, text):
        embed_calls.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(Embedder, "embed", spy_embed)

    # Sample the resident set around the drain. A background sampler captures the
    # peak so a transient mid-drain spike cannot hide between the before/after
    # reads. The drain is the only meaningful work on this thread during the run.
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    peak = {"rss": rss_before}
    stop = threading.Event()

    def _sampler():
        while not stop.is_set():
            r = proc.memory_info().rss
            if r > peak["rss"]:
                peak["rss"] = r
            time.sleep(0.005)

    sampler = threading.Thread(target=_sampler, name="rss-peak-sampler", daemon=True)
    sampler.start()
    try:
        counts = drain_deferred_captures(store)
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    rss_after = proc.memory_info().rss
    peak_rss = max(peak["rss"], rss_after)
    peak_delta = peak_rss - rss_before

    # (a) the drain embedded nothing — all embedding is deferred.
    assert embed_calls == [], (
        f"the drain must not embed any turn; saw {len(embed_calls)} embed calls. "
        f"counts={counts!r}"
    )

    # (b) every genuinely-new event landed as a pending row.
    assert counts["events_inserted"] == TOTAL_EVENTS, counts
    assert _count_pending(store) == TOTAL_EVENTS, (
        f"all {TOTAL_EVENTS} new turns must be pending; pending={_count_pending(store)}"
    )
    assert _count_active(store) == TOTAL_EVENTS, _count_active(store)

    # (c) the peak resident-set delta during the drain is bounded — the secondary
    # sanity check. The structural proof is (a): no embedder is materialised.
    assert peak_delta < RSS_DELTA_CEILING_BYTES, (
        f"drain peak RSS delta {peak_delta / 1024 / 1024:.1f} MiB exceeded the "
        f"{RSS_DELTA_CEILING_BYTES / 1024 / 1024:.0f} MiB ceiling — the drain is "
        f"not bounded (a synchronous embed of {TOTAL_EVENTS} turns would climb "
        f"far past this)"
    )

    # Surface the measured number for the report log.
    print(
        f"\n[drain-rss-bound] events={TOTAL_EVENTS} files={N_FILES} "
        f"rss_before={rss_before / 1024 / 1024:.1f}MiB "
        f"peak={peak_rss / 1024 / 1024:.1f}MiB "
        f"delta={peak_delta / 1024 / 1024:.1f}MiB "
        f"ceiling={RSS_DELTA_CEILING_BYTES / 1024 / 1024:.0f}MiB "
        f"embed_calls={len(embed_calls)}"
    )


# ---------------------------------------------------------------------------
# 2. Deferred-embed-is-bounded — phase two cannot run away either
# ---------------------------------------------------------------------------


class _SpyEmbedder:
    """Counts embed calls and returns a deterministic unit vector per text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str):
        import numpy as np

        self.calls.append(text)
        h = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(h)
        v = rng.random(EMBED_DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()


def test_deferred_embed_pass_stops_early_under_injected_over_cap(iai_home, monkeypatch):
    """The deferred-embed wake sequence is windowed + RSS-capped + stops early.

    Drives the real wake-sequence entry point over a pending backlog with a small
    window and an injected over-cap resident-set reader. The pass embeds exactly
    one window and yields, leaving the remaining rows pending for the next cycle —
    so even phase two is resident-set bounded and never runs the whole backlog
    through the embedder in one go.
    """
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.hippo import HippoDB

    store = _open_store()

    # A modest pending backlog drained writes-only-pending first.
    n = 12
    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{SESSION_PREFIX}-embed-backlog.jsonl"
    base = datetime(2026, 7, 5, 0, 0, 0, tzinfo=timezone.utc)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": f"{SESSION_PREFIX}-embed",
            "cwd": "/tmp/test",
        }) + "\n")
        for i in range(n):
            ts = datetime.fromtimestamp(base.timestamp() + i * 60, tz=timezone.utc).isoformat()
            ev = _make_event(
                f"Pending turn {i:03d} for the bounded deferred-embed pass with text",
                role="user", ts=ts, source_uuid=str(uuid.uuid4()),
            )
            fh.write(json.dumps(ev) + "\n")

    drain_deferred_captures(store)
    assert _count_pending(store) == n, _count_pending(store)

    # Window of 3 → the soft cap is checked before reading the second window.
    # The cap is growth-relative (max(requested, start + growth budget)), so
    # the injected reader returns the baseline on the pass-start sample and a
    # value past the budget on the per-window checks: exactly the first
    # window (3 rows) embeds and the remaining nine stay pending.
    from iai_mcp.hippo._db import REEMBED_RSS_GROWTH_BUDGET_BYTES

    monkeypatch.setenv("IAI_MCP_REEMBED_BATCH_SIZE", "3")
    monkeypatch.setenv("IAI_MCP_REEMBED_RSS_SOFT_CAP_BYTES", "1000")

    _rss_calls = {"n": 0}

    def _growing_rss() -> int:
        _rss_calls["n"] += 1
        if _rss_calls["n"] == 1:
            return 10_000
        return 10_000 + REEMBED_RSS_GROWTH_BUDGET_BYTES + 1

    monkeypatch.setattr(
        HippoDB, "_reembed_rss_bytes", staticmethod(_growing_rss)
    )

    spy = _SpyEmbedder()
    result = store.db.pending_embeddings_wake_sequence(spy)

    assert result["action"] == "wake_sequence", result
    assert result["reembed_count"] == 3, (
        f"only the first window may embed before the injected over-cap reading; "
        f"got {result['reembed_count']}"
    )
    assert len(spy.calls) == 3, spy.calls
    assert _count_pending(store) == n - 3, (
        f"the rows beyond the soft cap must stay pending for the next cycle; "
        f"pending={_count_pending(store)}"
    )

    # A second wake (cap not tripped) drains the remainder — nothing is lost.
    monkeypatch.setattr(
        HippoDB, "_reembed_rss_bytes", staticmethod(lambda: 100)
    )
    spy2 = _SpyEmbedder()
    result2 = store.db.pending_embeddings_wake_sequence(spy2)
    assert result2["reembed_count"] == n - 3, result2
    assert _count_pending(store) == 0, _count_pending(store)

    print(
        f"\n[deferred-embed-bound] pending={n} window=3 "
        f"first_pass_embedded={result['reembed_count']} "
        f"left_pending_after_first={n - 3} "
        f"second_pass_embedded={result2['reembed_count']}"
    )


# ---------------------------------------------------------------------------
# 3. Watchdog backstop — over-cap kills immediately
# ---------------------------------------------------------------------------

HARD_CAP = 2_684_354_560
FLOOR = 1_610_612_736
DEBOUNCE_N = 3
GRACE = 600.0
MAX_RECOVERIES = 3
WINDOW = 600.0
RSS_OVER_CAP = HARD_CAP + (512 * 1024 * 1024)
NORMAL = 1


def _evaluate(rss, *, uptime=GRACE + 1.0, consecutive=0):
    from iai_mcp import daemon

    return daemon._evaluate_watchdog(
        True,
        rss,
        NORMAL,
        uptime,
        consecutive,
        [],
        1_000_000.0,
        hard_cap=HARD_CAP,
        contributor_floor=FLOOR,
        debounce_n=DEBOUNCE_N,
        cold_start_grace_sec=GRACE,
        max_recoveries=MAX_RECOVERIES,
        recovery_window_sec=WINDOW,
    )


def test_watchdog_decision_kills_immediately_over_hard_cap():
    """An over-hard-cap sample is a runaway: kill on the first tick, no debounce,
    no grace. This is the backstop that bounds the worst case if both phases were
    somehow defeated."""
    # First over-cap tick, well out of grace.
    assert _evaluate(RSS_OVER_CAP, consecutive=0) == ("kill", "leak")
    # Even inside the cold-start grace (a runaway during the post-boot drain).
    assert _evaluate(RSS_OVER_CAP, uptime=5.0, consecutive=0) == ("kill", "leak")
    # A below-cap reading does not fast-path — the transient paths keep debounce.
    assert _evaluate(FLOOR - 1, consecutive=0) == ("none", "healthy")


def test_watchdog_tick_sigkills_over_cap(tmp_path, monkeypatch):
    """The integrated tick issues a SIGKILL on the first over-cap sample. The
    terminal kill is mocked — the test process is never killed."""
    from iai_mcp import daemon

    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")

    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)
    # Boot timestamp inside the grace window — proves the cap ignores grace.
    monkeypatch.setattr(daemon, "_daemon_started_monotonic", time.monotonic())

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    async def _probe_ok(_sock, _timeout):
        return True

    try:
        _interval, _consec = daemon._watchdog_tick(
            object(),
            sock_path,
            log_path,
            0,  # consecutive_failures starts at zero
            probe_fn=_probe_ok,
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: daemon.WATCHDOG_RSS_HARD_CAP_BYTES + (512 * 1024 * 1024),
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    assert kill_calls == [(os.getpid(), signal.SIGKILL)], (
        "an over-hard-cap sample must SIGKILL on the first tick, even within the "
        "cold-start grace and with consecutive_failures == 0"
    )
    crumb = log_path.read_text(encoding="utf-8")
    assert daemon.DAEMON_MEMORY_PRESSURE_KILL in crumb
    assert "reason=leak" in crumb
