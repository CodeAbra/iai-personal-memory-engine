"""Dual-driver safety net for the sensory tier (`.live.jsonl` pre-encode buffer).

Covers the bounded-read cap, the recall-critical-path no-op proof (SC-d),
and the permission-hardening RED guard (SC-e). The permission test is
EXPECTED TO FAIL on current HEAD (files are created mode 0644) — it guards
the fix that lands in the follow-up plan; it must not be xfail-marked so it
flips visibly GREEN once the fix lands.

No production source file is modified by this test file.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import make_tmp_store  # noqa: E402 — test helper; sys.path manipulated above

from iai_mcp import core
from iai_mcp.capture import (
    MAX_CAPTURE_LEN,
    _LIVE_SELECT_MAX_FILES,
    _TAIL_MAX_EVENT_LINES,
    read_pending_live_events,
    write_deferred_event,
)
from iai_mcp.sensory import sensory_pending, trace_promotion
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))


def _monkeypatch_env_with_store(monkeypatch, tmp_path: Path) -> Path:
    """Same as _monkeypatch_env, plus a real crypto passphrase so the
    write-side compaction path's internal MemoryStore() open succeeds.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-sensory-tier-write-bound-passphrase")
    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    import keyring.core
    keyring.core._keyring_backend = None
    return store_root


def _seed_one_record(store: MemoryStore, text: str = "reference content") -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
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
    store.insert(rec)


# ---------------------------------------------------------------------------
# SC-b: bounded-tail read cap (files + lines)
# ---------------------------------------------------------------------------

def test_bounded_tail_read_caps_lines_and_files(tmp_path, monkeypatch):
    """read_pending_live_events() never returns more than the existing caps.

    Writes more session files than _LIVE_SELECT_MAX_FILES and more event
    lines than _TAIL_MAX_EVENT_LINES into one file, then asserts the read
    path honors both caps. This proves the existing read bound holds and
    gives the write-side bound decision a companion assertion.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    n_files = _LIVE_SELECT_MAX_FILES + 5
    for i in range(n_files):
        write_deferred_event(f"sess-cap-{i}", "user", f"file cap probe turn {i} content")

    n_lines = _TAIL_MAX_EVENT_LINES + 50
    heavy_session = "sess-heavy-tail"
    for i in range(n_lines):
        write_deferred_event(heavy_session, "user", f"heavy tail line {i} probe content")

    events = read_pending_live_events()
    files_seen = {ev["session_id"] for ev in events}
    assert len(files_seen) <= _LIVE_SELECT_MAX_FILES, (
        f"read_pending_live_events() surfaced {len(files_seen)} distinct "
        f"session files; cap is {_LIVE_SELECT_MAX_FILES}"
    )

    heavy_events = read_pending_live_events(session_id=heavy_session)
    heavy_from_session = [ev for ev in heavy_events if ev["session_id"] == heavy_session]
    assert len(heavy_from_session) <= _TAIL_MAX_EVENT_LINES, (
        f"read_pending_live_events(session_id={heavy_session!r}) returned "
        f"{len(heavy_from_session)} events; per-file cap is {_TAIL_MAX_EVENT_LINES}"
    )


# ---------------------------------------------------------------------------
# SC-d: recall critical path never calls sensory-tier functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_path_no_sensory_calls(driver, tmp_path, monkeypatch):
    """memory_recall dispatch never touches the sensory tier, on either driver.

    Monkeypatches the three sensory-tier entry points to raise if called, then
    invokes the production recall dispatch. The dispatch must return normally
    (no raise) -- a measurable no-op proof for SC-d.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    _monkeypatch_env(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError(
            "sensory-tier function called during memory_recall dispatch"
        )

    monkeypatch.setattr("iai_mcp.capture.drain_active_live_captures", _boom)
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", _boom)
    monkeypatch.setattr("iai_mcp.capture.write_deferred_event", _boom)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        _seed_one_record(store, "reference content for recall no-op probe")

        resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content",
            "session_id": f"sess-noop-{driver}",
            "cue_embedding": [0.1] * EMBED_DIM,
        })
        assert "error" not in resp, (
            f"[driver={driver}] recall dispatch errored: {resp.get('error')}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# SC-e: permission hardening RED guard (guards a follow-up fix)
# ---------------------------------------------------------------------------

def test_live_jsonl_permission_mode_is_0600(tmp_path, monkeypatch):
    """write_deferred_event() must create .live.jsonl at mode 0600, not 0644.

    RED on current HEAD: write_deferred_event opens the file with the
    default umask (0644). This test guards the follow-up permission-hardening
    fix and is expected to flip GREEN only once that fix lands. Do NOT
    xfail this test -- it must surface as a visible failure until fixed.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    path = write_deferred_event("sess-perm-probe", "user", "permission probe content here")

    mode = oct(path.stat().st_mode & 0o777)
    assert mode == "0o600", (
        f"expected .live.jsonl mode 0o600, got {mode} for {path}"
    )


def test_live_jsonl_permission_tightened_on_preexisting_0644(tmp_path, monkeypatch):
    """A pre-existing 0644 .live.jsonl is tightened to 0600 on the next open.

    O_CREAT only sets the mode on create; a legacy world-readable file must be
    hardened to 0600 on every open, not only on the fresh-create path.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    session_id = "sess-perm-tighten"
    deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    legacy = deferred_dir / f"{session_id}.live.jsonl"
    legacy.write_text(
        '{"version": 1, "session_id": "sess-perm-tighten", "cwd": "/x"}\n',
        encoding="utf-8",
    )
    os.chmod(legacy, 0o644)
    assert oct(legacy.stat().st_mode & 0o777) == "0o644"

    write_deferred_event(session_id, "user", "tightening probe content here")

    mode = oct(legacy.stat().st_mode & 0o777)
    assert mode == "0o600", (
        f"expected pre-existing .live.jsonl tightened to 0o600, got {mode}"
    )


def test_compaction_preserves_concurrent_append(tmp_path, monkeypatch):
    """A line appended during compaction (between the read and os.replace) must
    never be lost. The lossless fence re-stats before the swap and aborts on any
    change, leaving the full file (including the concurrent append) intact.

    Simulates the same-session cross-process race by injecting an O_APPEND write
    at the exact moment os.replace is about to run.
    """
    from iai_mcp import capture as capture_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(capture_mod, "_WRITE_SIDE_MAX_EVENT_LINES", _TEST_WRITE_SIDE_CAP)
    monkeypatch.setattr(capture_mod, "_WRITE_SIDE_MIN_BYTES_PER_LINE", 1)

    session_id = "sess-compact-race"
    deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    live_path = deferred_dir / f"{session_id}.live.jsonl"
    n_events = _TEST_WRITE_SIDE_CAP + 5
    lines = ['{"version": 1, "session_id": "sess-compact-race", "cwd": "/x"}\n']
    for i in range(n_events):
        lines.append(
            '{"text": "existing compaction probe line %d content", "role": "user", '
            '"tier": "episodic", "ts": "2026-07-04T00:00:00+00:00"}\n' % i
        )
    live_path.write_text("".join(lines), encoding="utf-8")

    # Half the events already promoted (drain-offset set), so compaction would
    # otherwise rewrite the file down to the un-promoted tail.
    (state_dir / f"{session_id}.drain-offset").write_text(str(n_events // 2))

    # Drain is a no-op here (offset already set on disk); avoid a real store open.
    monkeypatch.setattr(
        capture_mod, "drain_active_live_captures", lambda *a, **k: {}
    )

    race_marker = "concurrent append during compaction MUST survive"
    real_open = os.open
    injected = {"done": False}

    def _open_with_concurrent_append(pathname, flags, *args, **kwargs):
        # The compaction opens the tmp file for writing AFTER it has read the
        # source (and captured pre_stat) and BEFORE the pre-replace re-stat.
        # Injecting a same-session O_APPEND write to the live file here lands a
        # fresh un-promoted line squarely inside the read->replace race window.
        fd = real_open(pathname, flags, *args, **kwargs)
        if not injected["done"] and str(pathname).endswith(".jsonl.compact-tmp"):
            injected["done"] = True
            afd = real_open(str(live_path), os.O_WRONLY | os.O_APPEND)
            try:
                os.write(
                    afd,
                    ('{"text": "%s", "role": "user", "tier": "episodic", '
                     '"ts": "2026-07-04T00:00:01+00:00"}\n' % race_marker).encode(),
                )
            finally:
                os.close(afd)
        return fd

    monkeypatch.setattr(capture_mod.os, "open", _open_with_concurrent_append)

    # This append triggers the compaction path; the injected concurrent append
    # lands mid-compaction and must survive.
    write_deferred_event(session_id, "user", "trigger line that fires compaction")

    surviving = live_path.read_text(encoding="utf-8")
    assert race_marker in surviving, (
        "a line appended concurrently during compaction was lost -- the "
        "lossless fence must abort the swap and keep the un-promoted append"
    )
    assert injected["done"], "test did not exercise the compaction os.replace path"


# ---------------------------------------------------------------------------
# SC-2: write-side bound -- lossless wins over bounded when nothing promoted;
# compaction removes only already-promoted lines when a drain succeeds.
# ---------------------------------------------------------------------------

_TEST_WRITE_SIDE_CAP = 30


def test_write_side_bound_never_drops_unpromoted_when_no_drain_possible(tmp_path, monkeypatch):
    """Filling past the write-side cap with un-promoted content, where the
    drain itself cannot run, must NOT truncate the file -- lossless wins
    over bounded.

    Forces drain_active_live_captures to fail (simulating e.g. an
    unreachable store) so nothing is promoted; the self-compact guard must
    catch this and leave the file fully intact rather than lose un-promoted
    raw content. The cap is monkeypatched down so the test exercises the
    real code path without writing thousands of lines.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setattr("iai_mcp.capture._WRITE_SIDE_MAX_EVENT_LINES", _TEST_WRITE_SIDE_CAP)

    def _boom_drain(*args, **kwargs):
        raise RuntimeError("simulated drain failure — store unreachable")

    monkeypatch.setattr("iai_mcp.capture.drain_active_live_captures", _boom_drain)

    session_id = "sess-nodrain-forced-fail"
    n_lines = _TEST_WRITE_SIDE_CAP + 10
    for i in range(n_lines):
        write_deferred_event(session_id, "user", f"undrainable probe line {i} content")

    path = Path.home() / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.endswith("\n")]
    # header + every written event line must still be present: the drain
    # raised, so compaction must have been a no-op -- lossless wins over
    # bounded.
    assert len(lines) - 1 == n_lines, (
        f"expected all {n_lines} event lines to survive (no promotion "
        f"possible), found {len(lines) - 1}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_write_side_bound_compacts_only_promoted_lines(driver, tmp_path, monkeypatch):
    """Filling past the write-side cap WITH promotion available compacts the
    file down to only the un-promoted tail; total on-disk lines stay bounded.

    The cap is monkeypatched down so the test exercises the real code path
    (real drain + real store inserts) without writing thousands of lines.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    store_root = _monkeypatch_env_with_store(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setattr("iai_mcp.capture._WRITE_SIDE_MAX_EVENT_LINES", _TEST_WRITE_SIDE_CAP)
    # Byte-size gate: force it to trip well before the real event-line size
    # would naturally cross the (now tiny) cap, so the compaction path is
    # exercised deterministically at this small n.
    monkeypatch.setattr("iai_mcp.capture._WRITE_SIDE_MIN_BYTES_PER_LINE", 1)

    # Open (and immediately close) a store once up-front so the on-disk store
    # exists before any write triggers the internal compaction store-open.
    store = MemoryStore(path=store_root)
    store.close()

    session_id = f"sess-drainable-{driver}"
    n_lines = _TEST_WRITE_SIDE_CAP + 10
    for i in range(n_lines):
        write_deferred_event(session_id, "user", f"drainable probe line {i} content")

    path = Path.home() / ".iai-mcp" / ".deferred-captures" / f"{session_id}.live.jsonl"
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.endswith("\n")]
    event_line_count = len(lines) - 1

    # This session's own file was drained by the self-compaction sentinel
    # (exclude_session_id="-" != session_id), so at least the lines written
    # before the cap tripped were promoted and compacted away.
    assert event_line_count < n_lines, (
        f"[driver={driver}] expected compaction to shrink the file below "
        f"{n_lines} lines, found {event_line_count}"
    )
    assert event_line_count <= _TEST_WRITE_SIDE_CAP + 20, (
        f"[driver={driver}] compacted file still has {event_line_count} "
        f"lines, expected it bounded near the cap ({_TEST_WRITE_SIDE_CAP})"
    )

    store2 = MemoryStore(path=store_root)
    try:
        promoted = [
            r for r in store2.iter_records(where="tier = 'episodic'")
            if "drainable probe line" in r.literal_surface
        ]
        assert len(promoted) > 0, (
            f"[driver={driver}] expected at least some lines promoted into "
            f"the store by the write-side compaction drain"
        )
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# SC-a: raw content lands pre-encode -- resident in the sensory buffer before
# any embed/HD-encode call.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_content_lands_pre_encode(driver, tmp_path, monkeypatch):
    """A sensory-appended turn is readable back from the pending buffer
    before any Hippo record (and therefore any embedding) exists for it.

    write_deferred_event() never touches the embedder or the store -- it is
    a plain JSONL append. This proves raw content is observable pre-encode:
    the pending event carries the raw text while no matching Hippo record
    exists yet.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    store_root = _monkeypatch_env_with_store(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    session_id = f"sess-preencode-{driver}"
    nonce = f"pre encode residency nonce marker {driver} content probe"

    write_deferred_event(session_id, "user", nonce)

    pending = sensory_pending(session_id=session_id)
    texts = [ev["text"] for ev in pending if ev["session_id"] == session_id]
    assert nonce in texts, (
        f"[driver={driver}] expected raw nonce resident in pending sensory "
        f"events before any promotion; got: {texts!r}"
    )

    store = MemoryStore(path=store_root)
    try:
        existing = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if nonce in (r.literal_surface or "")
        ]
        assert len(existing) == 0, (
            f"[driver={driver}] expected NO Hippo record for the nonce "
            f"before promotion (content must land pre-encode); found "
            f"{len(existing)}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# SC-c: a sensory entry is traceable end-to-end into a resulting Hippo
# record via the drain -> capture_turn -> embed -> store.insert chain.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_promotion_observable_end_to_end(driver, tmp_path, monkeypatch):
    """trace_promotion() observes a raw sensory entry landing in Hippo.

    Writes a unique-nonce sensory entry, runs promotion via
    sensory.trace_promotion, and asserts a Hippo record now carries that
    nonce in its literal_surface -- the full sensory -> Hippo trace, on
    both drivers.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    store_root = _monkeypatch_env_with_store(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    session_id = f"sess-e2e-trace-{driver}"
    nonce = f"end to end promotion trace nonce marker {driver} content probe"

    write_deferred_event(session_id, "user", nonce)

    store = MemoryStore(path=store_root)
    try:
        trace = trace_promotion(store, session_id, exclude_session_id="-none-")
        assert trace["counts"]["events_inserted"] >= 1, (
            f"[driver={driver}] expected trace_promotion to insert the "
            f"nonce turn, got counts: {trace['counts']}"
        )

        promoted = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if nonce in (r.literal_surface or "")
        ]
        assert len(promoted) == 1, (
            f"[driver={driver}] expected exactly 1 Hippo record carrying "
            f"the nonce after promotion; found {len(promoted)}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# SC-e: promotion preserves literal_surface byte-for-byte, modulo only the
# existing leading/trailing strip + MAX_CAPTURE_LEN truncate.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_promotion_preserves_literal_surface(driver, tmp_path, monkeypatch):
    """Promoted literal_surface is byte-identical to the raw sensory text,
    modulo only strip()/MAX_CAPTURE_LEN truncation -- no paraphrase, no
    whitespace normalization, no case-folding.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    store_root = _monkeypatch_env_with_store(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    store = MemoryStore(path=store_root)
    try:
        # Case 1: internal irregular whitespace, mixed case, odd punctuation;
        # no leading/trailing ws -- must round-trip byte-identical.
        session_a = f"sess-verbatim-a-{driver}"
        irregular = (
            f"Odd   Spacing,MixedCase!!  and\ttabs -- {driver} verbatim probe??"
        )
        write_deferred_event(session_a, "user", irregular)
        trace_a = trace_promotion(store, session_a, exclude_session_id="-none-")
        assert trace_a["counts"]["events_inserted"] >= 1

        episodic_a = list(store.iter_records(where="tier = 'episodic'"))
        matches_a = [r for r in episodic_a if r.literal_surface == irregular]
        candidates_a = [r.literal_surface for r in episodic_a if driver in (r.literal_surface or "")]
        assert len(matches_a) == 1, (
            f"[driver={driver}] expected byte-identical literal_surface for "
            f"the irregular-formatting probe; candidates found: {candidates_a!r}"
        )

        # Case 2: leading/trailing whitespace -- only strip() applies.
        session_b = f"sess-verbatim-b-{driver}"
        padded = f"   leading and trailing ws stripped only {driver} probe   "
        expected_b = padded.strip()
        write_deferred_event(session_b, "user", padded)
        trace_promotion(store, session_b, exclude_session_id="-none-")

        matches_b = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if r.literal_surface == expected_b
        ]
        assert len(matches_b) == 1, (
            f"[driver={driver}] expected literal_surface stripped of only "
            f"leading/trailing ws (got no exact match for {expected_b!r})"
        )
        assert len(expected_b) < MAX_CAPTURE_LEN, (
            "sanity: this probe must stay well under MAX_CAPTURE_LEN so the "
            "truncate boundary is never in play for this assertion"
        )
    finally:
        store.close()
