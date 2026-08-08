from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX socket; HOME isolation pattern",
)


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))

    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _write_deferred_jsonl(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    version: int = 1,
    ts_suffix: int | None = None,
) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = ts_suffix if ts_suffix is not None else int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": version,
        "deferred_at": "2026-04-26T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    lines = [json.dumps(header)] + [json.dumps(e) for e in events]
    out.write_text("\n".join(lines) + "\n")
    return out


def _make_event(text: str, role: str = "user") -> dict:
    return {
        "text": text,
        "cue": f"test cue: {text[:24]}",
        "tier": "episodic",
        "role": role,
        "ts": "2026-04-26T00:00:00Z",
    }


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_drain_consumes_jsonl_and_deletes_file(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    events = [
        _make_event("alice said: drain test event one — must be at least 12 chars"),
        _make_event("assistant reply with sufficient length to pass MIN_CAPTURE", role="assistant"),
        _make_event("third event for the round-trip drain count assertion"),
    ]
    fpath = _write_deferred_jsonl(deferred_dir, "session-A", events)
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "deferred file must be unlinked after drain"

    n_rows = store.db.open_table("records").count_rows()
    assert n_rows >= 3, f"expected ≥3 records inserted, got {n_rows}"


def test_drain_handles_malformed_event_line(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    fpath = deferred_dir / "session-B-12345.jsonl"
    fpath.write_text(
        json.dumps({
            "version": 1,
            "deferred_at": "2026-04-26T00:00:00Z",
            "session_id": "session-B",
            "cwd": "/tmp",
        }) + "\n"
        + json.dumps(_make_event("first valid event with adequate length")) + "\n"
        + "this line is not valid JSON {{{ broken\n"
        + json.dumps(_make_event("never reached because file-level error")) + "\n"
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert counts["files_drained"] == 0, counts
    assert not fpath.exists(), "original must be renamed away on per-file error"
    failed = list(deferred_dir.glob("session-B-12345.failed-*.jsonl"))
    assert len(failed) == 1, f"expected exactly 1 .failed-* file, got {failed}"


def test_drain_skips_future_version(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-C",
        [_make_event("event from a future format version that we cannot parse")],
        version=99,
    )

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert fpath.exists(), "version>1 file must remain for a future daemon to handle"
    assert not list(deferred_dir.glob("*.failed-*.jsonl"))

    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "drain must create a log file when it skips a future version"
    log_content = log_files[0].read_text()
    assert "skip" in log_content
    assert "session-C" in log_content
    assert "version=99" in log_content


def test_drain_no_deferred_dir(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    assert not deferred_dir.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not deferred_dir.exists(), "drain should not create .deferred-captures/"


def test_drain_empty_jsonl(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    fpath = deferred_dir / "session-E-empty.jsonl"
    fpath.write_text("")
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "0-byte file must be unlinked"


def test_drain_multiple_files_processed_in_order(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    distinct_texts = [
        "apples are red and grow on trees in orchards across the world",
        "quantum chromodynamics describes the strong nuclear force precisely",
        "hummingbirds beat their wings about eighty times per second in flight",
    ]
    paths = []
    for i, base_ts in enumerate([1000, 2000, 3000]):
        events = [_make_event(distinct_texts[i])]
        paths.append(
            _write_deferred_jsonl(
                deferred_dir, f"session-F-{i}", events, ts_suffix=base_ts,
            )
        )
    assert all(p.exists() for p in paths)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 3, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert counts["files_failed"] == 0, counts
    for p in paths:
        assert not p.exists(), f"{p} must be unlinked after drain"


def test_drain_partial_insert_failure_preserves_file(iai_home, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.hippo import HippoDB

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"

    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-H",
        [
            _make_event("first good event with adequate length here"),
            _make_event("INSERT_FAIL_SENTINEL_07_9 — this event triggers a failure"),
            _make_event("third good event after the failing one in the middle"),
        ],
        ts_suffix=42,
    )
    assert fpath.exists()

    # The two-phase drain writes each genuinely-new turn as a pending row via
    # HippoDB.insert_pending_row (no synchronous embed). Fail that write for the
    # sentinel event and assert the whole file is preserved as .failed-* — the
    # per-file insert-failure contract is identical, just at the new write site.
    real_insert_pending = HippoDB.insert_pending_row

    def insert_pending_or_fail(self, *, literal_surface, **kwargs):
        if "INSERT_FAIL_SENTINEL_07_9" in literal_surface:
            raise RuntimeError("simulated pending-row write failure")
        return real_insert_pending(self, literal_surface=literal_surface, **kwargs)

    monkeypatch.setattr(HippoDB, "insert_pending_row", insert_pending_or_fail)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert not fpath.exists(), "original file must be renamed when any insert fails"
    failed_files = list(deferred_dir.glob("session-H-42.failed-*.jsonl"))
    assert len(failed_files) == 1, (
        f"expected 1 .failed-* file; got {failed_files} "
        f"(deferred_dir contents: {list(deferred_dir.iterdir())})"
    )

    assert counts["events_inserted"] == 2, counts
    assert counts["events_skipped_insert_failed"] == 1, counts
    assert counts["events_skipped_intentional"] == 0, counts
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 1, counts

    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "log file must record the insert-failed event"
    log_content = log_files[0].read_text()
    assert "insert-failed" in log_content
    assert "session-H" in log_content


def test_drain_intentional_skip_does_not_fail_file(iai_home):
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-I",
        [
            _make_event("ok this is a long enough event for the min-length gate"),
            {"cue": "x", "text": "tiny", "tier": "episodic", "role": "user",
             "ts": "2026-04-26T00:00:00Z"},
        ],
        ts_suffix=43,
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert not fpath.exists()
    assert list(deferred_dir.glob("*.failed-*.jsonl")) == []
    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 1, counts
    assert counts["events_skipped_intentional"] == 1, counts
    assert counts["events_skipped_insert_failed"] == 0, counts


def _spawn_daemon(sock_path: Path, store_dir: Path, home: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    env["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-drain-integration-pass"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _kill_daemon_by_socket(sock_path: Path) -> None:
    target = str(sock_path)
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            try:
                env = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("IAI_DAEMON_SOCKET_PATH") == target:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        p.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def test_daemon_main_drain_does_not_crash_on_bad_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-integration-pass")

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)
    store_dir = iai_dir / "lancedb"
    store_dir.mkdir(parents=True, exist_ok=True)
    deferred_dir = iai_dir / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    bad = deferred_dir / "session-G-99999.jsonl"
    bad.write_text(
        json.dumps({"version": 1, "session_id": "session-G",
                    "deferred_at": "2026-04-26T00:00:00Z", "cwd": "/tmp"}) + "\n"
        + "totally not JSON ===invalid===\n"
    )
    assert bad.exists()

    sock_dir = Path(f"/tmp/iai-drn-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    proc = None
    try:
        proc = _spawn_daemon(
            sock_path, store_dir, home=Path(os.environ["HOME"])
        )
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            f"daemon never bound socket within 30s; pid={proc.pid} "
            f"poll_status={proc.poll()}"
        )

        # The socket binds BEFORE the embedder builds and the startup drain
        # runs — socket-up no longer implies boot work done. Poll with a
        # deadline instead of a fixed sleep.
        deadline = time.monotonic() + 60.0
        while bad.exists() and time.monotonic() < deadline:
            assert proc.poll() is None, (
                f"daemon exited unexpectedly with code {proc.returncode} — "
                f"startup-drain probably propagated an exception"
            )
            time.sleep(0.25)

        assert proc.poll() is None, (
            f"daemon exited unexpectedly with code {proc.returncode} — "
            f"startup-drain probably propagated an exception"
        )
        assert not bad.exists(), (
            "malformed file should have been renamed away by drain"
        )
        failed = list(deferred_dir.glob("session-G-99999.failed-*.jsonl"))
        assert len(failed) == 1, (
            f"expected exactly 1 .failed-* file, got {failed}"
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
        _kill_daemon_by_socket(sock_path)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
        import keyring.core
        keyring.core._keyring_backend = None


def test_drain_streams_caps_and_preserves_remainder(iai_home, monkeypatch):
    """A single deferred file larger than the per-run cap is drained in bounded
    batches: each run processes at most the cap, the unread tail is preserved
    verbatim for the next run, no event is lost across runs, and the file is
    never materialized whole — the read step interleaves parse with read so its
    resident batch tracks the cap, not the file size."""
    import iai_mcp.capture as capture_mod
    from iai_mcp.capture import drain_deferred_captures

    # Shrink the per-run cap so the test stays fast while still exercising the
    # cap-and-remainder path with a file several times larger than the cap.
    cap = 5
    monkeypatch.setattr(capture_mod, "MAX_DRAIN_EVENTS_PER_RUN", cap)

    total_events = 12  # 2 full cap batches + a short final batch (5 + 5 + 2)
    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    events = [
        _make_event(
            f"distinct streamed drain event number {i:03d} padded to clear "
            f"the minimum-capture length gate with room to spare"
        )
        for i in range(total_events)
    ]
    fpath = _write_deferred_jsonl(deferred_dir, "session-stream", events, ts_suffix=7)
    assert fpath.exists()

    # Streaming proof: wrap the deferred file's handle so we can observe how many
    # lines have been pulled from it at the moment the first EVENT is parsed. A
    # read-all (list comprehension) drains every line before parsing the first
    # event; a streaming drain parses the first event after pulling only the
    # header + that one event line.
    real_open = Path.open
    lines_pulled_when_first_event_parsed: dict[str, int] = {}
    peak_lines_pulled_before_first_parse: dict[str, int] = {}

    class _CountingHandle:
        def __init__(self, inner):
            self._inner = inner
            self.lines_pulled = 0

        def __iter__(self):
            return self

        def __next__(self):
            line = next(self._inner)
            self.lines_pulled += 1
            return line

        def readline(self, *a, **k):
            line = self._inner.readline(*a, **k)
            if line:
                self.lines_pulled += 1
            return line

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    active_handle: dict[str, _CountingHandle] = {}

    def counting_open(self, *args, **kwargs):
        inner = real_open(self, *args, **kwargs)
        # Only wrap the deferred-capture jsonl we care about (the claimed file
        # carries a .processing- marker; match on the session stem).
        if "session-stream" in self.name and self.suffix == ".jsonl":
            handle = _CountingHandle(inner)
            active_handle["h"] = handle
            return handle
        return inner

    real_loads = json.loads
    first_event_seen = {"done": False}

    def spy_loads(s, *a, **k):
        obj = real_loads(s, *a, **k)
        # The first non-header object parsed is an event (the header is parsed
        # first; events carry a "text" field, the header does not).
        if (
            isinstance(obj, dict)
            and "text" in obj
            and not first_event_seen["done"]
            and "h" in active_handle
        ):
            first_event_seen["done"] = True
            peak_lines_pulled_before_first_parse["v"] = active_handle["h"].lines_pulled
        return obj

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(capture_mod.json, "loads", spy_loads)

    store = _open_isolated_store()

    # --- First drain: cap-bounded, remainder preserved ------------------------
    counts1 = drain_deferred_captures(store)
    assert counts1["events_inserted"] == cap, counts1
    assert counts1["files_drained"] == 1, counts1
    assert counts1["files_failed"] == 0, counts1

    # Streaming proven: when the first event was parsed, only header + 1 event
    # line had been pulled from the handle (== 2). A read-all would have pulled
    # all total_events + 1 lines before the first parse.
    assert first_event_seen["done"], "no event was ever parsed"
    assert peak_lines_pulled_before_first_parse["v"] <= 2, (
        "drain materialized the file before parsing the first event "
        f"(pulled {peak_lines_pulled_before_first_parse['v']} lines first)"
    )

    # Original file consumed; a .partial.jsonl remainder must exist with the rest.
    assert not fpath.exists(), "original deferred file must be unlinked after cap"
    partials = list(deferred_dir.glob("session-stream-7*.partial.jsonl"))
    assert len(partials) == 1, f"expected 1 .partial.jsonl, got {partials}"
    partial = partials[0]
    partial_lines = [
        ln for ln in partial.read_text().splitlines() if ln.strip()
    ]
    # header + (total_events - cap) event lines preserved.
    assert len(partial_lines) == 1 + (total_events - cap), partial_lines
    # Remainder header is byte-identical to the original header.
    assert json.loads(partial_lines[0])["session_id"] == "session-stream"
    # Remainder events are exactly events[cap:] in order (no loss, no reorder).
    remainder_texts = [json.loads(ln)["text"] for ln in partial_lines[1:]]
    assert remainder_texts == [e["text"] for e in events[cap:]], "remainder reordered/dropped"

    # Restore the partial to a plain .jsonl so the next drain claims it (the
    # daemon promotes .partial.jsonl back into the rotation between cycles).
    promoted = deferred_dir / "session-stream-7next.jsonl"
    partial.rename(promoted)

    # --- Second drain: next cap-bounded batch ---------------------------------
    counts2 = drain_deferred_captures(store)
    assert counts2["events_inserted"] == cap, counts2

    partials2 = list(deferred_dir.glob("session-stream-7next*.partial.jsonl"))
    assert len(partials2) == 1, partials2
    promoted2 = deferred_dir / "session-stream-7final.jsonl"
    partials2[0].rename(promoted2)

    # --- Third drain: finishes the tail (< cap remaining) ---------------------
    counts3 = drain_deferred_captures(store)
    assert counts3["events_inserted"] == total_events - 2 * cap, counts3
    assert not list(deferred_dir.glob("*.partial.jsonl")), "no remainder should survive"
    assert not list(deferred_dir.glob("session-stream*.jsonl")), "all files consumed"

    # No event lost: inserts across the three runs sum to the full file.
    assert (
        counts1["events_inserted"]
        + counts2["events_inserted"]
        + counts3["events_inserted"]
        == total_events
    )


def test_drain_dedup_before_embed_skips_pending_write(iai_home, monkeypatch):
    """A duplicate event on the drain path is skipped by the exact-key idem
    check BEFORE any pending-row write, and the embedder is never invoked at all
    during the drain (the two-phase drain defers all embedding)."""
    import iai_mcp.capture as capture_mod
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.embed import Embedder

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    dup_event = _make_event(
        "this exact turn appears twice and must dedup before any write or embed"
    )

    # First drain: stores the record (one pending-row write, no embed).
    f1 = _write_deferred_jsonl(deferred_dir, "session-dup", [dup_event], ts_suffix=100)
    store = _open_isolated_store()
    counts1 = drain_deferred_captures(store)
    assert counts1["events_inserted"] == 1, counts1
    assert not f1.exists()

    # Second drain: the SAME turn (same session/role/ts/text → same idem tag).
    f2 = _write_deferred_jsonl(deferred_dir, "session-dup", [dup_event], ts_suffix=200)

    pending_calls: list[str] = []
    real_pending = capture_mod._drain_write_pending

    def spy_pending(store, *, text, **kwargs):
        pending_calls.append(text)
        return real_pending(store, text=text, **kwargs)

    embed_calls: list[str] = []
    real_embed = Embedder.embed

    def spy_embed(self, text):
        embed_calls.append(text)
        return real_embed(self, text)

    monkeypatch.setattr(capture_mod, "_drain_write_pending", spy_pending)
    monkeypatch.setattr(Embedder, "embed", spy_embed)

    counts2 = drain_deferred_captures(store)

    # The duplicate is reinforced, never re-inserted.
    assert counts2["events_inserted"] == 0, counts2
    assert counts2["events_reinforced"] == 1, counts2
    # Dedup fired BEFORE the pending-row write: the write site was never reached
    # for the duplicate.
    assert pending_calls == [], (
        f"_drain_write_pending was called for a duplicate: {pending_calls}"
    )
    # The embedder is never touched on the drain path (two-phase deferral).
    assert embed_calls == [], f"embedder invoked during drain: {embed_calls}"
    assert not f2.exists()
