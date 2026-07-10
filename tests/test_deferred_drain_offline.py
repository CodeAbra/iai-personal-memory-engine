"""Offline deferred-capture backlog recovery — behavior contract.

These tests pin every observable behavior of the offline ``iai-mcp
deferred-drain`` operator command: enumeration of all three backlog file types
(plain ``.jsonl``, ``.crash-*.jsonl``, and the stranded ``.quarantine/*.jsonl``
sink), byte-bounded batching, dry-run inertness, zero-loss reconciliation,
idempotent re-run via the ``source_uuid`` idem-tag, quarantine consumption,
per-batch process isolation (one spawned child per batch), two-writer refusal
when a live daemon is detected, reconciliation-authority agreement between the
parent and the children, the AES-key process-boundary fence, and dead-PID
pre-flight reclaim.

All fixtures are hermetic: a tmp store under ``tmp_path / "hippo"`` and a
synthetic backlog under ``tmp_path / "deferred-captures"``. The real home store
is never touched. The synthetic backlog is small and reproduces the verified
live capture-file schema exactly (header line 0 with four keys, event lines
with six keys including ``source_uuid`` and ``ts`` on every event).
"""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

from tests._helpers import make_tmp_store, set_tmp_env

from iai_mcp.deferred_drain import (
    enumerate_backlog,
    plan_batches,
    run_deferred_drain,
)


# ---- synthetic-backlog helpers --------------------------------------------


def _write_backlog_file(path: Path, session_id: str, texts: list[str]) -> int:
    """Write one capture-backlog file reproducing the verified live schema.

    Line 0 is the header with the four production keys
    ``{cwd, deferred_at, session_id, version}`` (NO ``ts``, NO ``source_uuid``).
    Lines 1..N are events with the six production keys
    ``{cue, role, source_uuid, text, tier, ts}``. The ``source_uuid`` is
    deterministic per ``(session_id, index)`` so re-synthesizing the same texts
    reproduces identical ``source_uuid`` values — this is what makes a re-drain
    reinforce rather than double-insert, mirroring production.

    Returns the number of event lines written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "cwd": "/tmp/work",
        "deferred_at": "2026-01-01T00:00:00+00:00",
        "session_id": session_id,
        "version": 1,
    }
    lines = [json.dumps(header)]
    for i, text in enumerate(texts):
        event = {
            "cue": text,
            "role": "user",
            "source_uuid": f"{session_id}-{i}",
            "text": text,
            "tier": "episodic",
            "ts": "2026-01-01T00:00:00+00:00",
        }
        lines.append(json.dumps(event))
    path.write_text("\n".join(lines) + "\n")
    return len(texts)


def _make_three_type_backlog(deferred_dir: Path) -> int:
    """Create one plain, one crash, and one quarantine backlog file.

    Each file carries a distinct ``session_id`` and unique-content texts, so
    every ``source_uuid`` is globally unique across the corpus. Returns the
    total event count written across all three files.
    """
    deferred_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    total += _write_backlog_file(
        deferred_dir / "plain.jsonl",
        "sess-plain",
        [
            "plainzzz alpha memory line one for drain",
            "plainzzz beta memory line two for drain",
        ],
    )
    total += _write_backlog_file(
        deferred_dir / "plain.crash-1.jsonl",
        "sess-crash",
        [
            "crashzzz gamma memory line one for drain",
            "crashzzz delta memory line two for drain",
        ],
    )
    quarantine_dir = deferred_dir / ".quarantine"
    total += _write_backlog_file(
        quarantine_dir / "20260101T000000Z-plain.jsonl",
        "sess-quar",
        [
            "quarzzz epsilon memory line one for drain",
            "quarzzz zeta memory line two for drain",
        ],
    )
    return total


def _all_files(deferred_dir: Path) -> set[Path]:
    """Snapshot every file under the backlog dir, recursively."""
    return {p for p in deferred_dir.rglob("*") if p.is_file()}


def _surfaces(tmp_path: Path) -> list[str]:
    store = make_tmp_store(tmp_path)
    try:
        return [r.literal_surface for r in store.all_records()]
    finally:
        store.close()


def _spawn_dead_pid() -> int:
    """Spawn a Python child that exits immediately; return its (now-dead) pid."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ---- enumeration + dry-run -------------------------------------------------


def test_enumerate_includes_all_three_types(tmp_path):
    deferred_dir = tmp_path / "deferred-captures"
    _make_three_type_backlog(deferred_dir)

    # Decoys that must NOT be enumerated.
    _write_backlog_file(
        deferred_dir / "live.live.jsonl", "sess-live", ["live decoy line should be skipped"]
    )
    _write_backlog_file(
        deferred_dir / "plain.processing-99999.jsonl",
        "sess-proc",
        ["processing decoy line should be skipped"],
    )

    files = enumerate_backlog(deferred_dir)
    names = {p.name for p in files}

    assert "plain.jsonl" in names
    assert "plain.crash-1.jsonl" in names
    assert "20260101T000000Z-plain.jsonl" in names
    quarantine_paths = {p for p in files if p.parent.name == ".quarantine"}
    assert quarantine_paths, "quarantine file must be enumerated"

    assert "live.live.jsonl" not in names
    assert "plain.processing-99999.jsonl" not in names

    # Deterministic ordering.
    assert enumerate_backlog(deferred_dir) == files


def test_dry_run_mutates_nothing_and_reports_counts(tmp_path, monkeypatch):
    deferred_dir = tmp_path / "deferred-captures"
    total_events = _make_three_type_backlog(deferred_dir)
    before = _all_files(deferred_dir)

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=True,
    )

    assert _all_files(deferred_dir) == before, "dry-run must remove nothing"
    assert result["dry_run"] is True
    assert result["events_seen"] == total_events
    assert result["inserted"] == 0
    assert result["batches"], "batch plan must be present in dry-run"
    assert result["total_bytes"] > 0
    assert make_tmp_store(tmp_path).active_records_count() == 0


# ---- real-run zero-loss, idempotency, byte-bound, AES-fence ----------------


def test_real_run_drains_all_three_types_zero_loss(tmp_path, monkeypatch):
    deferred_dir = tmp_path / "deferred-captures"
    total_events = _make_three_type_backlog(deferred_dir)

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )

    assert result["reconciled"] is True
    assert result["events_seen"] == total_events
    assert result["inserted"] == result["events_seen"]
    assert result["remaining_events"] == 0
    assert result["count_after"] == result["count_before"] + result["inserted"]

    surfaces = " ".join(_surfaces(tmp_path))
    assert "plainzzz" in surfaces
    assert "crashzzz" in surfaces
    assert "quarzzz" in surfaces


def test_reconciliation_authority_agrees(tmp_path, monkeypatch):
    deferred_dir = tmp_path / "deferred-captures"
    _make_three_type_backlog(deferred_dir)

    # A header-only file and an empty file contribute exactly 0 events to both
    # the parent's denominator and the children's re-count.
    _write_backlog_file(deferred_dir / "headeronly.jsonl", "sess-empty", [])
    (deferred_dir / "blank.jsonl").write_text("")

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )

    assert result["events_seen"] == result["child_events_seen"]
    assert result["reconciled"] is True


def test_quarantine_files_are_consumed(tmp_path, monkeypatch):
    """Regression guard: the stranded ``.quarantine/`` sink must be drained."""
    deferred_dir = tmp_path / "deferred-captures"
    quarantine_dir = deferred_dir / ".quarantine"
    events = _write_backlog_file(
        quarantine_dir / "20260101T000000Z-only.jsonl",
        "sess-quar-only",
        [
            "quaronly token one for the stranded sink",
            "quaronly token two for the stranded sink",
        ],
    )

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )

    assert result["events_seen"] == events
    assert result["inserted"] == result["events_seen"]
    assert "quaronly" in " ".join(_surfaces(tmp_path))


def test_rerun_is_idempotent(tmp_path, monkeypatch):
    # Idempotency rests on source_uuid: real events always carry it (20/20 live),
    # so _idem_tag keys on session_id|role|source_uuid. On a re-drain the
    # pre-embed idem check catches every already-present event and skips it
    # before embedding, so a re-run neither double-inserts nor re-reinforces.
    deferred_dir = tmp_path / "deferred-captures"
    total_events = _make_three_type_backlog(deferred_dir)

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    first = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )
    assert first["inserted"] == total_events
    count_after_first = make_tmp_store(tmp_path).active_records_count()

    # Re-synthesize the identical backlog: same session_id + same deterministic
    # source_uuid per (session_id, index).
    _make_three_type_backlog(deferred_dir)

    second = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )

    assert second["inserted"] == 0
    assert second["reinforced"] == 0
    assert second["skipped_existing"] == second["events_seen"]
    assert second["reconciled"] is True
    assert make_tmp_store(tmp_path).active_records_count() == count_after_first


def test_batches_are_byte_bounded(tmp_path):
    deferred_dir = tmp_path / "deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    # Ten small files of a few KB each.
    sizes = []
    for n in range(10):
        text = f"bytebound token {n} " + ("padding word " * 200)
        path = deferred_dir / f"small-{n}.jsonl"
        _write_backlog_file(path, f"sess-bb-{n}", [text])
        sizes.append(path.stat().st_size)

    # One file deliberately larger than batch_bytes (its own over-budget batch).
    big_text = "bytebound big " + ("padding word " * 4000)
    big = deferred_dir / "big.jsonl"
    _write_backlog_file(big, "sess-bb-big", [big_text])

    batch_bytes = max(sizes) * 2
    assert big.stat().st_size > batch_bytes, "big file must exceed batch_bytes"

    files = enumerate_backlog(deferred_dir)
    batches = plan_batches(files, batch_bytes)

    assert len(batches) > 1
    for batch in batches:
        if len(batch) > 1:
            assert sum(p.stat().st_size for p in batch) <= batch_bytes

    # The over-budget file is allowed as its own single-file batch.
    assert any(len(b) == 1 and b[0] == big for b in batches)

    # Every enumerated file appears in exactly one batch.
    flat = [p for b in batches for p in b]
    assert sorted(flat, key=str) == sorted(files, key=str)
    assert len(flat) == len(set(flat))


# ---- process isolation, two-writer refusal, AES-fence, dead-PID ------------


def test_large_corpus_spawns_distinct_child_processes(tmp_path, monkeypatch):
    deferred_dir = tmp_path / "deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    sizes = []
    for n in range(6):
        text = f"isozzz token {n} for the per-batch isolation drain test"
        path = deferred_dir / f"iso-{n}.jsonl"
        _write_backlog_file(path, f"sess-iso-{n}", [text])
        sizes.append(path.stat().st_size)

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    # Tiny batch_bytes => the plan cuts the corpus into several batches.
    batch_bytes = max(sizes) * 2
    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        batch_bytes=batch_bytes,
        dry_run=False,
    )

    assert len(result["batches"]) >= 2
    assert len(set(result["child_pids"])) >= 2, (
        "each byte-bounded batch must run in its own spawned child process"
    )
    assert result["reconciled"] is True
    assert result["remaining_events"] == 0


def test_live_daemon_forces_refusal_without_force(tmp_path, monkeypatch):
    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    import iai_mcp.migrate._dead_pid_unlock as dpu
    import iai_mcp.deferred_drain as dd
    from iai_mcp.cli._maintenance import cmd_deferred_drain

    # A live daemon is detected.
    monkeypatch.setattr(dpu, "_read_live_daemon_pid", lambda: 12345)

    called = {"spawned": False}

    def _must_not_run(**kwargs):
        called["spawned"] = True
        raise AssertionError("must not spawn when a live daemon is detected")

    monkeypatch.setattr(dd, "run_deferred_drain", _must_not_run)

    import argparse

    ns = argparse.Namespace(
        dry_run=False,
        json=True,
        batch_bytes=100 * 1024 * 1024,
        force=False,
    )
    rc = cmd_deferred_drain(ns)
    assert rc != 0
    assert called["spawned"] is False, "no child must spawn under live-daemon refusal"

    # --force bypasses the refusal and relies on the store flock.
    def _stub_ok(**kwargs):
        return {
            "dry_run": False,
            "files_enumerated": 0,
            "events_seen": 0,
            "total_bytes": 0,
            "batches": [],
            "inserted": 0,
            "reinforced": 0,
            "skipped": 0,
            "count_before": 0,
            "count_after": 0,
            "remaining_events": 0,
            "child_events_seen": 0,
            "child_pids": [],
            "reconciled": True,
        }

    monkeypatch.setattr(dd, "run_deferred_drain", _stub_ok)
    ns.force = True
    assert cmd_deferred_drain(ns) == 0


def test_child_constructs_own_store_key_not_passed_as_value():
    """Structural guard: the AES key never crosses the spawn pipe."""
    import iai_mcp.deferred_drain_worker as w

    src = inspect.getsource(w)

    assert "MemoryStore(" in src, "child must construct its own store"
    assert "store_root" in src, "child receives a store-root path"

    for forbidden in ("crypto_key", "aes_key", ".crypto.key", "key_bytes"):
        assert forbidden not in src, f"key material must not appear in the worker: {forbidden}"

    # The heavy imports live inside the worker body; the only module-level
    # import is `from __future__ import annotations`. Parse with AST so prose in
    # the module docstring (which mentions the word "import") is not mistaken for
    # a real import statement.
    tree = ast.parse(src)
    module_imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        isinstance(n, ast.ImportFrom) and n.module == "__future__"
        for n in module_imports
    ), f"unexpected module-level import: {module_imports}"


def test_stale_processing_marker_unlocked_preflight(tmp_path, monkeypatch):
    deferred_dir = tmp_path / "deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    dead_pid = _spawn_dead_pid()
    _write_backlog_file(
        deferred_dir / f"stranded.processing-{dead_pid}.jsonl",
        "sess-stranded",
        [
            "strandedzzz token recovered by the dead-pid pre-flight",
            "strandedzzz token two recovered by pre-flight",
        ],
    )

    set_tmp_env(monkeypatch, tmp_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))

    result = run_deferred_drain(
        deferred_dir=deferred_dir,
        store_root=tmp_path / "hippo",
        dry_run=False,
    )

    assert result["inserted"] >= 2
    assert "strandedzzz" in " ".join(_surfaces(tmp_path))


# ---- pre-embed idem skip for already-present records -----------------------


def test_pre_embed_skip_does_not_embed_duplicate(tmp_path, monkeypatch):
    """An already-present event is skipped before embedding; only the new one embeds.

    Crash-rotation backlogs are >99% duplicates of records already in the store.
    The per-event loop must catch those by idem-tag and skip them BEFORE the
    expensive embed, both to save the work and to avoid re-reinforcing each
    duplicate. This drains a backlog holding one already-stored event plus one
    genuinely-new event and asserts the embedder is invoked for the new text but
    never for the duplicate's text.
    """
    import iai_mcp.embed as embed_mod
    from iai_mcp.deferred_drain_worker import _drain_files

    store = make_tmp_store(tmp_path)

    dup_text = "duplicate already-present line that must not be re-embedded"
    new_text = "genuinely new line that must be embedded once and inserted"
    session_id = "sess-preembed"
    role = "user"
    ts = "2026-01-01T00:00:00+00:00"
    dup_uuid = f"{session_id}-0"

    # Seed the store with the duplicate via the same path the worker uses, so its
    # idem-tag is exactly the tag the pre-check will recompute.
    from iai_mcp.capture import capture_turn

    seed = capture_turn(
        store,
        cue=dup_text,
        text=dup_text,
        tier="episodic",
        session_id=session_id,
        role=role,
        ts=ts,
        source_uuid=dup_uuid,
    )
    assert seed["status"] == "inserted"

    # Wrap the real embedder so every embedded surface is recorded and embedding
    # the duplicate's text is an outright failure.
    real_factory = embed_mod.embedder_for_store
    embedded: list[str] = []

    class _SpyEmbedder:
        def __init__(self, inner):
            self._inner = inner

        def embed(self, text):
            embedded.append(text)
            if dup_text in text:
                raise AssertionError(
                    "the already-present duplicate must never be embedded"
                )
            return self._inner.embed(text)

    monkeypatch.setattr(
        embed_mod,
        "embedder_for_store",
        lambda s: _SpyEmbedder(real_factory(s)),
    )

    backlog = tmp_path / "deferred-captures" / "preembed.jsonl"
    _write_backlog_file(backlog, session_id, [dup_text, new_text])

    try:
        tally = _drain_files(store, [str(backlog)])
    finally:
        store.close()

    assert tally["skipped_existing"] == 1
    assert tally["inserted"] == 1
    assert tally["reinforced"] == 0
    assert tally["events_seen"] == 2
    assert (
        tally["inserted"]
        + tally["reinforced"]
        + tally["skipped_existing"]
        + tally["skipped"]
        == tally["events_seen"]
    )

    # The new text was embedded exactly once; the duplicate's text never was.
    assert any(new_text in t for t in embedded)
    assert not any(dup_text in t for t in embedded)
