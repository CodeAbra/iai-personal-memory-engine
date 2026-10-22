"""Bounded retention + deterministic rotation for the command-output store.

Hermetic isolation, binding for every test here: every index call threads
an EXPLICIT `store_root = tmp_path / "tools-output"` -- the dir name
`tools-output` is what makes the structural secret-gate predicate fire
(see `iai_mcp.tools.is_command_output_store_root`), so without it the gate
never engages and a rotation-reapplies-the-gate assertion would prove
nothing. `default_command_output_store_root()` is never called in-process
-- it is bound to `Path.home()` at import time.

Run directories here are hand-built (mkdir + meta.json/stdout.log/
stderr.log matching the real daemon-written shape) rather than run through
the executor daemon -- this file tests the retention/rotation MECHANISM
over already-persisted artifact files, not the daemon's own write path
(covered by tests/test_command_output_insight.py and
tests/test_command_output_index.py).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from iai_mcp.tools import search
from iai_mcp.tools import output_index
from iai_mcp.tools.output_index import index_command_output, load_retention_manifest


def _make_run(
    artifacts_root: Path,
    run_id: str,
    *,
    started_at: int,
    stdout_text: str = "",
    stderr_text: str = "",
) -> Path:
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text(stdout_text)
    (run_dir / "stderr.log").write_text(stderr_text)
    meta = {
        "run_id": run_id,
        "cmd": "echo",
        "args": [stdout_text],
        "pid": 1,
        "pgid": 1,
        "started_at": started_at,
        "exit_code": 0,
        "stdout_offset": 0,
        "stderr_offset": 0,
        "rlimits_applied": True,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


def test_retention_cap_bounds_indexed_runs(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    cap = 3

    run_dirs = [
        _make_run(artifacts_root, f"run-{i}", started_at=1000 + i, stdout_text=f"marker-{i}")
        for i in range(6)
    ]

    for run_dir in run_dirs:
        index_command_output(run_dir, store_root=output_store, retention_max_runs=cap)

    manifest = load_retention_manifest(output_store)
    assert len(manifest) == cap

    retained_ids = {e["run_id"] for e in manifest}
    assert retained_ids == {"run-3", "run-4", "run-5"}, "oldest runs must be dropped first"


def test_rotation_leaves_no_stale_postings(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    cap = 3

    # Fully distinct per-run markers (no shared word stems) -- a partial
    # BM25 match on a shared substring would otherwise make a dropped
    # run's query hit a retained run's file that merely shares a stem.
    markers = [uuid.uuid4().hex for _ in range(6)]
    run_dirs = [
        _make_run(artifacts_root, f"run-{i}", started_at=2000 + i, stdout_text=markers[i])
        for i in range(6)
    ]

    for run_dir in run_dirs:
        index_command_output(run_dir, store_root=output_store, retention_max_runs=cap)

    # Dropped runs (0, 1, 2) must have no postings left in the rebuilt store.
    for i in (0, 1, 2):
        hits = search(markers[i], store_root=output_store)
        assert hits == [], f"rotated-out run-{i} must leave no stale posting"

    # Retained runs (3, 4, 5) must still be searchable.
    for i in (3, 4, 5):
        hits = search(markers[i], store_root=output_store)
        assert hits, f"retained run-{i} must still be searchable after rotation"


def test_rotation_reapplies_secret_gate(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    cap = 3
    secret_value = "api_key=sk-alice-fixture-rotation-0000000000"

    # A secret-shaped run whose started_at is the newest of the whole
    # corpus, so it stays in the newest-N retained set through every
    # rotation below.
    secret_run = _make_run(
        artifacts_root, "secret-run", started_at=9000, stdout_text=secret_value
    )
    index_command_output(secret_run, store_root=output_store, retention_max_runs=cap)

    hits = search("alice-fixture-rotation", store_root=output_store)
    assert hits == [], "secret-shaped output must never be indexed in the first place"

    # Push the manifest past the cap with older, clean runs -- secret_run
    # (started_at=9000) stays in the newest-N set throughout.
    for i in range(5):
        run_dir = _make_run(
            artifacts_root, f"clean-run-{i}", started_at=1000 + i, stdout_text=f"clean-{i}"
        )
        index_command_output(run_dir, store_root=output_store, retention_max_runs=cap)

    manifest = load_retention_manifest(output_store)
    assert "secret-run" in {e["run_id"] for e in manifest}, (
        "the secret-bearing run must be retained by the started_at ordering "
        "for this assertion to be meaningful"
    )

    hits_after_rotation = search("alice-fixture-rotation", store_root=output_store)
    assert hits_after_rotation == [], (
        "a retained run's secret-shaped artifact must stay absent from the "
        "rebuilt store -- rotation must route through the same structural gate"
    )


def test_retention_is_opt_in_consistent(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    source_store = tmp_path / "tools"  # NOT named "tools-output" -- the code store

    run_a = _make_run(artifacts_root, "run-a", started_at=100, stdout_text="alpha-content")
    index_command_output(run_a, store_root=output_store)

    segment_files_before = sorted(output_store.glob("*.lcidx"))
    assert segment_files_before, "the first index call must publish at least one segment"
    bytes_before = {p.name: p.read_bytes() for p in segment_files_before}

    run_b = _make_run(artifacts_root, "run-b", started_at=200, stdout_text="beta-content")
    index_command_output(run_b, store_root=output_store)  # well below the default cap

    # No rotation fired: every pre-existing segment file is untouched, byte-identical.
    for name, content in bytes_before.items():
        assert (output_store / name).read_bytes() == content, (
            "a below-cap index must never rewrite a pre-existing segment"
        )

    segment_files_after = sorted(output_store.glob("*.lcidx"))
    assert len(segment_files_after) > len(segment_files_before), (
        "a below-cap index still publishes its own new segment"
    )

    # Retention never touches any store other than the one it was called on.
    assert not source_store.exists(), "retention must never create/touch the source-code store"


def test_fully_refused_run_does_not_consume_retention_slot(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    cap = 2
    secret_value = "api_key=sk-alice-fixture-fully-refused-0000000000"

    refused_run = _make_run(
        artifacts_root,
        "refused-run",
        started_at=500,
        stdout_text=secret_value,
        stderr_text=secret_value,
    )
    report = index_command_output(refused_run, store_root=output_store, retention_max_runs=cap)
    assert report["indexed"] == 0
    assert len(report["refused"]) == 2

    assert load_retention_manifest(output_store) == [], (
        "a run refused in full must not consume a retention slot"
    )

    clean_dirs = [
        _make_run(artifacts_root, f"clean-{i}", started_at=600 + i, stdout_text=f"clean-{i}")
        for i in range(cap)
    ]
    for clean_dir in clean_dirs:
        index_command_output(clean_dir, store_root=output_store, retention_max_runs=cap)

    manifest_after = load_retention_manifest(output_store)
    assert len(manifest_after) == cap, "the earlier fully-refused run must not have shrunk cap headroom"
    assert {e["run_id"] for e in manifest_after} == {f"clean-{i}" for i in range(cap)}


def test_concurrent_index_calls_do_not_lose_manifest_entries(
    tmp_path: Path, monkeypatch
) -> None:
    """Widens the read-modify-write race window by sleeping right
    after the manifest is loaded, inside the locked section. Without the
    fix's `_retention_lock`, this reliably reproduces a lost update: two
    threads both read the same starting manifest, then both append and
    overwrite, and one entry vanishes. With the lock, the sleep merely
    serializes callers and every entry survives.
    """
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"
    run_count = 8

    original_load = output_index.load_retention_manifest

    def slow_load(store_root: Path) -> "list[dict]":
        entries = original_load(store_root)
        time.sleep(0.02)
        return entries

    monkeypatch.setattr(output_index, "load_retention_manifest", slow_load)

    run_dirs = [
        _make_run(artifacts_root, f"run-{i}", started_at=1000 + i, stdout_text=f"marker-{i}")
        for i in range(run_count)
    ]

    threads = [
        threading.Thread(
            target=index_command_output,
            args=(run_dir,),
            kwargs={"store_root": output_store, "retention_max_runs": 100},
        )
        for run_dir in run_dirs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest = load_retention_manifest(output_store)
    assert len(manifest) == run_count, (
        f"expected {run_count} manifest entries, got {len(manifest)} -- "
        "a lost update means the lock did not serialize concurrent callers"
    )
    assert {e["run_id"] for e in manifest} == {rd.name for rd in run_dirs}


def test_retention_manifest_roundtrips(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_store = tmp_path / "tools-output"

    run_a = _make_run(artifacts_root, "run-a", started_at=300, stdout_text="first")
    run_b = _make_run(artifacts_root, "run-b", started_at=400, stdout_text="second")

    index_command_output(run_a, store_root=output_store)
    manifest_after_first = load_retention_manifest(output_store)
    assert manifest_after_first == [
        {
            "run_id": "run-a",
            "started_at": 300,
            "stdout_path": str(run_a / "stdout.log"),
            "stderr_path": str(run_a / "stderr.log"),
        }
    ]

    index_command_output(run_b, store_root=output_store)

    # A fresh read (simulating a re-open) must see both entries, in order.
    manifest_after_second = load_retention_manifest(output_store)
    assert [e["run_id"] for e in manifest_after_second] == ["run-a", "run-b"]
    assert [e["started_at"] for e in manifest_after_second] == [300, 400]
