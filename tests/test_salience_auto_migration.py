"""One-time, marker-gated activation of ``backfill_salience`` from the sleep
pipeline. Hermetic only -- no live store is ever opened by this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep
from iai_mcp.migrate._salience_auto_migration import (
    _DONE_MARKER_KEY,
    _NATIVE_KEY,
    run_salience_auto_migration,
    stamp_native_if_fresh,
)
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _make_pipeline(store, tmp_path: Path) -> SleepPipeline:
    """Every construction is scoped to tmp_path -- a pipeline built with the
    default (real) lifecycle-state path would read/write the actual
    ~/.iai-mcp/lifecycle_state.json, which no test may touch."""
    return SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
    )


def _emb(i: int) -> list[float]:
    vec = [0.0] * EMBED_DIM
    vec[i % EMBED_DIM] = 1.0
    vec[(i * 37 + 11) % EMBED_DIM] = 0.5
    return vec


def _rec(i: int, *, text: str, directive: bool = False) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_emb(i),
        community_id=None,
        centrality=0.0,
        detail_level=2,
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
        directive=directive,
        epistemic_status="unknown",
        salience_level="unflagged",
    )


def _delete_native_stamp(store: MemoryStore) -> None:
    with store.db._conn_lock:
        store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (_NATIVE_KEY,),
        )
        store.db._conn.commit()


def _seed_legacy_corpus(store: MemoryStore) -> list[MemoryRecord]:
    """A pre-producer corpus: every row landed as ``unflagged`` with no
    entity/AAAK coverage, as if captured before salience classification and
    the native stamp both existed. One directive row must be promotable by
    the composite.

    The store's own construction stamps it native (zero live records at
    open) before any row is inserted here -- deleting that stamp after
    seeding is what makes the store genuinely legacy-shaped: content
    present, no provenance stamp, no done marker."""
    rows = [
        _rec(0, text="deploy the release before the end of the day", directive=True),
        _rec(1, text="alice prefers tea over coffee"),
        _rec(2, text="bob asked about the quarterly roadmap"),
    ]
    for r in rows:
        store.insert(r)
    flush_record_buffer(store)
    # Force the legacy shape explicitly: clear any capture-time salience/
    # entity signal a fresh insert might already have earned, simulating a
    # corpus that predates classify_salience.
    with store.db._conn_lock:
        placeholders = ",".join("?" for _ in rows)
        store.db._conn.execute(
            "UPDATE records SET salience_level = 'unflagged', aaak_index = ''"
            f" WHERE id IN ({placeholders})",
            [str(r.id) for r in rows],
        )
        store.db._conn.commit()
    _delete_native_stamp(store)
    return rows


def _marker_row(store: MemoryStore, key: str):
    with store.db._conn_lock:
        return store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (key,),
        ).fetchone()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fresh_store_takes_zero_backfill(driver, tmp_path, monkeypatch):
    """The load-bearing acceptance test: a store born under a
    salience-aware binary (native stamp written at zero-live-record
    construction) whose captured records are ALL legitimately unflagged
    must take zero pre-write snapshot and zero backfill sweep on its first
    sleep cycle. Fails red against a gate keyed on the classification value,
    which cannot tell a never-migrated legacy row from a legitimately
    unflagged modern one."""
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store = MemoryStore(path=store_root)
    assert _marker_row(store, _NATIVE_KEY) is not None, (
        "construction at zero live records must stamp the store native"
    )

    for i, text in enumerate([
        "alice prefers tea over coffee",
        "bob asked about the quarterly roadmap",
        "the weather in the office is fine today",
    ]):
        store.insert(_rec(i, text=text))
    flush_record_buffer(store)

    def _raiser(*_a, **_kw):
        raise AssertionError(
            "backfill_salience must not be called for a native-stamped store",
        )

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    pipeline = _make_pipeline(store, tmp_path)
    done, payload = pipeline._step_salience_auto_migration(None)

    assert done is True
    assert payload.get("status") == "not_needed"

    snapshot_dirs = list(store_root.glob("hippo-pre-salience-backfill-*"))
    assert snapshot_dirs == [], (
        "a native-stamped store must never take a pre-write snapshot"
    )
    assert _marker_row(store, _DONE_MARKER_KEY) is None
    assert _marker_row(store, _NATIVE_KEY) is not None


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stamp_native_refrains_on_content_store(driver, tmp_path, monkeypatch):
    """The inverse of the acceptance test, and the only test that exercises
    the branch that keeps real legacy stores migrating: a store carrying
    live records with neither marker present must NOT be stamped native by
    a direct call. A wrong refrain branch would stamp every content-bearing
    store native, silently no-op every real legacy store, and the
    fresh-store test above would still pass (it fabricates its own
    precondition)."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)
    assert _marker_row(store, _NATIVE_KEY) is None

    stamp_native_if_fresh(store)

    assert _marker_row(store, _NATIVE_KEY) is None, (
        "a store carrying live content must never be stamped native"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stamp_native_refrains_on_store_with_only_tombstoned_rows(
    driver, tmp_path, monkeypatch,
):
    """A store whose only rows are tombstoned (zero LIVE rows, but content
    physically present in ``records``) must NOT be stamped native. The
    probe checks for ANY row, not just live ones: a tombstoned row can
    later be restored (e.g. ``compact_hippo``'s pinned/never-decay
    untombstone), so a live-only probe re-sampled on every store open could
    land in a transient zero-live-row window, wrongly stamp the store
    native, and the later-restored row would then never take the one-time
    backfill."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    rows = _seed_legacy_corpus(store)
    assert _marker_row(store, _NATIVE_KEY) is None

    now_iso = datetime.now(timezone.utc).isoformat()
    with store.db._conn_lock:
        placeholders = ",".join("?" for _ in rows)
        store.db._conn.execute(
            f"UPDATE records SET tombstoned_at = ?, live = 0 "
            f"WHERE id IN ({placeholders})",
            [now_iso] + [str(r.id) for r in rows],
        )
        store.db._conn.commit()

    stamp_native_if_fresh(store)

    assert _marker_row(store, _NATIVE_KEY) is None, (
        "a store that has ever held a row -- live or tombstoned -- must "
        "never be stamped native"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_auto_migration_runs_once_on_legacy_corpus(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    rows = _seed_legacy_corpus(store)

    assert _marker_row(store, _NATIVE_KEY) is None, (
        "a pre-producer store must carry no native stamp"
    )
    assert _marker_row(store, _DONE_MARKER_KEY) is None, (
        "no done marker must exist before the first run"
    )

    pipeline = _make_pipeline(store, tmp_path)

    # interrupt_check=None short-circuits _check_interrupt before any state
    # I/O, so this exercises exactly the migration gate + backfill call.
    done, payload = pipeline._step_salience_auto_migration(None)

    assert done is True
    assert payload.get("status") == "migrated"
    assert payload["salience_promoted"] >= 1

    marker = _marker_row(store, _DONE_MARKER_KEY)
    assert marker is not None, "done marker must be set after a successful run"

    directive_row = store.get(rows[0].id)
    assert directive_row is not None
    assert directive_row.salience_level == "notable", (
        "the directive row must be promoted off 'unflagged' by the composite"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_gate_short_circuits_before_any_backfill_call(driver, tmp_path, monkeypatch):
    """Marker-present path never calls backfill_salience -- proven by
    patching it to raise if invoked."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    with store.db._conn_lock:
        store.db._conn.execute(
            "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (_DONE_MARKER_KEY, "1"),
        )
        store.db._conn.commit()

    def _raiser(*_a, **_kw):
        raise AssertionError(
            "backfill_salience must not be called when the marker is present",
        )

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    result = run_salience_auto_migration(store)
    assert result == {"status": "already_migrated"}


def test_gate_never_probes_salience_level_value():
    """The gate consults only provenance markers -- never the classification
    value, which cannot distinguish a never-migrated row from a
    legitimately-unflagged modern one (the defect this redesign closes).

    Walks the AST rather than substring-matching the source text: a plain
    substring check only catches one exact spelling of the defective
    predicate and would pass right past 'salience_level=?' or an IN(...)
    variant. String literals inside docstrings/comments are not AST string
    constants in executable positions the same way SQL literals are, but to
    stay unambiguous this scans every string constant in the module and
    fails on ANY occurrence of the identifier -- so the invariant is "the
    identifier never appears in code the interpreter evaluates as data",
    which is what a SQL predicate is and a docstring's prose is not (prose
    lives in the docstring's own leading string, extracted separately)."""
    import ast
    import inspect

    from iai_mcp.migrate import _salience_auto_migration as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc_node = ast.get_docstring(node, clean=False)
            if doc_node is not None:
                first_stmt = node.body[0] if node.body else None
                if isinstance(first_stmt, ast.Expr) and isinstance(
                    getattr(first_stmt, "value", None), ast.Constant,
                ):
                    docstring_nodes.add(id(first_stmt.value))

    offending = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "salience_level" in node.value
        and id(node) not in docstring_nodes
    ]
    assert offending == [], (
        "the gate must never key on the classification value in any "
        f"non-docstring string literal (SQL predicate or otherwise): {offending!r}"
    )
    assert "SELECT 1 FROM records LIMIT 1" in source, (
        "the provenance probe must stay a bounded LIMIT 1 existence check"
    )


def test_sleep_step_present_in_all_five_structures(tmp_path):
    from iai_mcp.lilli.cycle.sleep_pipeline import STEP_PHASE, SleepPhase, _LIVENESS_SPEC

    step = SleepStep.SALIENCE_AUTO_MIGRATION
    assert step.value == 21
    assert STEP_PHASE[step] == SleepPhase.NREM
    assert _LIVENESS_SPEC[step] == ("records_scanned", "salience_promoted")
    assert SleepPipeline._STEP_ORDER[-1] == step

    pipeline = _make_pipeline(None, tmp_path)
    assert pipeline._step_methods[step] == pipeline._step_salience_auto_migration
    assert callable(SleepPipeline._step_salience_auto_migration)


def test_interrupt_returns_false_empty_payload(tmp_path):
    """An interrupt at step entry mirrors every sibling step's contract:
    (False, {}) leaves the resume index at the prior step, which already
    saved its own progress -- so the next cycle re-enters only this tail
    step without skipping or losing any earlier work."""
    pipeline = _make_pipeline(None, tmp_path)
    done, payload = pipeline._step_salience_auto_migration(lambda: True)
    assert done is False
    assert payload == {}


def test_no_store_never_raises(tmp_path):
    pipeline = _make_pipeline(None, tmp_path)
    done, payload = pipeline._step_salience_auto_migration(None)
    assert done is True
    assert payload.get("status") == "no_store"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_manual_apply_sets_done_marker(driver, tmp_path, monkeypatch):
    """Manual and auto migration converge on one marker: a successful
    operator --apply run sets the same done marker the sleep-step gate
    reads, so neither path re-triggers the other's backfill. A dry-run
    must never write it."""
    from iai_mcp.cli._maintenance import cmd_salience_backfill

    _select_driver(driver, monkeypatch)

    apply_root = tmp_path / "apply_store"
    store = MemoryStore(path=apply_root)
    _seed_legacy_corpus(store)
    assert _marker_row(store, _DONE_MARKER_KEY) is None

    rc = cmd_salience_backfill(
        argparse.Namespace(apply=True, store_path=str(apply_root)),
    )
    assert rc == 0
    assert _marker_row(store, _DONE_MARKER_KEY) is not None, (
        "a successful manual apply must set the same marker the sleep-step gate reads"
    )

    dry_root = tmp_path / "dry_store"
    dry_store = MemoryStore(path=dry_root)
    _seed_legacy_corpus(dry_store)

    rc2 = cmd_salience_backfill(
        argparse.Namespace(apply=False, store_path=str(dry_root)),
    )
    assert rc2 == 0
    assert _marker_row(dry_store, _DONE_MARKER_KEY) is None, (
        "a dry-run must never write the done marker"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_manual_apply_on_native_store_never_sets_done_marker(driver, tmp_path, monkeypatch):
    """The done and native markers are mutually exclusive: an operator
    running --apply against a store that is already born-native must not
    end up with both markers set. The doctor must then still report the
    native state, not a done-and-hidden run."""
    from iai_mcp.cli._maintenance import cmd_salience_backfill

    _select_driver(driver, monkeypatch)

    store_root = tmp_path / "native_store"
    store = MemoryStore(path=store_root)
    assert _marker_row(store, _NATIVE_KEY) is not None

    rc = cmd_salience_backfill(
        argparse.Namespace(apply=True, store_path=str(store_root)),
    )
    assert rc == 0
    assert _marker_row(store, _DONE_MARKER_KEY) is None, (
        "--apply on a born-native store must never write the done marker"
    )
    assert _marker_row(store, _NATIVE_KEY) is not None

    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    from iai_mcp.doctor import check_ll_salience_auto_migration

    result = check_ll_salience_auto_migration()
    assert result.status == "PASS"
    assert "not needed" in result.detail, (
        "the doctor must still report the native state, not a hidden 'done'"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_second_run_after_success_is_zero_mutation_noop(driver, tmp_path, monkeypatch):
    """After a successful migration, a second run performs zero record
    mutations and never calls backfill_salience again."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    first = run_salience_auto_migration(store)
    assert first["status"] == "migrated"

    def _raiser(*_a, **_kw):
        raise AssertionError("backfill_salience must not run a second time")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    second = run_salience_auto_migration(store)
    assert second == {"status": "already_migrated"}


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fault_injected_failure_never_blocks_recall_or_quarantines(
    driver, tmp_path, monkeypatch,
):
    """A fault-injected backfill failure never raises into the sleep
    pipeline (no 3-strike quarantine), leaves the done marker UNSET, and
    the store still answers recall while the migration is incomplete."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    rows = _seed_legacy_corpus(store)

    def _raiser(*_a, **_kw):
        raise RuntimeError("synthetic backfill failure")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    pipeline = _make_pipeline(store, tmp_path)
    result = pipeline.run()

    assert result["failed_step"] is None
    assert result["quarantine_triggered"] is False
    assert result["error"] is None
    assert SleepStep.SALIENCE_AUTO_MIGRATION in result["completed_steps"]

    assert _marker_row(store, _DONE_MARKER_KEY) is None, (
        "a failed attempt must leave the done marker unset"
    )

    # Recall stays daemon-independent: the store answers reads correctly
    # while the migration is incomplete.
    still_there = store.get(rows[0].id)
    assert still_there is not None
    assert still_there.literal_surface == rows[0].literal_surface


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_literal_surface_decrypted_plaintext_unchanged(driver, tmp_path, monkeypatch):
    """Every record's decrypted literal_surface is byte-identical
    before and after the migration -- metadata-only, never touching the
    encrypted plaintext field."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    rows = _seed_legacy_corpus(store)
    before = {r.id: store.get(r.id).literal_surface for r in rows}

    result = run_salience_auto_migration(store)
    assert result["status"] == "migrated"

    for rid, text in before.items():
        after = store.get(rid)
        assert after is not None
        assert after.literal_surface == text, (
            "the migration is metadata-only; literal_surface must never change"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_bounded_give_up_after_max_attempts_with_crash_order_proof(
    driver, tmp_path, monkeypatch,
):
    """A persistently-failing store stops auto-retrying after MAX_ATTEMPTS
    (no further backfill call, no further snapshot). The attempts counter
    is proven committed BEFORE each backfill call -- the raiser reads it
    back from _hippo_meta at call time and must already see the
    incremented value, proving the counter survives a hard crash mid-call."""
    from iai_mcp.migrate._salience_auto_migration import _ATTEMPTS_KEY, MAX_ATTEMPTS

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    observed_attempts_at_call: list[int] = []

    def _raiser(*_a, **_kw):
        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = ?", (_ATTEMPTS_KEY,),
            ).fetchone()
        value = row["value"] if hasattr(row, "keys") else row[0]
        observed_attempts_at_call.append(int(value))
        raise RuntimeError("synthetic persistent failure")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    results = [run_salience_auto_migration(store) for _ in range(MAX_ATTEMPTS)]
    for result in results:
        assert result["status"] == "failed"
    assert observed_attempts_at_call == list(range(1, MAX_ATTEMPTS + 1)), (
        "the attempts counter must already be incremented to N by the time "
        "backfill_salience is invoked on the Nth call"
    )

    assert _marker_row(store, _DONE_MARKER_KEY) is None

    gave_up = run_salience_auto_migration(store)
    assert gave_up == {"status": "gave_up", "attempts": MAX_ATTEMPTS}

    calls_before_giveup_check = len(observed_attempts_at_call)
    run_salience_auto_migration(store)
    assert len(observed_attempts_at_call) == calls_before_giveup_check, (
        "no further backfill call (and no further snapshot) once given up"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_failed_attempt_removes_only_its_own_snapshot(driver, tmp_path, monkeypatch):
    """Each failed attempt cleans up the pre-write snapshot it itself
    created; across MAX_ATTEMPTS failures the store accumulates zero
    leftover auto-migration snapshots."""
    from iai_mcp.migrate._salience_auto_migration import MAX_ATTEMPTS

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    calls = {"n": 0}

    def _fake_backfill_with_snapshot(store, *, apply=True, store_path=None):
        calls["n"] += 1
        root = Path(store_path) if store_path is not None else Path(store.root)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = root / f"hippo-pre-salience-backfill-{ts}-{calls['n']}"
        snap.mkdir()
        raise RuntimeError("synthetic failure after snapshot creation")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience",
        _fake_backfill_with_snapshot,
    )

    for _ in range(MAX_ATTEMPTS):
        result = run_salience_auto_migration(store)
        assert result["status"] == "failed"

    assert calls["n"] == MAX_ATTEMPTS
    remaining = list(Path(store.root).glob("hippo-pre-salience-backfill-*"))
    assert remaining == [], (
        "every failed attempt must clean up the snapshot it created"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pre_existing_snapshot_survives_a_failed_attempt(driver, tmp_path, monkeypatch):
    """A pre-existing (manual) snapshot dir present before a failing attempt
    is left untouched by that attempt's cleanup."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    manual_snap = Path(store.root) / "hippo-pre-salience-backfill-manual-20260101T000000Z"
    manual_snap.mkdir()

    def _raiser(*_a, **_kw):
        raise RuntimeError("synthetic failure, no snapshot created this attempt")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    result = run_salience_auto_migration(store)
    assert result["status"] == "failed"
    assert manual_snap.exists(), (
        "a pre-existing snapshot must never be removed by cleanup"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pre_run_enumeration_oserror_disables_cleanup_fail_closed(
    driver, tmp_path, monkeypatch,
):
    """A transient OSError enumerating pre-existing snapshot dirs before a
    failing attempt must disable that attempt's cleanup entirely, never
    fall back to treating "unknown" as "empty" -- an empty-set fallback
    would make the failed attempt's cleanup treat every existing snapshot,
    including a pre-existing manual one, as newly created and delete it."""
    from iai_mcp.migrate import _salience_auto_migration as mod

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    manual_snap = Path(store.root) / "hippo-pre-salience-backfill-manual-20260101T000000Z"
    manual_snap.mkdir()

    # Raise ONLY on the pre-run ("before") call, then delegate to the real
    # enumeration for any later call -- a patch that raises unconditionally
    # would make _cleanup_new_snapshot_dirs's own inner `except OSError:
    # return` swallow the raise on ITS "after" enumeration too, so the test
    # would pass even with the fail-open bug restored (before=set()) and
    # prove nothing about the fail-closed sentinel.
    real_list_snapshot_dirs = mod._list_snapshot_dirs
    calls = {"n": 0}

    def _raise_first_then_real(root):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("synthetic transient enumeration failure")
        return real_list_snapshot_dirs(root)

    monkeypatch.setattr(mod, "_list_snapshot_dirs", _raise_first_then_real)

    rmtree_calls: list[Path] = []
    monkeypatch.setattr(
        mod.shutil,
        "rmtree",
        lambda path, *a, **kw: rmtree_calls.append(Path(path)),
    )

    def _raiser(*_a, **_kw):
        raise RuntimeError("synthetic failure, no snapshot created this attempt")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )

    result = run_salience_auto_migration(store)

    assert result["status"] == "failed"
    assert calls["n"] == 1, (
        "the fail-closed sentinel must short-circuit cleanup before it ever "
        "re-enumerates -- a second enumeration call means before=None was "
        "not honored"
    )
    assert rmtree_calls == [], (
        "an unreliable pre-run enumeration must disable cleanup entirely, "
        "never fall back to wiping every existing snapshot"
    )
    assert manual_snap.exists(), (
        "the fail-closed sentinel must leave every existing snapshot untouched"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_partial_backfill_errors_block_done_marker_and_keep_attempts(
    driver, tmp_path, monkeypatch,
):
    """A backfill that returns normally with a non-empty per-record
    ``errors`` list must not be treated as fully done: the done marker
    stays unset and the attempts counter (already incremented before the
    call) stays incremented, so a persistently-failing record eventually
    reaches the bounded gave_up state via retry instead of silently
    vanishing into a false "done" state."""
    from iai_mcp.migrate._salience_auto_migration import _ATTEMPTS_KEY

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    def _partial_backfill(store, *, apply=True, store_path=None):
        return {
            "mode": "apply",
            "records_scanned": 3,
            "salience_would_promote": 1,
            "salience_promoted": 1,
            "salience_distribution_current": {},
            "salience_distribution_resulting": {},
            "aaak_nonempty_rate": 0.0,
            "entity_tag_rate": 0.0,
            "snapshot_dir": None,
            "graph_cache_invalidated": False,
            "entity_backfill": {},
            "errors": ["<id>: apply: RuntimeError: synthetic per-record failure"],
        }

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _partial_backfill,
    )

    result = run_salience_auto_migration(store)

    assert result["status"] == "failed"
    assert _marker_row(store, _DONE_MARKER_KEY) is None, (
        "a non-empty errors list must never be treated as fully done"
    )
    assert _marker_row(store, _ATTEMPTS_KEY) is not None, (
        "the attempts counter must stay incremented so a persistently "
        "failing record eventually reaches the bounded gave_up state"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_entity_backfill_errors_block_done_marker_on_auto_path(
    driver, tmp_path, monkeypatch,
):
    """A backfill whose top-level ``errors`` is empty but whose embedded
    ``entity_backfill["errors"]`` is non-empty must gate the done marker
    exactly like a top-level salience error: the auto (nightly) path must
    not diverge from the manual ``--apply`` path on which error signal it
    trusts."""
    from iai_mcp.migrate._salience_auto_migration import _ATTEMPTS_KEY

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    def _entity_only_failure(store, *, apply=True, store_path=None):
        return {
            "mode": "apply",
            "records_scanned": 3,
            "salience_would_promote": 1,
            "salience_promoted": 1,
            "salience_distribution_current": {},
            "salience_distribution_resulting": {},
            "aaak_nonempty_rate": 0.0,
            "entity_tag_rate": 0.0,
            "snapshot_dir": None,
            "graph_cache_invalidated": False,
            "entity_backfill": {
                "errors": ["<id>: apply: RuntimeError: synthetic entity anchor failure"],
            },
            "errors": [],
        }

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _entity_only_failure,
    )

    result = run_salience_auto_migration(store)

    assert result["status"] == "failed"
    assert _marker_row(store, _DONE_MARKER_KEY) is None, (
        "an entity/AAAK-only error must never be treated as fully done"
    )
    assert _marker_row(store, _ATTEMPTS_KEY) is not None, (
        "the attempts counter must stay incremented so a persistently "
        "failing record eventually reaches the bounded gave_up state"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_snapshot_survives_a_post_backfill_failure(driver, tmp_path, monkeypatch):
    """If backfill_salience SUCCEEDS but a subsequent step (the done-marker
    write) raises, the snapshot must be KEPT -- that run already mutated
    the store, so its snapshot is the load-bearing rollback point. Cleanup
    is scoped strictly to the backfill call's own failure, never to steps
    after it."""
    from iai_mcp.migrate import _salience_auto_migration as mod

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    def _raise_after_success(*_a, **_kw):
        raise RuntimeError("synthetic failure writing the done marker")

    monkeypatch.setattr(mod, "mark_migration_done", _raise_after_success)

    with pytest.raises(RuntimeError):
        run_salience_auto_migration(store)

    snapshots = list(Path(store.root).glob("hippo-pre-salience-backfill-*"))
    assert len(snapshots) == 1, (
        "the real backfill_salience call succeeded and minted a snapshot; a "
        "failure in the subsequent done-marker write must not remove it"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stale_attempts_and_last_error_cleared_on_success(driver, tmp_path, monkeypatch):
    """Attempts and last-error metadata from prior failed attempts are
    cleared once a run finally succeeds -- a fresh nightly cycle after
    recovery starts from a clean slate, not a stale failure count."""
    from iai_mcp.migrate._salience_auto_migration import _ATTEMPTS_KEY, _LAST_ERROR_KEY

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    _seed_legacy_corpus(store)

    def _raiser(*_a, **_kw):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        "iai_mcp.migrate._salience_backfill.backfill_salience", _raiser,
    )
    failed = run_salience_auto_migration(store)
    assert failed["status"] == "failed"
    assert _marker_row(store, _ATTEMPTS_KEY) is not None
    assert _marker_row(store, _LAST_ERROR_KEY) is not None

    monkeypatch.undo()  # restore the real backfill_salience
    succeeded = run_salience_auto_migration(store)
    assert succeeded["status"] == "migrated"

    assert _marker_row(store, _ATTEMPTS_KEY) is None
    assert _marker_row(store, _LAST_ERROR_KEY) is None
