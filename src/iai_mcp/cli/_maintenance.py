"""Maintenance, schema, lifecycle, and drain commands for the iai-mcp CLI."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_schema_cleanup(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=_cli.sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))

    store = MemoryStore(path=store_path)
    summary = cleanup_schema_duplicates(
        store, apply=apply, store_path=store_path,
    )

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp schema-cleanup [{mode_str}]")
    print(f"  groups (patterns with N>1 duplicates): {summary.get('groups', 0)}")
    print(f"  keepers (one per group):               {summary.get('keepers', 0)}")
    print(
        f"  pruned (soft-deleted, tier=semantic_pruned): "
        f"{summary.get('pruned', 0)}"
    )
    print(
        f"  of which noise (idem-hash patterns):   "
        f"{summary.get('noise_pruned', 0)}"
    )
    print(
        f"  edges to reinforce onto keepers:       "
        f"{summary.get('edges_reinforced', 0)}"
    )
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:                    {summary['snapshot_dir']}")
    if mode_str == "dry-run" and summary.get("groups", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    return 0


def _maintenance_compact_preflight_daemon_alive() -> str | None:
    from iai_mcp import cli as _cli
    import json as _json
    import os as _os

    if not _cli.STATE_PATH.exists():
        return None
    try:
        state = _json.loads(_cli.STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = state.get("daemon_pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    from iai_mcp.lifecycle_lock import pid_exists

    if not pid_exists(pid):
        return None
    try:
        import psutil
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except Exception as exc:
        logger.debug("psutil inspect pid %d failed: %s", pid, exc)
        return (
            f"daemon running (pid {pid}); run `iai-mcp daemon stop` "
            f"first, then retry"
        )
    if "iai_mcp.daemon" not in cmdline:
        return None
    return (
        f"daemon running (pid {pid}); run `iai-mcp daemon stop` first, "
        f"then retry"
    )


def _maintenance_compact_dry_run(
    store_path: Path, hippo_dir: Path,
) -> int:
    from iai_mcp import cli as _cli
    import json as _json
    from iai_mcp.store import MemoryStore

    store = None
    try:
        store = MemoryStore(path=store_path)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("compact dry-run MemoryStore open failed: %s", exc)
        print(
            f"warning: could not open MemoryStore (records_count + "
            f"record_id_set will be 0): {exc}",
            file=_cli.sys.stderr,
        )
    metrics = _cli._maintenance_compact_metrics(hippo_dir, store=store)
    out = {
        "mode": "dry-run",
        "metrics": {
            "pre": {
                k: v for k, v in metrics.items() if k != "record_id_set"
            },
            "post": None,
        },
        "would_invoke": "optimize_hippo_storage()",
    }
    print(_json.dumps(out, indent=2))
    return 0


def _maintenance_compact_apply(
    store_path: Path, hippo_dir: Path,
) -> int:
    from iai_mcp import cli as _cli
    import json as _json
    import time as _time
    from datetime import datetime, timezone
    from iai_mcp.maintenance import optimize_hippo_storage
    from iai_mcp.store import MemoryStore

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = (
        Path.home() / ".iai-mcp" / f".maintenance-compact-{ts}.json"
    )

    store = MemoryStore(path=store_path)
    pre_metrics = _cli._maintenance_compact_metrics(hippo_dir, store=store)
    pre_id_set = pre_metrics["record_id_set"]

    t0 = _time.monotonic()
    report = optimize_hippo_storage(store)
    elapsed = round(_time.monotonic() - t0, 3)

    store_after = MemoryStore(path=store_path)
    post_metrics = _cli._maintenance_compact_metrics(hippo_dir, store=store_after)
    post_id_set = post_metrics["record_id_set"]

    if pre_id_set != post_id_set:
        missing = pre_id_set - post_id_set
        extra = post_id_set - pre_id_set
        failed_path = (
            Path.home() / ".iai-mcp"
            / f".maintenance-compact-FAILED-{ts}.json"
        )
        failed_payload = {
            "command": "iai-mcp maintenance compact-hippo --apply",
            "timestamp_utc": ts,
            "status": "aborted",
            "reason": "record_id_set divergence post-optimize",
            "metrics_pre": {
                k: v for k, v in pre_metrics.items()
                if k != "record_id_set"
            },
            "metrics_post": {
                k: v for k, v in post_metrics.items()
                if k != "record_id_set"
            },
            "missing_ids_count": len(missing),
            "extra_ids_count": len(extra),
            "missing_ids_sample": list(sorted(missing))[:10],
            "extra_ids_sample": list(sorted(extra))[:10],
            "optimize_report": report,
            "elapsed_sec": elapsed,
        }
        try:
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(_json.dumps(failed_payload, indent=2), encoding="utf-8")
        except OSError:
            pass
        print(
            f"ABORT: record_id_set divergence — missing={len(missing)} "
            f"extra={len(extra)}; details written to {failed_path}",
            file=_cli.sys.stderr,
        )
        return 1

    payload = {
        "command": "iai-mcp maintenance compact-hippo --apply",
        "timestamp_utc": ts,
        "status": "ok",
        "metrics_pre": {
            k: v for k, v in pre_metrics.items() if k != "record_id_set"
        },
        "metrics_post": {
            k: v for k, v in post_metrics.items() if k != "record_id_set"
        },
        "elapsed_sec": elapsed,
        "optimize_report": report,
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(
            f"warning: could not write audit file {audit_path}: {exc}",
            file=_cli.sys.stderr,
        )
    print(_json.dumps({
        "mode": "apply",
        "metrics": {
            "pre": payload["metrics_pre"],
            "post": payload["metrics_post"],
        },
        "elapsed_sec": elapsed,
        "audit_file": str(audit_path),
        "status": "ok",
    }, indent=2))
    return 0


def cmd_maintenance_compact_hippo(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    if getattr(args, "maintenance_cmd", None) == "compact-records":
        print(
            "warning: compact-records is the deprecated name for "
            "compact-hippo; use compact-hippo going forward",
            file=_cli.sys.stderr,
        )

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    hippo_dir = store_path / "hippo"
    if not hippo_dir.exists():
        print(
            f"error: hippo storage not found at {hippo_dir}",
            file=_cli.sys.stderr,
        )
        return 1

    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    if not apply:
        return _maintenance_compact_dry_run(store_path, hippo_dir)

    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=_cli.sys.stderr)
        return 1

    if not yes and not _cli.sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to proceed "
            "without interactive consent or explicit --yes)",
            file=_cli.sys.stderr,
        )
        return 2

    if not yes:
        prompt = (
            "About to compact Hippo storage via wal_checkpoint + VACUUM + "
            "hnswlib rebuild. Daemon must be stopped. Type 'y' to proceed: "
        )
        try:
            response = input(prompt)
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("aborted: user did not consent", file=_cli.sys.stderr)
            return 1

    return _maintenance_compact_apply(store_path, hippo_dir)


def cmd_maintenance_compact_records(args: argparse.Namespace) -> int:
    args.maintenance_cmd = "compact-records"
    return cmd_maintenance_compact_hippo(args)


def cmd_maintenance_symmetrize_self_loops(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.maintenance import symmetrize_self_loops
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    hippo_dir = store_path / "hippo"
    if not hippo_dir.exists():
        print(
            f"error: hippo storage not found at {hippo_dir}",
            file=_cli.sys.stderr,
        )
        return 1

    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))

    if not apply:
        store = MemoryStore(path=store_path)
        result = symmetrize_self_loops(store, dry_run=True)
        print(json.dumps(result, indent=2))
        return 0

    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=_cli.sys.stderr)
        return 1

    if not yes and not _cli.sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to "
            "proceed without interactive consent or explicit --yes)",
            file=_cli.sys.stderr,
        )
        return 2

    if not yes:
        prompt = (
            "About to backfill missing hebbian self-loops on records. "
            "Daemon must be stopped. Type 'y' to proceed: "
        )
        try:
            response = input(prompt)
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("aborted: user did not consent", file=_cli.sys.stderr)
            return 1

    store = MemoryStore(path=store_path)
    result = symmetrize_self_loops(store, dry_run=False)
    print(json.dumps(result, indent=2))
    return 0


def _format_relative(ts_iso: str, now: datetime | None = None) -> str:
    try:
        ts = datetime.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = moment - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def cmd_lifecycle_force_unlock(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.lifecycle_lock import DEFAULT_LOCK_PATH, LifecycleLock

    lock_path = getattr(args, "lock_path", None)
    if lock_path is not None:
        lock = LifecycleLock(Path(lock_path))
    else:
        lock = LifecycleLock(DEFAULT_LOCK_PATH)

    existing = lock.read()
    if existing is None:
        print("No lockfile present; nothing to unlock.")
        return 0

    print(
        f"Existing lockfile: pid={existing['pid']} "
        f"hostname={existing['hostname']} "
        f"started_at={existing['started_at']}"
    )

    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Force unlock and remove the lockfile? [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Force-unlock cancelled.", file=_cli.sys.stderr)
            return 1

    previous = lock.force_unlock()
    if previous is None:
        print("Lockfile already removed by another process.")
        return 0
    print("Lockfile removed.")
    return 0


def cmd_lifecycle_status(args: argparse.Namespace) -> int:
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, load_state

    record = load_state(LIFECYCLE_STATE_PATH)
    print(f"state: {record['current_state']}")
    print(
        f"since: {record['since_ts']} "
        f"({_format_relative(record['since_ts'])})"
    )
    print(f"last_activity: {record['last_activity_ts']}")
    print(f"wrapper_event_seq: {record['wrapper_event_seq']}")

    progress = record.get("sleep_cycle_progress")
    if progress is None:
        print("sleep_cycle_progress: none")
    else:
        step = progress.get(
            "last_completed_index",
            progress.get("last_completed_step", 0),
        )
        attempt = progress.get("attempt", 0)
        last_error = progress.get("last_error") or "none"
        started_at = progress.get("started_at", "?")
        print(
            f"sleep_cycle_progress: step={step} attempt={attempt} "
            f"last_error={last_error} started_at={started_at}"
        )

    quarantine = record.get("quarantine")
    if quarantine is None:
        print("quarantine: none")
    else:
        print(
            f"quarantine: until={quarantine['until_ts']} "
            f"reason={quarantine['reason']} since={quarantine['since_ts']}"
        )

    shadow = record.get("shadow_run", True)
    if shadow:
        print(
            "shadow_run: true (legacy RSS-watchdog still owns shutdown)"
        )
    else:
        print("shadow_run: false")

    return 0


def cmd_maintenance_sleep_cycle(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from iai_mcp.store import MemoryStore

    if getattr(args, "store_path", None) is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    try:
        store = MemoryStore(path=store_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("sleep-cycle MemoryStore open failed: %s", exc)
        print(
            f"error: could not open MemoryStore at {store_path}: {exc}",
            file=_cli.sys.stderr,
        )
        return 2

    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=LIFECYCLE_STATE_PATH,
        event_log=LifecycleEventLog(),
    )

    reset_quarantine = bool(getattr(args, "reset_quarantine", False))
    force = bool(getattr(args, "force", False))

    if reset_quarantine:
        if pipeline.is_quarantined():
            pipeline.reset_quarantine()
            print("Quarantine cleared.")
        else:
            print("Quarantine not active; --reset-quarantine had no effect.")

    if pipeline.is_quarantined() and not force:
        from iai_mcp.lifecycle_state import load_state

        record = load_state(LIFECYCLE_STATE_PATH)
        quarantine = record.get("quarantine") or {}
        until_ts = quarantine.get("until_ts", "?")
        reason = quarantine.get("reason", "unknown")
        print(
            f"Sleep cycle quarantined until {until_ts}.",
            file=_cli.sys.stderr,
        )
        print(f"Reason: {reason}", file=_cli.sys.stderr)
        print(
            "Use --force to override OR --reset-quarantine to clear.",
            file=_cli.sys.stderr,
        )
        return 1

    step_index = {
        step: i + 1 for i, step in enumerate(SleepPipeline._STEP_ORDER)
    }
    total_steps = len(SleepPipeline._STEP_ORDER)

    print("Sleep cycle started.")
    runner = pipeline.force_run if force else pipeline.run
    result = runner()

    for step in result["completed_steps"]:
        idx = step_index.get(step, "?")
        print(f"[{idx}/{total_steps}] {step.name.lower()} ... ok")

    duration = result.get("duration_sec", 0.0)
    failed = result.get("failed_step")
    interrupted = result.get("interrupted", False)
    quarantine_triggered = result.get("quarantine_triggered", False)

    if failed is not None:
        idx = step_index.get(failed, "?")
        err = result.get("error") or "unknown"
        print(
            f"[{idx}/{total_steps}] {failed.name.lower()} ... FAILED: {err}",
            file=_cli.sys.stderr,
        )
        if quarantine_triggered:
            print(
                "Sleep cycle quarantined for 24h after 3rd consecutive "
                "failure of this step. Use --reset-quarantine to clear.",
                file=_cli.sys.stderr,
            )
        else:
            print(
                "Sleep cycle aborted; rerun to retry from this step.",
                file=_cli.sys.stderr,
            )
        return 1

    if interrupted:
        print(
            f"Sleep cycle deferred (bounded interrupt; "
            f"{duration:.1f}s elapsed). Resume on next invocation.",
        )
        return 0

    print(f"Sleep cycle complete ({duration:.1f}s total).")
    return 0


def cmd_drain_permanent_failed(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    dry_run = bool(getattr(args, "dry_run", False))

    resp = _cli._send_jsonrpc_request("drain_permanent_failed", {"dry_run": dry_run}, read_timeout=120.0)
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _print_drain_result(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore
    from iai_mcp.capture import drain_permanent_failed_files

    try:
        store = MemoryStore()
        result = drain_permanent_failed_files(store, dry_run=dry_run)
    except HippoLockHeldError:
        print(
            "Daemon holds the store lock — is it running? "
            "Ensure the daemon is reachable or stopped before using the direct-open fallback.",
            file=_cli.sys.stderr,
        )
        return 1

    _print_drain_result(result)
    return 0


def cmd_deferred_drain(args: argparse.Namespace) -> int:
    import os

    from iai_mcp import cli as _cli

    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    env_store = os.environ.get("IAI_MCP_STORE")
    if env_store:
        store_root = Path(env_store)
    else:
        from iai_mcp.store import DEFAULT_STORAGE_PATH
        store_root = Path(DEFAULT_STORAGE_PATH)

    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))

    from iai_mcp.migrate._dead_pid_unlock import _read_live_daemon_pid

    if _read_live_daemon_pid() is not None and not force:
        print(
            "a live daemon is running — stop the daemon and unload the launch "
            "service before draining the backlog, or pass --force to rely on "
            "the store lock as the single-writer guard",
            file=_cli.sys.stderr,
        )
        return 1

    # Permanently-failed capture files are NOT drained here; the operator should
    # also run `iai-mcp drain-permanent-failed` to recover those. This keeps the
    # backlog drain additive.
    print(
        "note: .permanent-failed-* capture files are not drained by this "
        "command — run `iai-mcp drain-permanent-failed` to recover them",
        file=_cli.sys.stderr,
    )

    from iai_mcp.deferred_drain import run_deferred_drain
    from iai_mcp.hippo import HippoLockHeldError

    try:
        result = run_deferred_drain(
            deferred_dir=deferred_dir,
            store_root=store_root,
            batch_bytes=args.batch_bytes,
            dry_run=dry_run,
        )
    except HippoLockHeldError:
        print(
            "the store is locked — another writer (the daemon?) holds it; stop "
            "it and retry",
            file=_cli.sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result))
        return 0

    if result.get("dry_run"):
        print(
            f"dry-run: {result['files_enumerated']} file(s), "
            f"{result['events_seen']} event(s), "
            f"{len(result['batches'])} batch(es), "
            f"{result['total_bytes']} bytes; nothing mutated"
        )
    else:
        print(
            f"drained {result['events_seen']} event(s): "
            f"inserted={result['inserted']} reinforced={result['reinforced']} "
            f"skipped_existing={result.get('skipped_existing', 0)} "
            f"skipped={result['skipped']} across {len(result['batches'])} "
            f"batch(es); remaining={result['remaining_events']}; "
            f"reconciled={result['reconciled']}"
        )
    return 0


def _print_drain_result(result: dict) -> None:
    files = result.get("files") or []
    if result.get("dry_run"):
        count = result.get("count", len(files))
        print(f"dry-run: {count} permanent-failed file(s) found")
        for f in files:
            print(f"  {f['name']}  ({f.get('line_count', '?')} lines)")
        return
    inserted = result.get("inserted", 0)
    dropped = result.get("dropped", 0)
    recovered = result.get("files_recovered") or []
    q_dir = result.get("quarantine_dir", "")
    print(f"recovered {len(recovered)} file(s): inserted={inserted} dropped={dropped}")
    for name in recovered:
        print(f"  {name}")
    if q_dir:
        print(f"quarantine copies at: {q_dir}")


def cmd_idem_dedup(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=_cli.sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))
    store = MemoryStore(path=store_path)
    summary = cleanup_idem_duplicates(store, apply=apply, store_path=store_path)

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp idem-dedup [{mode_str}]")
    print(f"  duplicate groups (same idem key):  {summary.get('groups', 0)}")
    print(f"  extra live copies:                 {summary.get('extra_copies', 0)}")
    print(f"  tombstoned:                        {summary.get('tombstoned', 0)}")
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:                {summary['snapshot_dir']}")
    if mode_str == "dry-run" and summary.get("groups", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    return 0


def cmd_blob_quarantine(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate._blob_quarantine import quarantine_notification_blobs
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=_cli.sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))
    store = MemoryStore(path=store_path)
    summary = quarantine_notification_blobs(
        store, apply=apply, store_path=store_path
    )

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp blob-quarantine [{mode_str}]")
    print(f"  records scanned:      {summary.get('records_scanned', 0)}")
    print(f"  notification blobs:   {summary.get('blobs_found', 0)}")
    print(f"  tombstoned:           {summary.get('tombstoned', 0)}")
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:   {summary['snapshot_dir']}")
    if summary.get("journal"):
        print(f"  journal:              {summary['journal']}")
    if mode_str == "dry-run" and summary.get("blobs_found", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    errors = summary.get("errors") or []
    if errors:
        print(f"error: {len(errors)} record(s) failed:", file=_cli.sys.stderr)
        for line in errors[:10]:
            print(f"  {line}", file=_cli.sys.stderr)
        return 1
    return 0


def cmd_entity_backfill(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=_cli.sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))
    refresh = bool(getattr(args, "refresh", False))
    store = MemoryStore(path=store_path)
    summary = backfill_entity_anchors(
        store, apply=apply, store_path=store_path, refresh=refresh
    )

    mode_str = summary.get("mode", "dry-run")
    refresh_str = " --refresh" if summary.get("refresh") else ""
    print(f"iai-mcp entity-backfill [{mode_str}{refresh_str}]")
    print(f"  records scanned:        {summary.get('records_scanned', 0)}")
    print(f"  already anchored:       {summary.get('already_anchored', 0)}")
    print(f"  records to anchor:      {summary.get('records_to_anchor', 0)}")
    print(f"  anchors proposed:       {summary.get('anchors_proposed', 0)}")
    if summary.get("anchors_to_remove"):
        print(f"  anchors to remove:      {summary['anchors_to_remove']}")
    print(f"  records written:        {summary.get('records_written', 0)}")
    print(f"  aaak indexes refreshed: {summary.get('aaak_refreshed', 0)}")
    if summary.get("journal"):
        print(f"  undo journal:           {summary['journal']}")
    if mode_str == "apply" and summary.get("records_written", 0) > 0:
        print()
        if summary.get("graph_cache_invalidated"):
            print(
                "  Graph cache invalidated; restart the daemon to serve "
                "the new anchors."
            )
        else:
            print(
                "  Live recall serves the new anchors after the next "
                "index rebuild (nightly, or a daemon restart)."
            )
    if mode_str == "dry-run" and summary.get("records_to_anchor", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    errors = summary.get("errors") or []
    if errors:
        print(
            f"error: {len(errors)} record(s) failed:",
            file=_cli.sys.stderr,
        )
        for line in errors[:10]:
            print(f"  {line}", file=_cli.sys.stderr)
        if len(errors) > 10:
            print(
                f"  ... and {len(errors) - 10} more",
                file=_cli.sys.stderr,
            )
        return 1
    return 0


def cmd_edge_backfill(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.migrate import backfill_consolidated_edges
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=_cli.sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))
    store = MemoryStore(path=store_path)
    summary = backfill_consolidated_edges(store, apply=apply, store_path=store_path)

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp edge-backfill [{mode_str}]")
    print(f"  live summaries:                    {summary.get('summaries_live', 0)}")
    print(f"  anchored (>=1 surviving edge):     {summary.get('summaries_anchored', 0)}")
    print(f"  edgeless (left to natural re-mint):{summary.get('summaries_edgeless', 0)}")
    print(f"  existing links:                    {summary.get('links_existing', 0)}")
    print(f"  proposed links:                    {summary.get('links_proposed', 0)}")
    print(f"  written links:                     {summary.get('links_written', 0)}")
    print(f"  oversize communities skipped:      {summary.get('communities_oversize_skipped', 0)}")
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:                {summary['snapshot_dir']}")
    if mode_str == "dry-run" and summary.get("links_proposed", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    return 0
