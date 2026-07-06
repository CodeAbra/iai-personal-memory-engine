"""Analytics, query, build, and migrate commands for the iai-mcp CLI."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def cmd_health(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from datetime import datetime as _dt

    from iai_mcp.tz import load_user_tz, to_local

    tz = load_user_tz()

    def _render_event(e: dict) -> None:
        ts_raw = e.get("ts")
        if isinstance(ts_raw, str):
            ts_raw = _dt.fromisoformat(ts_raw.replace("Z", "+00:00"))
        local = to_local(ts_raw, tz) if ts_raw is not None else None
        severity = e.get("severity") or "?"
        ts_str = local.isoformat() if local is not None else str(ts_raw)
        print(f"llm_health: {severity} at {ts_str} ({tz.key})")
        print(f"  data: {e.get('data', {})}")

    resp = _cli._send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            events = payload["events"]
            if not events:
                print("llm_health: no events recorded")
                return 0
            _render_event(events[0])
            return 0

    from iai_mcp.events import query_events
    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        events = query_events(store, kind="llm_health", limit=1)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    if not events:
        print("llm_health: no events recorded")
        return 0
    _render_event(events[0])
    return 0


def cmd_build_native(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import shutil
    from pathlib import Path

    if shutil.which("cargo") is None:
        print(
            "cargo not found on PATH.\n"
            "Install Rust: https://rustup.rs/",
            file=_cli.sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parents[3]
    native_dir = repo_root / "rust" / "iai_mcp_native"
    if not native_dir.exists():
        print(
            f"Rust source not found at {native_dir}.\n"
            "Are you running from an installed wheel? "
            "build-native requires the repo checkout.",
            file=_cli.sys.stderr,
        )
        return 1

    cmd = [
        _cli.sys.executable, "-m", "maturin", "develop", "--release",
        "--manifest-path", str(native_dir / "Cargo.toml"),
    ]
    result = _cli.subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        print(
            "\nbuild-native failed (see cargo output above).\n"
            "Common fix: rustup update",
            file=_cli.sys.stderr,
        )
        return result.returncode
    print("iai_mcp_native built successfully. Restart the daemon or MCP server.")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.store import MemoryStore
    store = MemoryStore()

    # --reembed-from-text owns --resume / --rollback when combined with it:
    # resume continues the reembed from its last committed window, rollback
    # discards the reembed checkpoint. Checked before the schema-migration
    # resume/rollback so the two machines never cross wires.
    if bool(getattr(args, "reembed_from_text", False)):
        from iai_mcp.migrate import migrate_reembed_from_text
        from iai_mcp.migrate._reembed_from_text import _clear_resume_cursor
        if bool(getattr(args, "rollback", False)):
            _clear_resume_cursor(store)
            print("reembed checkpoint cleared")
            return 0
        dry_run = bool(getattr(args, "dry_run", False))
        batch_size = int(getattr(args, "reembed_batch_size", 256))
        resume = bool(getattr(args, "resume", False))
        result = migrate_reembed_from_text(
            store, dry_run=dry_run, batch_size=batch_size, resume=resume
        )
        prefix = "[dry-run] would re-embed" if dry_run else "re-embedded"
        print(
            f"{prefix} {result['reembedded']} records; "
            f"skipped={result['skipped']} total={result['total']}"
        )
        return 0

    if bool(getattr(args, "rollback", False)):
        from iai_mcp import migrate
        return migrate._rollback(store.db, store)
    if bool(getattr(args, "resume", False)):
        from iai_mcp import migrate
        from iai_mcp.embed import embedder_for_store
        target = embedder_for_store(store)
        return migrate._resume(store.db, store, target)

    if bool(getattr(args, "rederive_timestamps", False)):
        from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
        dry_run = bool(getattr(args, "dry_run", False))
        result = migrate_rederive_collapsed_timestamps(store, dry_run=dry_run)
        prefix = "[dry-run] would update" if dry_run else "updated"
        print(
            f"{prefix} {result['records_updated']} records; "
            f"skipped_no_transcript={result['skipped_no_transcript']} "
            f"skipped_no_match={result['skipped_no_match']}"
        )
        return 0

    if bool(getattr(args, "dedupe_episodic", False)):
        from iai_mcp.migrate import migrate_dedupe_episodic_captures
        dry_run = bool(getattr(args, "dry_run", False))
        result = migrate_dedupe_episodic_captures(store, dry_run=dry_run)
        prefix = "[dry-run] would tombstone" if dry_run else "tombstoned"
        print(
            f"{prefix} {result['tombstoned']} duplicate records "
            f"across {result['groups']} group(s)"
        )
        return 0

    if bool(getattr(args, "salvage_torn_permanent_failed", False)):
        from iai_mcp.migrate import migrate_salvage_torn_permanent_failed
        dry_run = bool(getattr(args, "dry_run", False))
        result = migrate_salvage_torn_permanent_failed(dry_run=dry_run)
        prefix = "[dry-run] would salvage" if dry_run else "salvaged"
        print(
            f"{prefix} {result['records_salvaged']} record(s) from "
            f"{result['files_salvaged']} file(s); "
            f"dropped {result['bytes_dropped']} torn bytes"
        )
        return 0

    from_v = int(getattr(args, "from_", 1))
    to_v = int(getattr(args, "to", 2))
    dry_run = bool(getattr(args, "dry_run", False))
    verbose = bool(getattr(args, "verbose", False))

    def _progress(i: int, n: int) -> None:
        if verbose:
            print(f"[{i + 1}/{n}] migrating...")

    if from_v == 1 and to_v == 2:
        from iai_mcp.migrate import migrate_v1_to_v2
        result = migrate_v1_to_v2(store, dry_run=dry_run, progress=_progress)
        prefix = "would migrate" if dry_run else "migrated"
        print(
            f"{prefix} {result['records_migrated']} records in "
            f"{result['duration_sec']:.2f}s "
            f"({result['previous_model']} -> {result['new_model']})"
        )
        return 0

    if from_v == 2 and to_v == 3:
        from iai_mcp.migrate import migrate_encryption_v2_to_v3
        result = migrate_encryption_v2_to_v3(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would encrypt" if dry_run else "encrypted"
        print(
            f"{prefix} {result['records_migrated']} records + "
            f"{result['events_migrated']} events in "
            f"{result['duration_sec']:.2f}s "
            f"(AES-256-GCM, iai:enc:v1:)"
        )
        return 0

    if from_v == 3 and to_v == 4:
        from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4
        result = migrate_hd_vector_to_structure_hv_v3_to_v4(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would rename" if dry_run else "renamed"
        print(
            f"{prefix} {result['updated']} records' "
            f"hd_vector_json->structure_hv column in "
            f"{result['duration_ms'] / 1000:.2f}s "
            f"(schema v3->v4, TEM factorization, D=10000 BSC packed)"
        )
        return 0

    print(
        f"unsupported migration --from={from_v} --to={to_v}; "
        f"supported: 1->2 (schema), 2->3 (encryption), "
        f"3->4 (TEM factorization)",
        file=_cli.sys.stderr,
    )
    return 2


def cmd_deferred_unlock_dead_pids(args: argparse.Namespace) -> int:
    """Unlock `.processing-<pid>.jsonl` files in the deferred-captures dir whose owner
    PID is dead or differs from the live daemon's PID. Renames each back to `.jsonl`
    so the next drain tick claims it. Idempotent; collision-safe."""
    import json as _json
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    dry_run = bool(getattr(args, "dry_run", False))
    emit_json = bool(getattr(args, "json", False))
    result = migrate_unlock_dead_pid_processing_files(dry_run=dry_run)
    if emit_json:
        print(_json.dumps(result, indent=2, sort_keys=True))
        return 0
    prefix = "[dry-run] would unlock" if dry_run else "unlocked"
    print(
        f"{prefix} {result['files_unlocked']} file(s) "
        f"(scanned {result['files_scanned']}; "
        f"skipped_live_current_daemon={result['skipped_live_current_daemon']}; "
        f"collision_safe_renames={result['collision_safe_renames']}; "
        f"live_daemon_pid={result['live_daemon_pid']})"
    )
    return 0


def cmd_bank_recall(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.memory_bank import bank_recall_substring

    include_processed = not getattr(args, "recent_only", False)
    include_recent = not getattr(args, "processed_only", False)

    result = bank_recall_substring(
        args.query,
        limit=args.limit,
        include_processed=include_processed,
        include_recent=include_recent,
    )
    print(_json.dumps(result, ensure_ascii=False))
    return 0


def cmd_topology(args: argparse.Namespace) -> int:  # noqa: ARG001 -- argparse contract
    from iai_mcp import cli as _cli

    def _fmt(v) -> str:
        if v is None:
            return "insufficient_data"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _render(d: dict) -> None:
        print(f"C: {_fmt(d.get('C'))}")
        print(f"L: {_fmt(d.get('L'))}")
        print(f"sigma: {_fmt(d.get('sigma'))}")
        print(f"communities: {_fmt(d.get('community_count'))}")
        print(f"rich_club_ratio: {_fmt(d.get('rich_club_ratio'))}")
        print(f"N: {_fmt(d.get('N'))}")
        print(f"regime: {_fmt(d.get('regime'))}")

    resp = _cli._send_jsonrpc_request("topology", {})
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _render(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp import sigma as sigma_mod
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        records_count = int(store.db.open_table("records").count_rows())
        inline_ceil = sigma_mod._env_int(
            "IAI_MCP_TOPOLOGY_INLINE_N_CEIL",
            sigma_mod.TOPOLOGY_INLINE_N_CEIL,
        )
        if records_count > inline_ceil:
            snap = sigma_mod.latest_topology_snapshot(store)
            if snap is None:
                snap = sigma_mod.topology_payload(
                    N=records_count,
                    regime="insufficient_data",
                    source="insufficient",
                )
            _render(snap)
            return 0
        graph, assignment, rich_club = build_runtime_graph(store)
        snap = sigma_mod.compute_topology_snapshot(
            graph, assignment=assignment, rich_club=rich_club
        )
    except HippoLockHeldError:
        _render({})
        return 0

    _render(snap)
    return 0


def _aggregate_trajectory_from_events(
    events: list[dict],
) -> dict[str, list[tuple]]:
    from iai_mcp.trajectory import METRIC_NAMES

    out: dict[str, list[tuple]] = {m: [] for m in METRIC_NAMES}
    for e in events:
        data = e.get("data") or {}
        m = data.get("metric")
        v = data.get("value")
        if m in METRIC_NAMES and v is not None:
            try:
                out[m].append((e.get("ts"), float(v)))
            except (TypeError, ValueError):
                continue
    return out


def _render_trajectory(data: dict, metric_names: list) -> None:
    if not any(data.get(m) for m in metric_names):
        print("no trajectory data recorded")
        return
    for metric in metric_names:
        points = data.get(metric, [])
        if not points:
            print(f"{metric.upper()}: (no data)")
            continue
        values = [v for _, v in points]
        n = len(values)
        mean = sum(values) / n
        print(
            f"{metric.upper()}: n={n} mean={mean:.3f} "
            f"min={min(values):.3f} max={max(values):.3f}"
        )


def cmd_trajectory(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from datetime import datetime, timedelta, timezone

    from iai_mcp.trajectory import METRIC_NAMES

    weeks = getattr(args, "since", None)
    since = None
    since_iso = None
    if weeks is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(weeks))
        since_iso = since.isoformat()

    socket_params: dict = {"kind": "trajectory_metric", "limit": 1000}
    if since_iso:
        socket_params["since"] = since_iso
    resp = _cli._send_jsonrpc_request("events_query", socket_params)
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            data = _aggregate_trajectory_from_events(payload["events"])
            _render_trajectory(data, METRIC_NAMES)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore
    from iai_mcp.trajectory import aggregate_trajectory

    try:
        store = MemoryStore()
        data = aggregate_trajectory(store, since=since)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    _render_trajectory(data, METRIC_NAMES)
    return 0


def _redact_shield_data(data: dict) -> str:
    matched = data.get("matched") or []
    tier = data.get("tier", "-")
    record_id = data.get("record_id", "-")
    action = data.get("action", "-")
    return (
        f"tier={tier} action={action} "
        f"matched_count={len(matched)} record_id={record_id}"
    )


def _format_audit_event(event: dict, tz) -> str:
    from datetime import datetime as _dt

    from iai_mcp.tz import to_local

    ts = event.get("ts")
    if isinstance(ts, str):
        try:
            ts = _dt.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = None
    try:
        local_ts = to_local(ts, tz) if ts is not None else None
    except (ValueError, TypeError, OSError):
        local_ts = None
    ts_str = local_ts.isoformat() if local_ts is not None else str(event.get("ts"))

    kind = event.get("kind", "?")
    sev = event.get("severity") or "-"
    data = event.get("data") or {}
    if kind in ("shield_rejection", "shield_flag", "shield_log"):
        data_str = _redact_shield_data(data)
    else:
        data_str = str(data)[:200]
    return f"[{ts_str}] {kind:32s} [{sev:8s}] {data_str}"


def cmd_audit(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from datetime import datetime, timedelta, timezone

    from iai_mcp.tz import load_user_tz

    tz = load_user_tz()

    since_raw = getattr(args, "since", None)
    since = None
    since_iso = None
    if since_raw is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(since_raw))
        since_iso = since.isoformat()

    sub = getattr(args, "audit_sub", None)

    if sub == "drift":
        resp = _cli._send_jsonrpc_request("detect_drift", {})
        if isinstance(resp, dict) and "result" in resp:
            payload = resp["result"]
            if isinstance(payload, dict) and "alerts" in payload:
                alerts = payload["alerts"]
                if not alerts:
                    print("drift: no anomaly detected (M4 variance stable)")
                else:
                    for a in alerts:
                        print(
                            f"drift: variance increasing across "
                            f"{a.get('window_sessions')} sessions; "
                            f"first={a.get('first_value'):.3f} "
                            f"last={a.get('last_value'):.3f}"
                        )
                return 0

        from iai_mcp.hippo import HippoLockHeldError
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        try:
            store = MemoryStore()
            alerts = detect_drift_anomaly(store)
        except HippoLockHeldError:
            print("daemon holds store lock; retry when daemon is idle")
            return 0

        if not alerts:
            print("drift: no anomaly detected (M4 variance stable)")
        else:
            for a in alerts:
                print(
                    f"drift: variance increasing across "
                    f"{a.get('window_sessions')} sessions; "
                    f"first={a.get('first_value'):.3f} "
                    f"last={a.get('last_value'):.3f}"
                )
        return 0

    SHIELD_KINDS = ("shield_rejection", "shield_flag", "shield_log")
    IDENTITY_KINDS = (
        "s5_invariant_update",
        "s5_invariant_proposal",
        "s5_cooldown_block",
        "s5_drift_alert",
        "identity_cross_lingual_warning",
    )

    if sub == "shield":
        audit_kinds = list(SHIELD_KINDS)
        empty_msg = "audit shield: no events recorded"
    elif sub == "identity":
        audit_kinds = list(IDENTITY_KINDS)
        empty_msg = "audit identity: no events recorded"
    else:
        from iai_mcp.s5 import AUDIT_EVENT_KINDS
        audit_kinds = list(AUDIT_EVENT_KINDS)
        empty_msg = "No identity events recorded"

    severity = getattr(args, "severity", None)

    socket_params: dict = {"kinds": audit_kinds}
    if since_iso:
        socket_params["since"] = since_iso
    resp = _cli._send_jsonrpc_request("audit_query", socket_params)
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            events = payload["events"]
            if severity:
                events = [e for e in events if e.get("severity") == severity]
            if not events:
                print(empty_msg)
                return 0
            for e in events:
                print(_format_audit_event(e, tz))
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        events = audit_identity_events(store, since=since, kinds=tuple(audit_kinds))
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    if severity:
        events = [e for e in events if e.get("severity") == severity]
    if not events:
        print(empty_msg)
        return 0
    for e in events:
        print(_format_audit_event(e, tz))
    return 0
