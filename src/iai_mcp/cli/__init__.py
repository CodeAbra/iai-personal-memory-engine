from __future__ import annotations

import argparse
import importlib.resources as _res
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lock"
SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"
STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

LAUNCHD_TARGET: Path = Path.home() / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist"
SYSTEMD_TARGET: Path = Path.home() / ".config" / "systemd" / "user" / "iai-mcp-daemon.service"
WINDOWS_TASK_XML: Path = Path.home() / ".iai-mcp" / "iai-mcp-daemon.task.xml"
WINDOWS_START_CMD: Path = Path.home() / ".iai-mcp" / "daemon-start.cmd"

DAEMON_LABEL: str = "com.iai-mcp.daemon"
SERVICE_NAME: str = "iai-mcp-daemon.service"
WINDOWS_TASK_NAME: str = "iai-mcp-daemon"

CONSENT_BANNER: str = """\
==============================================================================
iai Sleep Daemon -- First Install Consent
==============================================================================

The sleep daemon runs in the background between Claude Code sessions to
perform neural consolidation (REM cycles, schema induction, drift detection).

Resource cost:
  - RAM: ~400 MB (bge-small-en-v1.5 embedding model kept warm to avoid cold-start)
  - CPU: brief bursts during REM cycles inside your learned quiet window
  - Disk: ~50MB/week in event logs + schema candidates

Claude subscription impact:
  - Max 1 `claude -p` call per night ("lucid moment" main insight)
  - Hard cap: 1% of daily subscription quota, 7% weekly buffer
  - ZERO API costs (no paid-API key -- uses your subscription only)

Opt out anytime:
  iai-mcp daemon uninstall

Continue? [y/N]: """


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _ensure_crypto_key_present():
    if os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        return None
    from iai_mcp.crypto import KEY_BYTES, CryptoKey
    ck = CryptoKey(user_id="default")
    path = ck._key_file_path()
    if path.exists():
        return None
    import secrets as _secrets
    fresh = _secrets.token_bytes(KEY_BYTES)
    ck._try_file_set(fresh)
    print(f"crypto: created {path} (mode 0o600, {KEY_BYTES} bytes)")
    return path


def _try_short_timeout_connect(timeout_ms: int = 250) -> bool:
    import socket as _socket

    from iai_mcp._ipc import make_sync_ipc_socket, send_sync_auth_token
    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
    try:
        s, _addr = make_sync_ipc_socket(addr=sock_path)
    except FileNotFoundError:
        return False
    s.settimeout(timeout_ms / 1000.0)
    try:
        s.connect(_addr)
        send_sync_auth_token(s)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError, _socket.timeout):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass




def _send_jsonrpc_request(
    method: str,
    params: dict,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
) -> dict | None:
    import asyncio
    from iai_mcp.cli._capture import _is_custom_store as _isc
    if not os.environ.get("IAI_DAEMON_SOCKET_PATH") and _isc():
        return None

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)

    async def _runner() -> dict | None:
        try:
            from iai_mcp._ipc import open_ipc_connection
            reader, writer = await asyncio.wait_for(
                open_ipc_connection(sock_path),
                timeout=connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError, asyncio.TimeoutError):
            return None
        try:
            req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=read_timeout)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            logger.debug("jsonrpc request failed: %s", exc)
            return None
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    try:
        return asyncio.run(_runner())
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("jsonrpc asyncio.run failed: %s", exc)
        return None




def _send_socket_request(req: dict, *, timeout: float = 30.0) -> dict | None:
    import asyncio

    async def _runner() -> dict | None:
        _sock = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
        try:
            from iai_mcp._ipc import open_ipc_connection
            reader, writer = await asyncio.wait_for(
                open_ipc_connection(_sock),
                timeout=5.0,
            )
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        except OSError:
            return None
        try:
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return asyncio.run(_runner())


def compute_session_start_tokens_p90(store: "MemoryStore") -> dict[str, int | None]:
    import statistics

    from iai_mcp.events import query_events

    events = query_events(store, kind="session_started", limit=100)
    samples = [
        int(e["data"]["total_cached_tokens"])
        for e in events
        if isinstance(e.get("data"), dict) and "total_cached_tokens" in e["data"]
    ]
    if not samples:
        return {"p90": None, "n_samples": 0}
    if len(samples) == 1:
        return {"p90": samples[0], "n_samples": 1}
    q = statistics.quantiles(samples, n=10, method="inclusive")
    p90 = int(round(q[8]))
    return {"p90": p90, "n_samples": len(samples)}






def _claude_desktop_config_path() -> Path | None:
    import platform as _plat
    home = Path.home()
    sysname = _plat.system()
    if sysname == "Darwin":
        p = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sysname == "Windows":
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        p = Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(home / ".config")
        p = Path(xdg) / "Claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None






def _maintenance_compact_metrics(
    hippo_dir: Path,
    store: object | None = None,
) -> dict:
    db_path = hippo_dir / "brain.sqlite3"
    size_bytes = 0
    try:
        if db_path.exists():
            size_bytes = db_path.stat().st_size
    except OSError:
        pass
    size_mb = round(size_bytes / (1024 * 1024), 1)
    records_count = 0
    record_id_set: set[str] = set()
    if store is not None:
        try:
            tbl = store.db.open_table("records")
            records_count = int(tbl.count_rows())
            df = tbl.search().select(["id"]).to_pandas()
            record_id_set = {str(x) for x in df["id"].tolist()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("compact metrics read failed: %s", exc)
    return {
        "db_size_mb": size_mb,
        "records_count": records_count,
        "record_id_set": record_id_set,
    }


from ._analytics import (
    cmd_health,
    cmd_topology,
    cmd_trajectory,
    cmd_audit,
    cmd_bank_recall,
    cmd_build_native,
    cmd_migrate,
    cmd_migrate_to_lilli,
    cmd_deferred_unlock_dead_pids,
)

from ._maintenance import (
    cmd_edge_backfill,
    cmd_idem_dedup,
    cmd_schema_cleanup,
    cmd_maintenance_compact_hippo,
    cmd_maintenance_compact_records,
    cmd_maintenance_symmetrize_self_loops,
    cmd_lifecycle_force_unlock,
    cmd_lifecycle_status,
    cmd_maintenance_sleep_cycle,
    cmd_drain_permanent_failed,
    cmd_deferred_drain,
    _maintenance_compact_preflight_daemon_alive,
    _maintenance_compact_dry_run,
    _maintenance_compact_apply,
    _format_relative,
    _print_drain_result,
)

from ._crypto import (
    cmd_crypto_status,
    cmd_crypto_rotate,
    cmd_crypto_recover_prior_key,
    cmd_crypto_redact_undecryptable,
    cmd_crypto_migrate_to_file,
    cmd_crypto_init,
)

from ._capture import (
    _truncate_for_claude_code_hook,
    _is_custom_store,
    cmd_session_start,
    get_other_sessions_live_size,
    read_live_fingerprint,
    write_live_fingerprint,
    get_max_created_at,
    _utc_iso,
    read_watermark,
    write_watermark,
    cmd_session_refresh_if_stale,
    cmd_capture_transcript,
    cmd_capture_turn_deferred,
    _capture_hook_paths,
    _turn_hook_paths,
    _resolve_wrapper_path,
    _build_iai_mcp_server_entry,
    _patch_claude_desktop_config,
    _patch_claude_code_config,
    _CAPTURE_HOOK_MARKER,
    _TURN_HOOK_MARKER,
    _SESSION_RECALL_HOOK_MARKER,
    _session_recall_hook_paths,
    _load_settings,
    cmd_capture_hooks_install,
    cmd_capture_hooks_uninstall,
    cmd_capture_hooks_status,
)

from ._cowork import (
    cmd_cowork_install,
    cmd_cowork_uninstall,
    cmd_cowork_status,
)

from ._daemon import (
    cmd_daemon_install,
    cmd_daemon_uninstall,
    cmd_daemon_restart,
    cmd_daemon_start,
    cmd_daemon_stop,
    cmd_self_update,
    cmd_daemon_status,
    cmd_daemon_logs,
    cmd_daemon_force_rem,
    cmd_daemon_pause,
    cmd_daemon_resume,
    cmd_daemon_stats,
    cmd_daemon_rss_stats,
    cmd_daemon_configure,
    _launchd_template,
    _render_launchd_plist,
    _render_systemd_unit,
    _render_windows_task_xml,
    _render_windows_start_cmd,
    _install_windows_task,
    _prompt_consent,
    _record_consent_receipt,
    _remove_state_files,
    _compute_p90_from_events,
    _render_daemon_stats,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iai-mcp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="show LLM health status")
    h.set_defaults(func=cmd_health)

    bn = sub.add_parser(
        "build-native",
        help=(
            "compile the Rust native extension (iai_mcp_native) in-place. "
            "Run after Python upgrade or on fresh clone. Requires cargo."
        ),
    )
    bn.set_defaults(func=cmd_build_native)

    m = sub.add_parser(
        "migrate",
        help=(
            "migrate records: 1->2 (schema) or 2->3 (encryption); "
            "OR --resume / --rollback a partial reembed migration"
        ),
    )
    m.add_argument("--from", dest="from_", type=int, default=1)
    m.add_argument("--to", type=int, default=2)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--verbose", "-v", action="store_true")
    m.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial reembed migration from migration_progress.json checkpoint.",
    )
    m.add_argument(
        "--rollback",
        action="store_true",
        help=(
            "Roll back a partial reembed migration: drop records_v_new and "
            "(if needed) restore records from records_old_<ts>."
        ),
    )
    m.add_argument(
        "--rederive-timestamps",
        action="store_true",
        help=(
            "Re-derive collapsed created_at timestamps from on-disk transcripts. "
            "One-time operation; idempotent. Records with no recoverable transcript "
            "are left unchanged."
        ),
    )
    m.add_argument(
        "--reembed-from-text",
        action="store_true",
        help=(
            "Re-embed every episodic record from its verbatim literal_surface "
            "text and rebuild the recall index. Repairs records whose stored "
            "vector was an embedding of the provenance cue label rather than "
            "the message content. One-time operation; idempotent. Only the "
            "embedding column is rewritten -- literal_surface is never touched."
        ),
    )
    m.add_argument(
        "--reembed-to-configured-provider",
        action="store_true",
        help=(
            "Re-embed every record with the configured embedding provider, "
            "including dimension changes. Uses the crash-safe staging-table "
            "migration and retains the previous table until cleanup."
        ),
    )
    m.add_argument(
        "--reembed-batch-size",
        type=int,
        default=256,
        help=(
            "Records per batch for --reembed-from-text or "
            "--reembed-to-configured-provider. Default 256."
        ),
    )
    m.add_argument(
        "--dedupe-episodic",
        action="store_true",
        help=(
            "Tombstone duplicate episodic records sharing an idem-tag (cleanup "
            "for the capture_turn() check-then-insert race, fixed separately). "
            "One-time operation; idempotent. Soft-delete only -- literal_surface, "
            "provenance, and embeddings are never touched."
        ),
    )
    m.add_argument(
        "--salvage-torn-permanent-failed",
        action="store_true",
        help=(
            "Recover torn .permanent-failed-*.jsonl files (drained mid-write "
            "before the atomic-write fix). Trims each file to the last "
            "newline-terminated record, re-deferres the salvageable prefix, "
            "and quarantines the original. Idempotent; soft-recovery only."
        ),
    )
    m.set_defaults(func=cmd_migrate)

    mtl = sub.add_parser(
        "migrate-to-lilli",
        help=(
            "copy a SQLite Hippo store verbatim into a fresh lilli store, then "
            "verify equality (does NOT swap the live store)"
        ),
    )
    mtl.add_argument(
        "--src",
        required=True,
        help=(
            "source SQLite store db file "
            "(e.g. /path/to/alice-store/hippo/brain.sqlite3)"
        ),
    )
    mtl.add_argument(
        "--dst",
        required=True,
        help=(
            "NEW destination store root (must not be the live store; the lilli "
            "schema is auto-created there)"
        ),
    )
    mtl.add_argument(
        "--batch",
        type=int,
        default=500,
        help="rows per streamed batch (default 500).",
    )
    mtl.add_argument(
        "--prune-telemetry-before",
        dest="prune_telemetry_before",
        default=None,
        help=(
            "skip events older than this UTC datetime string (telemetry only; "
            "off by default -- lossless, every event copied)."
        ),
    )
    mtl.add_argument("--verbose", "-v", action="store_true")
    mtl.set_defaults(func=cmd_migrate_to_lilli)

    udp = sub.add_parser(
        "deferred-unlock-dead-pids",
        help=(
            "Unlock .processing-<pid>.jsonl files in ~/.iai-mcp/.deferred-captures/ "
            "whose owner PID is dead or differs from the live daemon's PID. Renames "
            "each back to .jsonl so the next drain claims it. Idempotent; "
            "collision-safe. Run while the daemon is stopped to avoid racing the drain."
        ),
    )
    udp.add_argument("--dry-run", action="store_true")
    udp.add_argument("--json", action="store_true", help="emit a JSON result")
    udp.add_argument(
        "--force",
        action="store_true",
        help=(
            "proceed even when a live daemon is detected; "
            "use only when you are certain no daemon is writing to the drain"
        ),
    )
    udp.set_defaults(func=cmd_deferred_unlock_dead_pids)

    c = sub.add_parser(
        "crypto",
        help="encryption key management",
    )
    crypto_sub = c.add_subparsers(dest="crypto_cmd", required=True)

    cs = crypto_sub.add_parser(
        "status",
        help=(
            "show file-backend key status: backend, path, "
            "mode, uid, length validation, passphrase-fallback flag"
        ),
    )
    cs.add_argument("--user-id", dest="user_id", default="default")
    cs.set_defaults(func=cmd_crypto_status)

    cr = crypto_sub.add_parser(
        "rotate", help="rotate encryption key + re-encrypt all records"
    )
    cr.add_argument("--user-id", dest="user_id", default="default")
    cr.set_defaults(func=cmd_crypto_rotate)

    mtf = crypto_sub.add_parser(
        "migrate-to-file",
        help=(
            "one-time: read existing key from macOS Keychain "
            "and write to .crypto.key file (interactive Terminal only)"
        ),
    )
    mtf.add_argument("--user-id", dest="user_id", default="default")
    mtf_group = mtf.add_mutually_exclusive_group()
    mtf_group.add_argument(
        "--keep-keychain",
        dest="keep_keychain",
        action="store_true",
        default=True,
        help="leave the existing macOS Keychain entry in place (default)",
    )
    mtf_group.add_argument(
        "--delete-keychain",
        dest="keep_keychain",
        action="store_false",
        help="delete the macOS Keychain entry after successful migration",
    )
    mtf.set_defaults(func=cmd_crypto_migrate_to_file)

    ci = crypto_sub.add_parser(
        "init",
        help=(
            "generate a fresh .crypto.key file "
            "(fresh installs only — refuses if file exists)"
        ),
    )
    ci.add_argument("--user-id", dest="user_id", default="default")
    ci.set_defaults(func=cmd_crypto_init)

    rwpk = crypto_sub.add_parser(
        "recover-with-prior-key",
        help=(
            "stage all records, decrypt literal/provenance/gain with current "
            "then prior key, re-encrypt under current key; atomic Lance swap"
        ),
    )
    rwpk.add_argument(
        "--prior-key-file",
        type=Path,
        required=True,
        help="path to exactly 32 raw AES key bytes (same format as .crypto.key)",
    )
    rwpk.add_argument("--user-id", dest="user_id", default="default")
    rwpk.add_argument(
        "--dry-run",
        action="store_true",
        help="report rows that need the prior key without mutating tables",
    )
    rwpk.set_defaults(func=cmd_crypto_recover_prior_key)

    cred = crypto_sub.add_parser(
        "redact-undecryptable",
        help=(
            "replace literal_surface that fails AES-GCM decrypt with a redacted "
            "marker (preserves embeddings, edges, metadata)"
        ),
    )
    cred.add_argument("--user-id", dest="user_id", default="default")
    cred.set_defaults(func=cmd_crypto_redact_undecryptable)

    t = sub.add_parser(
        "trajectory",
        help="aggregate M1..M6 trajectory events",
    )
    t.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    t.set_defaults(func=cmd_trajectory)

    topo = sub.add_parser(
        "topology",
        help="live small-world topology snapshot: C, L, sigma, communities, rich-club ratio, N, regime",
    )
    topo.set_defaults(func=cmd_topology)

    cap = sub.add_parser(
        "capture-transcript",
        help=(
            "batch-capture a Claude Code JSONL transcript into episodic tier. "
            "Used by the Stop hook for ambient WRITE-side observation capture."
        ),
    )
    cap.add_argument("transcript_path", help="path to the Claude Code JSONL transcript file")
    cap.add_argument("--session-id", default="-", help="session id for provenance")
    cap.add_argument("--max-turns", type=int, default=200,
                     help="cap on turns to scan (default 200; older turns skipped)")
    cap.add_argument(
        "--no-spawn",
        action="store_true",
        default=False,
        help=(
            "Hook-only mode: try connect with 250ms timeout. On miss, write "
            "transcript to ~/.iai-mcp/.deferred-captures/ and exit 0 within 2s. "
            "NEVER spawn daemon. Used by ~/.claude/hooks/iai-mcp-session-capture.sh "
            "to eliminate spawn vector."
        ),
    )
    cap.set_defaults(func=cmd_capture_transcript)

    ctd = sub.add_parser(
        "capture-turn-deferred",
        help=(
            "append a single JSONL event per new transcript turn to "
            "{session_id}.live.jsonl. UserPromptSubmit-hook backend."
        ),
    )
    ctd.add_argument("--session-id", required=True)
    ctd.add_argument("--transcript-path", required=True)
    ctd.add_argument(
        "--max-turns-per-call",
        type=int,
        default=200,
        help="max new turns to process per invocation (default 200)",
    )
    ctd.set_defaults(func=cmd_capture_turn_deferred)

    ssp = sub.add_parser(
        "session-start",
        help=(
            "print the session-start recall payload as markdown on stdout. "
            "Hook target for ~/.claude/hooks/iai-mcp-session-recall.sh."
        ),
    )
    ssp.add_argument("--session-id", default="-", help="session id for provenance")
    ssp.set_defaults(func=cmd_session_start)

    sris = sub.add_parser(
        "session-refresh-if-stale",
        help=(
            "UserPromptSubmit hook gate: compare MAX(created_at) against the "
            "per-session watermark sidecar; call session_refresh_if_stale RPC "
            "only when new memory exists; emit additionalContext JSON on trigger."
        ),
    )
    sris.add_argument("--session-id", default="-", help="session id for watermark sidecar")
    sris.set_defaults(func=cmd_session_refresh_if_stale)

    ch = sub.add_parser(
        "capture-hooks",
        help=(
            "install/uninstall/status the ambient-capture hooks "
            "(Claude Code, Codex, Cursor, Antigravity, Hermes, OpenClaw)"
        ),
    )
    ch_sub = ch.add_subparsers(dest="capture_hooks_cmd", required=True)
    _target_kwargs = {
        "choices": (
            "claude", "codex", "cursor", "antigravity", "hermes",
            "openclaw", "all",
        ),
        "default": "claude",
        "help": (
            "host to wire: claude (default), codex, cursor, antigravity "
            "(CLI), hermes (>= 0.5.0), openclaw (MCP tools only), or all"
        ),
    }
    ch_install = ch_sub.add_parser(
        "install",
        help="copy the hook scripts and register them with the host",
    )
    ch_install.add_argument("--target", **_target_kwargs)
    ch_install.set_defaults(func=cmd_capture_hooks_install)
    ch_uninstall = ch_sub.add_parser(
        "uninstall",
        help="remove the hook scripts and their host registration",
    )
    ch_uninstall.add_argument("--target", **_target_kwargs)
    ch_uninstall.set_defaults(func=cmd_capture_hooks_uninstall)
    ch_status = ch_sub.add_parser(
        "status",
        help="show whether the hooks are installed and active",
    )
    ch_status.add_argument("--target", **_target_kwargs)
    ch_status.set_defaults(func=cmd_capture_hooks_status)

    cw = sub.add_parser(
        "cowork",
        help=(
            "install/uninstall/status the iai-mcp plugin for Claude Cowork "
            "(desktop local-agent mode): recall + capture hooks in every "
            "Cowork session"
        ),
    )
    cw_sub = cw.add_subparsers(dest="cowork_cmd", required=True)
    cw_sub.add_parser(
        "install",
        help="stage the plugin marketplace under ~/.iai-mcp/ and wire it into "
             "each Cowork home's cowork_settings.json",
    ).set_defaults(func=cmd_cowork_install)
    cw_sub.add_parser(
        "uninstall",
        help="unwire the plugin from Cowork and remove the staged marketplace",
    ).set_defaults(func=cmd_cowork_uninstall)
    cw_sub.add_parser(
        "status",
        help="show plugin staging and per-home Cowork wiring",
    ).set_defaults(func=cmd_cowork_status)

    a = sub.add_parser(
        "audit",
        help="identity + shield audit log",
    )
    a.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    a.add_argument(
        "--severity",
        choices=["info", "warning", "critical"],
        default=None,
        help="filter by severity",
    )
    audit_sub = a.add_subparsers(dest="audit_sub")
    for name, helptext in (
        ("shield", "shield-only audit (match counts redacted)"),
        ("drift", "detect M4 drift anomaly and surface it"),
        ("identity", "s5_* identity events only"),
    ):
        sp = audit_sub.add_parser(name, help=helptext)
        sp.add_argument("--since", type=int, default=None)
        sp.add_argument(
            "--severity",
            choices=["info", "warning", "critical"],
            default=None,
        )
    a.set_defaults(func=cmd_audit)

    d = sub.add_parser(
        "daemon",
        help="sleep daemon: install/uninstall/start/stop/status/logs/...",
    )
    daemon_sub = d.add_subparsers(dest="daemon_cmd", required=True)

    di = daemon_sub.add_parser(
        "install",
        help=(
            "install launchd plist (macOS) / systemd user unit (Linux); "
            "first-run consent banner unless --yes"
        ),
    )
    di.add_argument(
        "--dry-run",
        action="store_true",
        help="print plist/unit contents without writing or invoking launchctl/systemctl",
    )
    di.add_argument(
        "--yes", "-y",
        action="store_true",
        help="skip the consent banner (records --yes audit-trail still)",
    )
    di.set_defaults(func=cmd_daemon_install)

    du = daemon_sub.add_parser(
        "uninstall",
        help="C4 clean uninstall: remove plist/unit + 3 state files",
    )
    du.add_argument("--yes", "-y", action="store_true")
    du.set_defaults(func=cmd_daemon_uninstall)

    daemon_sub.add_parser(
        "start", help="launchctl kickstart / systemctl --user start",
    ).set_defaults(func=cmd_daemon_start)

    daemon_sub.add_parser(
        "stop", help="launchctl kill SIGTERM / systemctl --user stop",
    ).set_defaults(func=cmd_daemon_stop)

    daemon_sub.add_parser(
        "restart", help="stop (waits for the old pid to die) then start",
    ).set_defaults(func=cmd_daemon_restart)

    su = sub.add_parser(
        "self-update",
        help=(
            "upgrade the iai-pme wheel from PyPI AND restart the daemon — "
            "`pip install -U` alone leaves the old engine running. Refuses "
            "editable/source checkouts. --check prints the pending version "
            "without changing anything."
        ),
    )
    su.add_argument(
        "--check", action="store_true", default=False,
        help="only report whether a newer version exists; change nothing",
    )
    su.add_argument(
        "--yes", "-y", action="store_true", default=False,
        help="skip the confirmation prompt",
    )
    su.set_defaults(func=cmd_self_update)

    daemon_sub.add_parser(
        "status",
        help=(
            "socket round-trip: print daemon FSM state, uptime, version "
            "(warns on version skew vs installed package)"
        ),
    ).set_defaults(func=cmd_daemon_status)

    dlogs = daemon_sub.add_parser(
        "logs",
        help="tail daemon log file (macOS ~/.iai-mcp/logs) or journalctl (Linux)",
    )
    dlogs.add_argument("-f", "--follow", action="store_true")
    dlogs.add_argument("-n", "--lines", type=int, default=50)
    dlogs.set_defaults(func=cmd_daemon_logs)

    daemon_sub.add_parser(
        "force-rem",
        help="cooperative force: trigger one REM cycle out-of-schedule",
    ).set_defaults(func=cmd_daemon_force_rem)

    dpause = daemon_sub.add_parser(
        "pause", help="pause daemon scheduler for N seconds",
    )
    dpause.add_argument("seconds", type=int)
    dpause.set_defaults(func=cmd_daemon_pause)

    daemon_sub.add_parser(
        "resume", help="resume daemon scheduler after a pause",
    ).set_defaults(func=cmd_daemon_resume)

    daemon_sub.add_parser(
        "stats",
        help=(
            "VAL-02 longitudinal metrics: session_start_tokens_p90 over the "
            "most recent 100 session_started events (persisted in the events table)"
        ),
    ).set_defaults(func=cmd_daemon_stats)

    daemon_sub.add_parser(
        "rss-stats",
        help=(
            "print the current resident-set picture (RSS, vmmap regions, "
            "allocator pools) plus the most recent per-step resident-set deltas"
        ),
    ).set_defaults(func=cmd_daemon_rss_stats)

    dconf = daemon_sub.add_parser(
        "configure",
        help=(
            "per-setting override: set-budget / set-cycle-count / "
            "set-quiet-window / disable-claude / enable-claude"
        ),
    )
    dconf.add_argument(
        "key",
        choices=[
            "set-budget",
            "set-cycle-count",
            "set-quiet-window",
            "disable-claude",
            "enable-claude",
        ],
    )
    dconf.add_argument("value", nargs="?", default=None)
    dconf.set_defaults(func=cmd_daemon_configure)

    sc = sub.add_parser(
        "schema-cleanup",
        help=(
            "soft-delete duplicate schema records. Default "
            "mode is --dry-run; --apply snapshots the memory store dir and "
            "performs the cleanup. Idempotent (re-running is a no-op)."
        ),
    )
    sc_mode = sc.add_mutually_exclusive_group()
    sc_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print the cleanup diff without mutating the store",
    )
    sc_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="snapshot the store dir + soft-delete duplicates",
    )
    sc.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    sc.set_defaults(func=cmd_schema_cleanup)

    idm = sub.add_parser(
        "idem-dedup",
        help=(
            "collapse live records sharing one idempotency key down to the "
            "earliest copy (tombstones the rest). Default mode is --dry-run; "
            "--apply snapshots the store dir first. Idempotent."
        ),
    )
    idm_mode = idm.add_mutually_exclusive_group()
    idm_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print the duplicate groups without mutating the store",
    )
    idm_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="snapshot the store dir + tombstone the extra copies",
    )
    idm.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    idm.set_defaults(func=cmd_idem_dedup)

    ebf = sub.add_parser(
        "edge-backfill",
        help=(
            "re-link live knowledge summaries to their source moments: a "
            "summary anchored by at least one surviving consolidation edge "
            "claims the rest of its community's live members. Default mode "
            "is --dry-run; --apply snapshots the store dir first. Idempotent."
        ),
    )
    ebf_mode = ebf.add_mutually_exclusive_group()
    ebf_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) report proposed links without mutating the store",
    )
    ebf_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="snapshot the store dir + write the proposed consolidation edges",
    )
    ebf.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    ebf.set_defaults(func=cmd_edge_backfill)

    mtn = sub.add_parser(
        "maintenance",
        help=(
            "one-shot maintenance ops. Currently: compact-hippo "
            "(PRAGMA wal_checkpoint + VACUUM + hnswlib rebuild)."
        ),
    )
    mtn_sub = mtn.add_subparsers(dest="maintenance_cmd", required=True)
    mtn_compact = mtn_sub.add_parser(
        "compact-hippo",
        help=(
            "compact Hippo storage: wal_checkpoint + VACUUM + hnswlib rebuild. "
            "DAEMON MUST BE STOPPED. Default --dry-run; --apply requires "
            "--yes for non-tty."
        ),
    )
    mtn_compact_mode = mtn_compact.add_mutually_exclusive_group()
    mtn_compact_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print metrics-only JSON; do NOT call optimize",
    )
    mtn_compact_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="run wal_checkpoint + VACUUM + hnswlib rebuild on Hippo storage",
    )
    mtn_compact.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_compact.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors `schema-cleanup` flag."
        ),
    )
    mtn_compact.set_defaults(func=cmd_maintenance_compact_hippo)
    mtn_compact_legacy = mtn_sub.add_parser(
        "compact-records",
        help="Deprecated alias for compact-hippo (kept for one release).",
    )
    mtn_compact_legacy_mode = mtn_compact_legacy.add_mutually_exclusive_group()
    mtn_compact_legacy_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print metrics-only JSON; do NOT call optimize",
    )
    mtn_compact_legacy_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="run wal_checkpoint + VACUUM + hnswlib rebuild on Hippo storage",
    )
    mtn_compact_legacy.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_compact_legacy.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors `schema-cleanup` flag."
        ),
    )
    mtn_compact_legacy.set_defaults(func=cmd_maintenance_compact_records)

    mtn_symmetrize = mtn_sub.add_parser(
        "symmetrize-self-loops",
        help=(
            "backfill missing hebbian self-loops on existing records. "
            "DAEMON MUST BE STOPPED. Default --dry-run; --apply requires "
            "--yes for non-tty."
        ),
    )
    mtn_symmetrize_mode = mtn_symmetrize.add_mutually_exclusive_group()
    mtn_symmetrize_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print counts JSON; do NOT write self-loops",
    )
    mtn_symmetrize_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="write missing self-loops at delta=0.1 (hebbian edge_type)",
    )
    mtn_symmetrize.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_symmetrize.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors compact-hippo flag."
        ),
    )
    mtn_symmetrize.set_defaults(func=cmd_maintenance_symmetrize_self_loops)

    mtn_sleep = mtn_sub.add_parser(
        "sleep-cycle",
        help=(
            "run the 5-step sleep pipeline once: "
            "schema_mine, knob_tune, dream_decay, optimize_hippo, "
            "compact_records. 3-strike auto-quarantine; use --force "
            "to override, --reset-quarantine to clear."
        ),
    )
    mtn_sleep.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="run even if quarantined (operator override)",
    )
    mtn_sleep.add_argument(
        "--reset-quarantine",
        dest="reset_quarantine",
        action="store_true",
        default=False,
        help="clear quarantine state before running",
    )
    mtn_sleep.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    mtn_sleep.set_defaults(func=cmd_maintenance_sleep_cycle)

    doc = sub.add_parser(
        "doctor",
        help=(
            "Diagnose daemon health (incl. (g) duplicate-binder detection). "
            "With --apply, attempt safe repairs "
            "(unlink stale socket, kill duplicate binders, cleanup orphans, "
            "respawn daemon). With --apply --yes, skip confirmations. "
            "Exit 0=all green, 1=any FAIL, 2=--apply tried but FAIL persists."
        ),
    )
    doc.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="attempt safe repairs after diagnosis; prompts before each destructive action",
    )
    doc.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip confirmation prompts; equivalent to typing 'y' to all",
    )
    doc.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help=(
            "unattended reflex mode: run ONLY the auto-safe heal subset "
            "(no prompts, no process kills, no store mutations), damped to "
            "once per 6h. Spawned by the MCP wrapper when the daemon stays "
            "unreachable after a wake attempt."
        ),
    )
    doc.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "force headless mode (downgrade `(n) HID idle source` and "
            "`(b) socket file fresh` from FAIL to WARN). Auto-detected on "
            "Linux when DISPLAY/WAYLAND_DISPLAY are unset; on macOS use this "
            "flag explicitly."
        ),
    )
    def _cmd_doctor_lazy(args: argparse.Namespace) -> int:
        from iai_mcp.doctor import cmd_doctor
        return cmd_doctor(args)
    doc.set_defaults(func=_cmd_doctor_lazy)

    lc = sub.add_parser(
        "lifecycle",
        help=(
            "inspect lifecycle state machine "
            "(WAKE/DROWSY/SLEEP/HIBERNATION). Currently: status."
        ),
    )
    lc_sub = lc.add_subparsers(dest="lifecycle_cmd", required=True)
    lc_status = lc_sub.add_parser(
        "status",
        help=(
            "print current lifecycle state, since-ts, last activity, "
            "wrapper event seq, sleep-cycle progress, quarantine, and "
            "shadow_run flag"
        ),
    )
    lc_status.set_defaults(func=cmd_lifecycle_status)

    lc_unlock = lc_sub.add_parser(
        "force-unlock",
        help=(
            "clear a stale ~/.iai-mcp/.locked lockfile and "
            "print the prior PID / hostname / started_at"
        ),
    )
    lc_unlock.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive [y/N] prompt",
    )
    lc_unlock.set_defaults(func=cmd_lifecycle_force_unlock)

    br = sub.add_parser(
        "bank-recall",
        help=(
            "substring recall over bank/processed + bank/recent without "
            "booting the daemon. Used by the wrapper as a socket-dead "
            "fallback path."
        ),
    )
    br.add_argument("--query", required=True, help="cue substring to match")
    br.add_argument(
        "--limit", type=int, default=20, help="max hits (default 20)"
    )
    br.add_argument(
        "--processed-only", action="store_true", default=False
    )
    br.add_argument(
        "--recent-only", action="store_true", default=False
    )
    br.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="emit JSON to stdout (current default; --no-json is reserved)",
    )
    br.set_defaults(func=cmd_bank_recall)

    dpf = sub.add_parser(
        "drain-permanent-failed",
        help=(
            "recover terminal .permanent-failed-*.jsonl files from "
            ".deferred-captures/. Routes through daemon socket when daemon "
            "is running; direct-open fallback when daemon is down. "
            "--dry-run lists files without mutating anything."
        ),
    )
    dpf.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="list terminal files + event counts without inserting or renaming",
    )
    dpf.set_defaults(func=cmd_drain_permanent_failed)

    dd = sub.add_parser(
        "deferred-drain",
        help=(
            "recover the full deferred-capture backlog offline — plain, crash, "
            "and quarantined capture files — into the store. Runs one "
            "byte-bounded subprocess per batch so memory is reclaimed between "
            "batches. Requires the daemon stopped (single-writer). "
            "Permanently-failed files are recovered separately via "
            "drain-permanent-failed. --dry-run reports the plan without "
            "mutating."
        ),
    )
    dd.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="report the file list, event counts and batch plan without "
             "inserting or removing anything",
    )
    dd.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON result",
    )
    dd.add_argument(
        "--batch-bytes",
        type=int,
        default=100 * 1024 * 1024,
        help="maximum cumulative bytes of capture files per subprocess batch "
             "(default ~100 MB)",
    )
    dd.add_argument(
        "--force",
        action="store_true",
        help="proceed even if a live daemon is detected — relies on the store "
             "lock as the real guard",
    )
    dd.set_defaults(func=cmd_deferred_drain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
