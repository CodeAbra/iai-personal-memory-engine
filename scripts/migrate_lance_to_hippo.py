#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_log_fh: Any = None


def tee_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)
    if _log_fh is not None:
        file_kwargs = dict(kwargs)
        file_kwargs["file"] = _log_fh
        file_kwargs["flush"] = True
        print(*args, **file_kwargs)


def _import_lancedb_or_die():
    try:
        import lancedb  # noqa: F401
        return lancedb
    except ImportError as exc:
        print(
            "ERROR: lancedb is required for this migration but is not "
            "installed. Install via:\n"
            "    pip install iai-pme[migration]\n"
            f"(Original error: {type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        sys.exit(6)


def pre_flight_daemon_alive(store_root: Path) -> tuple[bool, str | None]:
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("iai_mcp.daemon" in part for part in cmdline):
                    return True, f"daemon process alive (pid={proc.info['pid']})"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass

    sock_path = store_root / ".daemon.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            probe.connect(str(sock_path))
        return True, "daemon socket responding"
    except FileNotFoundError:
        pass
    except ConnectionRefusedError:
        pass
    except OSError:
        pass

    return False, None


def backup_lancedb(store_root: Path, ts: str) -> Path:
    source = store_root / "lancedb"
    dest = store_root / f"lancedb.pre-migrate-{ts}"
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-a", str(source), str(dest)], check=True)
    else:
        import shutil
        shutil.copytree(str(source), str(dest), copy_function=shutil.copy2)
    tee_print(f"  backup: {source} -> {dest}")
    return dest


def move_to_trash(path: Path, label: str) -> Path:
    if platform.system() == "Darwin":
        if subprocess.run(["which", "trash"], capture_output=True).returncode == 0:
            subprocess.run(["trash", str(path)], check=True)
            return Path.home() / ".Trash" / path.name
        dest = Path.home() / ".Trash" / label
        import shutil
        shutil.move(str(path), str(dest))
        return dest
    else:
        if subprocess.run(["which", "gio"], capture_output=True).returncode == 0:
            subprocess.run(["gio", "trash", str(path)], check=True)
            return Path("~/.local/share/Trash/files") / label
        import shutil
        dest = Path.home() / ".local" / "share" / "Trash" / "files" / label
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        return dest


def write_failure_json(
    store_root: Path,
    ts: str,
    mismatches: list[dict],
    stage_results: dict,
    hnsw_result: dict | None = None,
    backup_path: Path | None = None,
) -> Path:
    fail_path = store_root / f".migrate-FAILED-{ts}.json"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage_results": stage_results,
        "hnsw_rebuild": hnsw_result,
        "mismatches": mismatches[:100],
        "backup_path": str(backup_path) if backup_path else None,
        "recovery_steps": [
            f"Run: python scripts/migrate_lance_to_hippo.py --rollback --rollback-ts {ts}",
            f"Or manually: mv {store_root}/hippo ~/.Trash/hippo-failed-{ts}/",
            f"And: mv {store_root}/lancedb.pre-migrate-{ts}/ {store_root}/lancedb/",
            "Investigate the mismatches list above",
        ],
    }
    try:
        fail_path.write_text(json.dumps(payload, indent=2, default=str))
        os.chmod(str(fail_path), 0o600)
    except OSError as exc:
        tee_print(
            f"WARNING: could not write failure JSON to {fail_path}: {exc}",
            file=sys.stderr,
        )
    return fail_path


def rollback(store_root: Path, ts: str | None) -> dict:
    if ts is None:
        failed_files = sorted(
            store_root.glob(".migrate-FAILED-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not failed_files:
            print(
                "ERROR: no .migrate-FAILED-*.json found in store; "
                "provide --rollback-ts <ts> explicitly.",
                file=sys.stderr,
            )
            sys.exit(7)
        ts = failed_files[0].stem.replace(".migrate-FAILED-", "")

    backup_path = store_root / f"lancedb.pre-migrate-{ts}"
    hippo_path = store_root / "hippo"
    failed_lance = store_root / "lancedb"

    if not backup_path.exists():
        print(
            f"ERROR: backup not found at {backup_path}",
            file=sys.stderr,
        )
        sys.exit(8)

    now_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if hippo_path.exists():
        move_to_trash(hippo_path, f"hippo-failed-{now_ts}")
        print(f"Moved failed hippo tree to Trash")

    if failed_lance.exists():
        move_to_trash(failed_lance, f"lancedb-mid-migrate-{now_ts}")
        print(f"Moved partial lancedb tree to Trash")

    os.rename(str(backup_path), str(failed_lance))
    print(f"Restored {backup_path} -> {failed_lance}")

    return {"action": "rollback", "ts": ts, "restored_to": str(failed_lance)}


_RECORDS_COLS = [
    ("id", "id", "direct"),
    ("tier", "tier", "direct"),
    ("literal_surface", "literal_surface", "direct"),
    ("aaak_index", "aaak_index", "direct"),
    ("embedding", "embedding", "embed"),
    ("structure_hv", "structure_hv", "bytes"),
    ("community_id", "community_id", "direct"),
    ("centrality", "centrality", "direct"),
    ("detail_level", "detail_level", "direct"),
    ("pinned", "pinned", "direct"),
    ("stability", "stability", "direct"),
    ("difficulty", "difficulty", "direct"),
    ("last_reviewed", "last_reviewed", "ts"),
    ("never_decay", "never_decay", "direct"),
    ("never_merge", "never_merge", "direct"),
    ("tombstoned_at", "tombstoned_at", "ts"),
    ("schema_bypass", "schema_bypass", "direct"),
    ("labile_until", "labile_until", "ts"),
    ("provenance_json", "provenance_json", "direct"),
    ("created_at", "created_at", "ts"),
    ("updated_at", "updated_at", "ts"),
    ("tags_json", "tags_json", "direct"),
    ("language", "language", "direct"),
    ("s5_trust_score", "s5_trust_score", "direct"),
    ("profile_modulation_gain_json", "profile_modulation_gain_json", "direct"),
    ("schema_version", "schema_version", "direct"),
    ("wing", "wing", "direct"),
    ("room", "room", "direct"),
    ("drawer", "drawer", "direct"),
    ("valence", "valence", "direct"),
]

_EDGES_COLS = [
    ("src", "src", "direct"),
    ("dst", "dst", "direct"),
    ("edge_type", "edge_type", "direct"),
    ("weight", "weight", "direct"),
    ("updated_at", "updated_at", "ts"),
]

_EVENTS_COLS = [
    ("id", "id", "direct"),
    ("kind", "kind", "direct"),
    ("severity", "severity", "direct"),
    ("domain", "domain", "direct"),
    ("ts", "ts", "ts"),
    ("data_json", "data_json", "direct"),
    ("session_id", "session_id", "direct"),
    ("source_ids_json", "source_ids_json", "direct"),
]

_BUDGET_LEDGER_COLS = [
    ("date", "date", "direct"),
    ("usd_spent", "usd_spent", "direct"),
    ("kind", "kind", "direct"),
    ("ts", "ts", "ts"),
]

_RATELIMIT_LEDGER_COLS = [
    ("ts", "ts", "ts"),
    ("status_code", "status_code", "direct"),
    ("endpoint", "endpoint", "direct"),
]

_TABLE_COLS: dict[str, list] = {
    "records": _RECORDS_COLS,
    "edges": _EDGES_COLS,
    "events": _EVENTS_COLS,
    "budget_ledger": _BUDGET_LEDGER_COLS,
    "ratelimit_ledger": _RATELIMIT_LEDGER_COLS,
}


def _transform_value(val: Any, transform: str) -> Any:
    if val is None:
        return None
    if transform == "embed":
        return np.array(val, dtype=np.float32).tobytes()
    if transform == "bytes":
        if isinstance(val, (bytes, bytearray, memoryview)):
            return bytes(val)
        return val
    if transform == "ts":
        if hasattr(val, "isoformat"):
            return val.isoformat()
        try:
            import pandas as pd
            if hasattr(pd, "Timestamp") and isinstance(val, pd.Timestamp):
                return val.isoformat()
            if hasattr(pd, "isna") and pd.isna(val):
                return None
        except (ImportError, TypeError, ValueError):
            pass
        return str(val) if val is not None else None
    return val


def _build_insert_or_ignore(table_name: str, col_defs: list) -> tuple[str, list[str]]:
    hippo_cols = [d[1] for d in col_defs]
    placeholders = ", ".join("?" for _ in hippo_cols)
    col_names = ", ".join(hippo_cols)
    sql = (
        f"INSERT OR IGNORE INTO {table_name} ({col_names}) VALUES ({placeholders})"
    )
    return sql, hippo_cols


def stream_copy_table(
    lance_db: Any,
    hippo_conn: sqlite3.Connection,
    table_name: str,
    batch_size: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    col_defs = _TABLE_COLS.get(table_name)
    if col_defs is None:
        tee_print(f"  {table_name}: skipped (no column map defined)")
        return 0, 0

    try:
        lance_tbl = lance_db.open_table(table_name)
    except Exception:  # noqa: BLE001
        tee_print(f"  {table_name}: not found in LanceDB source, skipping")
        return 0, 0

    try:
        source_schema_names = set(lance_tbl.schema.names)
    except Exception:  # noqa: BLE001
        source_schema_names = None

    if source_schema_names is not None:
        filtered_defs = [(s, h, t) for s, h, t in col_defs if s in source_schema_names]
        if not filtered_defs:
            tee_print(f"  {table_name}: no matching columns, skipping")
            return 0, 0
    else:
        filtered_defs = col_defs

    sql, hippo_cols = _build_insert_or_ignore(table_name, filtered_defs)
    source_cols = [d[0] for d in filtered_defs]
    transforms = [d[2] for d in filtered_defs]

    total_inserted = 0
    total_duplicates = 0
    batch: list[tuple] = []

    def _flush_batch() -> tuple[int, int]:
        nonlocal batch
        if not batch or dry_run:
            n = len(batch)
            batch = []
            return (n if dry_run else 0), 0
        hippo_conn.execute("BEGIN")
        inserted_in_batch = 0
        duplicates_in_batch = 0
        try:
            for row_values in batch:
                cursor = hippo_conn.execute(sql, row_values)
                if cursor.rowcount == 1:
                    inserted_in_batch += 1
                else:
                    duplicates_in_batch += 1
        except Exception:
            hippo_conn.execute("ROLLBACK")
            raise
        hippo_conn.execute("COMMIT")
        batch = []
        return inserted_in_batch, duplicates_in_batch

    try:
        arrow_table = lance_tbl.to_arrow()
        for record_batch in arrow_table.to_batches(max_chunksize=batch_size):
            batch_dict = record_batch.to_pydict()
            n_rows = record_batch.num_rows
            for i in range(n_rows):
                row_vals = tuple(
                    _transform_value(batch_dict[col][i], transform)
                    for col, transform in zip(source_cols, transforms)
                )
                batch.append(row_vals)
                if len(batch) >= batch_size:
                    ins, dup = _flush_batch()
                    total_inserted += ins
                    total_duplicates += dup
        ins, dup = _flush_batch()
        total_inserted += ins
        total_duplicates += dup
    except Exception as exc:
        try:
            ins, dup = _flush_batch()
            total_inserted += ins
            total_duplicates += dup
        except Exception:
            pass
        raise RuntimeError(
            f"stream_copy_table({table_name!r}) failed: {type(exc).__name__}: {exc}"
        ) from exc

    return total_inserted, total_duplicates


def rebuild_and_persist_hnsw(hippo_db: Any) -> dict:
    result: dict = {"action": "rebuild_and_save"}

    with hippo_db._hnsw_lock:
        rebuild_info = hippo_db._rebuild_index_from_sqlite()
        result["rebuilt_count"] = rebuild_info.get("rebuilt_count", 0)

    import hnswlib

    hnsw_path = hippo_db._hnsw_path
    if not hnsw_path.exists():
        raise RuntimeError(
            f"hnsw rebuild produced no file at {hnsw_path}"
        )

    verifier = hnswlib.Index(space="cosine", dim=hippo_db._embed_dim)
    verifier.load_index(str(hnsw_path))
    result["verified_count"] = verifier.get_current_count()
    result["file_size_bytes"] = hnsw_path.stat().st_size
    tee_print(
        f"  hnsw rebuild: {result['rebuilt_count']} vectors, "
        f"verified {result['verified_count']}, "
        f"file {result['file_size_bytes']} bytes"
    )
    return result


def verify_record_parity(lance_db: Any, hippo_conn: sqlite3.Connection) -> list[dict]:
    mismatches: list[dict] = []

    try:
        lance_tbl = lance_db.open_table("records")
        lance_df = lance_tbl.to_pandas()
    except Exception:  # noqa: BLE001
        lance_df = None

    if lance_df is not None and not lance_df.empty:
        hippo_idx: dict[str, Any] = {}
        for row in hippo_conn.execute("SELECT * FROM records"):
            hippo_idx[row["id"]] = row

        for _, lr in lance_df.iterrows():
            rid = str(lr.get("id", ""))
            hr = hippo_idx.get(rid)
            if hr is None:
                mismatches.append({
                    "table": "records", "id": rid,
                    "field": "<row>", "reason": "missing in hippo",
                })
                continue

            lance_emb = lr.get("embedding")
            if lance_emb is not None:
                lance_bytes = np.array(lance_emb, dtype=np.float32).tobytes()
                hippo_bytes = bytes(hr["embedding"]) if hr["embedding"] is not None else b""
                if lance_bytes != hippo_bytes:
                    mismatches.append({
                        "table": "records", "id": rid,
                        "field": "embedding",
                        "lance_first_bytes": lance_bytes[:32].hex(),
                        "hippo_first_bytes": hippo_bytes[:32].hex(),
                    })

            lance_hv = lr.get("structure_hv")
            if lance_hv is not None:
                lance_hv_b = bytes(lance_hv) if isinstance(lance_hv, (bytes, bytearray, memoryview)) else b""
                hippo_hv_b = bytes(hr["structure_hv"]) if hr["structure_hv"] is not None else b""
                if lance_hv_b != hippo_hv_b:
                    mismatches.append({
                        "table": "records", "id": rid,
                        "field": "structure_hv",
                        "reason": "bytes mismatch",
                    })

            for col in ("literal_surface", "provenance_json", "profile_modulation_gain_json"):
                lv = lr.get(col)
                hv = hr[col]
                if lv is None and hv is None:
                    continue
                if (lv is None) != (hv is None):
                    mismatches.append({
                        "table": "records", "id": rid, "field": col,
                        "reason": "null mismatch",
                    })
                    continue
                if str(lv) != str(hv):
                    mismatches.append({
                        "table": "records", "id": rid, "field": col,
                        "reason": "ciphertext mismatch",
                    })

    try:
        lance_edges = lance_db.open_table("edges").to_pandas()
    except Exception:  # noqa: BLE001
        lance_edges = None

    if lance_edges is not None and not lance_edges.empty:
        hippo_edges: dict[tuple, Any] = {}
        for row in hippo_conn.execute("SELECT * FROM edges"):
            key = (row["src"], row["dst"], row["edge_type"])
            hippo_edges[key] = row

        for _, lr in lance_edges.iterrows():
            key = (str(lr.get("src", "")), str(lr.get("dst", "")), str(lr.get("edge_type", "")))
            hr = hippo_edges.get(key)
            if hr is None:
                mismatches.append({
                    "table": "edges", "key": key,
                    "field": "<row>", "reason": "missing in hippo",
                })
                continue
            lw = float(lr.get("weight", 0.0) or 0.0)
            hw = float(hr["weight"] or 0.0)
            if abs(lw - hw) > 1e-6:
                mismatches.append({
                    "table": "edges", "key": key,
                    "field": "weight",
                    "lance": lw, "hippo": hw,
                })

    try:
        lance_events = lance_db.open_table("events").to_pandas()
    except Exception:  # noqa: BLE001
        lance_events = None

    if lance_events is not None and not lance_events.empty:
        hippo_events: dict[str, Any] = {}
        for row in hippo_conn.execute("SELECT * FROM events"):
            hippo_events[row["id"]] = row

        for _, lr in lance_events.iterrows():
            eid = str(lr.get("id", ""))
            hr = hippo_events.get(eid)
            if hr is None:
                mismatches.append({
                    "table": "events", "id": eid,
                    "field": "<row>", "reason": "missing in hippo",
                })
                continue
            ld = lr.get("data_json")
            hd = hr["data_json"]
            if (ld is None) != (hd is None):
                mismatches.append({
                    "table": "events", "id": eid, "field": "data_json",
                    "reason": "null mismatch",
                })
            elif ld is not None and str(ld) != str(hd):
                mismatches.append({
                    "table": "events", "id": eid, "field": "data_json",
                    "reason": "ciphertext mismatch",
                })
            ls = lr.get("source_ids_json")
            hs = hr["source_ids_json"]
            if (ls is None) != (hs is None):
                mismatches.append({
                    "table": "events", "id": eid, "field": "source_ids_json",
                    "reason": "null mismatch",
                })
            elif ls is not None and str(ls) != str(hs):
                mismatches.append({
                    "table": "events", "id": eid, "field": "source_ids_json",
                    "reason": "string mismatch",
                })

    try:
        lance_budget = lance_db.open_table("budget_ledger").to_pandas()
    except Exception:  # noqa: BLE001
        lance_budget = None

    if lance_budget is not None and not lance_budget.empty:
        hippo_budget = list(hippo_conn.execute("SELECT * FROM budget_ledger"))
        if len(lance_budget) != len(hippo_budget):
            mismatches.append({
                "table": "budget_ledger",
                "reason": f"row count mismatch: lance={len(lance_budget)} hippo={len(hippo_budget)}",
            })

    try:
        lance_ratelimit = lance_db.open_table("ratelimit_ledger").to_pandas()
    except Exception:  # noqa: BLE001
        lance_ratelimit = None

    if lance_ratelimit is not None and not lance_ratelimit.empty:
        hippo_ratelimit = list(hippo_conn.execute("SELECT * FROM ratelimit_ledger"))
        if len(lance_ratelimit) != len(hippo_ratelimit):
            mismatches.append({
                "table": "ratelimit_ledger",
                "reason": f"row count mismatch: lance={len(lance_ratelimit)} hippo={len(hippo_ratelimit)}",
            })

    return mismatches


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-shot migration from LanceDB to Hippo storage backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store",
        metavar="PATH",
        default=str(Path.home() / ".iai-mcp"),
        help="Root storage directory containing lancedb/ (default: ~/.iai-mcp)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all stages but skip the final trash-move.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Allow duplicates_skipped > 0 without failing (for re-runs after partial failure).",
    )
    p.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the lancedb backup and move the failed hippo tree to Trash.",
    )
    p.add_argument(
        "--rollback-ts",
        metavar="TS",
        help="Timestamp of the failed run to roll back (used with --rollback).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt before the trash-move stage.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Rows per batch during stream copy (default: 1000).",
    )
    p.add_argument(
        "--log-file",
        metavar="PATH",
        help="Mirror all output lines to this file (append mode).",
    )
    return p


def main() -> None:
    global _log_fh

    parser = _build_parser()
    args = parser.parse_args()

    store_root = Path(args.store).expanduser().resolve()

    if args.log_file:
        try:
            _log_fh = open(args.log_file, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        except OSError as exc:
            print(f"ERROR: cannot open log file {args.log_file}: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.rollback:
        result = rollback(store_root, args.rollback_ts)
        tee_print(f"Rollback complete: {result}")
        if _log_fh:
            _log_fh.close()
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_results: dict = {}
    backup_path: Path | None = None
    hippo_db = None
    exit_code = 0

    try:
        tee_print("=== pre-flight ===")

        alive, reason = pre_flight_daemon_alive(store_root)
        if alive:
            print(
                f"ERROR: {reason}. Stop it first:\n"
                "    iai-mcp daemon stop\n"
                "    launchctl bootout gui/$(id -u)/com.iai-mcp.daemon",
                file=sys.stderr,
            )
            sys.exit(2)

        lance_root = store_root / "lancedb"
        if not lance_root.exists():
            tee_print("Nothing to migrate: lancedb/ not found.")
            sys.exit(0)

        hippo_root = store_root / "hippo"
        if hippo_root.exists():
            print(
                f"ERROR: hippo/ already exists at {hippo_root}. "
                "Remove it or run --rollback before retrying.",
                file=sys.stderr,
            )
            sys.exit(3)

        stage_results["pre_flight"] = "ok"
        tee_print("  daemon check: ok")
        tee_print(f"  source: {lance_root}")

        lancedb = _import_lancedb_or_die()

        tee_print("=== backup ===")
        if not args.dry_run:
            backup_path = backup_lancedb(store_root, ts)
        else:
            backup_path = store_root / f"lancedb.pre-migrate-{ts}"
            tee_print(f"  dry-run: would backup to {backup_path}")
        stage_results["backup"] = "ok"

        tee_print("=== opening stores ===")
        lance_db = lancedb.connect(str(lance_root))
        tee_print(f"  LanceDB: {lance_root}")

        from iai_mcp.hippo import HippoDB
        hippo_db = HippoDB(path=store_root, crypto_key_provider=None)
        tee_print(f"  HippoDB: {hippo_root}")

        tee_print("=== stream copy ===")
        tables_ordered = ["records", "edges", "events", "budget_ledger", "ratelimit_ledger"]
        per_table_counts: dict[str, dict] = {}
        total_inserted = 0
        total_duplicates = 0

        for tbl_name in tables_ordered:
            ins, dup = stream_copy_table(
                lance_db,
                hippo_db._conn,
                tbl_name,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            per_table_counts[tbl_name] = {"inserted": ins, "duplicates_skipped": dup}
            total_inserted += ins
            total_duplicates += dup
            tee_print(f"  {tbl_name}: inserted={ins}, duplicates_skipped={dup}")

        stage_results["stream_copy"] = per_table_counts

        if total_duplicates > 0 and not args.resume:
            mismatch_rows = [{
                "phase": "stream_copy",
                "reason": "duplicates_skipped > 0 on fresh run",
                "counts": per_table_counts,
            }]
            fail_path = write_failure_json(
                store_root, ts, mismatch_rows, stage_results,
                backup_path=backup_path,
            )
            print(
                f"ERROR: fresh migration encountered {total_duplicates} duplicate(s). "
                "Run with --resume if intentional. "
                f"Failure report: {fail_path}",
                file=sys.stderr,
            )
            sys.exit(5)

        tee_print("=== hnsw rebuild ===")
        if not args.dry_run:
            hnsw_result = rebuild_and_persist_hnsw(hippo_db)
            stage_results["hnsw_rebuild"] = hnsw_result
        else:
            tee_print("  dry-run: skipping hnsw rebuild")
            hnsw_result = {"action": "dry_run_skipped"}
            stage_results["hnsw_rebuild"] = hnsw_result

        tee_print("=== verification ===")
        if not args.dry_run:
            mismatches = verify_record_parity(lance_db, hippo_db._conn)
            if mismatches:
                stage_results["verification"] = "failed"
                fail_path = write_failure_json(
                    store_root, ts, mismatches, stage_results,
                    hnsw_result=hnsw_result,
                    backup_path=backup_path,
                )
                print(
                    f"ERROR: verification failed with {len(mismatches)} mismatch(es). "
                    f"Failure report: {fail_path}",
                    file=sys.stderr,
                )
                sys.exit(4)
            tee_print(f"  verification passed (0 mismatches)")
            stage_results["verification"] = "ok"
        else:
            tee_print("  dry-run: skipping verification")
            stage_results["verification"] = "dry_run_skipped"

        if args.dry_run:
            tee_print("=== dry-run complete ===")
            tee_print(
                f"Would migrate: {total_inserted} rows across "
                f"{len(tables_ordered)} tables."
            )
            tee_print("No changes made to disk.")
            return

        if not args.yes:
            tee_print("")
            tee_print(
                f"Migration verified. About to move lancedb/ to Trash.\n"
                f"  source:  {lance_root}\n"
                f"  backup:  {backup_path}\n"
                f"Type 'yes' to continue: ",
                end="",
            )
            user_input = input().strip().lower()
            if user_input != "yes":
                tee_print("Aborted by user.")
                return

        tee_print("=== trash lancedb sources ===")
        move_to_trash(lance_root, f"lancedb-final-{ts}")
        tee_print(f"  moved {lance_root} to Trash")

        if backup_path and backup_path.exists():
            move_to_trash(backup_path, f"lancedb-pre-migrate-backup-{ts}")
            tee_print(f"  moved backup {backup_path} to Trash")

        stage_results["trash"] = "ok"

        hippo_db.close()
        hippo_db = None

        tee_print("")
        tee_print("=== Migration complete ===")
        tee_print("Restart the daemon to use the new storage backend.")

    except SystemExit:
        raise
    except KeyboardInterrupt:
        tee_print("\nInterrupted by user. Backup preserved.", file=sys.stderr)
        if hippo_db is not None:
            try:
                hippo_db.close()
            except Exception:
                pass
        fail_path = write_failure_json(
            store_root, ts,
            [{"reason": "interrupted by user"}],
            stage_results,
            backup_path=backup_path,
        )
        tee_print(f"State saved to {fail_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(
            f"ERROR: unhandled exception during migration: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if hippo_db is not None:
            try:
                hippo_db.close()
            except Exception:
                pass
        fail_path = write_failure_json(
            store_root, ts,
            [{"reason": f"{type(exc).__name__}: {exc}"}],
            stage_results,
            backup_path=backup_path,
        )
        tee_print(f"Failure report: {fail_path}", file=sys.stderr)
        sys.exit(exit_code or 1)
    finally:
        if _log_fh:
            try:
                _log_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
