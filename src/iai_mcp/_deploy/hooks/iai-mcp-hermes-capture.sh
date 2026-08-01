#!/usr/bin/env bash
# Hermes on_session_end adapter. Hermes keeps no transcript file — the
# conversation lives in ~/.hermes/state.db (sqlite, table `messages`). Read
# it with the SYSTEM python's stdlib sqlite3 (the daemon never runs this
# script, so the stdlib-sqlite quarantine is not breached) and append the
# turns to the deferred spool in the exact record shape the core
# turn-capture hook writes. A per-session watermark on the last message id
# keeps re-runs from duplicating. Fail-safe: exit 0 on any error.
set -u
input=$(cat 2>/dev/null || true)
PY_SCRIPT='
import json, os, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path

try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    sys.exit(0)
if not isinstance(payload, dict):
    sys.exit(0)
session_id = str(payload.get("session_id") or "")
if not session_id:
    sys.exit(0)

hermes_home = Path(os.environ.get("IAI_MCP_HERMES_HOME") or (Path.home() / ".hermes"))
db_path = hermes_home / "state.db"
if not db_path.is_file():
    sys.exit(0)

home = Path.home()
state_dir = home / ".iai-mcp" / ".capture-state"
deferred_dir = home / ".iai-mcp" / ".deferred-captures"
state_dir.mkdir(parents=True, exist_ok=True)
deferred_dir.mkdir(parents=True, exist_ok=True)
watermark = state_dir / f"hermes-{session_id}.watermark"
last_id = ""
if watermark.exists():
    try:
        last_id = watermark.read_text().strip()
    except OSError:
        last_id = ""

try:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    rows = conn.execute(
        "SELECT id, role, content, timestamp FROM messages"
        " WHERE session_id = ? AND role IN (\x27user\x27, \x27assistant\x27)"
        " ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
except Exception:
    sys.exit(0)

fresh = []
seen_last = last_id == ""
for row in rows:
    rid = str(row[0])
    if not seen_last:
        if rid == last_id:
            seen_last = True
        continue
    fresh.append(row)
if not seen_last:
    # Watermark id vanished (session pruned): take everything once.
    fresh = rows
if not fresh:
    sys.exit(0)

# The final .live-*.jsonl name is drain-claimable the instant it exists, so
# the file must appear only complete: stream to a .jsonl.tmp sibling (which
# the drain never claims), fsync, then atomic-rename.
final_name = f"{session_id}.live-{int(time.time())}-{os.getpid()}.jsonl"
tmp_path = deferred_dir / f"{final_name}.tmp"
fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    with os.fdopen(fd, "w") as out:
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": str(payload.get("cwd") or ""),
        }
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for rid, role, content, ts in fresh:
            text = str(content or "").strip()
            if not text:
                continue
            event = {
                "text": text,
                "cue": f"session {session_id} turn",
                "tier": "episodic",
                "role": str(role),
                "ts": str(ts) if ts else datetime.now(timezone.utc).isoformat(),
                "source_uuid": f"hermes-{session_id}-{rid}",
            }
            out.write(json.dumps(event, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, deferred_dir / final_name)
except OSError:
    try:
        tmp_path.unlink()
    except OSError:
        pass
    sys.exit(0)

try:
    tmp = watermark.with_name(watermark.name + ".tmp")
    tmp.write_text(str(fresh[-1][0]))
    os.replace(tmp, watermark)
except OSError:
    pass
'
printf '%s' "$input" | /usr/bin/python3 -c "$PY_SCRIPT" 2>/dev/null || true
exit 0
