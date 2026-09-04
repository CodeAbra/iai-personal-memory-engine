"""Sweep Claude Code conversation transcripts already on disk and stage new
turns into the deferred capture spool.

Reads only transcript files on disk; writes only the deferred spool, its own
per-file sweep state, and a receipt log. Never opens the memory store and
never holds a store lock.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.capture import (
    _spool_root,
    _write_deferred_captures_impl,
    capture_state_dir,
)

log = logging.getLogger(__name__)

_RECEIPT_CHANNEL = "transcript-sweep"

# Set in the scheduled sweeper's own unit environment so its interval runs
# pass the live-home guard below without a flag on every fire.
SWEEP_SCHEDULED_ENV_VAR = "IAI_MCP_SWEEP_SCHEDULED"

# A re-swept file that changed less than this many seconds ago is left for
# a later pass -- it is still being actively written. Never applied to a
# file's first-ever sweep: a brand-new session must be captured promptly.
QUIET_WINDOW_SEC = 90

# An unseen transcript is staged in full up to this many lines. Beyond it,
# the pass stops here rather than reading further -- a multi-hundred-
# megabyte pre-existing transcript is never read whole in one pass.
FIRST_SWEEP_LINE_BOUND = 5_000

_SWEEP_STATE_SUFFIX = ".transcript-sweep"

# Same restricted set the hook scripts enforce on a host-supplied
# configuration directory: absolute, existing, no parent-directory
# traversal segment, letters/digits/space/"._/-" only.
_SAFE_ROOT_RE = re.compile(r"[A-Za-z0-9._/ -]+")


def _safe_iterdir(directory: Path) -> "list[Path]":
    try:
        return sorted(directory.iterdir())
    except OSError:
        return []


def _validated_env_root(raw: str) -> "Path | None":
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        return None
    if not _SAFE_ROOT_RE.fullmatch(raw):
        return None
    if ".." in candidate.parts:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _transcript_roots() -> "list[Path]":
    """Every live Claude Code ``projects/`` tree on this machine: the
    shared host home, every dormant agent home under each Cowork session
    root, and a validated configuration-directory override read from this
    process's own environment. De-duped by resolved path."""
    from iai_mcp.cli._cowork import _cowork_session_roots

    candidates: "list[Path]" = [Path.home() / ".claude"]
    for cowork_root in _cowork_session_roots():
        if not cowork_root.is_dir():
            continue
        for account_dir in _safe_iterdir(cowork_root):
            if not account_dir.is_dir():
                continue
            for device_dir in _safe_iterdir(account_dir):
                if not device_dir.is_dir():
                    continue
                agent_dir = device_dir / "agent"
                for ditto_dir in _safe_iterdir(agent_dir):
                    candidate = ditto_dir / ".claude"
                    if candidate.is_dir():
                        candidates.append(candidate)

    env_root = _validated_env_root(os.environ.get("CLAUDE_CONFIG_DIR", ""))
    if env_root is not None:
        candidates.append(env_root)

    seen: "set[Path]" = set()
    deduped: "list[Path]" = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _iter_transcript_files(claude_root: Path):
    """One level deep only: yields the top-level session transcripts under
    ``claude_root/projects/<slug>/*.jsonl``. A session's sibling directories
    (subagent transcripts, tool-result captures) sit one level below this
    glob and are never reached."""
    projects_dir = claude_root / "projects"
    if not projects_dir.is_dir():
        return
    for project_dir in _safe_iterdir(projects_dir):
        if not project_dir.is_dir():
            continue
        for candidate in _safe_iterdir(project_dir):
            if candidate.is_file() and candidate.suffix == ".jsonl":
                yield candidate


def _receipt_log_dir() -> Path:
    return _spool_root() / "logs"


def _append_receipt(*, session_id: str, lines: int) -> None:
    now = datetime.now(timezone.utc)
    log_dir = _receipt_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_RECEIPT_CHANNEL}-{now:%Y-%m-%d}.log"
    line = (
        f"{now:%Y-%m-%dT%H:%M:%SZ} session={session_id} "
        f"lines={lines} channel={_RECEIPT_CHANNEL}\n"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _count_staged_events(spool_path: Path) -> int:
    # One header line always precedes the event lines this function counts.
    try:
        with spool_path.open("r", encoding="utf-8") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except OSError:
        return 0


@dataclass(frozen=True)
class _SweepState:
    mtime_ns: int
    size: int
    lines_swept: int
    pending_tools: "tuple[str, ...]" = ()


def _sweep_state_path(session_id: str) -> Path:
    return capture_state_dir() / f"{session_id}{_SWEEP_STATE_SUFFIX}"


def _read_sweep_state(session_id: str) -> "_SweepState | None":
    path = _sweep_state_path(session_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        pending = raw.get("pending_tools", [])
        return _SweepState(
            mtime_ns=int(raw["mtime_ns"]),
            size=int(raw["size"]),
            lines_swept=int(raw["lines_swept"]),
            pending_tools=tuple(
                n for n in pending if isinstance(pending, list) and isinstance(n, str)
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_sweep_state(session_id: str, state: "_SweepState") -> None:
    state_dir = capture_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _sweep_state_path(session_id)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {
        "mtime_ns": state.mtime_ns,
        "size": state.size,
        "lines_swept": state.lines_swept,
        "pending_tools": list(state.pending_tools),
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _should_sweep(path: Path, prior: "_SweepState | None", *, now: float) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    if prior is not None and st.st_mtime_ns == prior.mtime_ns and st.st_size == prior.size:
        return False  # unchanged since the last pass
    if prior is not None and (now - st.st_mtime) < QUIET_WINDOW_SEC:
        return False  # actively being written; wait for it to go quiet
    return True  # unseen file (always swept promptly) or a quiet, changed one


def _reclaim_orphan_sweep_state(known_session_ids: "set[str]") -> None:
    state_dir = capture_state_dir()
    try:
        entries = list(state_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.endswith(_SWEEP_STATE_SUFFIX):
            continue
        session_id = entry.name[: -len(_SWEEP_STATE_SUFFIX)]
        if session_id in known_session_ids:
            continue
        try:
            entry.unlink()
        except OSError:
            pass


def _same_day_capture_hooks_session_ids(*, now: datetime) -> "set[str]":
    """Session ids with a same-day capture-hooks receipt -- reused from
    _read_hook_receipts rather than re-parsing the log. A "capture" kind
    receipt comes from the turn-capture log, written on every user prompt
    of a code session already served by the capture-hooks command; a
    same-day hit means the Stop hook's own final catch-up has that session
    covered too."""
    from iai_mcp.cli._cowork import _read_hook_receipts

    receipts = _read_hook_receipts(_spool_root(), days=1)
    today = now.date()
    return {
        r.session_id
        for r in receipts
        if r.kind == "capture" and r.timestamp is not None and r.timestamp.date() == today
    }


def sweep_once(
    roots: "list[Path] | None" = None,
    *,
    skip_receipted_sessions: bool = True,
) -> dict:
    """Stage every discovered transcript's new turns into the deferred
    spool once. Opens no store and acquires no lock -- filesystem-only,
    safe to run off the memory system's own awake path.

    A re-swept file stages only the lines added since the last pass, kept
    in a small per-file state alongside the tool-trailer state needed to
    keep a resumed slice byte-identical to a single full pass. An unseen
    file larger than the first-sweep bound is staged only up to that bound
    -- the pass stops there rather than reading the rest.

    A session that already carries a same-day capture-hooks receipt is
    skipped -- the Stop hook already captured it, and re-staging it here
    would be wasted I/O even though the store would dedup it. This guard
    is the waste-avoidance half of double-capture prevention; the store's
    own exact-key idem-dedup, at drain time, is the correctness half that
    holds even with ``skip_receipted_sessions=False``.
    """
    if roots is None:
        roots = _transcript_roots()
    now = time.time()
    receipted_session_ids = (
        _same_day_capture_hooks_session_ids(now=datetime.now(timezone.utc))
        if skip_receipted_sessions
        else set()
    )
    summary = {
        "files_seen": 0,
        "sessions_staged": 0,
        "lines_staged": 0,
        "files_failed": 0,
    }
    known_session_ids: "set[str]" = set()
    for root in roots:
        for transcript_path in _iter_transcript_files(root):
            summary["files_seen"] += 1
            session_id = transcript_path.stem
            known_session_ids.add(session_id)
            if session_id in receipted_session_ids:
                continue
            try:
                prior = _read_sweep_state(session_id)
                if not _should_sweep(transcript_path, prior, now=now):
                    continue
                since_line = prior.lines_swept if prior is not None else 0
                pending = list(prior.pending_tools) if prior is not None else None
                max_turns = 100_000 if prior is not None else FIRST_SWEEP_LINE_BOUND
                out_path, ending_pending, lines_read = _write_deferred_captures_impl(
                    session_id,
                    transcript_path,
                    cwd=str(transcript_path.parent),
                    since_line=since_line,
                    pending_tools=pending,
                    max_turns=max_turns,
                )
                staged = _count_staged_events(out_path)
                if staged > 0:
                    _append_receipt(session_id=session_id, lines=staged)
                    summary["sessions_staged"] += 1
                    summary["lines_staged"] += staged
                st = transcript_path.stat()
                _write_sweep_state(
                    session_id,
                    _SweepState(
                        mtime_ns=st.st_mtime_ns,
                        size=st.st_size,
                        # ``lines_read`` is already an absolute count from the
                        # start of the file (the writer's own ``seen`` counter),
                        # not a delta past ``since_line`` -- it is the correct
                        # value to persist directly.
                        lines_swept=lines_read,
                        pending_tools=tuple(ending_pending),
                    ),
                )
            except Exception:  # noqa: BLE001 -- one bad transcript must never stall the rest
                summary["files_failed"] += 1
                log.warning(
                    "transcript sweep failed for %s", transcript_path, exc_info=True
                )
                continue
    _reclaim_orphan_sweep_state(known_session_ids)
    return summary


def _system_home_dir() -> "Path | None":
    """HOME can be redirected; the password database's pw_dir cannot."""
    import pwd  # POSIX-only; deferred so a non-POSIX import never breaks the CLI

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        return None


def _targets_live_home() -> bool:
    """True when this process, with its current environment, would sweep
    the real account's own memory store -- i.e. nothing has redirected
    HOME away from the OS-reported home directory."""
    system_home = _system_home_dir()
    if system_home is None:
        return False
    try:
        return Path.home().resolve() == system_home.resolve()
    except OSError:
        return False


def cmd_transcript_sweep_run(args: argparse.Namespace) -> int:
    scheduled = os.environ.get(SWEEP_SCHEDULED_ENV_VAR) == "1"
    allowed = bool(getattr(args, "allow_live_home", False))
    if _targets_live_home() and not scheduled and not allowed:
        print(
            "refusing to run: this would sweep this machine's real memory "
            "store. Pass --allow-live-home to confirm.",
            file=sys.stderr,
        )
        return 2
    summary = sweep_once()
    print(json.dumps(summary, ensure_ascii=False))
    return 0
