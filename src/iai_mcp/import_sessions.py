"""Cold-start bootstrap: seed an empty store from existing transcript files.

Pure discovery-and-loop wrapper around the already-tested, already-
idempotent O(N) `capture_transcript` spine (see `iai_mcp.capture`) -- this
module never re-implements parsing or dedup. English-only-brain note: this
wrapper adds no translation step; it inherits whatever `capture_transcript`
already does for a given line (none).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from iai_mcp.capture import capture_transcript

DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_CODEX_ROOT = Path.home() / ".codex"

ProgressFn = Callable[[int, int, str, dict[str, int]], None]


def discover_claude_files(
    root: Path | str, *, include_subagents: bool = False,
) -> list[Path]:
    """Walk `root/**/*.jsonl`. Sub-agent transcripts (any path segment named
    `subagents`) are excluded by default -- they dilute the "who is the
    user" signal a cold-start bootstrap exists to capture. Codex-shaped
    `rollout-*.jsonl` files are always excluded here (they belong to
    `discover_codex_files`), so a shared parent directory partitions
    cleanly between the two sources."""
    root = Path(root)
    if not root.is_dir():
        return []
    files = sorted(root.glob("**/*.jsonl"))
    files = [f for f in files if not f.name.startswith("rollout-")]
    if include_subagents:
        return files
    return [f for f in files if "subagents" not in f.parts]


def discover_codex_files(root: Path | str) -> list[Path]:
    """Walk `root/sessions/**/rollout-*.jsonl` -- the real on-disk layout of
    Codex CLI rollout transcripts, already parsed by `capture_transcript`'s
    `response_item` branch."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(root.glob("sessions/**/rollout-*.jsonl"))


def resolve_source(path: Path | str | None, explicit: str | None) -> str | None:
    """An explicit `--source` always wins. Else infer from a supplied path
    shape: a `rollout-*.jsonl` filename or any path under a `.codex`
    segment -> codex; a path under `.claude/projects` -> claude. Returns
    None when neither an explicit source nor a recognizable path shape is
    available -- the caller then targets both default locations."""
    if explicit:
        return explicit
    if path is None:
        return None
    p = Path(path)
    if p.name.startswith("rollout-") and p.name.endswith(".jsonl"):
        return "codex"
    if ".codex" in p.parts:
        return "codex"
    if ".claude" in p.parts and "projects" in p.parts:
        return "claude"
    return None


def _count_turns_in_file(path: Path) -> int:
    """Lightweight, write-free turn count for `--dry-run` reporting. Reuses
    the same per-line parser `capture_transcript` calls (`_parse_transcript_obj`)
    -- not a new parser, just a preview that never opens a store."""
    import json as _json

    from iai_mcp.capture import _parse_transcript_obj

    n = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = _json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict) and _parse_transcript_obj(obj) is not None:
                    n += 1
    except OSError:
        return 0
    return n


def _resume_state_path(store: Any, source: str) -> Path:
    """Per-source resume-state file under the OPEN store's own root, so it
    always honors the hermetic `IAI_MCP_STORE` redirect a test or an
    alternate-store operator has set -- never a hardcoded real-home path."""
    return Path(store.db._store_root) / "import-state" / f"{source}.json"


def _load_resume_state(path: Path) -> dict[str, dict[str, Any]]:
    import json as _json

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_resume_entry(path: Path, key: str, entry: dict[str, Any]) -> None:
    import json as _json

    state = _load_resume_state(path)
    state[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(_json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def import_transcripts(
    store: Any,
    files: list[Path],
    *,
    source: str,
    dry_run: bool = False,
    progress: ProgressFn | None = None,
    resume: bool = True,
) -> dict[str, int | str]:
    """Loop `files` through `capture_transcript` (one call per file) and
    aggregate a summary. `dry_run=True` never touches `store` (it may be
    None) -- it only counts turns per file via the shared parser, writing
    nothing. `empty_files` counts a file that yielded zero usable turns
    (all-zero capture_transcript counts, or zero dry-run turns) -- kept
    distinct from `skipped` (which means turns were attempted and rejected,
    e.g. too short or a dedup hit) so a silent format-drift import doesn't
    read as "nothing to skip".

    When `resume=True` and not dry-run, a per-source JSON state file tracks
    which files are already durably imported (keyed by absolute path,
    matched by mtime). CRITICAL ordering: a file's state entry is written
    ONLY after `flush_record_buffer(store)` makes that file's turns durable
    -- a crash between capture and flush leaves the file absent from state,
    so the next run re-imports it in full (the exact-key idem tag collapses
    any partial overlap; never a duplicate).
    """
    summary: dict[str, int | str] = {
        "source": source,
        "files": 0,
        "inserted": 0,
        "reinforced": 0,
        "skipped": 0,
        "errors": 0,
        "empty_files": 0,
        "would_import": 0,
    }
    total = len(files)
    resume_path: Path | None = None
    resume_state: dict[str, dict[str, Any]] = {}
    if resume and not dry_run and store is not None:
        resume_path = _resume_state_path(store, source)
        resume_state = _load_resume_state(resume_path)

    for i, f in enumerate(files, start=1):
        summary["files"] = int(summary["files"]) + 1
        if dry_run:
            n_turns = _count_turns_in_file(f)
            summary["would_import"] = int(summary["would_import"]) + n_turns
            if n_turns == 0:
                summary["empty_files"] = int(summary["empty_files"]) + 1
            if progress is not None:
                progress(i, total, str(f), dict(summary))  # type: ignore[arg-type]
            continue

        resume_key = str(f.resolve())
        if resume_path is not None:
            try:
                current_mtime = f.stat().st_mtime
            except OSError:
                current_mtime = None
            seen = resume_state.get(resume_key)
            if (
                seen is not None
                and current_mtime is not None
                and seen.get("mtime") == current_mtime
            ):
                if progress is not None:
                    progress(i, total, str(f), dict(summary))  # type: ignore[arg-type]
                continue

        session_id = f.stem
        # capture_transcript is never called with directive_marker_allowed:
        # bulk-imported history must never mint a standing directive.
        counts = capture_transcript(store, f, session_id=session_id)
        for key in ("inserted", "reinforced", "skipped", "errors"):
            summary[key] = int(summary[key]) + int(counts.get(key, 0))
        usable = (
            int(counts.get("inserted", 0))
            + int(counts.get("reinforced", 0))
            + int(counts.get("skipped", 0))
        )
        if usable == 0:
            summary["empty_files"] = int(summary["empty_files"]) + 1

        if resume_path is not None:
            from iai_mcp.store import flush_record_buffer

            # Durability precedes the watermark: flush BEFORE this file's
            # entry lands in resume_path, never after.
            flush_record_buffer(store)
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            _write_resume_entry(resume_path, resume_key, {
                "mtime": mtime, "counts": counts,
            })
            resume_state[resume_key] = {"mtime": mtime, "counts": counts}

        if progress is not None:
            progress(i, total, str(f), dict(summary))  # type: ignore[arg-type]
    return summary
