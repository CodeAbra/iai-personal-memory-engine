"""Direct teaching: study a document and weave it into existing memory.

Runs entirely against the awake hippocampus (MemoryStore), daemon-free, and
is never on the recall or capture critical path.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

logger = logging.getLogger(__name__)

STUDY_LINK_TOP_K: int = 5
"""Existing-record neighbors woven per chunk."""

STUDY_EDGE_DELTA: float = 0.15
"""Hebbian boost per woven edge."""

STUDY_CONTRADICTION_ERR_THRESHOLD: float = 0.6
"""Critic prediction-error score at or above which an existing record is
treated as contradicted by the studied material."""

STUDY_SELF_TEST_SAMPLE: int = 5
"""Studied chunks probed through the real recall path after the weave."""

_MAX_CANDIDATES_PER_STUDY: int = 100
"""Bound on critic candidates per pass (mirrors the critic's own batch cap)."""

_WEAVE_ANN_K_CAP: int = 64
"""Hard bound on the per-chunk weave ANN k: a big-k query raises the SHARED
index's ef and degrades every later awake recall until the next rebuild."""

STUDY_MAX_CHUNKS_ENV = "IAI_MCP_STUDY_MAX_CHUNKS"
STUDY_MAX_CHUNKS_DEFAULT = 5000
"""Per-document chunk ceiling (matches upload's cap): an unbounded document
would run chunks x ANN-weave queries — O(N^2) against the store."""


def _study_max_chunks() -> int:
    import os as _os

    try:
        val = int(_os.environ.get(STUDY_MAX_CHUNKS_ENV, STUDY_MAX_CHUNKS_DEFAULT))
    except (TypeError, ValueError):
        return STUDY_MAX_CHUNKS_DEFAULT
    return val if val > 0 else STUDY_MAX_CHUNKS_DEFAULT


def _doc_tag(source_name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", source_name.lower()).strip("-")[:64]
    return f"doc:{slug or 'inline'}"


def study_document(
    store: Any,
    *,
    path: "str | Path | None" = None,
    text: "str | None" = None,
    source_name: "str | None" = None,
    session_id: str = "study",
    link_top_k: int = STUDY_LINK_TOP_K,
    critic: "Callable[[list[tuple[UUID, str]]], dict[UUID, float]] | None" = None,
) -> dict[str, Any]:
    """Ingest + weave one document. Returns a delta report.

    ``critic`` scores (record_id, comparison_surface) pairs with a 0..1
    prediction error; None disables contradiction RESOLUTION (candidates are
    still counted in the report). Pass
    ``iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation`` for the
    subscription-billed LLM critic.
    """
    from iai_mcp.capture import capture_turn
    from iai_mcp.daemon_config import _load_patsep_config
    from iai_mcp.ingest import chunk_text, extract_text
    from iai_mcp.retrieve import invalidate_temporal_validity_cache
    from iai_mcp.store import flush_edge_buffer, flush_record_buffer

    if text is None:
        if path is None:
            raise ValueError("study_document needs a path or a text")
        p = Path(path)
        text = extract_text(p)
        source_name = source_name or p.name
    source_name = source_name or "inline"
    doc_tag = _doc_tag(source_name)

    report: dict[str, Any] = {
        "source": source_name,
        "doc_tag": doc_tag,
        "session_id": session_id,
        "chunks_total": 0,
        "inserted": 0,
        "reinforced": 0,
        "skipped": 0,
        "edges_formed": 0,
        "neighbors_reinforced": 0,
        "contradiction_candidates": 0,
        "contradictions_resolved": 0,
        "schemas_induced": 0,
        "replay_queued": 0,
        "recall_tested": 0,
        "recall_verified": 0,
    }

    # The same extracted-char ceiling the CLI entry points apply: a caller
    # reaching study_document directly (relay verb, tests) must not chunk an
    # unbounded text.
    import os as _os
    try:
        _max_chars = int(_os.environ.get("IAI_UPLOAD_MAX_CHARS", 20_000_000))
    except (TypeError, ValueError):
        _max_chars = 20_000_000
    if len(text) > _max_chars:
        report["chars_truncated"] = len(text) - _max_chars
        text = text[:_max_chars]

    chunks = chunk_text(text)
    max_chunks = _study_max_chunks()
    if len(chunks) > max_chunks:
        report["chunks_truncated"] = len(chunks) - max_chunks
        logger.warning(
            "study %s: %d chunks exceeds the %d cap — studying the first %d "
            "(raise %s to study more)",
            source_name, len(chunks), max_chunks, max_chunks,
            STUDY_MAX_CHUNKS_ENV,
        )
        chunks = chunks[:max_chunks]
    report["chunks_total"] = len(chunks)
    if not chunks:
        # An emptied (or deleted) document still supersedes its old chunks.
        report["superseded_chunks"] = _supersede_missing(
            store, doc_tag=doc_tag, current_ids=set(), session_id=session_id,
        )
        return report

    cfg = _load_patsep_config()

    # 1. Ingest through the shared capture spine with BOTH dedup layers: the
    # exact-key idem tag (chunk SHA — byte-identical re-teach reinforces) and
    # the cosine near-duplicate gate (near_dup_gate=True — paraphrases reinforce
    # the existing record instead of inserting a twin; never_merge honored).
    batch: list[tuple[UUID, str, str]] = []
    try:
        from iai_mcp.concurrency import foreground_backoff as _fg_backoff
    except ImportError:
        _fg_backoff = None
    for i, chunk in enumerate(chunks):
        # Politeness yield between chunks: a relayed teach runs inside the
        # live daemon and would otherwise contend an in-flight recall on
        # _conn_lock + the GIL for the whole document.
        if _fg_backoff is not None and i:
            _fg_backoff(max_wait_s=0.5)
        chunk_sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        try:
            result = capture_turn(
                store,
                cue="",
                text=chunk,
                tier="episodic",
                session_id=session_id,
                role="user",
                source_uuid=chunk_sha,
                provenance_extra={
                    "source": "study",
                    "filename": source_name,
                    "chunk_index": i,
                },
                extra_tags=[doc_tag],
                near_dup_gate=True,
            )
        except Exception as exc:  # noqa: BLE001 -- one transient embed hiccup
            # (NativeError from the Rust encoder included) must not abort a
            # whole document/directory study; the chunk is counted, not lost
            # silently.
            logger.warning(
                "study chunk %d of %s failed (%s: %s) — skipping",
                i, source_name, type(exc).__name__, str(exc)[:120],
            )
            result = {"status": "skipped", "record_id": None}
        status = result.get("status", "skipped")
        report[status] = report.get(status, 0) + 1
        rid = result.get("record_id")
        if rid:
            try:
                batch.append((UUID(str(rid)), status, chunk))
            except ValueError:
                continue
    flush_record_buffer(store)

    batch_ids = {str(rid) for rid, _s, _c in batch}

    # 2. Weave: strengthen Hebbian edges from each studied chunk to its
    # nearest PRE-EXISTING records. Near-duplicates are the gate's business;
    # the weave links related-but-distinct memories.
    contradiction_candidates: list[tuple[UUID, str, UUID]] = []
    seen_candidate_olds: set[str] = set()
    woven_neighbor_ids: set[str] = set()
    for _wi, (rid, status, _chunk) in enumerate(batch):
        if _fg_backoff is not None and _wi:
            _fg_backoff(max_wait_s=0.5)
        rec = store.get(rid)
        if rec is None or not rec.embedding:
            continue
        try:
            # Bounded k: enough headroom to skip own-batch neighbors, never
            # proportional to document size (see _WEAVE_ANN_K_CAP).
            neighbors = store.query_similar(
                list(rec.embedding),
                k=min(link_top_k + len(batch_ids) + 1, _WEAVE_ANN_K_CAP),
            )
        except Exception as exc:  # noqa: BLE001 -- weave is best-effort per chunk
            logger.debug("study neighbor search failed for %s: %s", rid, exc)
            continue
        woven = 0
        for nrec, cos in neighbors:
            nid = str(nrec.id)
            if nid == str(rid) or nid in batch_ids:
                continue
            if cos >= cfg.near_dup_threshold or cos < cfg.link_threshold:
                continue
            store.boost_edges(
                [(rid, nrec.id)], delta=STUDY_EDGE_DELTA, edge_type="hebbian"
            )
            report["edges_formed"] += 1
            woven_neighbor_ids.add(nid)
            woven += 1
            if (
                status == "inserted"
                and nid not in seen_candidate_olds
                and len(contradiction_candidates) < _MAX_CANDIDATES_PER_STUDY
            ):
                seen_candidate_olds.add(nid)
                contradiction_candidates.append((nrec.id, nrec.literal_surface, rid))
            if woven >= link_top_k:
                break
    report["neighbors_reinforced"] = len(seen_candidate_olds)
    report["contradiction_candidates"] = len(contradiction_candidates)

    # 3. Reconcile: score each related existing record against the studied
    # chunk that touched it; a high prediction error marks the old belief as
    # contradicted BY the already-stored chunk (a contradicts edge feeding
    # the temporal-validity anti-hit machinery — no duplicate record is
    # created; the chunk itself is the corrector).
    if critic is not None and contradiction_candidates:
        chunk_text_by_id = {str(rid): c for rid, _s, c in batch}
        items = [
            (
                old_id,
                f"EXISTING MEMORY: {old_surface[:300]} || NEW STUDIED MATERIAL: "
                f"{chunk_text_by_id.get(str(chunk_id), '')[:300]}",
            )
            for old_id, old_surface, chunk_id in contradiction_candidates
        ]
        try:
            scores = critic(items) or {}
        except Exception as exc:  # noqa: BLE001 -- critic failure degrades to no resolution
            logger.warning("study critic failed: %s", exc)
            scores = {}
        for old_id, _old_surface, chunk_id in contradiction_candidates:
            err = scores.get(old_id)
            if err is None or err < STUDY_CONTRADICTION_ERR_THRESHOLD:
                continue
            old_rec = store.get(old_id)
            if old_rec is None:
                continue
            # A contradicts edge is non-lossy: the original stays verbatim
            # (never_merge/pinned untouched); only the anti-hit edge is added.
            store.add_contradicts_edge(old_id, chunk_id)
            report["contradictions_resolved"] += 1
        if report["contradictions_resolved"]:
            invalidate_temporal_validity_cache(store)

    # 4. Schema induction over the combined material (tier-0 tag
    # co-occurrence — the same miner the sleep pipeline runs), persisting
    # only auto-confidence candidates evidenced by this study batch.
    try:
        from iai_mcp.lilli.cycle.schema import (
            build_pattern_keeper_map,
            induce_schemas_tier0,
            persist_schema,
        )

        candidates = induce_schemas_tier0(store)
        # One keeper scan for the whole candidate loop — per-candidate corpus
        # rescans starve every other store consumer for the induction run.
        keeper_map = build_pattern_keeper_map(store) if candidates else {}
        for cand in candidates:
            if cand.status != "auto":
                continue
            evidence = {str(e) for e in cand.evidence_ids}
            if doc_tag in cand.pattern or (evidence & batch_ids):
                kid = persist_schema(store, cand, keeper_map=keeper_map)
                keeper_map.setdefault(f"pattern:{cand.pattern}", kid)
                report["schemas_induced"] += 1
    except Exception as exc:  # noqa: BLE001 -- schema pass is additive, never fails the study
        logger.warning("study schema induction failed: %s", exc)

    # 5. Supersede-on-restudy: chunks that belonged to an earlier version of
    # THIS document and no longer appear in it are hinted into the forgetting
    # funnel (never deleted). Pinned records refuse the hint, as everywhere.
    report["superseded_chunks"] = _supersede_missing(
        store, doc_tag=doc_tag, current_ids=set(batch_ids), session_id=session_id,
    )

    # 6. Targeted reactivation for the next sleep cycle: stamp the studied
    # chunks AND their woven anchors as reviewed-now, so the replay step
    # treats the study session as one co-review cluster.
    reviewed_ids = sorted(batch_ids | woven_neighbor_ids)
    if reviewed_ids:
        try:
            from iai_mcp.store import RECORDS_TABLE  # noqa: PLC0415
            from iai_mcp.store._store import _uuid_literal  # noqa: PLC0415

            tbl = store.db.open_table(RECORDS_TABLE)
            now_ts = datetime.now(timezone.utc)
            # Bounded IN(...) windows: one statement over every chunk + woven
            # neighbor of a large document is an unbounded literal list.
            for start in range(0, len(reviewed_ids), 400):
                window = reviewed_ids[start:start + 400]
                quoted = ", ".join(
                    f"'{_uuid_literal(_doc_id)}'" for _doc_id in window
                )
                tbl.update(
                    where=f"id IN ({quoted})",
                    values={"last_reviewed": now_ts},
                )
            report["replay_queued"] = len(reviewed_ids)
        except Exception as exc:  # noqa: BLE001 -- replay queueing is additive
            logger.warning("study replay stamp failed: %s", exc)

    flush_record_buffer(store)
    flush_edge_buffer(store)

    # 7. Close the loop: probe the real awake recall path with a sample of
    # the studied chunks and report how many actually surface. A miss is a
    # loud signal (event + warning), never a silent success.
    try:
        from iai_mcp.retrieve import recall as _recall

        step = max(1, len(batch) // STUDY_SELF_TEST_SAMPLE)
        sample = batch[::step][:STUDY_SELF_TEST_SAMPLE]
        for rid, _status, chunk in sample:
            if _fg_backoff is not None:
                _fg_backoff(max_wait_s=0.5)
            rec = store.get(rid)
            if rec is None or not rec.embedding:
                continue
            report["recall_tested"] += 1
            try:
                resp = _recall(
                    store, list(rec.embedding), chunk, session_id, k_hits=5,
                )
                hit_ids = {str(h.record_id) for h in resp.hits}
            except Exception as exc:  # noqa: BLE001 -- one failed probe must not abort the rest
                logger.debug("study self-test probe failed for %s: %s", rid, exc)
                hit_ids = set()
            if str(rid) in hit_ids:
                report["recall_verified"] += 1
        if report["recall_tested"] and (
            report["recall_verified"] < report["recall_tested"]
        ):
            logger.warning(
                "study self-test: %d/%d studied chunks not recallable",
                report["recall_tested"] - report["recall_verified"],
                report["recall_tested"],
            )
            try:
                from iai_mcp.events import write_event  # noqa: PLC0415

                write_event(
                    store,
                    "study_self_test_miss",
                    {
                        "source": source_name,
                        "tested": report["recall_tested"],
                        "verified": report["recall_verified"],
                    },
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- telemetry must not fail the study
                pass
    except Exception as exc:  # noqa: BLE001 -- the probe pass is additive
        logger.warning("study self-test pass failed: %s", exc)

    return report


def _safe_uuid(value: str):
    from uuid import UUID

    try:
        return UUID(value)
    except (TypeError, ValueError):
        return None


def _supersede_missing(
    store: Any, *, doc_tag: str, current_ids: "set[str]", session_id: str,
) -> int:
    """Hint every doc-tagged chunk that is absent from the current version
    into the decay funnel (never delete; pinned refuse). Returns the count."""
    count = 0
    try:
        # Exact, indexed tag lookup (idx_record_tags_tag): '_' in a doc_tag is
        # a LIKE wildcard that could fade a SIBLING document's records, so a
        # LIKE scan is unsafe here. Two bounded queries — the lilli SQL parser
        # has no JOIN/alias support.
        with store.db.ro_conn() as conn:
            rows = conn.execute(
                "SELECT record_id FROM record_tags WHERE tag = ?",
                (doc_tag,),
            ).fetchall()
            candidates = [
                str(r[0]) for r in rows if str(r[0]) not in current_ids
            ]
            stale_ids: "list[str]" = []
            for start in range(0, len(candidates), 400):
                window = candidates[start:start + 400]
                placeholders = ", ".join("?" for _ in window)
                live = conn.execute(
                    "SELECT id FROM records"  # nosemgrep: sql-injection
                    f" WHERE id IN ({placeholders})"
                    " AND tombstoned_at IS NULL",
                    tuple(window),
                ).fetchall()
                stale_ids.extend(str(r[0]) for r in live)
        if not stale_ids:
            return 0
        stale_records = store.get_batch(
            [u for u in (_safe_uuid(s) for s in stale_ids) if u is not None]
        )
        from iai_mcp.store import RECORDS_TABLE  # noqa: PLC0415
        from iai_mcp.store._store import _uuid_literal  # noqa: PLC0415

        for rec_id, rec in stale_records.items():
            # never_merge is the storage lock; pinned is the UI pin.
            # Either one refuses the supersede hint.
            if rec.pinned or getattr(rec, "never_merge", False):
                continue
            # Shared ownership: a chunk carrying OTHER documents' doc tags is
            # still live in those documents — this document merely disowns it
            # (its tag is removed); the fade fires only for the LAST owner.
            other_doc_tags = [
                t for t in (rec.tags or [])
                if t.startswith("doc:") and t != doc_tag
            ]
            if other_doc_tags:
                try:
                    store.remove_tags(rec_id, [doc_tag])
                except Exception as exc:  # noqa: BLE001 -- disown is best-effort
                    logger.debug("supersede disown failed for %s: %s", rec_id, exc)
                continue
            store.db.open_table(RECORDS_TABLE).update(
                where=f"id = '{_uuid_literal(rec_id)}'",
                values={"centrality": 0.0, "never_decay": False},
            )
            try:
                store.append_provenance(
                    rec_id,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "cue": "restudy-supersede",
                        "session_id": session_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- provenance is traceability
                logger.debug("supersede provenance failed: %s", exc)
            count += 1
    except Exception as exc:  # noqa: BLE001 -- supersede pass is additive
        logger.warning("study supersede pass failed: %s", exc)
    return count


def watch_scan_once(
    store: Any,
    root: "str | Path",
    seen: "dict[str, float]",
    *,
    session_id: str = "study",
    study_fn: "Callable[..., dict] | None" = None,
) -> "tuple[dict[str, float], dict[str, int]]":
    """One watch tick: restudy changed/new files, fade deleted ones.

    ``seen`` maps relative paths to mtimes from the previous tick; returns the
    fresh map plus a change report. Pure enough to test without a loop.
    ``study_fn`` overrides how a file is studied (the CLI passes a
    daemon-relay implementation when the daemon owns the store); the default
    studies directly against ``store``."""
    if study_fn is None:
        def study_fn(**kw):  # noqa: ANN003
            return study_document(store, session_id=session_id, **kw)

    root = Path(root).expanduser().resolve()
    current: "dict[str, float]" = {}
    for path in iter_study_files(root, max_files=1_000_000):
        try:
            current[str(path.relative_to(root))] = path.stat().st_mtime
        except OSError:
            continue

    changes = {"studied": 0, "faded": 0, "superseded_chunks": 0}
    for rel, mtime in current.items():
        if seen.get(rel) == mtime:
            continue
        try:
            report = study_fn(path=root / rel, source_name=rel)
            changes["studied"] += 1
            changes["superseded_chunks"] += int(report.get("superseded_chunks") or 0)
        except Exception as exc:  # noqa: BLE001 -- one file must not stop the watch
            logger.warning("watch restudy skipped %s: %s", rel, exc)
    for rel in set(seen) - set(current):
        try:
            report = study_fn(text="", source_name=rel)
            changes["faded"] += 1
            changes["superseded_chunks"] += int(report.get("superseded_chunks") or 0)
        except Exception as exc:  # noqa: BLE001 -- a vanished file must not stop the watch
            logger.warning("watch fade skipped %s: %s", rel, exc)
    return current, changes


STUDY_DIR_MAX_FILES_ENV = "IAI_MCP_STUDY_DIR_MAX_FILES"
STUDY_DIR_MAX_FILE_BYTES = 2 * 1024 * 1024
_SKIP_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "node_modules", "target", "dist", "build",
    "__pycache__", ".pytest_cache", ".claude", ".idea", ".vscode", "artifacts",
})


def iter_study_files(root: "str | Path", *, max_files: "int | None" = None) -> "list[Path]":
    """The one directory scanner every study consumer shares (direct passes
    and the CLI's daemon-relay loop): allowed suffixes, skip-dirs, size cap,
    and the no-symlink rule live here and only here.

    os.walk with IN-PLACE dir pruning: skip-dirs are cut BEFORE descent so the
    walk never enumerates files under .git/node_modules."""
    import os as _os

    from iai_mcp.ingest._parsers import PLAINTEXT_SUFFIXES

    root = Path(root).expanduser().resolve()
    allowed = set(PLAINTEXT_SUFFIXES) | {".csv", ".pdf"}
    if max_files is None:
        try:
            max_files = int(_os.environ.get(STUDY_DIR_MAX_FILES_ENV, "500"))
        except (TypeError, ValueError):
            max_files = 500

    files: "list[Path]" = []
    for dirpath, dirnames, filenames in _os.walk(str(root), followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_DIR_NAMES
        )
        for name in sorted(filenames):
            if len(files) >= max_files:
                return files
            path = Path(dirpath) / name
            # Symlinked files can point outside the studied tree; never follow.
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue
            try:
                if path.stat().st_size > STUDY_DIR_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return files


def study_directory(
    store: Any,
    root: "str | Path",
    *,
    session_id: str = "study",
    critic: "Callable[[list[tuple[UUID, str]]], dict[UUID, float]] | None" = None,
) -> dict[str, Any]:
    """Study every supported file under a directory — a codebase or a docs
    tree becomes woven memory, file by file, each idempotent on restudy."""
    root = Path(root).expanduser().resolve()
    files = iter_study_files(root)

    totals: dict[str, Any] = {
        "root": str(root), "files": 0, "files_skipped": 0,
        "chunks_total": 0, "inserted": 0, "reinforced": 0, "skipped": 0,
        "edges_formed": 0, "contradictions_resolved": 0,
        "schemas_induced": 0, "superseded_chunks": 0,
        "recall_tested": 0, "recall_verified": 0, "replay_queued": 0,
    }
    for path in files:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path.name
        try:
            report = study_document(
                store, path=path,
                source_name=str(rel),
                session_id=session_id,
                critic=critic,
            )
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            logger.warning("study_directory skipped %s: %s", path, exc)
            totals["files_skipped"] += 1
            continue
        totals["files"] += 1
        for key in (
            "chunks_total", "inserted", "reinforced", "skipped",
            "edges_formed", "contradictions_resolved", "schemas_induced",
            "superseded_chunks", "recall_tested", "recall_verified",
            "replay_queued",
        ):
            totals[key] += int(report.get(key) or 0)
    return totals
