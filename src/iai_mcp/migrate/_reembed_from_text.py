"""Re-embed records from their verbatim text.

The stored embedding can diverge from ``embed(literal_surface)`` in two known
ways: the capture path once embedded the provenance cue label instead of the
message content, and an embedder-model swap (same dimension, different
weights) leaves every record written during the swap window in a foreign
vector space where cosine against current-model cues is noise. Either way the
ANN/cosine index built from those vectors is semantically dead.

This migration rebuilds every active record's embedding — all tiers — from
its intact ``literal_surface`` (the verbatim text, which was always stored
correctly), then rebuilds the recall index from the corrected vectors and
stamps the store with the identity of the embedder that produced them.

Boundary held by design: only the ``embedding`` column is rewritten.
``literal_surface`` is never modified, and the at-rest encryption boundary is
untouched -- only ``literal_surface`` is decrypted in-process via the normal
record-read path, exactly as graph build and recall already do.

Throughput: records are processed in id-ordered windows. Within each window the
read decrypts only ``literal_surface`` (not the whole record), the texts are
embedded in one batch call, and the corrected vectors are written under a single
transaction -- one commit per window, not per record. A keyset cursor over the
primary key bounds memory regardless of corpus size, and the last committed
cursor is checkpointed so an interrupted run resumes from the next window.

Idempotent: re-embedding the same text yields the same vector, so a second run
is a no-op in effect. Records whose text is missing or undecryptable are
skipped and counted -- no vector is ever fabricated.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

from iai_mcp.events import write_event

log = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 256
PROGRESS_FILE = "reembed_progress.json"


def _progress_path(store) -> str:
    return os.path.join(str(store.root), PROGRESS_FILE)


def _load_resume_cursor(store) -> str:
    """Return the last committed id cursor, or "" if no checkpoint exists."""
    path = _progress_path(store)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cursor = data.get("last_id")
        return cursor if isinstance(cursor, str) else ""
    except (OSError, ValueError, TypeError):
        return ""


def _save_resume_cursor(store, last_id: str, stats: dict) -> None:
    """Atomically persist the resume cursor after a window commits."""
    path = _progress_path(store)
    payload = {
        "last_id": last_id,
        "reembedded": int(stats.get("reembedded", 0)),
        "skipped": int(stats.get("skipped", 0)),
        "total": int(stats.get("total", 0)),
    }
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".reembed_progress.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _clear_resume_cursor(store) -> None:
    try:
        os.unlink(_progress_path(store))
    except OSError:
        pass


def migrate_reembed_from_text(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
) -> dict:
    """Re-embed every active record (all tiers) from its ``literal_surface``.

    Streams record ids in id-ordered windows so the whole corpus is never
    loaded at once. Per window: a light read decrypts only ``literal_surface``,
    the window's texts are embedded in one batch call, and the corrected vectors
    are written under one transaction. After all windows land, the HNSW recall
    index is rebuilt from the corrected vectors.

    Returns a dict with keys: reembedded, skipped, total, dry_run.

    With ``resume=True`` the run continues from the last committed window
    recorded in the on-disk checkpoint, so already-corrected windows are not
    re-read or re-embedded.

    Safe to call multiple times (idempotent): the same text re-embeds to the
    same vector, so re-running has no net effect. Records whose text is empty
    or undecryptable are skipped and counted, never re-embedded with a
    fabricated vector.
    """
    from iai_mcp.crypto import is_encrypted
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.hippo import HippoDB, HippoIntegrityError, _encode_embedding

    db = store.db
    if not isinstance(db, HippoDB):
        return {"reembedded": 0, "skipped": 0, "total": 0, "dry_run": dry_run}

    if batch_size < 1:
        batch_size = DEFAULT_BATCH_SIZE

    # The migration is the sanctioned way OUT of an identity mismatch, so it
    # must be able to obtain the embedder while the guard would refuse it.
    embedder = embedder_for_store(store, allow_identity_mismatch=True)

    reembedded = 0
    skipped = 0
    total = 0

    # Total active episodic count for an observable "done X/total" line. Bounded
    # single-row read; cheap relative to the per-window embed work.
    with db._conn_lock:
        total_target_row = db._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    total_target = int(total_target_row["n"]) if total_target_row is not None else 0

    # Keyset cursor over the primary key. Stable under the in-place embedding
    # updates this loop performs (updates never change id), and resumable from
    # the last committed window.
    last_id = _load_resume_cursor(store) if resume else ""
    if resume and last_id:
        log.info("reembed_from_text: resuming from last_id=%s", last_id)

    while True:
        with db._conn_lock:
            # Pending rows are excluded: their vector is a placeholder owned by
            # the deferred-embed pass, which also flips embedding_pending —
            # writing here would leave the flag lying about the vector.
            rows = db._conn.execute(
                "SELECT id, literal_surface FROM records"
                " WHERE tombstoned_at IS NULL"
                "   AND COALESCE(embedding_pending, 0) = 0"
                "   AND id > ?"
                " ORDER BY id"
                " LIMIT ?",
                (last_id, int(batch_size)),
            ).fetchall()
        if not rows:
            break
        window_last_id = rows[-1]["id"]

        # Light decrypt: only literal_surface, mirroring the graph-build read
        # path. Records with empty or undecryptable text are skipped and never
        # fabricated.
        window_ids: list[str] = []
        window_texts: list[str] = []
        for row in rows:
            total += 1
            rid_str = row["id"]
            literal_raw = row["literal_surface"] or ""
            try:
                from uuid import UUID as _UUID
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(
                        _UUID(rid_str), literal_raw
                    )
            except (HippoIntegrityError, ValueError, TypeError) as exc:
                log.warning(
                    "reembed_from_text: skip id=%s (decrypt failed: %s)",
                    rid_str,
                    type(exc).__name__,
                )
                skipped += 1
                continue
            except Exception as exc:  # noqa: BLE001 -- InvalidTag / OSError fail-safe
                log.warning(
                    "reembed_from_text: skip id=%s (decrypt failed: %s)",
                    rid_str,
                    type(exc).__name__,
                )
                skipped += 1
                continue

            text = (literal_raw or "").strip()
            if not text:
                skipped += 1
                continue
            window_ids.append(rid_str)
            window_texts.append(text)

        # Batch embed: one call per window, id<->text<->vector alignment exact.
        if window_texts:
            try:
                vectors = embedder.embed_batch(window_texts)
            except Exception as exc:  # noqa: BLE001 -- per-window fail-safe
                log.warning(
                    "reembed_from_text: skip window ending id=%s (embed failed: %s)",
                    window_last_id,
                    type(exc).__name__,
                )
                skipped += len(window_texts)
                window_ids = []
                vectors = []
        else:
            vectors = []

        # Batch write: one transaction per window, embedding column only. The
        # raw UPDATE keeps the AES boundary on literal_surface untouched -- only
        # the plaintext embedding blob is rewritten, encoded the same way the
        # per-record update path encoded it (float32 little-endian), so the
        # vectors are byte-identical to embed(literal_surface).
        if not dry_run and window_ids:
            blobs = [_encode_embedding(vec) for vec in vectors]
            with db._conn_lock:
                db._conn.execute("BEGIN")
                try:
                    db._conn.executemany(
                        "UPDATE records SET embedding = ? WHERE id = ?",
                        list(zip(blobs, window_ids)),
                    )
                    db._conn.execute("COMMIT")
                except Exception:
                    db._conn.execute("ROLLBACK")
                    raise
            reembedded += len(window_ids)
        else:
            # dry-run: count what would be re-embedded without writing.
            reembedded += len(window_ids)

        # Advance the cursor only after the window's write commits, so an
        # interrupted run resumes from the next uncommitted window.
        last_id = window_last_id
        if not dry_run:
            _save_resume_cursor(
                store,
                last_id,
                {"reembedded": reembedded, "skipped": skipped, "total": total},
            )

        log.info(
            "reembed: done %d/%d, reembedded %d, skipped %d, last_id=%s",
            total,
            total_target,
            reembedded,
            skipped,
            last_id,
        )

    # The derived structures go stale the moment any prior invocation wrote —
    # a zero-write resume after an interrupted final window must still rebuild
    # and clear the checkpoint, or the index keeps serving pre-correction
    # vectors while the operator reads "re-embedded 0" as done.
    if not dry_run:
        rebuild = db._rebuild_index_from_sqlite()
        # The corpus vectors changed in place but the corpus count did not, so the
        # warm graph's staleness-window cache key never flips. Drop the snapshot so
        # the next build re-streams the corrected vectors into community gating and
        # centrality instead of reusing the stale node set. Key-free unlink; the
        # AES fence is untouched.
        try:
            from iai_mcp import runtime_graph_cache as _rgc
            _rgc.invalidate(store)
        except Exception:  # noqa: BLE001 -- invalidation must never break migration
            pass
        # The resident exact-cosine matrix (if warm) is now stale in the same
        # way: it holds the pre-correction vectors for these ids. Invalidate
        # it alongside the other two derived structures so the next
        # exact_top_k rebuild reads the corrected embeddings, never a stale
        # snapshot with live ids but wrong scores.
        try:
            _inv_exact = getattr(store, "invalidate_exact_index", None)
            if callable(_inv_exact):
                _inv_exact()
        except Exception:  # noqa: BLE001 -- invalidation must never break migration
            pass
        try:
            write_event(
                store,
                "migration_reembed_from_text",
                {
                    "reembedded": reembedded,
                    "skipped": skipped,
                    "total": total,
                    "rebuild": rebuild,
                },
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_reembed_from_text event write failed: %s", exc)
        # The corpus is fully corrected; drop the checkpoint so a later run
        # starts clean rather than resuming a completed migration.
        _clear_resume_cursor(store)
        # The identity stamp asserts that EVERY vector belongs to this
        # embedder, so it lands only when no row was skipped — a corpus with
        # undecryptable leftovers keeps its legacy (unguarded) status and the
        # skip count in the result is the operator's signal.
        if skipped == 0:
            from iai_mcp.embed import (
                EmbedderConfigError,
                EmbedIdentityMismatch,
                stamp_store_embed_identity,
            )

            try:
                stamp_store_embed_identity(store, embedder)
            except (EmbedderConfigError, EmbedIdentityMismatch):
                # A typed stamp refusal means the rewritten vectors CANNOT
                # be attested — swallowing it would leave the old stamp
                # standing as a false attestation over a success-shaped
                # result.
                raise
            except Exception as exc:  # noqa: BLE001 -- stamp fault must never undo the reembed
                log.error("reembed identity stamp failed: %s", exc)

    return {
        "reembedded": reembedded,
        "skipped": skipped,
        "total": total,
        "dry_run": dry_run,
    }
