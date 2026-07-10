from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKey, decrypt_field, encrypt_field


log = logging.getLogger(__name__)


_RECENT_APPEND_LOCK = threading.Lock()

_RECENT_WINDOW_PREFIX = "window-"
_RECENT_WINDOW_SUFFIX = ".jsonl"
_RECENT_KEEP_DAYS_DEFAULT = 30


def write_processed_salience_top_n(store: Any, n: int = 1000) -> None:
    try:
        from iai_mcp import retrieve

        embed_dim = int(store.embed_dim)
        graph, _assignment, _rc = retrieve.build_runtime_graph(store)

        centrality_by_id: dict[str, float] = {
            str(nid): graph.get_centrality(nid) for nid in graph.iter_nodes()
        }

        processed_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(processed_dir, 0o700)
        target = processed_dir / "salience-top-N.jsonl"

        entries: list[tuple[float, dict[str, Any]]] = []
        for rec in store.iter_records():
            embedding = getattr(rec, "embedding", None)
            if embedding is None or len(embedding) != embed_dim:
                actual_len = -1 if embedding is None else len(embedding)
                log.warning(
                    "skipping record %s: embedding dim %s != store.embed_dim %s",
                    getattr(rec, "id", "<unknown-id>"),
                    actual_len,
                    embed_dim,
                )
                continue

            try:
                emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
            except (TypeError, ValueError) as exc:
                log.warning(
                    "skipping record %s: embedding not float-coercible (%s)",
                    getattr(rec, "id", "<unknown-id>"),
                    exc,
                )
                continue

            embedding_b64 = base64.b64encode(emb_bytes).decode("ascii")
            salience = centrality_by_id.get(str(rec.id), 0.0)
            entries.append(
                (
                    salience,
                    {
                        "id": str(rec.id),
                        "text": (rec.literal_surface or "")[:200],
                        "embedding_b64": embedding_b64,
                        "tier": rec.tier,
                        "ts": rec.created_at.isoformat(),
                        "salience": float(salience),
                    },
                )
            )

        entries.sort(key=lambda pair: pair[0], reverse=True)
        top = entries[: max(0, int(n))]
        payload_lines = [
            json.dumps(line, separators=(",", ":")) for _s, line in top
        ]
        body = "\n".join(payload_lines)
        if payload_lines:
            body += "\n"

        fd, tmp = tempfile.mkstemp(
            prefix=".salience-top-n.",
            suffix=".tmp",
            dir=str(processed_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except (OSError, ValueError, TypeError):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning("processed salience writer failed: %s", exc)


def write_processed_batch_results(
    batch_id: str,
    content_hash: str,
    results: list[dict],
) -> Path | None:
    try:
        processed_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(processed_dir, 0o700)
        target = processed_dir / f"batch-{batch_id}.jsonl"

        payload_lines = [
            json.dumps(
                {
                    "batch_id": batch_id,
                    "content_hash": content_hash,
                    "result": r,
                },
                separators=(",", ":"),
                default=str,
            )
            for r in results
        ]
        body = "\n".join(payload_lines)
        if payload_lines:
            body += "\n"

        fd, tmp = tempfile.mkstemp(
            prefix=f".batch-{batch_id}.",
            suffix=".tmp",
            dir=str(processed_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except (OSError, ValueError, TypeError):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target
    except OSError as exc:
        import errno
        if getattr(exc, "errno", None) == errno.ENOSPC:
            log.warning(
                "batch results writer ENOSPC (disk full) for %s: %s — "
                "operator: free space under ~/.iai-mcp/.memory-bank/processed/",
                batch_id, exc,
            )
        else:
            log.warning(
                "batch results writer OS error for %s (%s): %s",
                batch_id, errno.errorcode.get(getattr(exc, "errno", 0), "?"), exc,
            )
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("batch results writer failed for %s: %s", batch_id, exc)
        return None


def append_recent_record(
    store: Any,
    record: Any,
    *,
    now: datetime | None = None,
) -> None:
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    recent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(recent_dir, 0o700)

    ts = now or datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    target = recent_dir / f"{_RECENT_WINDOW_PREFIX}{date_str}{_RECENT_WINDOW_SUFFIX}"
    window_aad = date_str.encode("utf-8")

    role = "user"
    if record.provenance:
        role = record.provenance[-1].get("role", "user") or "user"

    emb_bytes = np.asarray(record.embedding, dtype=np.float32).tobytes()
    embedding_b64 = base64.b64encode(emb_bytes).decode("ascii")

    payload = {
        "id": str(record.id),
        "text": record.literal_surface,
        "embedding_b64": embedding_b64,
        "tier": record.tier,
        "ts": record.created_at.isoformat(),
        "role": role,
    }
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        associated_data=window_aad,
    )

    line = (ciphertext + "\n").encode("utf-8")

    with _RECENT_APPEND_LOCK:
        fd = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)


def prune_recent_windows(
    *,
    keep_days: int = _RECENT_KEEP_DAYS_DEFAULT,
    now: datetime | None = None,
) -> int:
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    if not recent_dir.exists():
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=keep_days)
    cutoff_date = cutoff.date()
    deleted = 0
    for fpath in recent_dir.iterdir():
        if not fpath.is_file():
            continue
        name = fpath.name
        if not (
            name.startswith(_RECENT_WINDOW_PREFIX)
            and name.endswith(_RECENT_WINDOW_SUFFIX)
        ):
            continue
        date_part = name[len(_RECENT_WINDOW_PREFIX) : -len(_RECENT_WINDOW_SUFFIX)]
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff_date:
            try:
                fpath.unlink()
                deleted += 1
            except OSError:
                log.warning("prune_recent_windows: failed to unlink %s", fpath)
    return deleted


def read_processed_records() -> Iterator[dict[str, Any]]:
    processed_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "processed"
    target = processed_dir / "salience-top-N.jsonl"
    if not target.exists():
        return
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("read_processed_records: cannot read %s: %s", target, exc)
        return
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning(
                "read_processed_records: skipping malformed line %d: %s",
                line_no,
                exc,
            )
            continue
        if not isinstance(obj, dict):
            log.warning(
                "read_processed_records: skipping non-object line %d",
                line_no,
            )
            continue
        yield obj


def read_recent_records(*, key: bytes | None = None) -> Iterator[dict[str, Any]]:
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    if not recent_dir.exists():
        return

    resolved_key = key
    if resolved_key is None:
        ck = CryptoKey(user_id="default")
        resolved_key = ck.get_or_create()

    candidates: list[tuple[Path, bytes]] = []
    for fpath in recent_dir.iterdir():
        if not fpath.is_file():
            continue
        name = fpath.name
        if not (
            name.startswith(_RECENT_WINDOW_PREFIX)
            and name.endswith(_RECENT_WINDOW_SUFFIX)
        ):
            continue
        date_part = name[len(_RECENT_WINDOW_PREFIX) : -len(_RECENT_WINDOW_SUFFIX)]
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        candidates.append((fpath, date_part.encode("utf-8")))

    candidates.sort(key=lambda pair: pair[0].name, reverse=True)

    for fpath, window_aad in candidates:
        try:
            text = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("read_recent_records: cannot read %s: %s", fpath, exc)
            continue
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                plaintext = decrypt_field(
                    line, resolved_key, associated_data=window_aad
                )
            except (InvalidTag, ValueError) as exc:
                log.warning(
                    "read_recent_records: skipping line %d in %s: %s",
                    line_no,
                    fpath.name,
                    exc,
                )
                continue
            try:
                obj = json.loads(plaintext)
            except json.JSONDecodeError as exc:
                log.warning(
                    "read_recent_records: bad JSON at line %d in %s: %s",
                    line_no,
                    fpath.name,
                    exc,
                )
                continue
            if not isinstance(obj, dict):
                log.warning(
                    "read_recent_records: skipping non-object line %d in %s",
                    line_no,
                    fpath.name,
                )
                continue
            yield obj


def bank_recall_substring(
    query: str,
    limit: int = 20,
    *,
    include_processed: bool = True,
    include_recent: bool = True,
    key: bytes | None = None,
) -> dict[str, Any]:
    needle = (query or "").lower()
    processed_hits: list[tuple[float, dict[str, Any]]] = []
    recent_hits: list[tuple[str, dict[str, Any]]] = []

    if include_processed:
        for row in read_processed_records():
            text = str(row.get("text", "") or "")
            if needle and needle not in text.lower():
                continue
            try:
                salience = float(row.get("salience", 0.0) or 0.0)
            except (TypeError, ValueError):
                salience = 0.0
            hit = {
                "record_id": str(row.get("id", "")),
                "score": salience,
                "reason": "bank-substring-match (processed)",
                "literal_surface": text,
                "adjacent_suggestions": [],
                "valid_from": None,
                "valid_to": None,
            }
            processed_hits.append((salience, hit))

    if include_recent:
        for row in read_recent_records(key=key):
            text = str(row.get("text", "") or "")
            if needle and needle not in text.lower():
                continue
            ts = str(row.get("ts", "") or "")
            hit = {
                "record_id": str(row.get("id", "")),
                "score": 0.0,
                "reason": "bank-substring-match (recent)",
                "literal_surface": text,
                "adjacent_suggestions": [],
                "valid_from": None,
                "valid_to": None,
            }
            recent_hits.append((ts, hit))

    processed_hits.sort(key=lambda pair: pair[0], reverse=True)
    recent_hits.sort(key=lambda pair: pair[0], reverse=True)

    ordered = [h for _s, h in processed_hits] + [h for _t, h in recent_hits]
    capped = ordered[: max(0, int(limit))]

    return {
        "hits": capped,
        "anti_hits": [],
        "activation_trace": [],
        "budget_used": 0,
        "cue_mode": "verbatim",
        "patterns_observed": [],
        "_knobs_applied": {},
    }
