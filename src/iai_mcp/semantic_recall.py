from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_RPC_TIMEOUT_MS = 500


def _embed_rpc_timeout_ms() -> int:
    raw = os.environ.get("IAI_MCP_EMBED_RPC_TIMEOUT_MS", "")
    try:
        v = int(raw)
        return max(50, v)
    except (ValueError, TypeError):
        return _DEFAULT_EMBED_RPC_TIMEOUT_MS


_DEFAULT_CONSTRUCT_BUDGET_MS = 1000


def _construct_budget_ms() -> int:
    raw = os.environ.get("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return _DEFAULT_CONSTRUCT_BUDGET_MS


_SMOKE_ENCODE_TEXT = "warmup"


def _construct_with_budget(root: "str | Path") -> "tuple[Any, float]":
    import time as _time

    box: "dict[str, Any]" = {}

    def _work() -> None:
        t0 = _time.monotonic()
        try:
            import iai_mcp.embed as _embed_mod

            emb = _embed_mod.embedder_for_store(None)
            emb.embed(_SMOKE_ENCODE_TEXT)
            box["emb"] = emb
        except Exception as exc:  # noqa: BLE001 — construct/encode failure: stay floor
            box["err"] = exc
            logger.debug("construct_with_budget_failed: %s", exc)
        finally:
            box["ms"] = (_time.monotonic() - t0) * 1000.0

    th = threading.Thread(target=_work, daemon=True, name="iai-embed-construct")
    t0 = _time.monotonic()
    th.start()
    th.join(timeout=_construct_budget_ms() / 1000.0)
    if th.is_alive() or "emb" not in box:
        return None, (_time.monotonic() - t0) * 1000.0
    return box["emb"], box.get("ms", (_time.monotonic() - t0) * 1000.0)


def _send_embed_cue_rpc(cue: str, timeout_ms: int) -> "list[float] | None":
    import asyncio
    import json

    from iai_mcp.concurrency import SOCKET_PATH

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
    connect_timeout = timeout_ms / 1000.0

    async def _runner() -> "list[float] | None":
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(sock_path),
                timeout=connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError, asyncio.TimeoutError):
            return None
        try:
            req = {"type": "embed_cue", "cue": cue}
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(
                reader.readline(),
                timeout=connect_timeout,
            )
            if not line:
                return None
            resp = json.loads(line.decode("utf-8"))
            if not isinstance(resp, dict) or not resp.get("ok"):
                return None
            vec = resp.get("embedding")
            if not isinstance(vec, list) or len(vec) != 384:
                return None
            return [float(x) for x in vec]
        except Exception:  # noqa: BLE001
            return None
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    try:
        return asyncio.run(_runner())
    except Exception:  # noqa: BLE001
        return None


_WARM_LOCAL_STORE: "Any" = None


def _get_or_open_warm_local_store(store_root: Path) -> "Any":
    global _WARM_LOCAL_STORE  # noqa: PLW0603
    if _WARM_LOCAL_STORE is not None:
        return _WARM_LOCAL_STORE
    try:
        from iai_mcp.store import MemoryStore
        store = MemoryStore(str(store_root))
        _WARM_LOCAL_STORE = store
        return store
    except Exception as exc:  # noqa: BLE001 — store open failure: fall back to ANN-only
        logger.debug("local_store_open_failed: %s", exc)
        return None


def _recall_daemon_down_post_warm(
    store_root: Path,
    cue: str,
    embedder: "Any",
    n: int,
    session_id: "str | None",
    profile_state: "dict | None" = None,
) -> "list[dict]":
    from iai_mcp.hippo import (
        degraded_semantic_recall as _degrade,
        EMBED_DIM,
    )

    try:
        _test_vec = embedder.embed(cue)
        if not isinstance(_test_vec, (list, tuple)) or len(_test_vec) != EMBED_DIM:
            raise ValueError(f"embed returned unexpected dim {len(_test_vec) if hasattr(_test_vec, '__len__') else '?'}")
    except Exception as exc:  # noqa: BLE001
        logger.debug("daemon_down_local_embed_failed: %s", exc)
        rows = _degrade(store_root, cue, limit=n, session_id=session_id)
        return _stamp_degrade_source(rows)

    local_store = _get_or_open_warm_local_store(store_root)
    if local_store is None:
        return _ann_only_daemon_down(store_root, list(_test_vec), n, cue, session_id)

    try:
        import iai_mcp.embed as _embed_mod
        from iai_mcp import core as _core_mod

        _captured_embedder = embedder

        def _injected_embedder_for_store(_store: "Any") -> "Any":
            return _captured_embedder

        # INVARIANT: single-process, single-recall-per-process only.
        # The warm embedder is injected by swapping the module-global
        # embedder_for_store because core.dispatch resolves the model via that
        # global rather than an explicit parameter. This swap is NOT
        # concurrency-safe: two recalls in the same process would race the
        # global (one could observe the other's injected or restored binding,
        # or see a torn binding while finally restores mid-dispatch). It is
        # safe here because the sole caller is the short-lived daemon-down CLI,
        # which runs exactly one recall per process. Do NOT call this path from
        # the daemon or any threaded/concurrent context without first plumbing
        # the embedder as an explicit argument through core.dispatch.
        _orig_efs = _embed_mod.embedder_for_store
        try:
            _embed_mod.embedder_for_store = _injected_embedder_for_store
            params = {
                "cue": cue,
                "session_id": session_id or "daemon-down",
                "budget_tokens": n * 300,
            }
            resp_dict = _core_mod.dispatch(local_store, "memory_recall", params)
        finally:
            _embed_mod.embedder_for_store = _orig_efs

        hits_raw = resp_dict.get("hits") or []
        if hits_raw:
            return [
                {
                    "literal_surface": h.get("literal_surface", "") or h.get("surface", "") or "",
                    "score": float(h.get("score") or h.get("final_score") or 0.0),
                    "_source": "daemon-down-full",
                    "record_id": str(h.get("record_id") or h.get("id") or ""),
                }
                for h in hits_raw[:n]
            ]
    except Exception as exc:  # noqa: BLE001 — structural path failure: ANN-only fallback
        logger.debug("daemon_down_structural_path_failed: %s", exc)

    return _ann_only_daemon_down(store_root, list(_test_vec), n, cue, session_id)


def _ann_only_daemon_down(
    store_root: Path,
    vec: list[float],
    n: int,
    cue: str,
    session_id: "str | None",
) -> "list[dict]":
    from iai_mcp.hippo import (
        degraded_semantic_recall as _degrade,
        EMBED_DIM,
    )
    from iai_mcp.hippo import _ann_lookup_client

    try:
        labels = _ann_lookup_client(store_root, vec, k=n, embed_dim=EMBED_DIM)
        if labels:
            rows = _fetch_records_by_labels(store_root, labels, n=n)
            if rows:
                return rows
    except Exception:  # noqa: BLE001
        pass

    rows = _degrade(store_root, cue, limit=n, session_id=session_id)
    return _stamp_degrade_source(rows)


def _stamp_degrade_source(rows: "list[dict]") -> "list[dict]":
    result = []
    for row in rows:
        stamped = dict(row)
        stamped["_source"] = "daemon-down-degrade"
        result.append(stamped)
    return result


def recall_semantic_warm(
    store_root: "str | Path",
    cue: str,
    n: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    from iai_mcp.hippo import degraded_semantic_recall as _degrade

    root = Path(store_root)

    embedder, _construct_ms = _construct_with_budget(root)
    if embedder is None:
        rows = _degrade(root, cue, limit=n, session_id=session_id)
        _emit_recall_source(
            "recency-degrade",
            construct_ms=_construct_ms,
            reason="construct_timeout_or_fail",
            session_id=session_id,
        )
        return _stamp_degrade_source(rows)

    result = _recall_daemon_down_post_warm(
        root, cue, embedder, n, session_id, profile_state=None
    )
    _fell_to_recency = (not result) or all(
        r.get("_source") == "daemon-down-degrade" for r in result
    )
    if _fell_to_recency:
        _emit_recall_source(
            "recency-degrade",
            construct_ms=_construct_ms,
            reason="construct_timeout_or_fail",
            session_id=session_id,
        )
    else:
        _emit_recall_source(
            "semantic-inprocess",
            construct_ms=_construct_ms,
            session_id=session_id,
        )
    return result


def _emit_recall_source(
    source: str,
    *,
    construct_ms: "float | None" = None,
    encode_ms: "float | None" = None,
    reason: "str | None" = None,
    session_id: "str | None" = None,
) -> None:
    try:
        from iai_mcp.events import emit_best_effort, TELEMETRY_RECALL_SOURCE

        data: "dict[str, Any]" = {"source": source}
        if construct_ms is not None:
            data["construct_ms"] = round(float(construct_ms), 2)
        if encode_ms is not None:
            data["encode_ms"] = round(float(encode_ms), 2)
        if reason is not None:
            data["reason"] = reason
        emit_best_effort(
            _WARM_LOCAL_STORE,
            TELEMETRY_RECALL_SOURCE,
            data,
            severity="info",
            session_id=session_id or "-",
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break recall
        try:
            logger.debug("recall_source_emit_failed source=%s", source)
        except Exception:  # noqa: BLE001
            pass


def _fetch_records_by_labels(
    store_root: "str | Path",
    vec_labels: "list[int]",
    n: int = 10,
) -> "list[dict]":
    from iai_mcp.hippo import AccessMode, HippoDB

    if not vec_labels:
        return []

    root = Path(store_root)
    db: "HippoDB | None" = None
    try:
        db = HippoDB(root, access_mode=AccessMode.SHARED, read_only=True, _lock_timeout_override=0.25)

        _crypto_key: "bytes | None" = None
        try:
            from iai_mcp.crypto import CryptoKey as _CK
            _crypto_key = _CK(store_root=root).get_or_create()
        except Exception:  # noqa: BLE001
            pass

        try:
            from iai_mcp.crypto import decrypt_field as _df, is_encrypted as _ie
        except Exception:  # noqa: BLE001
            _df = None  # type: ignore[assignment]
            _ie = None  # type: ignore[assignment]

        results: list[dict] = []
        for label in vec_labels[:n]:
            # This vec_label = ? equality is a second beneficiary of the records
            # vec_label index: the engine serves it through the ColIndex
            # ``col = <literal>`` branch (single-value probe + full-predicate
            # re-apply), so it is index-served and scan-parity on migrated stores
            # exactly like the ``vec_label IN (...)`` recall lookup.
            # Recall read: route through the RO pool (lock-free on lilli,
            # falls back to the shared writer conn on stdlib) so this read
            # never serializes behind background write activity.
            with db.ro_conn() as _rc:
                row = _rc.execute(
                    "SELECT id, literal_surface, created_at FROM records"
                    " WHERE vec_label = ? AND tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0",
                    (label,),
                ).fetchone()
            if row is None:
                continue
            row_id = str(row["id"] or "")
            surface = row["literal_surface"] or ""
            if surface and _crypto_key is not None and _ie is not None and _df is not None:
                try:
                    if _ie(surface):
                        surface = _df(surface, _crypto_key, row_id.encode("utf-8"))
                except Exception:  # noqa: BLE001
                    pass
            results.append({
                "literal_surface": surface,
                "score": 1.0,
                "_source": "direct-store",
            })
        return results
    except Exception:  # noqa: BLE001
        return []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass


def recall_semantic_degraded(
    store_root: "str | Path",
    cue: str,
    n: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    from iai_mcp.hippo import degraded_semantic_recall as _degrade
    return _degrade(store_root, cue, limit=n, session_id=session_id)
