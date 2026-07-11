"""Local loopback-only (127.0.0.1) brain-view server: watch the memory graph,
add a memory, or hint one into the forgetting queue.

Reads go to the awake MemoryStore, writes through the capture spine. No delete:
a forget hint sets centrality -> 0 and never_decay -> off, leaving the record
for the sleep pipeline's eligibility gate. Pinned records refuse the hint.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

BRAINVIEW_DEFAULT_PORT: int = 4477

#: Per-call token overhead for one avoided memory search (request framing +
#: tool schema + answer scaffolding), used as the savings floor.
SEARCH_CALL_OVERHEAD_TOK: int = 220
GRAPH_NODE_LIMIT: int = 400
GRAPH_EDGE_LIMIT: int = 3000
SURFACE_TRUNC: int = 280
_MAX_POST_BYTES: int = 40_000_000
_BROWSE_ROOT_SCAN_CAP: int = 20_000
_READ_CACHE_MAX: int = 128


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

#: Verbs served off the lock-free RO snapshot when the writable store is
#: unavailable (relay busy + write-open fenced). Read-only by construction.
BRAIN_VIEW_RO_VERBS = frozenset({"overview", "graph", "economy", "events", "browse"})

#: OBSERVATION, not user work: these must NEVER reset the daemon's activity
#: clock, or a dashboard polling them keeps the sleep pipeline from ever
#: seeing an idle window (browsing one's own memory is watching, not working).
BRAIN_VIEW_OBSERVATION_VERBS = BRAIN_VIEW_RO_VERBS | {"search", "surface"}


def _graph_node_selectors(
    central_share: int, recent_share: int
) -> list[tuple[str, tuple]]:
    """Node selection with per-tier representation floors so a dominant tier
    cannot evict the smaller tiers. Unused floor capacity is reclaimed by the
    global centrality pass."""
    base = (
        "SELECT id FROM records WHERE tombstoned_at IS NULL"
        " AND COALESCE(embedding_pending, 0) = 0"
    )
    procedural_floor = max(10, central_share // 10)
    semantic_floor = max(30, central_share // 4)
    return [
        (
            base + " AND tier = 'procedural'"
            " ORDER BY centrality DESC, created_at DESC LIMIT ?",
            (procedural_floor,),
        ),
        (
            base + " AND tier = 'semantic'"
            " ORDER BY centrality DESC, created_at DESC LIMIT ?",
            (semantic_floor,),
        ),
        (base + " ORDER BY centrality DESC LIMIT ?", (central_share,)),
        (base + " ORDER BY created_at DESC LIMIT ?", (recent_share,)),
    ]


def _savings_lower_est(
    served: int, served_bytes: int, avg_search_tokens: int
) -> int:
    """Lower-bound token savings: per-call overhead per served pack, plus the
    observed premium when real search responses run larger than the packs."""
    if served <= 0:
        return 0
    savings = served * SEARCH_CALL_OVERHEAD_TOK
    if avg_search_tokens:
        avg_pack = (served_bytes // 4) // served if served else 0
        savings += served * max(0, avg_search_tokens - avg_pack)
    return savings


def _load_page() -> bytes:
    ref = resources.files("iai_mcp") / "_deploy" / "brainview" / "index.html"
    return ref.read_bytes()


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


class BrainView:
    """Data layer for the HTTP handlers, with an adaptive backend.

    While the daemon's socket answers, every data verb relays to the daemon —
    the single writer owns the store, the view is a thin client. When the
    daemon is away, the view opens the store directly (SHARED, non-persisting)
    and serves from the awake hippocampus itself. A view constructed WITH a
    store is pinned direct (tests, and the daemon's own server-side view).
    """

    _RELAY_TIMEOUT_S = 90.0
    _RELAY_MAX_RESP = 64 * 1024 * 1024
    # Long enough that a UI polling every few seconds pays the slow-relay
    # probe once per window, not on every tick.
    _RELAY_RETRY_COOLDOWN_S = 30.0

    def __init__(
        self,
        store: Any = None,
        *,
        store_root: "str | Path | None" = None,
        pinned_direct: bool = False,
    ) -> None:
        if store is None and store_root is None:
            raise ValueError("BrainView requires a store or a store_root")
        self.store = store
        self.root = Path(store_root) if store_root is not None else Path(store.root)
        self._pinned_direct = bool(pinned_direct or store is not None)
        self._lock = threading.Lock()
        self._backend_lock = threading.Lock()
        self._relay_down_until = 0.0
        self._direct_down_until = 0.0
        self._direct_down_reason = ""
        self._ro_cached = None
        self._ro_lock = threading.Lock()
        self._read_cache: dict = {}
        self._read_inflight: set = set()
        self._last_request_ts = time.monotonic()
        self._direct_active = 0  # in-flight direct store ops, guarded by _backend_lock
        if not self._pinned_direct:
            threading.Thread(
                target=self._idle_janitor, daemon=True,
            ).start()

    #: An idle direct-mode view must not keep holding the store: the engine
    #: admits ONE writer per file, and a daemon (re)booting overnight would
    #: crash against a viewer nobody is looking at. Reads reopen lazily.
    _IDLE_RELEASE_S = 30.0

    def _idle_janitor(self) -> None:
        while True:
            time.sleep(15.0)
            try:
                with self._backend_lock:
                    idle = (
                        self.store is not None
                        and self._direct_active == 0
                        and time.monotonic() - self._last_request_ts
                        > self._IDLE_RELEASE_S
                    )
                    store = self.store if idle else None
                    if idle:
                        self.store = None
                if store is not None:
                    store.close()
                    logger.debug("idle janitor released the direct store")
            except Exception:  # noqa: BLE001 -- the janitor must outlive any hiccup
                logger.debug("idle janitor pass failed", exc_info=True)

    # -- backend ----------------------------------------------------------

    def _socket_path(self) -> Path:
        return self.root / ".daemon.sock"

    def _relay_alive(self, ignore_cooldown: bool = False) -> bool:
        # The retry cooldown exists for READS, which have an instant RO
        # fallback. Writes and search have none — for them a live socket is
        # always worth the wait, so they check only that it exists.
        if self._pinned_direct:
            return False
        if not ignore_cooldown and time.monotonic() < self._relay_down_until:
            return False
        try:
            return self._socket_path().is_socket()
        except OSError:
            return False

    def _open_direct(self) -> Any:
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        # SHARED coexists with a booting/adjacent holder; a non-owning view
        # never persists the hnsw index (disk hnsw belongs to the daemon).
        return MemoryStore(
            self.root, access_mode=AccessMode.SHARED, persist_index=False,
        )

    def _backend(self) -> str:
        if self._relay_alive():
            with self._backend_lock:
                if self.store is not None and not self._pinned_direct:
                    # Yield the writer back to the daemon: the engine admits
                    # a single writer per store file.
                    try:
                        self.store.close()
                    except Exception:  # noqa: BLE001 -- yield must not fail the request
                        pass
                    self.store = None
            return "relay"
        with self._backend_lock:
            if self.store is None:
                if time.monotonic() < self._direct_down_until:
                    raise RuntimeError(
                        f"writable store unavailable: {self._direct_down_reason}"
                    )
                try:
                    self.store = self._open_direct()
                except Exception as exc:
                    # Same economics as the relay cooldown: pay the fenced
                    # writable probe once per window, not on every poll.
                    self._direct_down_until = (
                        time.monotonic() + self._RELAY_RETRY_COOLDOWN_S
                    )
                    self._direct_down_reason = str(exc)[:200]
                    raise
        return "direct"

    #: Verbs served straight off a lock-free read-only snapshot whenever the
    #: daemon is too busy to answer and the writable open is fenced — the
    #: brain stays visible through its own consolidation.
    _RO_VERBS = BRAIN_VIEW_RO_VERBS
    # A grinding daemon loses the read after 3s; the RO snapshot answers in
    # milliseconds and the cooldown keeps the whole window on the snapshot.
    _RELAY_READ_TIMEOUT_S = 3.0

    def _call(self, verb: str, kwargs: dict) -> Any:
        if verb not in self._RO_VERBS:
            self._last_request_ts = time.monotonic()
        if verb in self._RO_VERBS:
            return self._read_stale_while_revalidate(verb, kwargs)
        if self._relay_alive(ignore_cooldown=True):
            return self._relay(verb, kwargs)
        try:
            backend = self._backend()
        except Exception as exc:  # noqa: BLE001 -- writable open fenced/locked
            return {"status": "memory_busy", "reason": str(exc)[:200]}
        if backend == "relay":
            return self._relay(verb, kwargs)
        with self._backend_lock:
            self._direct_active += 1
        try:
            return getattr(self, f"{verb}_direct")(**kwargs)
        finally:
            with self._backend_lock:
                self._direct_active -= 1

    @staticmethod
    def _is_busy_result(res: Any) -> bool:
        return isinstance(res, dict) and res.get("status") in (
            "memory_busy", "daemon_down", "error",
        )

    def _read_stale_while_revalidate(self, verb: str, kwargs: dict) -> Any:
        """Reads never hang the UI: one request per verb refreshes at a time;
        everyone else (and any refresh that fails against a busy brain) gets
        the last good payload marked stale. Consolidation becomes something
        you can WATCH instead of something that blanks the screen."""
        key = f"{verb}:{sorted(kwargs.items())!r}"
        # Every _read_cache access is under _backend_lock: ThreadingHTTPServer
        # runs one thread per request on this shared view, and an unguarded
        # eviction loop racing an insert raises "dictionary changed size".
        with self._backend_lock:
            cached = self._read_cache.get(key)
            leader = key not in self._read_inflight
            if leader:
                self._read_inflight.add(key)
        if not leader:
            if cached is not None:
                return self._mark_stale(cached[1])
            # Cold-cache herd: several pollers land before the first result
            # exists. Followers wait briefly for the leader instead of each
            # paying its own read; if the leader vanishes without caching
            # (busy backend), fall through and read once ourselves.
            for _ in range(10):
                time.sleep(0.15)
                with self._backend_lock:
                    warmed = self._read_cache.get(key)
                    still_inflight = key in self._read_inflight
                if warmed is not None:
                    return self._mark_stale(warmed[1])
                if not still_inflight:
                    break
        try:
            res = self._read_once(verb, kwargs)
            if not self._is_busy_result(res):
                with self._backend_lock:
                    self._read_cache[key] = (time.time(), res)
                    while len(self._read_cache) > _READ_CACHE_MAX:
                        self._read_cache.pop(next(iter(self._read_cache)))
            elif cached is not None:
                return self._mark_stale(cached[1])
            return res
        finally:
            if leader:
                with self._backend_lock:
                    self._read_inflight.discard(key)

    @staticmethod
    def _mark_stale(payload: Any) -> Any:
        if isinstance(payload, dict):
            out = dict(payload)
            out["stale"] = True
            return out
        return payload

    def _read_once(self, verb: str, kwargs: dict) -> Any:
        # Reads NEVER open the writable store: while a daemon is down, a
        # watched dashboard polling every few seconds must not hold the
        # single-writer engine — or the daemon could never boot back. The
        # RO snapshot serves reads; the writable store is touched only if
        # a recent write already holds it (fresh and free).
        if self._relay_alive():
            res = self._relay(verb, kwargs, timeout=self._RELAY_READ_TIMEOUT_S)
            if self._is_busy_result(res):
                return self._ro_read(verb, kwargs)
            return res
        if self.store is not None:
            try:
                return getattr(self, f"{verb}_direct")(**kwargs)
            except Exception:  # noqa: BLE001 -- racing the janitor's close
                return self._ro_read(verb, kwargs)
        return self._ro_read(verb, kwargs)

    def _relay(self, verb: str, kwargs: dict, timeout: float | None = None) -> Any:
        import socket as _socket

        req = {
            "jsonrpc": "2.0", "id": 1, "method": "brain_view",
            "params": {"verb": verb, "kwargs": kwargs},
        }
        budget = timeout if timeout is not None else self._RELAY_TIMEOUT_S
        deadline = time.monotonic() + budget
        try:
            from iai_mcp._ipc import make_sync_ipc_socket, send_sync_auth_token
            s, _addr = make_sync_ipc_socket()
            s.settimeout(budget)
            s.connect(_addr)
            send_sync_auth_token(s)
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n") and len(buf) < self._RELAY_MAX_RESP:
                # A slowly-streaming daemon must not evade the budget one
                # chunk at a time: the deadline is TOTAL, not per-recv.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _socket.timeout("relay budget exhausted")
                s.settimeout(remaining)
                chunk = s.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
            s.close()
            resp = json.loads(buf or b"{}")
        except (OSError, ValueError) as exc:
            self._relay_down_until = time.monotonic() + self._RELAY_RETRY_COOLDOWN_S
            return {"status": "daemon_down", "reason": str(exc)[:200]}
        if "error" in resp:
            return {"status": "error", "reason": str(resp["error"])[:200]}
        return resp.get("result")

    # -- lock-free read-only snapshot fallback ------------------------------

    def _ro_conn(self):
        """Short-lived lock-free RO snapshot on the store file. Never touches
        the writer lock; a concurrent checkpoint invalidates it (the caller
        retries once on the fence error)."""
        import sqlite3 as _sqlite3

        from iai_mcp.hippo._raw_open import open_store_conn

        db_path = str(self.root / "hippo" / "brain.sqlite3")
        conn = open_store_conn(db_path, read_only=True)
        if conn is None:
            conn = _sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False,
            )
        conn.row_factory = _sqlite3.Row
        return conn

    _EMPTY_SHELLS = {
        "overview": lambda: {
            "counts": {"episodic": 0, "semantic": 0, "procedural": 0,
                       "tombstoned": 0, "edges": 0},
            "lifecycle": "awake", "daemon_up": False,
        },
        "graph": lambda: {"nodes": [], "edges": []},
        "events": lambda: [],
        "economy": lambda: {
            "packs_built": 0, "packs_served": 0, "tokens_injected_est": 0,
            "agent_searches": 0, "avg_search_tokens_est": 0,
            "tokens_saved_lower_est": 0,
        },
        "browse": lambda: {"folders": {"time": [], "topics": [], "teachings": [],
                                       "sessions": [], "special": []}},
    }

    def _ro_read(self, verb: str, kwargs: dict) -> Any:
        # A brand-new install has no store file yet: show an empty brain, not
        # a scary 'memory_busy'.
        db_path = self.root / "hippo" / "brain.sqlite3"
        if not db_path.is_file():
            shell = self._EMPTY_SHELLS.get(verb)
            if shell is not None:
                return shell()
        # The snapshot conn is CACHED: the engine pays catalog/column-index
        # decode on open, far too heavy per poll. A checkpoint fences the
        # cached snapshot -> drop + reopen once (which also refreshes the
        # data the snapshot sees).
        with self._ro_lock:
            for attempt in (1, 2):
                try:
                    if self._ro_cached is None:
                        self._ro_cached = self._ro_conn()
                    return getattr(self, f"_ro_{verb}")(
                        self._ro_cached, **kwargs
                    )
                except Exception as exc:  # noqa: BLE001 -- fence or busy: reopen once, then honest
                    try:
                        if self._ro_cached is not None:
                            self._ro_cached.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._ro_cached = None
                    if attempt == 1:
                        continue
                    logger.debug(
                        "brainview ro fallback %s failed: %s", verb, exc,
                    )
                    return {"status": "memory_busy", "reason": str(exc)[:200]}

    def _decrypt_record_col(self, rid: str, value):
        from iai_mcp.crypto import CryptoKey, decrypt_field, is_encrypted

        if value is None or not is_encrypted(value):
            return value
        key = self._ro_key_cached()
        try:
            return decrypt_field(value, key, associated_data=rid.lower().encode("ascii"))
        except Exception:  # noqa: BLE001 -- a torn row must not blank the view
            return ""

    def _ro_key_cached(self) -> bytes:
        key = getattr(self, "_ro_key", None)
        if key is None:
            from iai_mcp.crypto import CryptoKey

            key = CryptoKey(store_root=self.root).get_or_create()
            self._ro_key = key
        return key

    def _ro_overview(self, conn) -> dict:
        counts: dict[str, int] = {}
        for tier in ("episodic", "semantic", "procedural"):
            row = conn.execute(
                "SELECT COUNT(*) FROM records WHERE tier = ?"
                " AND tombstoned_at IS NULL",
                (tier,),
            ).fetchone()
            counts[tier] = int(row[0]) if row else 0
        row = conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NOT NULL"
        ).fetchone()
        counts["tombstoned"] = int(row[0]) if row else 0
        row = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        counts["edges"] = int(row[0]) if row else 0
        return {
            "counts": counts,
            "lifecycle": self._lifecycle_label(),
            "daemon_up": self._socket_path().is_socket(),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def _lifecycle_label(self) -> str:
        try:
            from iai_mcp.lifecycle_state import lifecycle_state_path

            lc_path = lifecycle_state_path(self.root)
            if lc_path.is_file():
                return (
                    json.loads(lc_path.read_text(encoding="utf-8")).get("state")
                    or "awake"
                )
        except Exception:  # noqa: BLE001 -- display is cosmetic
            pass
        return "awake"

    _RO_NODE_COLS = (
        "id, tier, centrality, community_id, pinned, never_decay,"
        " created_at, last_reviewed, tags_json, provenance_json,"
        " literal_surface, tombstoned_at"
    )

    def _ro_node_dict(self, row) -> dict:
        rid = str(row["id"])
        surface = self._decrypt_record_col(rid, row["literal_surface"]) or ""
        prov_raw = self._decrypt_record_col(rid, row["provenance_json"])
        hinted = False
        try:
            for p in json.loads(prov_raw or "[]"):
                cue = str(p.get("cue") or "")
                if cue == "user-forget-hint":
                    hinted = True
                elif cue == "user-rescue":
                    hinted = False
        except (ValueError, TypeError, AttributeError):
            pass
        try:
            tags = [
                t for t in json.loads(row["tags_json"] or "[]")
                if not str(t).startswith("idem:")
            ][:6]
        except (ValueError, TypeError):
            tags = []
        return {
            "id": rid,
            "surface": str(surface)[:SURFACE_TRUNC],
            "tier": str(row["tier"] or ""),
            "centrality": float(row["centrality"] or 0.0),
            "community": str(row["community_id"] or ""),
            "pinned": bool(row["pinned"]),
            "never_decay": bool(row["never_decay"]),
            "hinted": hinted,
            "created_at": _iso(row["created_at"]),
            "last_reviewed": _iso(row["last_reviewed"]),
            "tags": tags,
        }

    def _ro_graph(self, conn, limit: int = GRAPH_NODE_LIMIT) -> dict:
        limit = max(10, min(int(limit), GRAPH_NODE_LIMIT))
        recent_share = max(20, limit // 4)
        central_share = limit - recent_share
        nodes: list[dict] = []
        seen: set[str] = set()
        selectors = [
            (sql.replace("SELECT id FROM", f"SELECT {self._RO_NODE_COLS} FROM", 1), args)
            for sql, args in _graph_node_selectors(central_share, recent_share)
        ]
        selectors.append((
            f"SELECT {self._RO_NODE_COLS} FROM records"
            " WHERE tombstoned_at IS NOT NULL"
            " ORDER BY tombstoned_at DESC LIMIT 40",
            (),
        ))
        for sql, args in selectors:
            try:
                rows = conn.execute(sql, args).fetchall()
            except Exception as exc:  # noqa: BLE001 -- one selector must not blank the view
                logger.debug("ro graph select failed: %s", exc)
                rows = []
            for row in rows:
                rid = str(row["id"])
                if rid in seen:
                    continue
                seen.add(rid)
                node = self._ro_node_dict(row)
                if row["tombstoned_at"] is not None:
                    node["ghost"] = True
                nodes.append(node)
        node_ids = {n["id"] for n in nodes}
        edges = []
        try:
            rows = conn.execute(
                "SELECT src, dst, edge_type, weight FROM edges LIMIT ?",
                (GRAPH_EDGE_LIMIT * 10,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ro edge select failed: %s", exc)
            rows = []
        for row in rows:
            src, dst = str(row["src"]), str(row["dst"])
            if src in node_ids and dst in node_ids and src != dst:
                edges.append({
                    "src": src, "dst": dst,
                    "type": str(row["edge_type"]),
                    "w": float(row["weight"] or 0.0),
                })
                if len(edges) >= GRAPH_EDGE_LIMIT:
                    break
        return {"nodes": nodes, "edges": edges}

    def _ro_events(self, conn, limit: int = 30) -> list[dict]:
        try:
            rows = conn.execute(
                "SELECT kind, severity, ts FROM events ORDER BY ts DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ro events select failed: %s", exc)
            return []
        return [
            {
                "kind": str(r["kind"] or "?"),
                "severity": str(r["severity"] or "info"),
                "ts": _iso(r["ts"]),
            }
            for r in rows
        ]

    def _ro_economy(self, conn) -> dict:
        from iai_mcp.crypto import decrypt_field, is_encrypted

        packs_built = 0
        searches = 0
        search_chars = 0
        try:
            rows = conn.execute(
                "SELECT id, kind, data_json FROM events WHERE kind IN"
                " ('foresight_pack_built', 'recall_dispatched')"
                " ORDER BY ts DESC LIMIT 1000",
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ro economy select failed: %s", exc)
            rows = []
        for r in rows:
            raw = r["data_json"]
            if raw is not None and is_encrypted(raw):
                try:
                    raw = decrypt_field(
                        raw, self._ro_key_cached(),
                        associated_data=str(r["id"]).lower().encode("ascii"),
                    )
                except Exception:  # noqa: BLE001
                    continue
            try:
                data = json.loads(raw or "{}")
            except (ValueError, TypeError):
                data = {}
            if r["kind"] == "foresight_pack_built":
                packs_built += 1
            else:
                searches += 1
                search_chars += int(data.get("resp_chars") or 0)
        served = 0
        served_bytes = 0
        try:
            ledger = self.root / "logs" / "foresight-served.jsonl"
            if ledger.is_file():
                for line in ledger.read_text(encoding="utf-8").splitlines()[-2000:]:
                    try:
                        rowd = json.loads(line)
                        served += 1
                        served_bytes += int(rowd.get("bytes") or 0)
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass
        avg_search_tokens = (search_chars // 4 // searches) if searches else 0
        savings_lower = _savings_lower_est(
            served, served_bytes, avg_search_tokens,
        )
        return {
            "packs_built": packs_built,
            "packs_served": served,
            "tokens_injected_est": served_bytes // 4,
            "agent_searches": searches,
            "avg_search_tokens_est": avg_search_tokens,
            "tokens_saved_lower_est": savings_lower,
        }

    # -- public verbs (backend-adaptive) ------------------------------------

    def overview(self) -> dict:
        return self._call("overview", {})

    def economy(self) -> dict:
        return self._call("economy", {})

    def events(self, limit: int = 30) -> list[dict]:
        return self._call("events", {"limit": limit})

    def browse(self, folder: "dict | None" = None, limit: int = 60) -> dict:
        return self._call("browse", {"folder": folder, "limit": limit})

    def graph(self, limit: int = GRAPH_NODE_LIMIT) -> dict:
        return self._call("graph", {"limit": limit})

    def search(self, query: str, k: int = 12) -> dict:
        return self._call("search", {"query": query, "k": k})

    def rescue(self, record_id: str) -> dict:
        return self._call("rescue", {"record_id": record_id})

    def surface(self, record_id: str) -> dict:
        return self._call("surface", {"record_id": record_id})

    def pin(self, record_id: str, on: bool = True) -> dict:
        return self._call("pin", {"record_id": record_id, "on": bool(on)})

    def capture(self, text: str) -> dict:
        return self._call("capture", {"text": text})

    def teach(
        self,
        filename: str,
        content_b64: str,
        session_id: str = "study",
        reconcile: bool = False,
        fade: bool = False,
    ) -> dict:
        return self._call("teach", {
            "filename": filename,
            "content_b64": content_b64,
            "session_id": session_id,
            "reconcile": reconcile,
            "fade": fade,
        })

    def forget_hint(self, record_id: str) -> dict:
        return self._call("forget_hint", {"record_id": record_id})

    # -- reads ------------------------------------------------------------

    def overview_direct(self) -> dict:
        db = self.store.db
        counts: dict[str, int] = {}
        with db.ro_conn() as conn:
            for tier in ("episodic", "semantic", "procedural"):
                row = conn.execute(
                    "SELECT COUNT(*) FROM records WHERE tier = ?"
                    " AND tombstoned_at IS NULL",
                    (tier,),
                ).fetchone()
                counts[tier] = int(row[0]) if row else 0
            row = conn.execute(
                "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NOT NULL"
            ).fetchone()
            counts["tombstoned"] = int(row[0]) if row else 0
            row = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
            counts["edges"] = int(row[0]) if row else 0

        lifecycle = "awake"
        try:
            from iai_mcp.lifecycle_state import lifecycle_state_path

            lc_path = lifecycle_state_path(self.store.root)
            if lc_path.is_file():
                lifecycle = (
                    json.loads(lc_path.read_text(encoding="utf-8")).get("state")
                    or "awake"
                )
        except Exception:  # noqa: BLE001 -- lifecycle display is cosmetic
            lifecycle = "awake"

        daemon_up = False
        try:
            daemon_up = (Path(self.store.root) / ".daemon.sock").is_socket()
        except Exception:  # noqa: BLE001 -- presence probe is cosmetic
            daemon_up = False
        return {
            "counts": counts,
            "lifecycle": lifecycle,
            "daemon_up": daemon_up,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def economy_direct(self) -> dict:
        """Economy counters: packs built, packs served, injected token
        estimate, agent searches, and a lower-bound token-savings estimate."""
        packs_built = 0
        searches = 0
        search_chars = 0
        try:
            from iai_mcp.events import query_events

            for ev in query_events(self.store, kind="foresight_pack_built", limit=500):
                packs_built += 1
            for ev in query_events(self.store, kind="recall_dispatched", limit=500):
                searches += 1
                search_chars += int((ev.get("data") or {}).get("resp_chars") or 0)
        except Exception as exc:  # noqa: BLE001 -- economy is observability, degrade to zeros
            logger.debug("economy events read failed: %s", exc)

        served = 0
        served_bytes = 0
        try:
            ledger = Path(self.store.root) / "logs" / "foresight-served.jsonl"
            if ledger.is_file():
                for line in ledger.read_text(encoding="utf-8").splitlines()[-2000:]:
                    try:
                        row = json.loads(line)
                        served += 1
                        served_bytes += int(row.get("bytes") or 0)
                    except (ValueError, TypeError):
                        continue
        except OSError as exc:
            logger.debug("economy ledger read failed: %s", exc)

        avg_search_tokens = (search_chars // 4 // searches) if searches else 0
        savings_lower = _savings_lower_est(
            served, served_bytes, avg_search_tokens,
        )
        return {
            "packs_built": packs_built,
            "packs_served": served,
            "tokens_injected_est": served_bytes // 4,
            "agent_searches": searches,
            "avg_search_tokens_est": avg_search_tokens,
            "tokens_saved_lower_est": savings_lower,
        }

    def browse_direct(self, folder: "dict | None" = None, limit: int = 60) -> dict:
        with self.store.db.ro_conn() as conn:
            return self._browse_impl(conn, folder, limit)

    def _ro_browse(self, conn, folder: "dict | None" = None, limit: int = 60) -> dict:
        return self._browse_impl(conn, folder, limit)

    #: Folders map to existing groupings: time (created_at), topics (Leiden
    #: communities), teachings (studied documents), sessions, and curation state.
    def _browse_impl(self, conn, folder: "dict | None", limit: int) -> dict:
        if folder is not None and not isinstance(folder, dict):
            return {"status": "error", "reason": "folder must be an object"}
        limit = max(1, min(int(limit), 200))
        if not folder:
            return {"folders": self._browse_folders_impl(conn)}
        kind = str(folder.get("kind") or "")
        key = str(folder.get("key") or "")
        order = "created_at DESC"
        where: str
        args: tuple
        if kind == "time":
            lo, hi = self._time_bucket_bounds().get(key, (None, None))
            clauses = ["tombstoned_at IS NULL"]
            targs: list = []
            if lo is not None:
                clauses.append("created_at >= ?")
                targs.append(lo)
            if hi is not None:
                clauses.append("created_at < ?")
                targs.append(hi)
            where, args = " AND ".join(clauses), tuple(targs)
        elif kind == "topic":
            where, args = "tombstoned_at IS NULL AND community_id = ?", (key,)
        elif kind == "teaching":
            where, args = (
                "tombstoned_at IS NULL AND tags_json LIKE ?",
                (f'%"{key}"%',),
            )
        elif kind == "session":
            return {"items": self._browse_session(conn, key, limit)}
        elif kind == "special" and key == "pinned":
            where, args = "tombstoned_at IS NULL AND pinned = 1", ()
        elif kind == "special" and key == "fading":
            where, args = "tombstoned_at IS NOT NULL", ()
            order = "tombstoned_at DESC"
        elif kind == "special" and key == "contradicted":
            return {"items": self._browse_contradicted(conn, limit)}
        else:
            return {"status": "error", "reason": f"unknown folder {kind}/{key}"}
        rows = conn.execute(
            f"SELECT {self._RO_NODE_COLS} FROM records WHERE {where}"
            f" ORDER BY {order} LIMIT ?",
            (*args, limit),
        ).fetchall()
        items = []
        for row in rows:
            node = self._ro_node_dict(row)
            if row["tombstoned_at"] is not None:
                node["ghost"] = True
            items.append(node)
        return {"items": items}

    def _record_session_id(self, rid: str, provenance_json) -> str:
        raw = self._decrypt_record_col(rid, provenance_json)
        try:
            prov = json.loads(raw or "[]")
            if prov and isinstance(prov, list):
                return str(prov[0].get("session_id") or "")
        except (ValueError, TypeError, AttributeError):
            pass
        return ""

    def _browse_session(self, conn, key: str, limit: int) -> list[dict]:
        # session lives inside encrypted provenance, not in a column: scan
        # newest-first, decrypt, match — bounded so one click stays cheap.
        rows = conn.execute(
            f"SELECT {self._RO_NODE_COLS} FROM records"
            " WHERE tombstoned_at IS NULL"
            " ORDER BY created_at DESC LIMIT 5000",
        ).fetchall()
        items = []
        for row in rows:
            if self._record_session_id(str(row["id"]), row["provenance_json"]) == key:
                items.append(self._ro_node_dict(row))
                if len(items) >= limit:
                    break
        return items

    def _browse_contradicted(self, conn, limit: int) -> list[dict]:
        rows = conn.execute(
            "SELECT dst FROM edges WHERE edge_type = 'contradicts' LIMIT 300",
        ).fetchall()
        seen: list[str] = []
        for r in rows:
            rid = str(r[0])
            if rid not in seen:
                seen.append(rid)
        items = []
        for rid in seen[: min(limit, 60)]:
            row = conn.execute(
                f"SELECT {self._RO_NODE_COLS} FROM records WHERE id = ?", (rid,),
            ).fetchone()
            if row is None:
                continue
            node = self._ro_node_dict(row)
            if row["tombstoned_at"] is not None:
                node["ghost"] = True
            items.append(node)
        return items

    @staticmethod
    def _time_bucket_bounds() -> dict:
        from datetime import datetime, timedelta, timezone

        midnight = (
            datetime.now().astimezone()
            .replace(hour=0, minute=0, second=0, microsecond=0)
        )

        def utc(dt) -> str:
            return str(dt.astimezone(timezone.utc))

        return {
            "today": (utc(midnight), None),
            "yesterday": (utc(midnight - timedelta(days=1)), utc(midnight)),
            "week": (utc(midnight - timedelta(days=7)),
                     utc(midnight - timedelta(days=1))),
            "month": (utc(midnight - timedelta(days=30)),
                      utc(midnight - timedelta(days=7))),
            "older": (None, utc(midnight - timedelta(days=30))),
        }

    def _browse_folders_impl(self, conn) -> dict:
        from collections import Counter

        # Bounded like _browse_session: the newest slice is what the folder
        # counts summarize; an unbounded scan + per-row AES-decrypt would
        # block a handler thread proportional to total corpus size.
        rows = conn.execute(
            "SELECT id, community_id, provenance_json, created_at, pinned,"
            " tombstoned_at, tags_json FROM records"
            " ORDER BY created_at DESC LIMIT ?",
            (_BROWSE_ROOT_SCAN_CAP,),
        ).fetchall()
        bounds = self._time_bucket_bounds()
        time_counts = {k: 0 for k in bounds}
        communities: Counter = Counter()
        sessions: Counter = Counter()
        session_last: dict = {}
        session_repr: dict = {}
        teachings: Counter = Counter()
        pinned = 0
        fading = 0
        for row in rows:
            if row["tombstoned_at"] is not None:
                fading += 1
                continue
            created = str(row["created_at"] or "")
            for k, (lo, hi) in bounds.items():
                if (lo is None or created >= lo) and (hi is None or created < hi):
                    time_counts[k] += 1
                    break
            cid = row["community_id"]
            if cid:
                communities[str(cid)] += 1
            sid = self._record_session_id(str(row["id"]), row["provenance_json"])
            if sid and sid not in ("-", "None"):
                sessions[sid] += 1
                if created > session_last.get(sid, ""):
                    session_last[sid] = created
                    session_repr[sid] = str(row["id"])
            if row["pinned"]:
                pinned += 1
            tj = row["tags_json"]
            if tj and '"doc:' in str(tj):
                try:
                    for t in json.loads(tj):
                        if str(t).startswith("doc:"):
                            teachings[str(t)] += 1
                except (ValueError, TypeError):
                    pass
        try:
            contradicted = int(conn.execute(
                "SELECT COUNT(*) FROM edges WHERE edge_type = 'contradicts'",
            ).fetchone()[0])
        except Exception:  # noqa: BLE001 -- optional count
            contradicted = 0

        def _repr_snippet(where: str, arg: str) -> str:
            row = conn.execute(
                f"SELECT id, literal_surface FROM records WHERE {where}"
                " AND tombstoned_at IS NULL"
                " ORDER BY centrality DESC, created_at DESC LIMIT 1",
                (arg,),
            ).fetchone()
            if row is None:
                return ""
            surf = self._decrypt_record_col(str(row["id"]), row["literal_surface"])
            return " ".join(str(surf or "").split())[:44]

        topics = [
            {"key": cid, "label": _repr_snippet("community_id = ?", cid) or cid[:8],
             "n": n}
            for cid, n in communities.most_common(12)
        ]
        def _surface_of(rid: str) -> str:
            row = conn.execute(
                "SELECT literal_surface FROM records WHERE id = ?", (rid,),
            ).fetchone()
            if row is None:
                return ""
            surf = self._decrypt_record_col(rid, row[0])
            return " ".join(str(surf or "").split())[:44]

        recent_sessions = sorted(
            sessions, key=lambda x: session_last.get(x, ""), reverse=True,
        )[:8]
        session_folders = [
            {"key": sid,
             "label": _surface_of(session_repr.get(sid, "")) or sid[:8],
             "n": sessions[sid]}
            for sid in recent_sessions
        ]
        return {
            "time": [
                {"key": k, "label": k, "n": time_counts[k]}
                for k in ("today", "yesterday", "week", "month", "older")
            ],
            "topics": topics,
            "teachings": [
                {"key": t, "label": t[4:], "n": n}
                for t, n in teachings.most_common(20)
            ],
            "sessions": session_folders,
            "special": [
                {"key": "pinned", "label": "pinned", "n": pinned},
                {"key": "fading", "label": "fading", "n": fading},
                {"key": "contradicted", "label": "contradicted",
                 "n": contradicted},
            ],
        }

    def events_direct(self, limit: int = 30) -> list[dict]:
        try:
            from iai_mcp.events import query_events

            rows = query_events(self.store, limit=limit)
        except Exception as exc:  # noqa: BLE001 -- feed is cosmetic
            logger.debug("brainview events read failed: %s", exc)
            return []
        feed = []
        for r in rows:
            feed.append(
                {
                    "kind": str(r.get("kind") or r.get("event_type") or "?"),
                    "severity": str(r.get("severity") or "info"),
                    "ts": _iso(r.get("ts") or r.get("created_at")),
                }
            )
        return feed

    def graph_direct(self, limit: int = GRAPH_NODE_LIMIT) -> dict:
        db = self.store.db
        limit = max(10, min(int(limit), GRAPH_NODE_LIMIT))
        recent_share = max(20, limit // 4)
        central_share = limit - recent_share

        ids: list[str] = []
        seen: set[str] = set()
        with db.ro_conn() as conn:
            for sql, args in _graph_node_selectors(central_share, recent_share):
                try:
                    rows = conn.execute(sql, args).fetchall()
                except Exception as exc:  # noqa: BLE001 -- one selector failing must not blank the view
                    logger.debug("brainview node select failed: %s", exc)
                    rows = []
                for row in rows:
                    rid = str(row[0])
                    if rid not in seen:
                        seen.add(rid)
                        ids.append(rid)

        uuids = []
        for rid in ids:
            try:
                uuids.append(UUID(rid))
            except ValueError:
                continue
        records = self.store.get_batch(uuids)

        nodes = [
            self._node_dict(records[rid]) for rid in uuids if records.get(rid)
        ]

        # Tombstoned-but-not-yet-erased records stay visible as ghosts so the
        # user can watch (and cancel) forgetting before erasure.
        ghost_ids: list[UUID] = []
        with db.ro_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT id FROM records WHERE tombstoned_at IS NOT NULL"
                    " ORDER BY tombstoned_at DESC LIMIT 40",
                ).fetchall()
            except Exception as exc:  # noqa: BLE001
                logger.debug("brainview ghost select failed: %s", exc)
                rows = []
        for row in rows:
            try:
                ghost_ids.append(UUID(str(row[0])))
            except ValueError:
                continue
        ghosts = self.store.get_batch(ghost_ids)
        for rid in ghost_ids:
            rec = ghosts.get(rid)
            if rec is None:
                continue
            node = self._node_dict(rec)
            node["ghost"] = True
            nodes.append(node)

        node_ids = {n["id"] for n in nodes}
        edges = []
        with db.ro_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT src, dst, edge_type, weight FROM edges LIMIT ?",
                    (GRAPH_EDGE_LIMIT * 10,),
                ).fetchall()
            except Exception as exc:  # noqa: BLE001
                logger.debug("brainview edge select failed: %s", exc)
                rows = []
        for row in rows:
            src, dst = str(row[0]), str(row[1])
            if src in node_ids and dst in node_ids and src != dst:
                edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "type": str(row[2]),
                        "w": float(row[3] or 0.0),
                    }
                )
                if len(edges) >= GRAPH_EDGE_LIMIT:
                    break
        return {"nodes": nodes, "edges": edges}

    def _node_dict(self, rec: Any) -> dict:
        hinted = False
        for p in rec.provenance or []:
            cue = str(p.get("cue") or "")
            if cue == "user-forget-hint":
                hinted = True
            elif cue == "user-rescue":
                hinted = False
        return {
            "id": str(rec.id),
            "surface": (rec.literal_surface or "")[:SURFACE_TRUNC],
            "tier": rec.tier,
            "centrality": float(rec.centrality or 0.0),
            "community": str(rec.community_id or ""),
            "pinned": bool(rec.pinned),
            "never_decay": bool(rec.never_decay),
            "hinted": hinted,
            "created_at": _iso(rec.created_at),
            "last_reviewed": _iso(rec.last_reviewed),
            "tags": [t for t in (rec.tags or []) if not t.startswith("idem:")][:6],
        }

    def search_direct(self, query: str, k: int = 12) -> dict:
        """Semantic find-a-memory: embed the cue, return the nearest records
        in the same node shape the graph uses (awake path, daemon-free)."""
        query = (query or "").strip()
        if not query:
            return {"hits": []}
        try:
            from iai_mcp.embed import embedder_for_store

            emb = embedder_for_store(self.store).embed(query[:512])
            raw = self.store.query_similar(list(emb), k=max(1, min(int(k), 24)))
        except Exception as exc:  # noqa: BLE001 -- search is navigation, degrade to empty
            logger.warning("brainview search failed: %s", exc)
            return {"hits": []}
        hits = []
        for rec, cos in raw:
            node = self._node_dict(rec)
            node["cos"] = round(float(cos), 3)
            hits.append(node)
        return {"hits": hits}

    def daemon_action(self, action: str) -> dict:
        """Best-effort operator verbs over the daemon socket; an absent socket
        is a normal answer, not a failure."""
        if action in ("start", "stop", "restart"):
            return self._daemon_lifecycle(action)
        # These are CONTROL messages (the `type` protocol), not JSON-RPC
        # methods — the daemon's control plane reads `type`, and a `method`
        # envelope would fall through to core.dispatch and error out.
        ctype = {
            "sleep": "user_initiated_sleep",
            "wake": "force_wake",
            "consolidate": "force_rem",
        }.get(action)
        if ctype is None:
            return {"status": "error", "reason": f"unknown action {action!r}"}
        import socket as _socket
        from datetime import datetime, timezone

        from iai_mcp._ipc import IS_WINDOWS, make_sync_ipc_socket, send_sync_auth_token
        sock_path = self._socket_path()
        # is_socket() is meaningless on Windows (TCP loopback, not a filesystem
        # object). Skip that check and let the port-file lookup below fail
        # instead if the daemon isn't running.
        if not IS_WINDOWS and not sock_path.is_socket():
            return {"status": "daemon_down", "reason": "daemon socket not present"}
        try:
            s, _addr = make_sync_ipc_socket()
            # All three are fire-and-forget control messages that reply
            # immediately (the pipeline runs later); a short timeout suffices.
            s.settimeout(5.0)
            s.connect(_addr)
            send_sync_auth_token(s)
            req = {"type": ctype, "ts": datetime.now(timezone.utc).isoformat()}
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n") and len(buf) < 65536:
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
            s.close()
            resp = json.loads(buf or b"{}")
            # Require an explicit ok:true — an empty/EOF reply (daemon crashed
            # mid-request) must NOT read as success.
            if resp.get("ok") is True:
                return {"status": "ok", "action": action, "result": resp}
            return {"status": "error",
                    "reason": str(resp.get("reason") or resp.get("error")
                                  or "no response from the daemon")[:200]}
        except (OSError, ValueError) as exc:
            return {"status": "daemon_down", "reason": str(exc)[:200]}

    def _daemon_lifecycle(self, action: str) -> dict:
        """Start/stop/restart the daemon through the platform service manager
        (launchd on macOS, systemd user unit on Linux)."""
        import os
        import platform
        import subprocess

        sysname = platform.system()
        if sysname == "Darwin":
            target = f"gui/{os.getuid()}/com.iai-mcp.daemon"
            if action == "stop":
                cmd = ["launchctl", "kill", "TERM", target]
            elif action == "restart":
                cmd = ["launchctl", "kickstart", "-k", target]
            else:  # start: never force-kill a running/mid-boot daemon
                cmd = ["launchctl", "kickstart", target]
        elif sysname == "Linux":
            cmd = ["systemctl", "--user", action, "iai-mcp-daemon.service"]
        else:
            return {"status": "error", "reason": f"unsupported platform {sysname}"}
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "error", "reason": str(exc)[:200]}
        if proc.returncode != 0:
            reason = (proc.stderr or proc.stdout or "").strip()
            if action == "stop" and "no such process" in reason.lower():
                return {"status": "ok", "action": action, "note": "already stopped"}
            return {
                "status": "error",
                "reason": reason[:200] or f"exit {proc.returncode}",
            }
        return {"status": "ok", "action": action}

    # -- writes -----------------------------------------------------------

    def rescue_direct(self, record_id: str) -> dict:
        """Mirror of forget_hint: un-tombstone a not-yet-erased ghost, stamp a
        fresh review, restore centrality above the gate threshold, and record
        provenance."""
        try:
            rid = UUID(record_id)
        except (TypeError, ValueError):
            return {"status": "error", "reason": "invalid id"}
        rec = self.store.get(rid)
        if rec is None:
            return {"status": "error", "reason": "unknown record (already erased?)"}
        with self._lock:
            from iai_mcp.store import RECORDS_TABLE

            self.store.db.open_table(RECORDS_TABLE).update(
                where=f"id = '{rid}'",
                values={
                    "tombstoned_at": None,
                    "live": 1,
                    "centrality": 1.0,
                    "last_reviewed": datetime.now(timezone.utc),
                },
            )
            try:
                self.store.append_provenance(
                    rid,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "cue": "user-rescue",
                        "session_id": "brainview",
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- provenance is traceability
                logger.debug("rescue provenance append failed: %s", exc)
        try:
            from iai_mcp.events import write_event

            write_event(
                self.store, "forget_rescue", {"record_id": str(rid)},
                severity="info", buffered=False,
            )
        except Exception:  # noqa: BLE001
            pass
        return {"status": "rescued", "record_id": str(rid)}

    def surface_direct(self, record_id: str) -> dict:
        with self.store.db.ro_conn() as conn:
            return self._ro_surface(conn, record_id)

    def _ro_surface(self, conn, record_id: str) -> dict:
        row = conn.execute(
            "SELECT id, literal_surface FROM records WHERE id = ?",
            (str(record_id),),
        ).fetchone()
        if row is None:
            return {"status": "error", "reason": "unknown record"}
        text = self._decrypt_record_col(str(row["id"]), row["literal_surface"])
        return {"id": str(row["id"]), "surface": str(text or "")}

    def pin_direct(self, record_id: str, on: bool = True) -> dict:
        """Pin/unpin a record. The erasure gate and lossy merges must refuse a
        pinned record."""
        try:
            rid = UUID(str(record_id))
        except (TypeError, ValueError):
            return {"status": "error", "reason": "invalid id"}
        rec = self.store.get(rid)
        if rec is None:
            return {"status": "error", "reason": "unknown record"}
        with self._lock:
            from iai_mcp.store import RECORDS_TABLE

            self.store.db.open_table(RECORDS_TABLE).update(
                where=f"id = '{rid}'",
                values={"pinned": bool(on), "never_merge": bool(on)},
            )
            try:
                self.store.append_provenance(
                    rid,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "cue": "user-pin" if on else "user-unpin",
                        "session_id": "brainview",
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- provenance is traceability
                logger.debug("pin provenance append failed: %s", exc)
        return {"status": "pinned" if on else "unpinned", "record_id": str(rid)}

    def capture_direct(self, text: str) -> dict:
        import hashlib

        from iai_mcp.capture import capture_turn

        text = (text or "").strip()
        # Same content ceiling as teach's decoded-blob cap: one dashboard
        # capture must not write a ~40MB record just because the transport
        # body cap is looser.
        if len(text.encode("utf-8", errors="ignore")) > 25 * 1024 * 1024:
            return {"status": "error", "reason": "capture too large (max 25 MB)"}
        with self._lock:
            # Content-bound identity (like a file upload, unlike a spoken
            # turn): re-capturing the same note reinforces instead of storing
            # a second copy.
            result = capture_turn(
                self.store,
                cue="",
                text=text,
                tier="episodic",
                session_id="brainview",
                role="user",
                source_uuid=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                provenance_extra={"source": "brainview"},
            )
        from iai_mcp.store import flush_record_buffer

        flush_record_buffer(self.store)
        return result

    def teach_direct(
        self,
        filename: str,
        content_b64: str,
        session_id: str = "study",
        reconcile: bool = False,
        fade: bool = False,
    ) -> dict:
        """Study an uploaded file via study_document. Binary formats (.pdf)
        round-trip through a temp file, text formats decode directly.
        fade=True studies an empty document under the same source name to
        supersede a deleted file (used by `iai watch`)."""
        import base64
        import tempfile

        from iai_mcp.study import study_document

        session_id = str(session_id or "study")[:64]
        # The temp file on disk uses the basename (no traversal), but the
        # source label keeps the caller's relative name — doc_tag identity
        # must match what a direct `iai teach`/`iai watch` of the same file
        # produces, or supersede-on-restudy misses across backends.
        label = str(filename or "upload.txt")[:200]
        name = Path(label).name

        if fade:
            with self._lock:
                try:
                    report = study_document(
                        self.store, text="", source_name=label,
                        session_id=session_id,
                    )
                except (ValueError, RuntimeError, NotImplementedError) as exc:
                    return {"status": "error", "reason": str(exc)[:200]}
            report["status"] = "studied"
            return report

        critic = None
        if reconcile:
            try:
                from iai_mcp.reconsolidation_critic import (
                    evaluate_batch_reconsolidation,
                )
                critic = evaluate_batch_reconsolidation
            except Exception:  # noqa: BLE001 -- critic optional, degrade to no resolution
                critic = None

        suffix = Path(name).suffix.lower() or ".txt"
        try:
            from iai_mcp.ingest._parsers import PLAINTEXT_SUFFIXES

            allowed = set(PLAINTEXT_SUFFIXES) | {".csv", ".pdf"}
        except Exception:  # noqa: BLE001 -- fall back to the core text formats
            allowed = {".txt", ".md", ".csv", ".pdf"}
        if suffix not in allowed:
            return {"status": "error", "reason": f"unsupported file type {suffix}"}
        try:
            import re as _re

            cleaned = _re.sub(r"\s+", "", content_b64 or "")
            blob = base64.b64decode(cleaned, validate=True)
        except (ValueError, TypeError):
            return {"status": "error", "reason": "malformed upload payload"}
        if not blob:
            return {"status": "error", "reason": "empty file"}
        if len(blob) > 25 * 1024 * 1024:
            return {"status": "error", "reason": "file too large (max 25 MB)"}

        with self._lock:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / name
                tmp_path.write_bytes(blob)
                try:
                    report = study_document(
                        self.store, path=tmp_path, source_name=label,
                        session_id=session_id, critic=critic,
                    )
                except (ValueError, RuntimeError, NotImplementedError) as exc:
                    return {"status": "error", "reason": str(exc)[:200]}
        report["status"] = "studied"
        return report

    def forget_hint_direct(self, record_id: str) -> dict:
        """Nudge a record into the existing decay/erasure funnel — no delete.

        The sleep pipeline's erasure gate selects on centrality, never_decay,
        pinned and age; the hint zeroes the first two levers and records who
        asked. A pinned record refuses the hint (unpin is a deliberate,
        separate act)."""
        try:
            rid = UUID(record_id)
        except (TypeError, ValueError):
            return {"status": "error", "reason": "invalid id"}
        rec = self.store.get(rid)
        if rec is None:
            return {"status": "error", "reason": "unknown record"}
        if rec.pinned:
            return {
                "status": "refused",
                "reason": "record is pinned; unpin explicitly before hinting",
            }
        with self._lock:
            from iai_mcp.store import RECORDS_TABLE

            self.store.db.open_table(RECORDS_TABLE).update(
                where=f"id = '{rid}'",
                values={"centrality": 0.0, "never_decay": False},
            )
            try:
                self.store.append_provenance(
                    rid,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "cue": "user-forget-hint",
                        "session_id": "brainview",
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- provenance is traceability, not the lever
                logger.debug("forget-hint provenance append failed: %s", exc)
        try:
            from iai_mcp.events import write_event

            write_event(
                self.store,
                "forget_hint",
                {"record_id": str(rid)},
                severity="info",
                buffered=False,
            )
        except Exception:  # noqa: BLE001 -- telemetry must not fail the hint
            pass
        return {"status": "queued_for_forgetting", "record_id": str(rid)}


class _Handler(BaseHTTPRequestHandler):
    view: BrainView
    page: bytes

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("brainview http: " + fmt, *args)

    #: Loopback-only is not browser-only: any web page the user visits can
    #: fire cross-origin requests at 127.0.0.1 (CSRF) and DNS-rebinding can
    #: point a hostile hostname here. A local viewer must therefore verify
    #: BOTH the Host it was addressed by and, for state-changing requests,
    #: that no foreign web Origin sent them.
    _ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")

    def _request_allowed(self, mutating: bool) -> bool:
        raw_host = (self.headers.get("Host") or "").strip().lower()
        if raw_host.startswith("["):
            # Bracketed IPv6 (e.g. "[::1]:4477"): the port follows the
            # bracket — a bare split(":") truncates the host to "[" and the
            # v6-loopback allow-list entry never matches.
            host = raw_host[: raw_host.index("]") + 1] if "]" in raw_host else raw_host
        else:
            host = raw_host.split(":", 1)[0]
        if host not in self._ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                from urllib.parse import urlsplit

                ohost = (urlsplit(origin).hostname or "").lower()
            except ValueError:
                return False
            if ohost not in self._ALLOWED_HOSTS:
                return False
        elif mutating:
            # Non-browser local clients (curl, the app shell) send no Origin;
            # browsers always do on cross-origin POSTs. Belt-and-braces for
            # the "always" assumption: a request that self-identifies its
            # fetch site as foreign is rejected even without an Origin.
            sec_site = (self.headers.get("Sec-Fetch-Site") or "").lower()
            if sec_site == "cross-site":
                return False
        return True

    def _method_not_allowed(self) -> None:
        self._json({"status": "error", "reason": "method not allowed"}, 405)

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _method_not_allowed

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if ctype.startswith("text/html"):
            # The page renders decrypted memory + ingested documents; the page
            # is fully self-contained (inline script/style, no external
            # loads), so everything beyond same-origin + inline is denied —
            # an escaped-sink slip cannot exfiltrate or load foreign code.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; form-action 'self'; "
                "frame-ancestors 'none'; base-uri 'none'",
            )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(
            code,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self._request_allowed(mutating=False):
                self._json({"status": "error", "reason": "forbidden origin"}, 403)
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, self.page, "text/html; charset=utf-8")
            elif path == "/api/overview":
                self._json(self.view.overview())
            elif path == "/api/graph":
                self._json(self.view.graph(limit=self._qint("limit", GRAPH_NODE_LIMIT)))
            elif path == "/api/events":
                self._json(self.view.events(limit=self._qint("limit", 30)))
            elif path == "/api/economy":
                self._json(self.view.economy())
            elif path == "/api/browse":
                self._json(self.view.browse())
            else:
                self._json({"status": "error", "reason": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 -- a handler fault must not kill the server
            logger.warning("brainview GET failed: %s", exc)
            self._json({"status": "error", "reason": str(exc)[:200]}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if not self._request_allowed(mutating=True):
                self._json({"status": "error", "reason": "forbidden origin"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = -1
            if length < 0 or length > _MAX_POST_BYTES:
                self._json({"status": "error", "reason": "invalid content length"}, 400)
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = {}
            if self.path == "/api/capture":
                text = str(payload.get("text") or "")
                self._json(self.view.capture(text))
            elif self.path == "/api/teach":
                self._json(
                    self.view.teach(
                        str(payload.get("filename") or ""),
                        str(payload.get("content_b64") or ""),
                    )
                )
            elif self.path == "/api/forget":
                self._json(self.view.forget_hint(str(payload.get("id") or "")))
            elif self.path == "/api/rescue":
                self._json(self.view.rescue(str(payload.get("id") or "")))
            elif self.path == "/api/pin":
                self._json(self.view.pin(str(payload.get("id") or ""),
                                         bool(payload.get("on", True))))
            elif self.path == "/api/surface":
                self._json(self.view.surface(str(payload.get("id") or "")))
            elif self.path == "/api/search":
                self._json(self.view.search(str(payload.get("q") or "")))
            elif self.path == "/api/browse":
                self._json(
                    self.view.browse(payload.get("folder"), _safe_int(payload.get("limit"), 60))
                )
            elif self.path == "/api/daemon":
                self._json(self.view.daemon_action(str(payload.get("action") or "")))
            else:
                self._json({"status": "error", "reason": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 -- a handler fault must not kill the server
            logger.warning("brainview POST failed: %s", exc)
            self._json({"status": "error", "reason": str(exc)[:200]}, 500)

    def _qint(self, key: str, default: int) -> int:
        if "?" not in self.path:
            return default
        from urllib.parse import parse_qs

        qs = parse_qs(self.path.split("?", 1)[1])
        try:
            return int(qs.get(key, [default])[0])
        except (TypeError, ValueError):
            return default


def make_server(
    store: Any = None, port: int = 0, *, view: "BrainView | None" = None
) -> ThreadingHTTPServer:
    """Bind the loopback server; port=0 picks an ephemeral port."""
    if view is None:
        view = BrainView(store)
    page = _load_page()
    handler = type("BoundHandler", (_Handler,), {"view": view, "page": page})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


#: Server-side views for the daemon's relay verb — one per live store.
import weakref as _weakref

_VIEWS: "_weakref.WeakKeyDictionary" = _weakref.WeakKeyDictionary()

BRAIN_VIEW_VERBS = frozenset({
    "overview", "graph", "search", "economy", "events", "browse", "surface",
    "capture", "teach", "forget_hint", "rescue", "pin",
})



def view_for_store(store: Any) -> BrainView:
    view = _VIEWS.get(store)
    if view is None:
        view = BrainView(store, pinned_direct=True)
        _VIEWS[store] = view
    return view


def open_app_window(url: str) -> None:
    """Open the page as a chromeless desktop window (Chrome app mode when
    available, default browser otherwise)."""
    import subprocess

    chrome = Path("/Applications/Google Chrome.app")
    try:
        if chrome.exists():
            subprocess.Popen(
                ["open", "-na", "Google Chrome", "--args", f"--app={url}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
    except OSError:
        pass
    import webbrowser

    webbrowser.open(url)


def serve(
    store_root: "str | Path",
    port: int = BRAINVIEW_DEFAULT_PORT,
    open_browser: bool = False,
    app_window: bool = False,
) -> None:
    # No store is opened up front: the backend is decided per request, so the
    # port binds instantly even mid-consolidation.
    view = BrainView(store_root=store_root)
    server = make_server(port=port, view=view)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"brain view: {url}  (Ctrl+C to stop)")
    if app_window:
        open_app_window(url)
    elif open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            if view.store is not None:
                view.store.close()
        except Exception:  # noqa: BLE001 -- shutdown must not fail the user
            pass
        try:
            ro = getattr(view, "_ro_cached", None)
            if ro is not None:
                ro.close()
        except Exception:  # noqa: BLE001 -- shutdown must not fail the user
            pass
