"""Hippo store, SQLite, HNSW index, native embedder, CQRS-event, AVX2 and SDK-presence health checks.

Read-only probes of the storage layer. They open the store at most read-only and
never touch key material. Some checks read through the daemon's socket while it
holds the store, falling back to a direct SHARED read-only open (which coexists
with the daemon's own post-boot SHARED hold) when the daemon is down.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from iai_mcp import _sqlite_stdlib
from iai_mcp import errors

from iai_mcp.doctor import CheckResult
from iai_mcp.hippo import open_store_conn

try:
    from iai_mcp.lillibrain.sql.parser import ParseError as _LilliParseError
except Exception:
    _LilliParseError = ()  # type: ignore[assignment]  # empty tuple never matches

logger = logging.getLogger(__name__)


def _resolve_hippo_db_path(*args, **kwargs):
    # re-fetch the package attribute per call so monkeypatches stay visible
    from iai_mcp import doctor as _pkg

    return _pkg._resolve_hippo_db_path(*args, **kwargs)


def _hippo_expected_schema_version():
    # re-fetch the package attribute per call so a future monkeypatch stays visible
    from iai_mcp import doctor as _pkg

    return _pkg._HIPPO_EXPECTED_SCHEMA_VERSION


def check_h_crypto_file_state() -> CheckResult:
    from iai_mcp.crypto import CryptoKey, CryptoKeyError, SERVICE_NAME_DEFAULT

    ck = CryptoKey(user_id="default")
    path = ck._key_file_path()

    if path.exists():
        try:
            ck._try_file_get()
            return CheckResult(
                "(h) crypto key file state",
                True,
                f"crypto key file present at {path} (mode 0o600, valid)",
                status="PASS",
            )
        except CryptoKeyError as exc:
            return CheckResult(
                "(h) crypto key file state",
                False,
                f"crypto key file is malformed: {exc}",
                status="FAIL",
            )

    keyring_has_key = False
    keyring_probe_failed = False
    try:
        import keyring as _keyring
        import keyring.errors as _keyring_errors
    except ImportError:
        _keyring = None
        _keyring_errors = None  # type: ignore[assignment]

    if _keyring is not None:
        try:
            existing = _keyring.get_password(SERVICE_NAME_DEFAULT, "default")
            keyring_has_key = existing is not None
        except _keyring_errors.NoKeyringError:
            pass
        except _keyring_errors.KeyringError:
            keyring_probe_failed = True
        except Exception as e:  # noqa: BLE001 — defensive against keyring backend quirks
            logger.debug("check_h: keyring probe failed: %s", e)
            keyring_probe_failed = True

    if keyring_has_key:
        return CheckResult(
            "(h) crypto key file state",
            True,
            (
                f"crypto key file missing at {path}, but a Keychain entry was found.\n"
                f"  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key."
            ),
            status="WARN",
        )
    if keyring_probe_failed:
        return CheckResult(
            "(h) crypto key file state",
            True,
            (
                f"crypto key file missing at {path}; Keychain probe could not complete "
                f"(may indicate non-interactive context). If you have an existing Keychain key, "
                f"run `iai-mcp crypto migrate-to-file` from a Terminal."
            ),
            status="WARN",
        )

    return CheckResult(
        "(h) crypto key file state",
        True,
        (
            f"crypto key file absent at {path} and no Keychain entry detected. "
            f"Fresh install — run `iai-mcp crypto init` or set IAI_MCP_CRYPTO_PASSPHRASE."
        ),
        status="PASS",
    )


def check_i_hippo_db_size() -> CheckResult:
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail="brain.sqlite3 not present yet (fresh install or no writes yet)",
            status="PASS",
        )
    try:
        size_bytes = db_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 500:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=f"{size_mb:.1f} MB — healthy",
            status="PASS",
        )
    if size_mb < 2048:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=(
                f"{size_mb:.1f} MB — consider "
                f"`iai-mcp maintenance compact-hippo --apply --yes`"
            ),
            status="WARN",
        )
    return CheckResult(
        name="(i) hippo db size",
        passed=False,
        detail=f"{size_mb:.1f} MB — run compaction immediately",
        status="FAIL",
    )


def check_w_no_permanent_failed() -> CheckResult:
    import fnmatch

    # Single authority for the spool location — the spool is home-fixed and
    # IAI_MCP_STORE never names its parent directory.
    from iai_mcp.capture import deferred_captures_dir

    deferred_dir = deferred_captures_dir()

    if not deferred_dir.exists():
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="deferred-captures dir absent — nothing to recover",
        )

    count = 0
    total_bytes = 0
    try:
        for entry in os.scandir(deferred_dir):
            if not entry.is_file():
                continue
            try:
                total_bytes += entry.stat().st_size
            except OSError:
                pass
            if fnmatch.fnmatch(entry.name, "*.permanent-failed-*.jsonl"):
                count += 1
    except OSError as exc:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail=f"could not scan deferred-captures dir: {exc}",
            status="WARN",
        )

    # Soft cap only — the spool is never evicted (lossless wins over bounded);
    # a swollen backlog means the drain is not keeping up and the operator
    # must be told loudly.
    try:
        cap_mb = int(os.environ.get("IAI_MCP_SPOOL_SOFT_CAP_MB", "100"))
    except ValueError:
        cap_mb = 100
    size_mb = total_bytes / (1024 * 1024)
    if cap_mb > 0 and size_mb > cap_mb:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail=(
                f"deferred spool at {size_mb:.0f} MB (soft cap {cap_mb} MB) — "
                "the drain is not keeping up; check the daemon, then "
                "'iai-mcp deferred-drain'"
                + (f"; {count} permanent-failed file(s)" if count else "")
            ),
            status="WARN",
        )

    if count == 0:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="No permanent-failed capture files",
        )
    return CheckResult(
        name="(w) no permanent-failed captures",
        passed=True,
        detail=(
            f"{count} permanent-failed capture file(s) — "
            "run 'iai-mcp drain-permanent-failed' to recover"
        ),
        status="WARN",
    )


def _deferred_at_rest_age_sec() -> float:
    try:
        return max(0.0, float(os.environ.get("IAI_MCP_DEFERRED_AT_REST_AGE_SEC", "300")))
    except ValueError:
        return 300.0


def _live_spool_pending(deferred_dir: Path) -> "tuple[int, list[str]]":
    """Read-only pending-beyond-persisted-offset count for stale live spools.

    Only ``.live.jsonl`` files older than ``_deferred_at_rest_age_sec()`` are
    considered -- a fresher file is normal ~30s wake-sweep cadence lag on an
    actively-capturing session, not a stranded spool.
    """
    from iai_mcp.capture import _LIVE_ACTIVE_RE, capture_state_dir

    total_pending = 0
    pending_names: list[str] = []
    age_threshold = _deferred_at_rest_age_sec()
    now = time.time()
    state_dir = capture_state_dir()

    try:
        entries = list(os.scandir(deferred_dir))
    except OSError:
        return 0, []

    for entry in entries:
        if not entry.is_file() or not _LIVE_ACTIVE_RE.search(entry.name):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime < age_threshold:
            continue

        session_id = _LIVE_ACTIVE_RE.sub("", entry.name)
        try:
            with open(entry.path, "r", encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except OSError:
            continue
        # Completeness rule must match the drain writer (capture.py): a torn tail is not drainable.
        complete_lines = [ln for ln in raw_lines if ln.endswith("\n")]
        event_lines = max(0, len(complete_lines) - 1)

        offset_path = state_dir / f"{session_id}.drain-offset"
        offset = 0
        try:
            if offset_path.exists():
                offset = int(offset_path.read_text(encoding="utf-8").strip() or "0")
        except (ValueError, OSError):
            offset = 0

        pending = max(0, event_lines - offset)
        if pending > 0:
            total_pending += pending
            pending_names.append(entry.name)

    return total_pending, pending_names


def check_ee_deferred_capture_backlog_at_rest() -> CheckResult:
    """FAIL when the deferred-capture spool holds undrained events at rest.

    Covers both the rotated/crashed/quarantine backlog (`enumerate_backlog`)
    AND stale `.live.jsonl` files past their persisted drain offset -- the
    latter is structurally invisible to `check_w_no_permanent_failed`.
    """
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.deferred_drain import _count_events, enumerate_backlog

    name = "(ee) deferred-capture backlog at rest"
    deferred_dir = deferred_captures_dir()

    if not deferred_dir.exists():
        return CheckResult(
            name=name,
            passed=True,
            detail="deferred-captures dir absent — nothing at rest",
            status="PASS",
        )

    try:
        rotated_count = sum(_count_events(f) for f in enumerate_backlog(deferred_dir))
    except OSError as exc:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"could not scan deferred-capture backlog: {exc}",
            status="WARN",
        )
    live_count, live_files = _live_spool_pending(deferred_dir)
    total = rotated_count + live_count

    if total == 0:
        return CheckResult(
            name=name,
            passed=True,
            detail="no undrained capture events at rest",
            status="PASS",
        )

    return CheckResult(
        name=name,
        passed=False,
        detail=(
            f"{total} undrained capture event(s) at rest "
            f"(rotated backlog={rotated_count}, stale live spool={live_count}"
            + (f" [{', '.join(live_files)}]" if live_files else "")
            + ") — run 'iai-mcp deferred-drain' or restart the daemon"
        ),
        status="FAIL",
    )


def check_aa_capture_state_hygiene() -> CheckResult:
    from iai_mcp.capture import sweep_capture_state

    name = "(aa) capture-state hygiene"
    try:
        counts = sweep_capture_state(apply=False)
    except OSError as exc:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"could not scan capture-state dir: {exc}",
            status="WARN",
        )
    stale_total = counts["stale"] + counts["tmp"]
    if stale_total == 0:
        return CheckResult(
            name=name,
            passed=True,
            detail=f"no stale files ({counts['kept']} live)",
        )
    return CheckResult(
        name=name,
        passed=False,
        detail=(
            f"{counts['stale']} month-old session state file(s) + "
            f"{counts['tmp']} crashed-writer tmp file(s) — the heal "
            "removes them; a swept offset costs one re-walk, deduped by "
            "source-uuid keys where the host provides them and absorbed "
            "by the cosine gate for text-keyed hosts"
        ),
    )


def check_x_no_collapsed_timestamps() -> CheckResult:
    """Warn when many episodic records share an identical created_at (time-collapsed session)."""
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail="db absent (fresh install)",
            status="PASS",
        )
    conn = None
    try:
        _eng = open_store_conn(db_path, read_only=True)
        if _eng is not None:
            conn = _eng
        else:
            conn = _sqlite_stdlib.connect(str(db_path), timeout=2.0)
        rows = conn.execute(
            "SELECT created_at, COUNT(*) AS n FROM records"
            " WHERE tier = 'episodic' AND tombstoned_at IS NULL"
            " GROUP BY created_at HAVING COUNT(*) >= 5 ORDER BY n DESC LIMIT 20"
        ).fetchall()
    except (errors.Error, _LilliParseError) as exc:  # type: ignore[misc]
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail=f"check skipped: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if not rows:
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail="no collapsed timestamp groups found",
            status="PASS",
        )
    # Groups the rederive repair already visited and left in place are
    # verified batch history (their transcripts really carry one second),
    # not damage — only groups formed AFTER the last repair warrant a WARN.
    last_repair = None
    try:
        _eng2 = open_store_conn(db_path, read_only=True)
        _conn2 = _eng2 if _eng2 is not None else _sqlite_stdlib.connect(
            str(db_path), timeout=2.0,
        )
        try:
            r2 = _conn2.execute(
                "SELECT ts FROM events"
                " WHERE kind = 'migration_rederive_timestamps'"
                " ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if r2 is not None:
                last_repair = str(r2[0]).replace(" ", "T")
        finally:
            try:
                _conn2.close()
            except Exception:
                pass
    except (errors.Error, _LilliParseError):
        last_repair = None

    def _norm(ts: object) -> str:
        return str(ts).replace(" ", "T")

    fresh = [
        r for r in rows
        if last_repair is None or _norm(r[0]) > last_repair
    ]
    if not fresh:
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail=(
                f"{len(rows)} same-second group(s) predate the last"
                " timestamp repair — verified batch history, not damage"
            ),
            status="PASS",
        )
    group_count = len(fresh)
    total_affected = sum(r[1] for r in fresh)
    worst_ts, worst_n = fresh[0]
    return CheckResult(
        name="(x) no collapsed-timestamp groups",
        passed=False,
        detail=(
            f"{group_count} group(s) with >= 5 records at one timestamp"
            f" ({total_affected} records total; worst group: {worst_n} records at {worst_ts})"
            " — run 'iai-mcp migrate --rederive-timestamps' to repair"
        ),
        status="WARN",
    )


def check_z_avx2_support() -> CheckResult:
    from iai_mcp.cpu_features import has_avx2

    try:
        avx2_ok = has_avx2()
    except Exception as exc:  # noqa: BLE001 -- defensive against probe quirks
        try:
            from iai_mcp.store import CPU_HAS_AVX2
            avx2_ok = CPU_HAS_AVX2
        except Exception:  # noqa: BLE001 -- store may itself be unimportable
            avx2_ok = True
        logger.debug(
            "check_z: has_avx2() probe failed: %s; fallback=%s",
            exc,
            avx2_ok,
        )

    if avx2_ok:
        return CheckResult(
            name="(z) AVX2 CPU support",
            passed=True,
            detail="AVX2 available (or N/A on this architecture)",
            status="PASS",
        )
    return CheckResult(
        name="(z) AVX2 CPU support",
        passed=False,
        detail=(
            "this host lacks AVX2 -- the native memory store cannot load; iai-mcp memory "
            "store is unavailable. Deploy on an AVX2-equipped host (any "
            "Intel CPU 2013+; AMD Excavator 2015+; Mac M-series ARM is "
            "unaffected)."
        ),
        status="FAIL",
    )


def check_r_hippo_hnsw_loadable() -> CheckResult:
    hnsw_path = _resolve_hippo_db_path().parent / "records.hnsw"
    if not hnsw_path.exists():
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=True,
            detail="records.hnsw absent (HippoDB rebuilds from SQLite on next boot)",
            status="WARN",
        )
    try:
        size = hnsw_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="FAIL",
        )
    if size == 0:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                "records.hnsw is zero bytes (corrupt; rebuild needed — "
                "restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    try:
        from iai_mcp.hippo import _vecindex as _vi
        from iai_mcp.types import EMBED_DIM

        idx = _vi.Index(space="cosine", dim=EMBED_DIM)
        idx.load_index(str(hnsw_path), max_elements=0)
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        logger.debug("check_r: hnswlib.load_index failed: %s", exc)
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                f"hnswlib.load_index failed: {type(exc).__name__}: {exc} "
                "(restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(r) hippo hnsw index",
        passed=True,
        detail=f"{size / (1024 * 1024):.1f} MB",
        status="PASS",
    )


def check_s_hippo_schema_version() -> CheckResult:
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail="db absent (fresh install)",
            status="PASS",
        )
    conn = None
    try:
        _eng = open_store_conn(db_path, read_only=True)
        if _eng is not None:
            conn = _eng
        else:
            conn = _sqlite_stdlib.connect(str(db_path), timeout=2.0)
        row = conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = 'schema_version'"
        ).fetchone()
    except (errors.Error, _LilliParseError) as exc:  # type: ignore[misc]
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail=f"check skipped: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if row is None:
        return CheckResult(
            name="(s) hippo schema version",
            passed=False,
            detail="_hippo_meta missing schema_version row",
            status="FAIL",
        )
    value = str(row[0])
    expected = _hippo_expected_schema_version()
    if value != expected:
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail=f"schema_version={value} (expected {expected})",
            status="WARN",
        )
    return CheckResult(
        name="(s) hippo schema version",
        passed=True,
        detail=f"schema_version={value}",
        status="PASS",
    )


_HIPPO_COMPACTED_CHECK_NAME = "(t) hippo_compacted freshness"
_CENTRALITY_CHECK_NAME = "(u) recall centrality regression"


def _hippo_compacted_verdict(ts_value: object, now: "object") -> CheckResult:
    """Shared PASS/WARN verdict for check_t, fed by either read path.

    ``ts_value`` is either an aware datetime (the direct path — ``query_events``
    always returns a datetime ``ts``) or an ISO-8601 string (the socket path —
    the dispatch isoformats it on the wire). Both paths must feed this single
    function so the two routes can never diverge in verdict.
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    if isinstance(ts_value, _dt):
        ts = ts_value if ts_value.tzinfo is not None else ts_value.replace(tzinfo=_tz.utc)
    else:
        try:
            ts = _dt.fromisoformat(str(ts_value))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
        except (TypeError, ValueError):
            return CheckResult(
                name=_HIPPO_COMPACTED_CHECK_NAME,
                passed=True,
                detail="last hippo_compacted event timestamp unparseable",
                status="WARN",
            )

    age_hours = (now - ts).total_seconds() / 3600.0
    if age_hours <= 24.0:
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail=f"last hippo_compacted event {age_hours:.1f}h ago",
            status="PASS",
        )
    return CheckResult(
        name=_HIPPO_COMPACTED_CHECK_NAME,
        passed=True,
        detail=(
            f"last hippo_compacted event {age_hours:.1f}h ago "
            f"(consider `iai-mcp maintenance compact-hippo --apply --yes`)"
        ),
        status="WARN",
    )


def _centrality_verdict(
    data_dicts: "list[dict]", *, emit_store: object = None,
) -> CheckResult:
    """Shared PASS/WARN verdict for check_u, fed by either read path.

    ``data_dicts`` are the ``recall_timing`` event ``data`` payloads. The
    ``health_concern`` best-effort emit only fires when ``emit_store`` is
    given (the direct path owns a store handle; the socket path is
    read-only and intentionally drops the emit — nothing consumes it).
    """
    import statistics

    if not data_dicts:
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail="no recall_timing events in last 24h (daemon idle or sampling missed)",
            status="WARN",
        )

    centrality_values: list[float] = []
    for payload in data_dicts:
        cv = (payload or {}).get("centrality_ms")
        if cv is None:
            continue
        try:
            centrality_values.append(float(cv))
        except (TypeError, ValueError):
            continue
    if not centrality_values:
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail="recall_timing events present but centrality_ms missing/invalid",
            status="WARN",
        )

    median_ms = statistics.median(centrality_values)
    if median_ms > 30.0:
        if emit_store is not None:
            try:
                from iai_mcp.events import write_event

                write_event(
                    emit_store,
                    kind="health_concern",
                    data={"centrality_median_ms": float(median_ms)},
                    severity="warning",
                )
            except Exception as exc:  # noqa: BLE001 — telemetry best-effort
                logger.debug("check_u: health_concern emit failed: %s", exc)
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail=(
                f"centrality_ms median {median_ms:.1f}ms > 30ms threshold "
                f"(n_events={len(centrality_values)})"
            ),
            status="WARN",
        )
    return CheckResult(
        name=_CENTRALITY_CHECK_NAME,
        passed=True,
        detail=(
            f"centrality_ms median {median_ms:.1f}ms <= 30ms "
            f"(n_events={len(centrality_values)})"
        ),
        status="PASS",
    )


def _check_t_via_socket(now: "object") -> CheckResult:
    """Read the latest ``hippo_compacted`` event through the daemon socket.

    Raises ``_SocketUnavailable`` when the daemon cannot be reached at all,
    so the caller falls back to a direct store open; any other unusable
    reply is a genuine WARN (the daemon answered, but the answer was not
    usable) — never a blind PASS.
    """
    from iai_mcp.cli import _send_jsonrpc_request
    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable

    resp = _send_jsonrpc_request(
        "events_query",
        {"kind": "hippo_compacted", "limit": 1},
        connect_timeout=1.0,
        read_timeout=5.0,
    )
    if resp is None:
        raise _SocketUnavailable()
    if not isinstance(resp, dict) or "result" not in resp:
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail="unable to read hippo_compacted freshness via daemon socket",
            status="WARN",
        )
    result = resp["result"]
    events = result.get("events") if isinstance(result, dict) else None
    if events is None:
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail="unable to read hippo_compacted freshness via daemon socket",
            status="WARN",
        )
    if not events:
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail="no hippo_compacted event found (fresh install or compaction not yet run)",
            status="WARN",
        )
    return _hippo_compacted_verdict(events[0].get("ts"), now)


def _check_u_via_socket(now: "object") -> CheckResult:
    """Read the last 24h of ``recall_timing`` events through the daemon socket.

    Same fall-back contract as ``_check_t_via_socket``. The socket path is
    read-only, so the ``health_concern`` best-effort emit is intentionally
    dropped here (preserved on the direct branch only).
    """
    from datetime import timedelta as _td

    from iai_mcp.cli import _send_jsonrpc_request
    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable

    since = now - _td(hours=24)
    resp = _send_jsonrpc_request(
        "events_query",
        {"kind": "recall_timing", "since": since.isoformat(), "limit": 1000},
        connect_timeout=1.0,
        read_timeout=5.0,
    )
    if resp is None:
        raise _SocketUnavailable()
    if not isinstance(resp, dict) or "result" not in resp:
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail="unable to read recall centrality via daemon socket",
            status="WARN",
        )
    result = resp["result"]
    events = result.get("events") if isinstance(result, dict) else None
    if events is None:
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail="unable to read recall centrality via daemon socket",
            status="WARN",
        )
    return _centrality_verdict([e.get("data") or {} for e in events], emit_store=None)


def check_t_hippo_compacted_freshness() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable
    from iai_mcp.hippo import HippoLockHeldError

    if not _store_file_present():
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail="no store yet (skip)",
        )

    now = _dt.now(_tz.utc)
    try:
        return _check_t_via_socket(now)
    except _SocketUnavailable:
        pass
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never crash the run
        logger.debug("check_t: socket read failed: %s", exc)
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    _store = None
    try:
        from iai_mcp.events import query_events
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        # SHARED + read-only coexists with the daemon's steady-state SHARED
        # lock; reaching a lock error here means something unexpected holds
        # the store exclusively, not a "daemon holds it, normal" condition.
        _store = MemoryStore(access_mode=AccessMode.SHARED, read_only=True)
        events = query_events(_store, kind="hippo_compacted", limit=1)
    except HippoLockHeldError as exc:
        logger.debug("check_t: store lock unavailable: %s", exc)
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail=f"unable to read hippo_compacted freshness: {type(exc).__name__}",
            status="WARN",
        )
    except errors.OperationalError as exc:
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if _store is not None and hasattr(_store, "close"):
            try:
                _store.close()
            except Exception:  # noqa: BLE001
                pass

    if not events:
        return CheckResult(
            name=_HIPPO_COMPACTED_CHECK_NAME,
            passed=True,
            detail="no hippo_compacted event found (fresh install or compaction not yet run)",
            status="WARN",
        )
    return _hippo_compacted_verdict(events[0].get("ts"), now)


def check_u_recall_centrality_regression() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.doctor._lifecycle_checks import _SocketUnavailable
    from iai_mcp.hippo import HippoLockHeldError

    if not _store_file_present():
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail="no store yet (skip)",
        )

    now = _dt.now(_tz.utc)
    try:
        return _check_u_via_socket(now)
    except _SocketUnavailable:
        pass
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never crash the run
        logger.debug("check_u: socket read failed: %s", exc)
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    store = None
    try:
        from iai_mcp.events import query_events
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        # SHARED + read-only coexists with the daemon's steady-state SHARED
        # lock; reaching a lock error here means something unexpected holds
        # the store exclusively, not a "daemon holds it, normal" condition.
        store = MemoryStore(access_mode=AccessMode.SHARED, read_only=True)
        since = now - _td(hours=24)
        events = query_events(store, kind="recall_timing", since=since, limit=1000)
        return _centrality_verdict(
            [ev.get("data") or {} for ev in events], emit_store=store,
        )
    except HippoLockHeldError as exc:
        logger.debug("check_u: store lock unavailable: %s", exc)
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail=f"unable to read recall centrality: {type(exc).__name__}",
            status="WARN",
        )
    except errors.OperationalError as exc:
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name=_CENTRALITY_CHECK_NAME,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if store is not None and hasattr(store, "close"):
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass


def check_v_native_embedder() -> CheckResult:
    import math

    backend = "unknown"
    try:
        from iai_mcp.embed import Embedder, embed_query

        emb = Embedder()
        backend = emb._backend
        vec = embed_query(emb, "smoke")
        assert len(vec) == emb.DIM, f"expected {emb.DIM} dims, got {len(vec)}"
        assert all(math.isfinite(float(x)) for x in vec[:3]), (
            "non-finite values in output"
        )
    except Exception as exc:  # noqa: BLE001
        remedy = (
            "rebuild with: cd rust/iai_mcp_native && maturin develop --release"
            if backend in {"unknown", "rust"}
            else "check IAI_MCP_EMBED_URL and the loopback embedder service"
        )
        return CheckResult(
            name="(v) configured embedder",
            passed=False,
            detail=f"{type(exc).__name__}: {exc} — {remedy}",
        )
    return CheckResult(
        name="(v) configured embedder",
        passed=True,
        detail=f"encode ok, backend={backend}, {emb.DIM}-dim, model={emb.model_key}",
    )


def check_dd_exact_index_coercions() -> CheckResult:
    """WARN when a non-finite (NaN/inf) row or query cue has been coerced to
    zero at the exact-cosine authority index — the degradation is countable,
    not invisible. Reads persisted ``TELEMETRY_EMBED_NONFINITE`` events
    read-only; never the live in-process counter (a doctor-opened store gets
    its own cold index with zero counters), never by calling ``exact_top_k``
    or ``_build_exact_index_sync`` (a diagnostic must not trigger a
    whole-corpus build)."""
    name = "(dd) exact-index coercions"
    if not _store_file_present():
        return CheckResult(name=name, passed=True, detail="no store yet (skip)")

    store = None
    try:
        from iai_mcp.events import TELEMETRY_EMBED_NONFINITE, query_events
        from iai_mcp.hippo import AccessMode
        from iai_mcp.store import MemoryStore

        store = MemoryStore(access_mode=AccessMode.SHARED, read_only=True)
        # No `since` window: legacy bad rows surface at the first cold build
        # after ANY daemon start, so a 24h window on a long-up daemon would
        # falsely PASS a still-degraded ranking.
        events = query_events(
            store, kind=TELEMETRY_EMBED_NONFINITE, severity="warning", limit=500
        )
        coerced = [
            e for e in events if (e.get("data") or {}).get("action") == "coerced"
        ]
        if not coerced:
            return CheckResult(
                name=name,
                passed=True,
                detail="no non-finite coercions recorded",
                status="PASS",
            )

        # A single build/cue emit can cover multiple coercions in one event
        # (`count`), so event-count != coercion-count — the payload's own
        # running `total` is exact on both lanes.
        row_total = max(
            (
                int(e["data"].get("total", 0))
                for e in coerced
                if e["data"].get("source") == "row"
            ),
            default=0,
        )
        cue_total = max(
            (
                int(e["data"].get("total", 0))
                for e in coerced
                if e["data"].get("source") == "cue"
            ),
            default=0,
        )
        latest = max((e.get("ts") for e in coerced), default=None)
        return CheckResult(
            name=name,
            passed=True,
            detail=(
                f"non-finite coercions observed (cue={cue_total}, row={row_total}); "
                f"latest {latest} — a stored row or cue carried NaN/inf; "
                "ranking degraded, not crashed"
            ),
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must never crash the run
        return CheckResult(
            name=name,
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if store is not None and hasattr(store, "close"):
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass


def _store_file_present() -> bool:
    """True when the store's database file already exists.

    Every diagnostic (and the heal audit writer) must consult this before
    MemoryStore(): opening an absent path CREATES an empty brain.sqlite3
    whose driver the next daemon boot then misdetects — a doctor run on a
    fresh host would invalidate its own no-store row within the same run.
    Path resolution MIRRORS MemoryStore's (env, then the store module's
    default) — an independent spelling here goes blind on a live store and
    keeps minting at the virgin root. Fail-open on any resolution error:
    the subsequent open then reports its own failure as one row instead of
    a silent skip.
    """
    try:
        from iai_mcp import store as _store_mod

        store_env = os.environ.get("IAI_MCP_STORE")
        root = Path(store_env) if store_env else Path(_store_mod.DEFAULT_STORAGE_PATH)
        return (root / "hippo" / "brain.sqlite3").exists()
    except Exception:  # noqa: BLE001
        return True


def _read_store_meta_values(
    store_root: Path, keys: "tuple[str, ...]"
) -> "dict[str, str | None]":
    from iai_mcp.hippo import AccessMode, HippoDB

    # A diagnostic read must never CREATE a store: opening an absent path
    # would mint an empty stdlib-driver brain.sqlite3 that a later lilli
    # daemon boot then misdetects.
    if not (store_root / "hippo" / "brain.sqlite3").exists():
        raise FileNotFoundError(f"store absent: {store_root}")

    db = HippoDB(
        store_root,
        access_mode=AccessMode.SHARED,
        read_only=True,
        _lock_timeout_override=0.25,
    )
    try:
        out: "dict[str, str | None]" = {}
        with db._conn_lock:
            for key in keys:
                row = db._conn.execute(
                    "SELECT value FROM _hippo_meta WHERE key = ?", (key,)
                ).fetchone()
                out[key] = row["value"] if row is not None else None
        return out
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass


def check_ii_embed_identity() -> CheckResult:
    """Store vector-identity stamp vs the embedder the runtime would use.

    Daemon up: the daemon's status payload is authoritative — it holds the
    warm embedder that passed the boot guard. Daemon down: read the stamp
    directly and compare against the configured embedder for the store's
    dimension (allow_mismatch, so the comparison itself is reported rather
    than raised).
    """
    import asyncio as _asyncio

    name = "(ii) store embed identity"
    from iai_mcp import doctor as _pkg

    status = None
    try:
        status = _asyncio.run(
            _pkg._socket_status_probe(_pkg._resolve_socket_path(), 2.0)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("check_ii: socket probe failed: %s", exc)

    if isinstance(status, dict) and not isinstance(
        status.get("embed_identity"), dict
    ):
        # The daemon answered but carries no identity block: an older
        # daemon or a broken boot wire. The direct read below would only
        # hit the daemon's own store lock, so name the real remedy.
        return CheckResult(
            name, True,
            "daemon serves no identity block (pre-upgrade daemon or broken "
            "boot wire) — restart the daemon to restore this row",
            status="WARN",
        )

    if isinstance(status, dict) and isinstance(status.get("embed_identity"), dict):
        ei = status["embed_identity"]
        if ei.get("state") == "warming":
            return CheckResult(
                name, True,
                "embedder still warming after boot — retry shortly",
                status="WARN",
            )
        if ei.get("error"):
            return CheckResult(
                name, True, f"daemon could not compute: {ei['error']}",
                status="WARN",
            )
        stored = ei.get("stored")
        if stored is None:
            return CheckResult(
                name, True,
                "store unstamped — protection begins at the first stamp",
                status="WARN",
            )
        if ei.get("match"):
            return CheckResult(name, True, f"match: {stored}")
        return CheckResult(
            name, False,
            f"MISMATCH: store={stored!r} runtime={ei.get('runtime')!r} — "
            "run the re-embedding migration "
            "(iai-mcp migrate --reembed-to-configured-provider) or restore "
            "the original embedder configuration",
        )

    # Daemon down: direct read.
    try:
        from types import SimpleNamespace

        from iai_mcp.embed import (
            EMBED_IDENTITY_META_KEY,
            effective_model_identity,
            embedder_for_store,
        )

        from iai_mcp import store as _store_mod

        store_env = os.environ.get("IAI_MCP_STORE")
        store_root = (
            Path(store_env) if store_env
            else Path(_store_mod.DEFAULT_STORAGE_PATH)
        )
        meta = _read_store_meta_values(
            store_root, (EMBED_IDENTITY_META_KEY, "embed_dim")
        )
        stored = meta[EMBED_IDENTITY_META_KEY]
        if stored is None:
            return CheckResult(
                name, True,
                "store unstamped — protection begins at the first stamp",
                status="WARN",
            )
        dim_raw = meta["embed_dim"]
        embed_dim = int(dim_raw) if dim_raw else None
        # db=None skips the guard inside embedder construction; the
        # comparison below is the diagnosis this row exists to report.
        runtime = effective_model_identity(
            embedder_for_store(
                SimpleNamespace(db=None, embed_dim=embed_dim),
                allow_identity_mismatch=True,
            )
        )
        if stored == runtime:
            return CheckResult(name, True, f"match: {stored}")
        return CheckResult(
            name, False,
            f"MISMATCH: store={stored!r} runtime={runtime!r} — "
            "run the re-embedding migration "
            "(iai-mcp migrate --reembed-to-configured-provider) or restore "
            "the original embedder configuration",
        )
    except FileNotFoundError:
        return CheckResult(
            name, True,
            "no store yet — nothing to compare",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name, True,
            f"could not compare (store busy or embedder unavailable): "
            f"{type(exc).__name__}: {str(exc)[:120]}",
            status="WARN",
        )


def check_p_anthropic_sdk_absent() -> CheckResult:
    try:
        import anthropic  # noqa: F401 -- presence-probe only
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,
            detail=(
                "anthropic SDK is importable in this venv. It was dropped "
                "as a runtime dependency; this is likely leftover site-packages "
                "from an older install. Run `pip uninstall anthropic` "
                "to clean up."
            ),
            status="WARN",
        )
    except ImportError:
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,
            detail="ImportError as expected (subscription-only path)",
            status="PASS",
        )
