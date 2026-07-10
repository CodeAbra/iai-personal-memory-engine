"""Hippo store, SQLite, HNSW index, native embedder, CQRS-event, AVX2 and SDK-presence health checks.

Read-only probes of the storage layer. They open the store at most read-only and
never touch key material; a store held by the live daemon (or a SQLite lock from
it) is reported as a normal, passing condition.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

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

    env_store = os.environ.get("IAI_MCP_STORE")
    if env_store:
        deferred_dir = Path(env_store).parent / ".deferred-captures"
    else:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    if not deferred_dir.exists():
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="deferred-captures dir absent — nothing to recover",
        )

    count = 0
    try:
        for entry in os.scandir(deferred_dir):
            if entry.is_file() and fnmatch.fnmatch(entry.name, "*.permanent-failed-*.jsonl"):
                count += 1
    except OSError as exc:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail=f"could not scan deferred-captures dir: {exc}",
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
            conn = sqlite3.connect(str(db_path), timeout=2.0)
        rows = conn.execute(
            "SELECT created_at, COUNT(*) AS n FROM records"
            " WHERE tier = 'episodic' AND tombstoned_at IS NULL"
            " GROUP BY created_at HAVING COUNT(*) >= 5 ORDER BY n DESC LIMIT 20"
        ).fetchall()
    except (sqlite3.Error, _LilliParseError) as exc:  # type: ignore[misc]
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
    group_count = len(rows)
    total_affected = sum(r[1] for r in rows)
    worst_ts, worst_n = rows[0]
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
        import hnswlib as _hnswlib
        from iai_mcp.types import EMBED_DIM

        idx = _hnswlib.Index(space="cosine", dim=EMBED_DIM)
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
            conn = sqlite3.connect(str(db_path), timeout=2.0)
        row = conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = 'schema_version'"
        ).fetchone()
    except (sqlite3.Error, _LilliParseError) as exc:  # type: ignore[misc]
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


def check_t_hippo_compacted_freshness() -> CheckResult:
    import sqlite3
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    events: list[dict] = []
    _store = None
    try:
        from iai_mcp.events import query_events
        from iai_mcp.store import MemoryStore

        _store = MemoryStore()
        events = query_events(_store, kind="hippo_compacted", limit=1)
    except HippoLockHeldError as exc:
        logger.debug("check_t: store held by running daemon: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_t: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(t) hippo_compacted freshness",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
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
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="no hippo_compacted event found (fresh install or compaction not yet run)",
            status="WARN",
        )

    last_event = events[0]
    ts_str = last_event.get("timestamp") or last_event.get("ts") or ""
    try:
        ts = _dt.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        now = _dt.now(_tz.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="last hippo_compacted event timestamp unparseable",
            status="WARN",
        )

    if age_hours <= 24.0:
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"last hippo_compacted event {age_hours:.1f}h ago",
            status="PASS",
        )
    return CheckResult(
        name="(t) hippo_compacted freshness",
        passed=True,
        detail=(
            f"last hippo_compacted event {age_hours:.1f}h ago "
            f"(consider `iai-mcp maintenance compact-hippo --apply --yes`)"
        ),
        status="WARN",
    )


def check_u_recall_centrality_regression() -> CheckResult:
    import sqlite3
    import statistics
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    store = None
    try:
        from iai_mcp.events import query_events, write_event
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
        since = _dt.now(_tz.utc) - _td(hours=24)
        events = query_events(
            store, kind="recall_timing", since=since, limit=1000
        )

        if not events:
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail="no recall_timing events in last 24h (daemon idle or sampling missed)",
                status="WARN",
            )

        centrality_values: list[float] = []
        for ev in events:
            payload = ev.get("data") or {}
            cv = payload.get("centrality_ms")
            if cv is None:
                continue
            try:
                centrality_values.append(float(cv))
            except (TypeError, ValueError):
                continue
        if not centrality_values:
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail="recall_timing events present but centrality_ms missing/invalid",
                status="WARN",
            )

        median_ms = statistics.median(centrality_values)
        if median_ms > 30.0:
            try:
                write_event(
                    store,
                    kind="health_concern",
                    data={"centrality_median_ms": float(median_ms)},
                    severity="warning",
                )
            except Exception as exc:  # noqa: BLE001 — telemetry best-effort
                logger.debug("check_u: health_concern emit failed: %s", exc)
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail=(
                    f"centrality_ms median {median_ms:.1f}ms > 30ms threshold "
                    f"(n_events={len(centrality_values)})"
                ),
                status="WARN",
            )
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail=(
                f"centrality_ms median {median_ms:.1f}ms <= 30ms "
                f"(n_events={len(centrality_values)})"
            ),
            status="PASS",
        )
    except HippoLockHeldError as exc:
        logger.debug("check_u: store held by running daemon: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_u: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
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

    try:
        import iai_mcp_native  # noqa: F401
        from iai_mcp.embed import Embedder

        emb = Embedder()
        assert emb._backend == "rust", f"backend={emb._backend!r}"
        vec = emb.embed("smoke")
        assert len(vec) == 384, f"expected 384 dims, got {len(vec)}"
        assert all(math.isfinite(float(x)) for x in vec[:3]), (
            "non-finite values in output"
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="(v) native Rust embedder",
            passed=False,
            detail=(
                f"{type(exc).__name__}: {exc} — rebuild with: "
                "cd rust/iai_mcp_native && maturin develop --release"
            ),
        )
    return CheckResult(
        name="(v) native Rust embedder",
        passed=True,
        detail="encode ok, backend=rust, 384-dim",
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
