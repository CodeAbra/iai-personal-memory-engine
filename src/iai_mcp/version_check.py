"""Notify-only version awareness for wheel installs.

The check is never on a critical path: network fetches happen at most once
per TTL and only from slow surfaces (doctor, self-update, the daemon's
daily tick); session-start reads the cache file only. Offline is silent —
no version data is a non-event, never an error surfaced to the user.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_NAME = "iai-pme"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
CHECK_TTL_SEC: float = 24 * 3600.0
FETCH_TIMEOUT_SEC: float = 3.0
CACHE_FILENAME = ".version-check.json"

#: Kill switch (set to "0"): hermetic tests and air-gapped installs must be
#: able to forbid the network probe entirely.
ENV_TOGGLE = "IAI_MCP_VERSION_CHECK"


def _cache_path() -> Path:
    env = os.environ.get("IAI_MCP_STORE")
    root = Path(env) if env else (Path.home() / ".iai-mcp")
    return root / CACHE_FILENAME


def check_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE, "1") != "0"


#: Pre-rename installs still report under the old distribution name.
_DIST_NAMES = (PACKAGE_NAME, "iai-mcp")


def installed_version() -> str | None:
    try:
        from importlib.metadata import version
    except Exception:  # noqa: BLE001 — stripped environment
        return None
    for name in _DIST_NAMES:
        try:
            return version(name)
        except Exception:  # noqa: BLE001 — try the next name
            continue
    return None


def is_editable_install() -> bool:
    # A legacy egg-info editable install carries no direct_url.json, so the
    # metadata probe alone misses exactly the dev checkouts this guard
    # exists for — a module living inside a repo tree is the ground truth.
    if (Path(__file__).resolve().parents[2] / "pyproject.toml").exists():
        return True
    try:
        from importlib.metadata import distribution
    except Exception:  # noqa: BLE001
        return False
    for name in _DIST_NAMES:
        try:
            raw = distribution(name).read_text("direct_url.json")
        except Exception:  # noqa: BLE001
            continue
        if raw and json.loads(raw).get("dir_info", {}).get("editable"):
            return True
    return False


def _parse_version(v: str) -> tuple[int, ...] | None:
    parts: list[int] = []
    for token in str(v).strip().split("."):
        if not token.isdigit():
            break
        parts.append(int(token))
    return tuple(parts) if parts else None


def _is_clean_release(v: str) -> bool:
    parts = _parse_version(v)
    return parts is not None and ".".join(str(p) for p in parts) == str(v).strip()


def is_newer(latest: str, current: str) -> bool:
    # `latest` must be a clean numeric release: the upgrade pins `==latest`,
    # and an explicit pin installs prereleases — never steer anyone onto an
    # rc. `current` only needs a numeric prefix (dev suffixes are fine; dev
    # checkouts are stopped earlier by the editable guard anyway).
    if not _is_clean_release(latest):
        return False
    lt, ct = _parse_version(latest), _parse_version(current)
    if lt is None or ct is None:
        return False
    return lt > ct


def fetch_latest_version(timeout: float = FETCH_TIMEOUT_SEC) -> str | None:
    if not check_enabled():
        return None
    try:
        from urllib.request import urlopen

        with urlopen(PYPI_JSON_URL, timeout=timeout) as resp:  # noqa: S310 — fixed https URL
            payload = json.loads(resp.read().decode("utf-8"))
        latest = payload.get("info", {}).get("version")
        return str(latest) if latest else None
    except Exception:  # noqa: BLE001 — offline/blocked is silent by design
        return None


def read_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent/corrupt cache reads as none
        return None


def _write_cache(data: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # pid-suffixed tmp + os.replace: concurrent writers (daemon tick,
        # doctor, self-update) must not interleave into one tmp file, and
        # rename() would refuse an existing destination on Windows.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.debug("version cache write failed: %s", e)


def refresh_cache(force: bool = False) -> dict | None:
    """Fetch-if-stale. Keeps the previous `latest` when offline, but stamps
    the attempt so offline machines retry daily, not per call."""
    cache = read_cache()
    now = time.time()
    if (
        not force
        and cache is not None
        and (now - float(cache.get("checked_at", 0))) < CHECK_TTL_SEC
    ):
        return cache
    latest = fetch_latest_version()
    updated = {
        "checked_at": now,
        "latest": latest or (cache or {}).get("latest"),
        "current": installed_version(),
    }
    _write_cache(updated)
    return updated


def pending_update() -> tuple[str, str] | None:
    """(current, latest) when the CACHE says a newer version exists.
    Cache-only — safe on any critical path. The kill switch mutes this
    too: "disabled entirely" must mean no notices, not just no network."""
    if not check_enabled():
        return None
    # Source checkouts update via git — a repo whose version string lags
    # PyPI must not nag its own developer every session.
    if is_editable_install():
        return None
    cache = read_cache()
    if not cache:
        return None
    latest = cache.get("latest")
    current = installed_version()
    if not latest or not current:
        return None
    if is_newer(str(latest), current):
        return current, str(latest)
    return None


def pending_update_line() -> str | None:
    pair = pending_update()
    if pair is None:
        return None
    current, latest = pair
    return (
        f"iai-pme {latest} is available (installed {current}) — "
        f"run `iai-mcp self-update` to upgrade AND restart the engine"
    )
