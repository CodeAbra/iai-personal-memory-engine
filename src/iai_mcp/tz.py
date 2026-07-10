from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

CONFIG_FILENAME = "config.json"


def _config_path() -> Path:
    env = os.environ.get("IAI_MCP_STORE")
    root = Path(env) if env else Path.home() / ".iai-mcp"
    return root / CONFIG_FILENAME


def detect_tz() -> str:
    try:
        tz = datetime.now().astimezone().tzinfo
        if tz is None:
            return "UTC"
        key = getattr(tz, "key", None)
        if key:
            return str(key)
        return "UTC"
    except (OSError, TypeError, ValueError, AttributeError):
        return "UTC"


def _seed_config(cfg_path: Path, tz_key: str) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.setdefault("user", {})
    if not isinstance(existing["user"], dict):
        existing["user"] = {}
    existing["user"]["timezone"] = tz_key
    tmp = cfg_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, cfg_path)


def load_user_tz() -> ZoneInfo:
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = None
        if cfg is not None and isinstance(cfg, dict):
            user = cfg.get("user")
            if isinstance(user, dict):
                tz_key = user.get("timezone")
                if isinstance(tz_key, str) and tz_key.strip():
                    return ZoneInfo(tz_key)

    detected = detect_tz()
    try:
        zi = ZoneInfo(detected)
    except (KeyError, ValueError, OSError):
        detected = "UTC"
        zi = ZoneInfo("UTC")
    _seed_config(cfg_path, detected)
    return zi


def to_local(
    utc_dt: datetime,
    tz: ZoneInfo | None = None,
) -> datetime:
    if tz is None:
        tz = load_user_tz()
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(tz)
