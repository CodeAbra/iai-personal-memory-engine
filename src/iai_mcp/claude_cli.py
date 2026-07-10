from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.daemon_state import load_state, save_state

_ANTHR = "ANTHR" + "OPIC_" + "API_" + "KEY"
_CLAUDE_KEY = "CLAUDE_" + "API_" + "KEY"
_CLAUDE_CODE_KEY = "CLAUDE_" + "CODE_" + "API_" + "KEY"
ENV_DENY_LIST: tuple[str, ...] = (_ANTHR, _CLAUDE_KEY, _CLAUDE_CODE_KEY)

CLAUDE_TIMEOUT_SEC: float = 120.0
FORCE_WAKE_GRACE_SEC: float = 60.0
TERMINATE_WAIT_SEC: float = 5.0
KILL_WAIT_SEC: float = 2.0
DAILY_QUOTA_BUDGET_PCT: float = 0.01
WEEKLY_BUFFER_PCT: float = 0.07
ESTIMATED_DAILY_TOKEN_CEILING: int = 1_000_000
CREDENTIALS_PATH: Path = Path.home() / ".claude" / ".credentials.json"
# On macOS, Claude Code stores OAuth credentials in the login Keychain rather
# than CREDENTIALS_PATH, so the file is absent on a normal desktop-app /
# `claude /login` setup. This generic-password item holds the same
# `{"claudeAiOauth": {...}}` payload and is used as a fallback.
KEYCHAIN_SERVICE: str = "Claude Code-credentials"
BUDGET_STATE_KEY: str = "claude_budget"


_VALID_SUBSCRIPTION_TYPES: frozenset[str] = frozenset({
    "pro", "pro_max", "max", "team", "enterprise",
})

_REQUIRED_SCOPE: str = "user:inference"


def _read_keychain_credentials() -> dict | None:
    """Best-effort read of Claude Code's macOS Keychain credentials item.

    Claude Code on macOS keeps OAuth credentials in the login Keychain instead
    of CREDENTIALS_PATH, so the file is absent on a normal desktop-app /
    ``claude /login`` setup. The generic-password item holds the same
    ``{"claudeAiOauth": {...}}`` payload. Returns None when unavailable
    (non-macOS, item absent, or the user denies the Keychain prompt).
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return json.loads(out.stdout.strip())
    except json.JSONDecodeError:
        return None


def verify_credentials_subscription() -> dict:
    if CREDENTIALS_PATH.exists():
        try:
            data = json.loads(CREDENTIALS_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "reason": "credentials_unreadable", "error": str(exc)}
    else:
        # macOS keeps credentials in the Keychain, not the JSON file.
        data = _read_keychain_credentials()
    if data is None:
        return {"ok": False, "reason": "credentials_file_missing"}

    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if isinstance(oauth, dict) and oauth.get("subscriptionType"):
        sub_type = str(oauth.get("subscriptionType") or "").strip()
        if sub_type not in _VALID_SUBSCRIPTION_TYPES:
            return {
                "ok": False,
                "reason": "not_subscription",
                "subscription_type": sub_type,
            }
        scopes = oauth.get("scopes") or []
        if not isinstance(scopes, list) or _REQUIRED_SCOPE not in scopes:
            return {
                "ok": False,
                "reason": "missing_inference_scope",
                "subscription_type": sub_type,
            }
        expires_at_ms = oauth.get("expiresAt")
        refresh_token = oauth.get("refreshToken") or ""
        if (
            isinstance(expires_at_ms, (int, float))
            and expires_at_ms > 0
            and not refresh_token
        ):
            now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000.0
            if expires_at_ms < now_ms:
                return {
                    "ok": False,
                    "reason": "credentials_expired",
                    "subscription_type": sub_type,
                    "expires_at_ms": int(expires_at_ms),
                }
        return {"ok": True, "subscription_type": sub_type}

    billing = data.get("billingType") or data.get("billing_type") or ""
    if billing == "stripe_subscription":
        return {"ok": True, "billing_type": billing}

    return {
        "ok": False,
        "reason": "not_subscription",
        "billing_type": billing,
    }


class BudgetTracker:

    def __init__(self, state: dict) -> None:
        self._state = state
        budget = state.get(BUDGET_STATE_KEY) or {}
        self._daily_used_tokens = int(budget.get("daily_used_tokens", 0) or 0)
        self._weekly_buffer_used_tokens = int(
            budget.get("weekly_buffer_used_tokens", 0) or 0,
        )
        self._last_reset_date = budget.get("last_reset_date")
        self._claude_disabled = bool(budget.get("claude_disabled", False))
        self._disabled_reason = budget.get("claude_disabled_reason")


    def claude_disabled_after_billing_event(self) -> bool:
        return self._claude_disabled

    def weekly_buffer_exceeded(self) -> bool:
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        return self._weekly_buffer_used_tokens >= weekly_cap

    def can_spend(self, estimated_tokens: int) -> bool:
        if self._claude_disabled:
            return False
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        if self._daily_used_tokens + estimated_tokens <= daily_cap:
            return True
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        overflow = (self._daily_used_tokens + estimated_tokens) - daily_cap
        return self._weekly_buffer_used_tokens + overflow <= weekly_cap


    def reset_if_new_day(self, now: datetime, tz) -> None:
        today_local = now.astimezone(tz).date().isoformat()
        if self._last_reset_date == today_local:
            return
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        weekly_cap = int(WEEKLY_BUFFER_PCT * ESTIMATED_DAILY_TOKEN_CEILING * 7)
        unused_today = max(0, daily_cap - self._daily_used_tokens)
        self._weekly_buffer_used_tokens = max(
            0,
            min(
                weekly_cap,
                self._weekly_buffer_used_tokens - unused_today,
            ),
        )
        self._daily_used_tokens = 0
        self._last_reset_date = today_local
        self._persist()

    def record(self, tokens_in: int, tokens_out: int, now: datetime) -> None:
        total = int(tokens_in) + int(tokens_out)
        daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
        if self._daily_used_tokens + total <= daily_cap:
            self._daily_used_tokens += total
        else:
            overflow = (self._daily_used_tokens + total) - daily_cap
            self._daily_used_tokens = daily_cap
            self._weekly_buffer_used_tokens += overflow
        self._persist()

    def disable_claude(self, reason: str) -> None:
        self._claude_disabled = True
        self._disabled_reason = str(reason)[:500]
        self._persist()


    def _persist(self) -> None:
        self._state[BUDGET_STATE_KEY] = {
            "daily_used_tokens": self._daily_used_tokens,
            "weekly_buffer_used_tokens": self._weekly_buffer_used_tokens,
            "last_reset_date": self._last_reset_date,
            "claude_disabled": self._claude_disabled,
            "claude_disabled_reason": self._disabled_reason,
        }
        save_state(self._state)


def _scrubbed_env() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in ENV_DENY_LIST:
            continue
        result[key] = value
    for hostile in ENV_DENY_LIST:
        result.pop(hostile, None)
    return result


def _bare_mode_enabled() -> bool:
    """Whether to pass ``--bare`` to ``claude`` for consolidation calls.

    ``--bare`` keeps the daemon's own ``claude -p`` calls from triggering hooks,
    auto-memory, etc., and is the default. On some macOS setups (credentials in
    the Keychain), ``claude --bare -p`` reports ``Not logged in`` while a plain
    ``claude -p`` authenticates normally; such users can set
    ``IAI_MCP_CLAUDE_BARE=0`` to drop the flag.
    """
    val = os.environ.get("IAI_MCP_CLAUDE_BARE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _build_cmd(prompt: str, model: str) -> list[str]:
    cmd = ["claude"]
    if _bare_mode_enabled():
        cmd.append("--bare")
    cmd += [
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        model,
    ]
    return cmd


async def _terminate_then_kill(proc, grace_sec: float) -> None:
    try:
        if proc.returncode is None:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=KILL_WAIT_SEC)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 -- best-effort
            pass


async def invoke_claude_once(
    prompt: str,
    *,
    model: str = "haiku",
) -> dict:
    env = _scrubbed_env()
    cmd = _build_cmd(prompt, model)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=CLAUDE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        await _terminate_then_kill(proc, TERMINATE_WAIT_SEC)
        return {
            "ok": False,
            "reason": "timeout",
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }
    except asyncio.CancelledError:
        await _terminate_then_kill(proc, FORCE_WAKE_GRACE_SEC)
        return {
            "ok": False,
            "reason": "force_wake_killed",
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": "nonzero_exit",
            "exit_code": proc.returncode,
            "stderr": stderr.decode("utf-8", errors="replace")[:500],
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "reason": "unparseable_output",
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    cost_usd = float(data.get("cost_usd", 0.0) or 0.0)
    usage = data.get("usage") or {}
    tokens_in = int(usage.get("input_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or 0)

    if cost_usd > 0.0:
        try:
            state = load_state()
            BudgetTracker(state).disable_claude(
                reason=f"api_billing_detected cost_usd={cost_usd}",
            )
        except Exception:  # noqa: BLE001 -- tripwire must not re-raise
            pass
        return {
            "ok": False,
            "reason": "api_billing_detected",
            "cost_usd": cost_usd,
            "data": data,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    return {
        "ok": True,
        "data": data,
        "cost_usd": cost_usd,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def invoke_claude_sync(
    prompt: str,
    *,
    model: str = "haiku",
    timeout_sec: float | None = None,
) -> dict:
    env = _scrubbed_env()
    cmd = _build_cmd(prompt, model)
    wall = timeout_sec if timeout_sec is not None else CLAUDE_TIMEOUT_SEC

    try:
        completed = subprocess.run(  # noqa: S603 -- argv list, no shell
            cmd,
            input=b"",
            capture_output=True,
            env=env,
            timeout=wall,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "reason": "timeout",
            "exit_code": -1,
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    if completed.returncode != 0:
        return {
            "ok": False,
            "reason": "nonzero_exit",
            "exit_code": completed.returncode,
            "stderr": completed.stderr.decode("utf-8", errors="replace")[:500],
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "reason": "unparseable_output",
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    cost_usd = float(data.get("cost_usd", 0.0) or 0.0)
    usage = data.get("usage") or {}
    tokens_in = int(usage.get("input_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or 0)

    if cost_usd > 0.0:
        try:
            state = load_state()
            BudgetTracker(state).disable_claude(
                reason=f"api_billing_detected cost_usd={cost_usd}",
            )
        except Exception:  # noqa: BLE001 -- tripwire must not re-raise
            pass
        return {
            "ok": False,
            "reason": "api_billing_detected",
            "cost_usd": cost_usd,
            "data": data,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    return {
        "ok": True,
        "data": data,
        "cost_usd": cost_usd,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
