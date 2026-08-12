"""Provider selection for the overnight reflection call.

The reflection rides the user's own logged-in subscription CLI. The
child process environment is built from an allow-list, so no vendor's
API key or endpoint override can reach it; each CLI runs in its most
restricted non-interactive mode with the prompt delivered on stdin. The
provider is chosen by the store's config.json (``reflection.provider``)
— config over environment, so a service environment can never silently
reroute the call to a different vendor.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

SUPPORTED_REFLECTION_PROVIDERS = ("claude", "codex", "gemini")
DEFAULT_REFLECTION_PROVIDER = "claude"

#: A vendor CLI's zero-exit stdout above this size is refused: raw agent
#: output becomes a stored memory record, and an unbounded banner/dump
#: must never be minted as an insight.
REFLECTION_OUTPUT_MAX_BYTES = 8192

_PROVIDER_BIN_ENV = {"codex": "IAI_MCP_CODEX_BIN", "gemini": "IAI_MCP_GEMINI_BIN"}

#: Only these parent variables reach a non-claude vendor child. The CLIs
#: authenticate from their own config directories under HOME; everything
#: else — foreign API keys, endpoint overrides — is denied by absence.
_CHILD_ENV_ALLOW = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR",
    "SHELL", "USER", "LOGNAME",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
)

_STDIN_PREAMBLE = "Follow the instructions provided on stdin."


def resolve_provider_bin(provider: str) -> "str | None":
    """Absolute path (or PATH-resolvable name) of the provider's CLI.

    The daemon runs under the service manager with a minimal PATH, so a
    bare binary name resolves in a shell but not in production; the
    ladder mirrors _resolve_claude_bin: explicit env override, PATH,
    then the standard install locations. None when nothing resolves.
    """
    if provider == "claude":
        from iai_mcp.claude_cli import _resolve_claude_bin

        path = _resolve_claude_bin()
        if Path(path).is_file() or shutil.which(path):
            return path
        return None
    env_name = _PROVIDER_BIN_ENV.get(provider)
    if env_name:
        explicit = os.environ.get(env_name)
        if explicit and Path(explicit).is_file():
            return explicit
    found = shutil.which(provider)
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / provider,
        Path("/opt/homebrew/bin") / provider,
        Path("/usr/local/bin") / provider,
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _provider_argv(provider: str, bin_path: str) -> "list[str] | None":
    # Most-restricted non-interactive mode per CLI; the prompt itself
    # travels on stdin, never in argv (argv is world-readable via ps).
    if provider == "codex":
        return [
            bin_path, "exec", "--sandbox", "read-only",
            "--skip-git-repo-check", "-",
        ]
    if provider == "gemini":
        # --skip-trust trusts only the empty scratch cwd the child runs
        # in; plan mode keeps every tool read-only.
        return [
            bin_path, "--approval-mode", "plan", "--skip-trust",
            "-e", "none", "-p", _STDIN_PREAMBLE,
        ]
    return None


def _subscription_child_env() -> dict[str, str]:
    from iai_mcp.claude_cli import _ensure_user_identity

    env = {k: os.environ[k] for k in _CHILD_ENV_ALLOW if k in os.environ}
    env["NO_COLOR"] = "1"
    return _ensure_user_identity(env)


def _reflection_cwd() -> "str | None":
    # A dedicated empty directory: the vendor CLI must never treat the
    # inherited working directory (the service manager pins the user's
    # home) as its workspace. Callers refuse when this returns None — a
    # fallback to the inherited cwd would silently undo the isolation.
    from iai_mcp.tz import store_root

    try:
        scratch = store_root() / ".reflection-scratch"
        scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
        return str(scratch)
    except OSError:
        return None


def _read_reflection_config() -> dict:
    """The store config as a dict; raises ValueError when it exists but
    cannot be parsed — a corrupt config must refuse, not silently reroute
    the nightly call to the default vendor."""
    from iai_mcp.tz import store_config_path

    path = store_config_path()
    if not path.exists():
        return {}
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"reflection config unreadable at {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"reflection config at {path} is not a JSON object")
    return cfg


def configured_reflection_provider() -> str:
    """The provider the store's config selects; fail-loud on unknown.

    An unknown name or an unreadable config is a refusal, not a silent
    fallback — a typo that quietly rerouted the nightly call to a
    different vendor would spend the wrong subscription without a trace.
    Only an absent config file defaults to claude.
    """
    cfg = _read_reflection_config()
    raw = ((cfg.get("reflection") or {}).get("provider") or "").strip().lower()
    if not raw:
        return DEFAULT_REFLECTION_PROVIDER
    if raw not in SUPPORTED_REFLECTION_PROVIDERS:
        from iai_mcp.tz import store_config_path

        raise ValueError(
            f"unknown reflection provider {raw!r} in {store_config_path()}; "
            f"supported: {', '.join(SUPPORTED_REFLECTION_PROVIDERS)}"
        )
    return raw


def set_reflection_provider(name: str) -> None:
    from iai_mcp.tz import store_config_path

    name = (name or "").strip().lower()
    if name not in SUPPORTED_REFLECTION_PROVIDERS:
        raise ValueError(
            f"unknown reflection provider {name!r}; supported: "
            f"{', '.join(SUPPORTED_REFLECTION_PROVIDERS)}"
        )
    cfg = _read_reflection_config()
    reflection = dict(cfg.get("reflection") or {})
    reflection["provider"] = name
    cfg["reflection"] = reflection
    path = store_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp, path)


async def invoke_reflection_once(
    prompt: str, *, model: str = "haiku", provider: "str | None" = None,
) -> dict:
    """One reflection call through the given (or configured) provider.

    The claude provider keeps its full machinery (credential check and
    budget happen at the caller). The adapter does not parse usage from
    other providers' CLIs, so their token counts are reported as zero;
    pacing comes from the nightly interval.
    """
    if provider is None:
        provider = configured_reflection_provider()
    if provider == "claude":
        from iai_mcp.claude_cli import invoke_claude_once

        return await invoke_claude_once(prompt, model=model)
    return await _invoke_subscription_cli(provider, prompt)


def invoke_reflection_sync(
    prompt: str, *, model: str = "haiku", provider: "str | None" = None,
) -> dict:
    """Synchronous variant for callers off the event loop (REM critic)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # asyncio.run below would raise and leak an un-awaited coroutine.
        return {"ok": False, "reason": "sync_call_on_event_loop",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    if provider is None:
        provider = configured_reflection_provider()
    if provider == "claude":
        from iai_mcp.claude_cli import invoke_claude_sync

        return invoke_claude_sync(prompt, model=model)
    return asyncio.run(_invoke_subscription_cli(provider, prompt))


async def _invoke_subscription_cli(provider: str, prompt: str) -> dict:
    from iai_mcp.claude_cli import (
        CLAUDE_TIMEOUT_SEC,
        FORCE_WAKE_GRACE_SEC,
        TERMINATE_WAIT_SEC,
        _terminate_then_kill,
    )

    if provider not in _PROVIDER_BIN_ENV:
        return {"ok": False, "reason": f"provider_unsupported_{provider}",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    bin_path = resolve_provider_bin(provider)
    if bin_path is None:
        return {"ok": False, "reason": f"provider_cli_missing_{provider}",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    argv = _provider_argv(provider, bin_path)
    cwd = _reflection_cwd()
    if cwd is None:
        return {"ok": False, "reason": f"{provider}_scratch_cwd_unavailable",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_subscription_child_env(),
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=CLAUDE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        await _terminate_then_kill(proc, TERMINATE_WAIT_SEC)
        return {"ok": False, "reason": "timeout",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    except asyncio.CancelledError:
        # A force-wake cancels the reflection; the vendor child must not
        # keep running unsupervised.
        await _terminate_then_kill(proc, FORCE_WAKE_GRACE_SEC)
        raise
    if proc.returncode != 0:
        detail = (stderr or b"").decode("utf-8", errors="replace")[:160]
        return {"ok": False,
                "reason": f"{provider}_exit_{proc.returncode}: {detail}",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    raw = stdout or b""
    if len(raw) > REFLECTION_OUTPUT_MAX_BYTES:
        return {"ok": False,
                "reason": f"{provider}_output_too_large_{len(raw)}b",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {"ok": False, "reason": f"{provider}_empty_output",
                "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
    # data.result mirrors the claude CLI's shape so every downstream
    # consumer extracts the text the same way regardless of provider.
    return {"ok": True, "text": text, "data": {"result": text},
            "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
