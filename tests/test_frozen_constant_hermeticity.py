from __future__ import annotations

import inspect
import os
from pathlib import Path

import iai_mcp.backup
import iai_mcp.capture_queue
import iai_mcp.cli
import iai_mcp.concurrency
import iai_mcp.crypto
import iai_mcp.daemon
import iai_mcp.daemon_state
import iai_mcp.directive_cache
import iai_mcp.hippo
import iai_mcp.lifecycle
import iai_mcp.lifecycle_event_log
import iai_mcp.lifecycle_state
import iai_mcp.store


def _real_root() -> Path:
    return Path(iai_mcp.hippo._REAL_IAI_ROOT)


def _tmp_home() -> Path:
    return Path(os.environ["HOME"])


def _resolves_under_real(value) -> bool:
    real = _real_root()
    try:
        p = Path(value).resolve()
    except (TypeError, ValueError):
        return False
    return p == real.resolve() or real.resolve() in p.parents


def _resolves_under_tmp(value) -> bool:
    home = _tmp_home().resolve()
    try:
        p = Path(value).resolve()
    except (TypeError, ValueError):
        return False
    return p == home or home in p.parents


def _redirected_constants() -> dict[str, object]:
    return {
        "hippo._DEFAULT_IAI_ROOT": iai_mcp.hippo._DEFAULT_IAI_ROOT,
        "store.DEFAULT_STORAGE_PATH": iai_mcp.store.DEFAULT_STORAGE_PATH,
        "concurrency.SOCKET_PATH": iai_mcp.concurrency.SOCKET_PATH,
        "daemon_state.STATE_PATH": iai_mcp.daemon_state.STATE_PATH,
        "lifecycle_state.LIFECYCLE_STATE_PATH": iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH,
        "cli.LOCK_PATH": iai_mcp.cli.LOCK_PATH,
        "cli.STATE_PATH": iai_mcp.cli.STATE_PATH,
        "lifecycle_event_log.DEFAULT_LOG_DIR": iai_mcp.lifecycle_event_log.DEFAULT_LOG_DIR,
        "capture_queue.DEFAULT_QUEUE_DIR": iai_mcp.capture_queue.DEFAULT_QUEUE_DIR,
        "lifecycle.DEFAULT_LOCK_PATH": iai_mcp.lifecycle.DEFAULT_LOCK_PATH,
        "daemon.SESSION_START_CACHE_PATH": iai_mcp.daemon.SESSION_START_CACHE_PATH,
        "crypto._DEFAULT_STORE_ROOT": iai_mcp.crypto._DEFAULT_STORE_ROOT,
        "backup.DEFAULT_STORE_PATH": iai_mcp.backup.DEFAULT_STORE_PATH,
        "directive_cache.DIRECTIVES_CACHE_PATH": iai_mcp.directive_cache.DIRECTIVES_CACHE_PATH,
    }


def _home_derived_defaults() -> list[tuple[str, object]]:
    """Function default-argument VALUES, not module constants.

    A default like ``def f(path: Path = SOME_HOME_CONSTANT)`` is bound once,
    at function-definition time (import time) -- before any fixture runs.
    ``monkeypatch.setattr(module, "SOME_HOME_CONSTANT", tmp)`` rebinds the
    module global but never touches the already-captured default, so a call
    site that omits the argument silently keeps writing to the operator's
    real home. ``_redirected_constants`` above is structurally blind to this:
    it only ever reads the module attribute, which the fixture patches
    successfully -- the leak is in the function's ``__defaults__``.
    """
    modules = (
        iai_mcp.backup, iai_mcp.capture_queue, iai_mcp.cli, iai_mcp.concurrency,
        iai_mcp.crypto, iai_mcp.daemon, iai_mcp.daemon_state, iai_mcp.directive_cache,
        iai_mcp.hippo, iai_mcp.lifecycle, iai_mcp.lifecycle_event_log,
        iai_mcp.lifecycle_state, iai_mcp.store,
    )
    out: list[tuple[str, object]] = []
    for mod in modules:
        for fname, fn in vars(mod).items():
            if not inspect.isroutine(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (ValueError, TypeError):
                continue
            for pname, p in params.items():
                if p.default is inspect.Parameter.empty:
                    continue
                if isinstance(p.default, (str, Path)):
                    out.append((f"{mod.__name__}.{fname}({pname}=)", p.default))
    return out


def test_no_function_default_argument_freezes_a_real_home_path() -> None:
    real = _real_root()
    offenders = [
        f"{name}={value!r}"
        for name, value in _home_derived_defaults()
        if _resolves_under_real(value)
    ]
    assert not offenders, (
        "function default argument(s) bound at import time to the operator's "
        f"real store {real} -- a caller that omits the argument writes there "
        f"even though the module-level constant is patched: {offenders}"
    )


def test_redirected_constants_resolve_under_tmp_not_real() -> None:
    real = _real_root()
    for name, value in _redirected_constants().items():
        assert not _resolves_under_real(value), (
            f"{name}={value!r} resolves under the operator's real store {real}"
        )
        assert _resolves_under_tmp(value), (
            f"{name}={value!r} is not under the per-test tmp HOME"
        )


def test_socket_path_shadowed_by_env_resolves_to_tmp() -> None:
    consumer_value = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        iai_mcp.cli.SOCKET_PATH
    )
    assert consumer_value, "IAI_DAEMON_SOCKET_PATH must be set by the fixture"
    assert not _resolves_under_real(consumer_value), (
        f"cli socket consumer resolution {consumer_value!r} lands on the real store"
    )
    assert _resolves_under_tmp(consumer_value), (
        f"cli socket consumer resolution {consumer_value!r} is not under tmp"
    )


def test_launchd_target_is_outside_the_store_boundary() -> None:
    assert not _resolves_under_real(iai_mcp.cli.LAUNCHD_TARGET), (
        f"cli.LAUNCHD_TARGET={iai_mcp.cli.LAUNCHD_TARGET!r} unexpectedly under the real store"
    )
    target = Path(iai_mcp.cli.LAUNCHD_TARGET)
    assert target.parent.name == "LaunchAgents"
    assert target.suffix == ".plist"


def test_no_frozen_home_constant_resolves_to_real_store() -> None:
    real = _real_root()
    offenders: list[str] = []

    for name, value in _redirected_constants().items():
        if _resolves_under_real(value):
            offenders.append(f"{name}={value!r}")

    socket_consumer = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        iai_mcp.cli.SOCKET_PATH
    )
    if _resolves_under_real(socket_consumer):
        offenders.append(f"cli.SOCKET_PATH(consumer)={socket_consumer!r}")

    assert not offenders, (
        "frozen home-derived constants resolve to the real store "
        f"{real} under pytest: {offenders}"
    )
