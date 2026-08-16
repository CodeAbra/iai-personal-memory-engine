"""Single source of truth for the live/socket-dispatch test harness.

Consolidates the spawn/kill/socket-wait/HF-cache/passphrase helpers and the
in-process JSON-RPC socket helpers that were previously scattered across
sibling test files, plus the runtime sender-surface extractor consumed by the
dispatch reachability sweep.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import psutil
import pytest

from _warm_recall_repro_support import (  # noqa: F401 -- re-exported
    await_socket,
    raw_recall,
    spawn_isolated_daemon,
)

REPO = Path(__file__).resolve().parent.parent

_TEST_CRYPTO_PASSPHRASE = "iai-mcp-test-passphrase"


def _hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _kill_test_daemons(sock_path: Path) -> None:
    target = str(sock_path)
    res = subprocess.run(
        ["lsof", "-U", "-F", "pn"],
        capture_output=True, text=True, check=False,
    )
    current: int | None = None
    pids: set[int] = set()
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and line[1:] == target:
            pids.add(current)
    for pid in pids:
        try:
            cl = " ".join(psutil.Process(pid).cmdline())
            if "iai_mcp.daemon" in cl:
                psutil.Process(pid).send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _wait_for_daemon_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


async def _send_jsonrpc(
    sock_path: Path,
    method: str,
    params: dict | None = None,
    req_id: int | str = 1,
    *,
    timeout: float = 10.0,
) -> dict:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        envelope: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            envelope["params"] = params
        writer.write((json.dumps(envelope) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError(f"daemon closed without reply (method={method})")
    return json.loads(line.decode("utf-8"))


async def _send_raw(sock_path: Path, raw_bytes: bytes, *, timeout: float = 5.0) -> dict:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        writer.write(raw_bytes)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError("daemon closed without reply")
    return json.loads(line.decode("utf-8"))


async def _with_socket_server(sock_path: Path, store, coro_fn):
    from iai_mcp.socket_server import SocketServer

    srv = SocketServer(store, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    for _ in range(250):
        if sock_path.exists():
            break
        await asyncio.sleep(0.01)
    if not sock_path.exists():
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
        raise AssertionError("socket never bound")

    try:
        result = await coro_fn(sock_path, store)
    finally:
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
    return result


def _live_daemon_env(tmp_home: Path, sock_path: Path) -> dict[str, str]:
    store_dir = tmp_home / ".iai-mcp"
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "120"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    env["IAI_MCP_EMBED_OFFLINE"] = "1"
    env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    hf_root = _hf_cache_root()
    env["HF_HOME"] = str(hf_root)
    env["HF_HUB_CACHE"] = str(hf_root / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")
    return env


def spawn_live_daemon(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
    *,
    cue: "str | None" = None,
    seed: "Callable[[object, list | None], None] | None" = None,
):
    """Generator yielding a SimpleNamespace wrapping a real, isolated
    ``iai_mcp.daemon`` subprocess plus ``iai()``/``recall_json()``/
    ``wait_until()`` CLI drivers.

    If ``seed`` is given it runs against a freshly opened ``MemoryStore`` at
    the isolated store root BEFORE the daemon subprocess is spawned, as
    ``seed(store, cue_vec)`` -- gold-fixture seeding (which records land in
    the store) stays the caller's responsibility; this factory only owns the
    process lifecycle.
    """
    hf_cache = _hf_cache_root()
    weights_dir = hf_cache / "hub" / "models--BAAI--bge-small-en-v1.5"
    if not weights_dir.exists():
        pytest.skip(
            f"bge-small weight cache absent ({weights_dir}); the offline "
            "live-gate construct cannot run."
        )

    tmp_home = tmp_path / "home"
    tmp_home.mkdir(parents=True)
    store_dir = tmp_home / ".iai-mcp"

    sock_dir = Path(tempfile.mkdtemp(prefix="iai-live-"))
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode()) < 104, (
        f"sun_path too long ({len(str(sock_path).encode())} >= 104): {sock_path}"
    )

    proc: "subprocess.Popen | None" = None
    try:
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_CRYPTO_PASSPHRASE)
        monkeypatch.setenv("HF_HOME", str(hf_cache))
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache / "hub"))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
        monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")

        cue_vec: "list | None" = None
        if cue is not None:
            from iai_mcp.embed import Embedder
            cue_vec = Embedder().embed(cue)

        if seed is not None:
            from iai_mcp.store import MemoryStore
            store = MemoryStore(str(store_dir))
            try:
                seed(store, cue_vec)
            finally:
                store.close()

        daemon_env = _live_daemon_env(tmp_home, sock_path)
        proc = subprocess.Popen(
            [sys.executable, "-m", "iai_mcp.daemon"],
            cwd=str(REPO),
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        bound = _wait_for_daemon_socket(sock_path, timeout_sec=30.0)
        assert bound, (
            f"daemon did not bind socket within 30 s: {sock_path}; "
            f"proc.poll()={proc.poll()!r}"
        )

        cli_env = dict(daemon_env)

        def iai(*argv: str, timeout: int = 60) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "iai_mcp.iai_cli", *argv],
                env=cli_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        def recall_json(cue_str: str) -> dict:
            result = iai("recall", "--json", "--limit", "50", cue_str)
            assert result.returncode == 0, (
                f"iai recall failed (rc={result.returncode}):\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
            )
            stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            assert stdout_lines, f"no JSON on stdout; stderr={result.stderr!r}"
            return json.loads(stdout_lines[-1])

        def wait_until(
            predicate: "Callable[[], bool]",
            timeout: float = 10.0,
            interval: float = 0.05,
        ) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(interval)
            return False

        lifecycle_path = store_dir / "lifecycle_state.json"

        ns = SimpleNamespace(
            cue=cue,
            cue_vec=cue_vec,
            store_dir=store_dir,
            sock_path=sock_path,
            lifecycle_path=lifecycle_path,
            proc=proc,
            iai=iai,
            recall_json=recall_json,
            wait_until=wait_until,
        )
        yield ns

    finally:
        try:
            _kill_test_daemons(sock_path)
        except Exception:  # noqa: BLE001
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(sock_dir, ignore_errors=True)


_METHOD_NAME_RE = r"[A-Za-z_][A-Za-z0-9_]*"


def _extract_wrapper_ts_methods(wrapper_src: Path) -> "dict[str, set[str]]":
    """Named anchor: first string arg of ``.call("m")`` and the TypeScript
    generic form ``.call<T>("m", ...)``."""
    pattern = re.compile(
        r'\.call\s*(?:<[^>]*>)?\s*\(\s*"(' + _METHOD_NAME_RE + r')"'
    )
    out: "dict[str, set[str]]" = {}
    if not wrapper_src.exists():
        return out
    for ts_file in sorted(wrapper_src.glob("*.ts")):
        text = ts_file.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            out.setdefault(m.group(1), set()).add(f"wrapper:{ts_file.name}")
    return out


def _extract_python_send_methods(py_files: "list[Path]") -> "dict[str, set[str]]":
    """Named anchor: first string-literal arg of ``_send_jsonrpc_request("m", ...)``
    and ``_relay_rpc("m", ...)`` -- never a blind literal scan of the file."""
    pattern = re.compile(
        r'(?:_send_jsonrpc_request|_relay_rpc)\(\s*"(' + _METHOD_NAME_RE + r')"'
    )
    out: "dict[str, set[str]]" = {}
    for py_file in py_files:
        if not py_file.exists():
            continue
        text = py_file.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            out.setdefault(m.group(1), set()).add(f"python:{py_file.name}")
    return out


def _extract_hook_envelope_methods(hooks_dir: Path) -> "dict[str, set[str]]":
    """Named anchor: the ``"method"`` key of a raw JSON-RPC envelope literal
    embedded in a deploy hook script (hooks send over a raw AF_UNIX socket,
    not through the Python send helpers)."""
    pattern = re.compile(r'"method"\s*:\s*"(' + _METHOD_NAME_RE + r')"')
    out: "dict[str, set[str]]" = {}
    if not hooks_dir.exists():
        return out
    hook_files = sorted(list(hooks_dir.glob("*.sh")) + list(hooks_dir.glob("*.ps1")))
    for hook_file in hook_files:
        text = hook_file.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            out.setdefault(m.group(1), set()).add(f"hook:{hook_file.name}")
    return out


def derive_client_sent_methods_with_provenance() -> "dict[str, set[str]]":
    """UNION of every real client-sent dispatch method, mapped to the
    sender-surface label(s) it was found on -- derived at call time from the
    wrapper, ``iai_cli.py``, ``cli/*`` and deploy-hook sources, minus
    ``SocketServer.CONTROL_MSG_TYPES``. Never a frozen list."""
    wrapper_src = REPO / "mcp-wrapper" / "src"
    cli_dir = REPO / "src" / "iai_mcp" / "cli"
    py_files = [REPO / "src" / "iai_mcp" / "iai_cli.py"]
    if cli_dir.exists():
        py_files.extend(sorted(cli_dir.glob("*.py")))
    hooks_dir = REPO / "src" / "iai_mcp" / "_deploy" / "hooks"

    mapping: "dict[str, set[str]]" = {}
    for source in (
        _extract_wrapper_ts_methods(wrapper_src),
        _extract_python_send_methods(py_files),
        _extract_hook_envelope_methods(hooks_dir),
    ):
        for method, surfaces in source.items():
            mapping.setdefault(method, set()).update(surfaces)

    from iai_mcp.socket_server import SocketServer
    for control_type in SocketServer.CONTROL_MSG_TYPES:
        mapping.pop(control_type, None)
    return mapping


def derive_client_sent_methods() -> "set[str]":
    """The client-sent dispatch method set, derived at call time. See
    ``derive_client_sent_methods_with_provenance`` for the per-method sender
    surface(s)."""
    return set(derive_client_sent_methods_with_provenance())


# Dispatch method branches parsed out of core.dispatch with no real client
# sender on any surface, awaiting removal by the dead-code cleanup.
KNOWN_DEAD_DISPATCH_METHODS: "frozenset[str]" = frozenset({
    "s5_propose",  # no client surface invokes S5 proposal review
    "shield_check",  # no client surface calls the shield check directly
})


def _is_method_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "method"


def _str_const(node: ast.AST) -> "str | None":
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def dispatch_method_branches(src_path: Path) -> "set[str]":
    """AST-parse ``src_path`` (never imported) for every string literal
    compared against a bare ``method`` name -- both ``method == "..."`` and
    ``method in {"...", ...}`` / ``(...)`` / ``[...]`` forms. The ``==``
    match is symmetric: ``method`` may sit on either side of the operator."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    methods: "set[str]" = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for op, left, right in zip(node.ops, operands, operands[1:]):
            if isinstance(op, ast.Eq):
                if _is_method_name(left):
                    literal = _str_const(right)
                elif _is_method_name(right):
                    literal = _str_const(left)
                else:
                    literal = None
                if literal is not None:
                    methods.add(literal)
            elif (
                isinstance(op, ast.In)
                and _is_method_name(left)
                and isinstance(right, (ast.Set, ast.Tuple, ast.List))
            ):
                for elt in right.elts:
                    literal = _str_const(elt)
                    if literal is not None:
                        methods.add(literal)
    return methods
