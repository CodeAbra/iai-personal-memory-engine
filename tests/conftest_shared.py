"""NOT a standard pytest conftest — this file is NOT auto-discovered.
Tests import helpers directly:  from tests.conftest_shared import make_tmp_store

The project-wide autouse fixtures in tests/conftest.py (crypto passphrase +
autoflush) already apply to every test here, so store.insert() is synchronous
in tests without any extra setup.

All helpers use tmp_path, never ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from iai_mcp.store import MemoryStore


def short_socket_path(prefix: str = "iai-sock-") -> Path:
    """Return a short, bindable AF_UNIX socket path under a new mkdtemp directory.

    Creates a process-unique directory under the system temp root (TMPDIR →
    /var/folders/.../T/ on macOS) whose total path stays well under the platform
    sun_path limit (~104 bytes on macOS).  The caller is responsible for removing
    the directory after use; the companion pytest fixture ``short_socket`` handles
    teardown automatically.

    Use this function (rather than the fixture) only in non-fixture call sites —
    e.g. module-level helpers — where you must manage cleanup via try/finally or
    request.addfinalizer yourself.
    """
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d / "d.sock"


@pytest.fixture()
def short_socket():
    """Pytest fixture: yields a short, bindable AF_UNIX socket path with cleanup.

    Allocates a process-unique directory under the system temp root (TMPDIR →
    /var/folders/.../T/ on macOS) whose total path stays well under the platform
    sun_path limit (~104 bytes on macOS), avoiding ``OSError: AF_UNIX path too
    long`` on bind() when the pytest tmp_path is deep.

    The store / HOME remain hermetic under ``tmp_path``; only the socket endpoint
    moves to this short root.  The directory and socket file are removed on
    teardown via ``shutil.rmtree``, preventing accumulation in the system temp dir.

    Usage:
        def test_something(short_socket):
            from tests.conftest_shared import short_socket  # import into module
            srv.bind(str(short_socket))
            ...
    """
    d = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    sock = d / "d.sock"
    yield sock
    shutil.rmtree(d, ignore_errors=True)


def make_tmp_store(tmp_path: Path) -> MemoryStore:
    """Construct an isolated MemoryStore rooted at tmp_path.

    Never touches ~/.iai-mcp/ or the live daemon.

    Usage in a test:
        def test_something(tmp_path):
            store = make_tmp_store(tmp_path)
            ...
    """
    store_root = tmp_path / "hippo"
    store_root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=store_root)


def set_tmp_env(monkeypatch, tmp_path: Path) -> None:
    """Monkeypatch IAI_MCP_STORE and IAI_DAEMON_SOCKET_PATH to tmp paths.

    Prevents any code path that reads these env vars from touching the
    live store or the live daemon socket.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))


def retry_once_on_assertion(test_fn):
    """Wall-clock perf tripwires ONLY. One retry absorbs OS-scheduler
    jitter on a loaded machine; a genuine regression misses BOTH
    attempts, so the tripwire keeps its teeth. Each attempt runs in a
    fresh tmp_path subdirectory so store files never collide across
    attempts. The first miss is reported loudly, never swallowed.
    """
    import functools
    import sys as _sys

    @functools.wraps(test_fn)
    def wrapper(tmp_path, *args, **kwargs):
        last: AssertionError | None = None
        for attempt in (1, 2):
            sub = tmp_path / f"attempt{attempt}"
            sub.mkdir(exist_ok=True)
            try:
                return test_fn(sub, *args, **kwargs)
            except AssertionError as exc:
                last = exc
                if attempt == 2:
                    raise
                print(
                    f"perf bound missed on attempt 1 ({test_fn.__name__}): "
                    f"{exc} — retrying once on a fresh store",
                    file=_sys.stderr,
                )
        raise last  # unreachable; keeps type-checkers honest

    return wrapper


def lsof_name_is_socket(name: str, target: str) -> bool:
    """Return True when an `lsof -F n` NAME field refers to the target socket.

    The NAME field is not portable. macOS emits the bare path:

        n/tmp/iai-x/d.sock

    Linux appends the socket type to the same field:

        n/tmp/iai-x/d.sock type=STREAM

    An `== target` comparison therefore matches nothing at all on Linux, and
    because every caller feeds the result into a PID set, the failure is silent:
    the set comes back empty and reads as "no daemon is bound" rather than as a
    parse error. Matching the path plus a space boundary accepts both platforms
    and still refuses a longer path — "/a/b.sock2 type=STREAM" does not start
    with "/a/b.sock ", so the sibling-path false positive the exact comparison
    was protecting against stays excluded.
    """
    return name == target or name.startswith(target + " ")


def _pids_bound_via_proc(target: str) -> set[int]:
    """Linux: resolve unix-socket binders through /proc, the way `ss -lxp` does.

    /proc/net/unix lists every bound unix socket with its inode but no owning
    pid; /proc/<pid>/fd/* symlinks read back as "socket:[<inode>]". Joining the
    two recovers the binder set without shelling out to `ss` (absent from some
    minimal images) or `lsof`.
    """
    import os as _os

    inodes: set[str] = set()
    try:
        for line in Path("/proc/net/unix").read_text().splitlines()[1:]:
            parts = line.split()
            # Num RefCount Protocol Flags Type St Inode Path
            if len(parts) >= 8 and parts[7] == target:
                inodes.add(parts[6])
    except OSError:
        return set()
    if not inodes:
        return set()

    wanted = {f"socket:[{ino}]" for ino in inodes}
    pids: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                if _os.readlink(fd) in wanted:
                    pids.add(int(entry.name))
                    break
        except OSError:
            continue  # process exited, or another uid's fds
    return pids


def pids_bound_to_unix_socket(sock_path) -> set[int]:
    """Return the pids holding the unix socket at ``sock_path``.

    Written because `lsof -U` does not reliably report unix-socket paths on
    Linux: measured in-container, a live daemon's socket appears in
    /proc/net/unix while `lsof -U` omits it entirely, so an lsof-only lookup
    returns an empty set and reads as "nothing is bound". That is what made
    test_socket_subagent_reuse fail on both Linux interpreters while staying
    green on the macOS gate. iai_mcp.doctor already compensates by falling back
    to `ss -lxp`; the test helpers had no such fallback.

    Linux goes through /proc directly (authoritative, and no dependency on `ss`,
    which some minimal images omit). macOS has no /proc, so it keeps lsof.
    """
    import platform as _platform
    import subprocess as _subprocess

    target = str(sock_path)
    if _platform.system() == "Linux":
        return _pids_bound_via_proc(target)

    res = _subprocess.run(
        ["lsof", "-U", "-F", "pn"], capture_output=True, text=True, check=False,
    )
    current: int | None = None
    pids: set[int] = set()
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and lsof_name_is_socket(line[1:], target):
            pids.add(current)
    return pids
