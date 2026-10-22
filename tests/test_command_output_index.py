"""Opt-in command-output auto-index + structural secret gate.

Hermetic isolation, binding for every test here: IAI_TOOLS_ROOT points at
a tmp dir (monkeypatch, read at call time) and every index call threads an
EXPLICIT output_store_root (tmp_path / "tools-output") so the store-
boundary predicate fires without ever touching the user's real
~/.iai-mcp store. `default_command_output_store_root()` is never called
in-process -- `_DEFAULT_IAI_ROOT` is bound to Path.home() at import time.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_MAX_CONNECT_RETRIES = 3


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


@pytest.fixture
def exec_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so the
    # socket lives in its own short-path directory directly under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", "1.0")
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _run_with_retry(run_and_render, *args, **kwargs):
    """Bounded retry around a known intermittent executor-daemon connect
    wobble (ConnectionRefused during daemon spawn, tracked separately) --
    not a defect of this test's own auto-index wiring.
    """
    last_error: "Exception | None" = None
    for attempt in range(_MAX_CONNECT_RETRIES):
        try:
            return run_and_render(*args, **kwargs)
        except ConnectionRefusedError as e:
            last_error = e
            time.sleep(0.2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def test_run_then_search_retrieves_output(exec_env: dict, tmp_path: Path) -> None:
    import uuid

    from iai_mcp.tools import search
    from iai_mcp.tools.exec_cli import run_and_render

    marker = uuid.uuid4().hex
    output_store = tmp_path / "tools-output"
    source_store = tmp_path / "source-store"

    result = _run_with_retry(
        run_and_render,
        "echo",
        [marker],
        auto_index=True,
        output_store_root=output_store,
    )
    assert result["run_id"]
    assert result.get("index_error") is None
    assert result["indexed"] >= 1
    assert result["refused"] == []

    hits = search(marker, store_root=output_store)
    assert hits, "the run's own stdout must be retrievable from the command-output store"
    assert any("stdout.log" in path for path, _score in hits)

    source_hits = search(marker, store_root=source_store)
    assert source_hits == [], "the same marker must NOT leak into a separate source-code store"


def test_secret_shaped_output_refused(exec_env: dict, tmp_path: Path) -> None:
    from iai_mcp.tools import search
    from iai_mcp.tools.exec_cli import run_and_render

    output_store = tmp_path / "tools-output"
    secret_value = "api_key=sk-alice-fixture-0000000000"

    result = _run_with_retry(
        run_and_render,
        "echo",
        [secret_value],
        auto_index=True,
        output_store_root=output_store,
    )
    assert result.get("index_error") is None
    # stderr.log is empty (echo writes nothing to stderr) and passes the
    # gate; only the secret-bearing stdout.log is refused.
    assert len(result["refused"]) == 1
    assert "stdout.log" in result["refused"][0]

    hits = search("alice-fixture", store_root=output_store)
    assert hits == [], "secret-shaped output must never land in the searchable store"


def test_direct_index_paths_to_output_store_is_gated(exec_env: dict, tmp_path: Path) -> None:
    from iai_mcp.tools import index_paths, search

    output_store = tmp_path / "tools-output"

    secret_file = tmp_path / "secret.log"
    secret_file.write_text("password=alice-fixture-secret-value\n")
    clean_file = tmp_path / "clean.log"
    clean_file.write_text("nothing sensitive here, just plain text\n")

    indexed = index_paths([str(secret_file), str(clean_file)], store_root=output_store)
    assert indexed == 1, "the direct index_paths call must still refuse the secret file"

    secret_hits = search("alice-fixture-secret-value", store_root=output_store)
    assert secret_hits == [], "a DIRECT index_paths bypass attempt must still be gated"

    clean_hits = search("nothing sensitive", store_root=output_store)
    assert clean_hits, "a non-secret file passed the same way must be indexed and retrievable"


def test_auto_index_error_is_noop_after_successful_run(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.tools import output_index
    from iai_mcp.tools.exec_cli import run_and_render

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(output_index, "index_command_output", _raise)

    output_store = tmp_path / "tools-output"
    result = _run_with_retry(
        run_and_render,
        "echo",
        ["still-succeeds"],
        auto_index=True,
        output_store_root=output_store,
    )

    assert result["run_id"]
    assert result["state"] == "exited:0"
    assert "still-succeeds" in result["bounded_tail"]["stdout"]
    assert result["index_error"] == "simulated index failure"
    assert "indexed" not in result
    assert "refused" not in result


def test_auto_index_with_background_signals_skip_not_silent_noop(
    exec_env: dict, tmp_path: Path
) -> None:
    from iai_mcp.tools.exec_cli import run_and_render

    output_store = tmp_path / "tools-output"
    result = _run_with_retry(
        run_and_render,
        "echo",
        ["backgrounded"],
        auto_index=True,
        background=True,
        output_store_root=output_store,
    )

    assert result["run_id"]
    assert "index_error" in result, "auto-index + background must signal, never silently no-op"
    assert "background" in result["index_error"]
    assert "indexed" not in result
    assert "refused" not in result
    assert not output_store.exists(), "a background run must never create the command-output store"


def test_default_run_is_byte_unchanged(exec_env: dict, tmp_path: Path) -> None:
    from iai_mcp import tools_exec_client
    from iai_mcp.tools.exec_cli import run_and_render

    direct = _run_with_retry(tools_exec_client.run, "echo", ["plain"])
    via_render = _run_with_retry(run_and_render, "echo", ["plain"])

    assert set(via_render.keys()) == set(direct.keys()) == {"run_id", "state", "bounded_tail"}
    assert via_render["state"] == direct["state"]
    assert via_render["bounded_tail"]["stdout"] == direct["bounded_tail"]["stdout"]

    output_store = tmp_path / "tools-output"
    assert not output_store.exists(), "a run without the opt-in flag must never create the store"
