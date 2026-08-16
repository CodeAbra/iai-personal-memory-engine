"""Dispatch-method reachability sweep.

Drives every method a real client (wrapper / CLI / deploy hook) can send,
over the real ``SocketServer._handle -> core.dispatch`` wire, and asserts
each reaches a handler. This file must NEVER import ``iai_mcp.core.dispatch``
directly -- reachability is proven only by driving the socket wire.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from _live_harness import (
    KNOWN_DEAD_DISPATCH_METHODS,
    _send_jsonrpc,
    _with_socket_server,
    derive_client_sent_methods,
    derive_client_sent_methods_with_provenance,
)

pytestmark = pytest.mark.live

# Minimal but real params per method, so the sweep tests reachability rather
# than tripping over -32602 param validation. Methods absent here get {}.
_MINIMAL_PARAMS: "dict[str, dict]" = {
    "memory_recall": {"cue": "reachability probe cue"},
    "memory_reinforce": {"ids": []},
    "memory_contradict": {
        "id": "00000000-0000-0000-0000-000000000000",
        "new_fact": "reachability probe fact",
    },
    "memory_capture": {"text": "reachability probe capture"},
    "profile_get": {"knob": None},
    "profile_set": {"knob": "literal_preservation", "value": "strong"},
    "drain_permanent_failed": {"dry_run": True},
}


def _params_for(method: str) -> dict:
    return _MINIMAL_PARAMS.get(method, {})


@pytest.fixture
def short_sock_path(tmp_path: Path):
    sock_dir = Path(f"/tmp/iai-dispsweep-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode()) < 104, (
        f"sun_path too long ({len(str(sock_path).encode())} >= 104): {sock_path}"
    )
    try:
        yield sock_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_derive_client_sent_methods_is_complete_and_precise() -> None:
    derived = derive_client_sent_methods()

    assert "session_start_payload" in derived, (
        "extractor dropped a real sender: the TypeScript generic-form call "
        "site .call<T>(\"session_start_payload\", ...) -- the completeness "
        "self-check has regressed"
    )
    assert "brain_view" in derived, (
        "extractor dropped a real sender surface: the top-level iai_cli.py "
        "_relay_rpc(\"brain_view\", ...) call site -- the completeness "
        "self-check has regressed"
    )
    assert "captured_at" not in derived, (
        "extractor over-collected a non-method literal (captured_at is a "
        "dict key in iai_cli.py, not a dispatch method) -- the precision "
        "self-check has regressed, likely from a blind string scan"
    )


def test_dead_set_has_no_client_sender() -> None:
    derived = derive_client_sent_methods()
    resurrected = KNOWN_DEAD_DISPATCH_METHODS & derived
    assert not resurrected, (
        f"dead handler(s) now have a real client sender: {sorted(resurrected)} "
        "-- a handler wired to a real client must leave the "
        "expected-dead/never-sent set"
    )


def test_every_client_sent_method_is_reachable(short_sock_path: Path) -> None:
    from iai_mcp.store import MemoryStore

    provenance = derive_client_sent_methods_with_provenance()
    derived = set(provenance)
    assert derived, "sender-surface extraction found zero methods -- extractor is broken"

    store = MemoryStore()

    async def _drive_all(sock_path: Path, store) -> "list[str]":
        failures: list[str] = []
        for req_id, method in enumerate(sorted(derived), start=1):
            resp = await _send_jsonrpc(
                sock_path, method, _params_for(method), req_id=req_id, timeout=30.0,
            )
            if resp.get("jsonrpc") != "2.0" or resp.get("id") != req_id:
                failures.append(
                    f"{method}: malformed JSON-RPC envelope "
                    f"(sender={sorted(provenance[method])}): {resp!r}"
                )
                continue
            error = resp.get("error")
            if error is not None and error.get("code") == -32601:
                failures.append(
                    f"{method}: unreachable (-32601 unknown method); sent by "
                    f"{sorted(provenance[method])} but no dispatch handler "
                    f"answers it"
                )
        return failures

    failures = asyncio.run(_with_socket_server(short_sock_path, store, _drive_all))
    assert not failures, "unreachable client-sent method(s):\n" + "\n".join(failures)
