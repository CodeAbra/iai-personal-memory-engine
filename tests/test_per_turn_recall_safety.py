"""Hermetic safety coverage for the per-turn-recall hook: bounded stdin,
strict session-id parsing, additive socket-path selection, symlink-safe
cache/state reads, a flock-protected ledger, and socket-ownership/deadline/
render-timeout guards -- all fail-open, none of them reordering or removing
the working-tier/live-state/agent-registry/directives emitters or their
double-fire suppression gate.

Never runs against the owner's live store or a real daemon socket -- every
fixture here builds a throwaway HOME with fake cache files and, where a
socket is needed, a local AF_UNIX listener bound under the test's own
temp directory.
"""
from __future__ import annotations

import ast
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_HOOKS = Path(__file__).resolve().parents[1] / "src/iai_mcp/_deploy/hooks"
_HOOK = _HOOKS / "iai-mcp-per-turn-recall.sh"


@dataclass(frozen=True)
class _HookRuntime:
    root: Path
    home: Path
    store: Path
    bin: Path
    hook: Path
    socket: Path


@pytest.fixture
def _hook_runtime():
    # Short root: AF_UNIX socket paths have a ~104 byte OS limit, well below
    # what pytest's deeply nested default tmp_path produces.
    root = Path(tempfile.mkdtemp(prefix="iai-safe-", dir="/tmp"))
    runtime = _HookRuntime(
        root, root / "home", root / "store", root / "bin",
        root / "hooks" / _HOOK.name, root / "recall.sock",
    )
    try:
        for path in (runtime.home, runtime.store, runtime.bin, runtime.hook.parent):
            path.mkdir(parents=True, mode=0o700)
        for name in ("head", "sed", "uname", "stat", "date", "tr", "cut", "dirname",
                     "wc", "mkdir", "tail", "mv", "base64", "bash", "cat", "mktemp", "rm"):
            executable = shutil.which(name)
            assert executable is not None, f"required coreutil missing: {name}"
            (runtime.bin / name).symlink_to(executable)
        (runtime.bin / "python3").symlink_to(sys.executable)
        for name in (_HOOK.name, "_recall_render.py"):
            shutil.copyfile(_HOOKS / name, runtime.hook.parent / name)
        yield runtime
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _hook_env(runtime: _HookRuntime, **overrides: str) -> dict[str, str]:
    env = {
        "HOME": str(runtime.home),
        "IAI_MCP_STORE": str(runtime.store),
        "PATH": str(runtime.bin),
        "IAI_MCP_PER_TURN_SOCKET_ACCEL": "0",
        "TMPDIR": str(runtime.root),
    }
    env.update(overrides)
    return env


def _run_hook(
    runtime: _HookRuntime, payload: bytes, *, env: dict[str, str] | None = None,
    hold_open: bool = False, timeout: float = 8,
) -> subprocess.CompletedProcess[bytes]:
    child = subprocess.Popen(
        ["/bin/bash", str(runtime.hook)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(runtime.root),
        env=_hook_env(runtime) if env is None else env, start_new_session=True,
    )
    try:
        if hold_open:
            assert child.stdin is not None
            writer = child.stdin

            def _feed() -> None:
                # The hook's bounded reader may stop draining before every
                # byte lands (that is the cap doing its job) -- a resulting
                # broken pipe here is an expected outcome, not a test error.
                try:
                    writer.write(payload)
                    writer.flush()
                except (BrokenPipeError, OSError):
                    pass

            feeder = threading.Thread(target=_feed, daemon=True)
            feeder.start()
            # communicate() must not close this writer to manufacture an EOF.
            child.stdin = None
            stdout, stderr = child.communicate(timeout=timeout)
            feeder.join(timeout=3)
            try:
                writer.close()
            except OSError:
                pass
        else:
            stdout, stderr = child.communicate(payload, timeout=timeout)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    return subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)


class _RecallPeer:
    """Minimal fake daemon: accept once, read to newline, send a fixed reply."""

    def __init__(self, path: Path, reply: bytes) -> None:
        self.path, self.reply = path, reply
        self.requests: list[dict] = []

    def __enter__(self) -> "_RecallPeer":
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(1)
        self.server.settimeout(5)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.thread.join(timeout=5)
        self.server.close()

    def _serve(self) -> None:
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            buf = bytearray()
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
            if buf:
                self.requests.append(json.loads(buf))
            try:
                conn.sendall(self.reply)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Bounded stdin: a host that never closes stdin must not hang the turn.
# ---------------------------------------------------------------------------

def test_oversized_stdin_is_capped_and_turn_completes(_hook_runtime: _HookRuntime) -> None:
    payload = b'{"session_id":"sess-cap","prompt":"' + b"x" * 200_000 + b'"}'
    started = time.monotonic()
    result = _run_hook(_hook_runtime, payload, hold_open=True)
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert elapsed < 5, "capped read must not wait on data past the byte limit"


def test_stalled_stdin_completes_under_the_read_deadline(_hook_runtime: _HookRuntime) -> None:
    payload = b'{"session_id":"sess-stall","promp'  # deliberately incomplete
    started = time.monotonic()
    result = _run_hook(_hook_runtime, payload, hold_open=True)
    elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert elapsed < 5, "a stalled host must not hang the turn past the read deadline"


def test_stalled_stdin_still_renders_unrelated_emitters(_hook_runtime: _HookRuntime) -> None:
    (_hook_runtime.store / ".directives.cached.md").write_text(
        "always answer in English\n", encoding="utf-8",
    )
    result = _run_hook(
        _hook_runtime, b'{"session_id":"sess-stall2","promp', hold_open=True,
    )
    assert result.returncode == 0
    assert b"<iai-mcp-directives>" in result.stdout, (
        "a malformed/incomplete stdin must not suppress session-independent emitters"
    )


# ---------------------------------------------------------------------------
# Strict JSON parse: a duplicate session_id key is malformed, not last-wins.
# ---------------------------------------------------------------------------

def test_duplicate_session_id_key_is_rejected_not_last_wins(
    _hook_runtime: _HookRuntime,
) -> None:
    (_hook_runtime.store / ".working-tier.sess-first.cached.md").write_text(
        "goal: first claimant\nnext action: do the first thing\n", encoding="utf-8",
    )
    (_hook_runtime.store / ".working-tier.sess-second.cached.md").write_text(
        "goal: second claimant\nnext action: do the second thing\n", encoding="utf-8",
    )
    payload = b'{"session_id":"sess-first","prompt":"hi","session_id":"sess-second"}'
    result = _run_hook(_hook_runtime, payload)
    assert result.returncode == 0
    assert b"<iai-mcp-working-tier>" not in result.stdout, (
        "a duplicate-key payload must degrade to no-incoming-session, "
        "never silently resolve to either claimant's snapshot"
    )


def test_well_formed_session_id_still_resolves(_hook_runtime: _HookRuntime) -> None:
    (_hook_runtime.store / ".working-tier.sess-ok.cached.md").write_text(
        "goal: legitimate task\nnext action: keep going\n", encoding="utf-8",
    )
    result = _run_hook(_hook_runtime, b'{"session_id":"sess-ok","prompt":"hi"}')
    assert result.returncode == 0
    assert b"<iai-mcp-working-tier>" in result.stdout
    assert b"legitimate task" in result.stdout


def test_absent_session_id_stays_silent_on_scoped_emitters(
    _hook_runtime: _HookRuntime,
) -> None:
    (_hook_runtime.store / ".working-tier.sess-any.cached.md").write_text(
        "goal: unreachable\nnext action: unreachable\n", encoding="utf-8",
    )
    result = _run_hook(_hook_runtime, b'{"prompt":"hi"}')
    assert result.returncode == 0
    assert b"<iai-mcp-working-tier>" not in result.stdout


# ---------------------------------------------------------------------------
# Socket-path selection: additive override, current default when unset.
# ---------------------------------------------------------------------------

def test_iai_daemon_socket_path_override_is_honored(_hook_runtime: _HookRuntime) -> None:
    custom_sock = _hook_runtime.root / "custom.sock"
    default_sock = _hook_runtime.store / ".daemon.sock"
    with _RecallPeer(custom_sock, b'{"result":{"hits":[{"text":"alice waters the garden"}]}}\n') as peer:
        env = _hook_env(
            _hook_runtime, IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
            IAI_DAEMON_SOCKET_PATH=str(custom_sock),
        )
        result = _run_hook(_hook_runtime, b'{"prompt":"alice garden"}', env=env)
        assert result.returncode == 0
        assert peer.requests, "the override path must receive the live-recall connection"
    assert not default_sock.exists(), "the default socket path must never be created/used"
    assert b"alice waters the garden" in result.stdout


def test_absent_override_falls_back_to_current_default(_hook_runtime: _HookRuntime) -> None:
    default_sock = _hook_runtime.store / ".daemon.sock"
    with _RecallPeer(default_sock, b'{"result":{"hits":[{"text":"alice waters the garden"}]}}\n') as peer:
        env = _hook_env(_hook_runtime, IAI_MCP_PER_TURN_SOCKET_ACCEL="1")
        env.pop("IAI_DAEMON_SOCKET_PATH", None)
        result = _run_hook(_hook_runtime, b'{"prompt":"alice garden"}', env=env)
        assert result.returncode == 0
        assert peer.requests, "an unset override must still reach the default socket"
    assert b"alice waters the garden" in result.stdout


# ---------------------------------------------------------------------------
# Structural fence: the pre-existing emitters and double-fire gate stay
# defined and ordered exactly as before this hardening pass.
# ---------------------------------------------------------------------------

def test_pre_existing_emitters_and_gate_are_still_defined_in_order() -> None:
    source = _HOOK.read_text(encoding="utf-8")
    required_emitters = [
        "emit_live_state", "emit_live_state_fallback",
        "emit_agent_registry", "emit_directives",
    ]
    for name in required_emitters:
        assert f"{name}() {{" in source, f"{name} must stay defined"
    assert "_WORKING_TIER_EMITTED" in source

    call_order = [
        "emit_foresight", "emit_working_tier", "emit_live_state",
        "emit_live_state_fallback", "emit_agent_registry", "emit_directives",
        "emit_socket_recall",
    ]
    positions = [source.index(f"\n{name}\n") for name in call_order]
    assert positions == sorted(positions), (
        "the emitter call sequence must stay in its original order"
    )


def test_at_most_two_python3_invocations_per_run() -> None:
    # Interpreter-invocation convention: exactly two spawn sites per run --
    # the validation preamble and emit_socket_recall's own render call.
    # Later guards must reuse one of these two, never add a third spawn site.
    source = _HOOK.read_text(encoding="utf-8")
    assert source.count('python3 -c "$_STDIN_PREAMBLE"') == 1
    assert source.count("python3 - <<'PYEOF'") == 1


# ---------------------------------------------------------------------------
# Positive co-existence: foresight and working-tier fire together in the
# same turn, and live-state does not double-fire underneath the
# working-tier block that already carried the same content.
# ---------------------------------------------------------------------------

def test_working_tier_and_foresight_coexist_without_live_state_double_fire(
    _hook_runtime: _HookRuntime,
) -> None:
    sid = "sess-coexist"
    (_hook_runtime.store / f".working-tier.{sid}.cached.md").write_text(
        "goal: shared focal goal\nnext action: a real next step\n", encoding="utf-8",
    )
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: the agent will need the deploy checklist next\n",
        encoding="utf-8",
    )
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    out = result.stdout
    assert out.count(b"<iai-mcp-working-tier>") == 1
    assert out.count(b"<iai-mcp-foresight>") == 1
    assert out.count(b"<iai-mcp-live-state>") == 0, (
        "emit_live_state must stay suppressed once emit_working_tier already "
        "carried this turn's live state, even while foresight independently fires"
    )


# ---------------------------------------------------------------------------
# Symlink-safe cache/state reads: a symlinked cache/state path is rejected,
# not followed, and the turn still renders (other emitters unaffected).
# ---------------------------------------------------------------------------

def test_symlinked_working_tier_cache_is_rejected(_hook_runtime: _HookRuntime) -> None:
    secret = _hook_runtime.root / "secret.md"
    secret.write_text(
        "goal: attacker planted secret\nnext action: leak me\n", encoding="utf-8",
    )
    (_hook_runtime.store / ".working-tier.sess-sym.cached.md").symlink_to(secret)
    result = _run_hook(_hook_runtime, b'{"session_id":"sess-sym","prompt":"hi"}')
    assert result.returncode == 0
    assert b"attacker planted secret" not in result.stdout
    assert b"<iai-mcp-working-tier>" not in result.stdout
    assert b"<iai-mcp-live-state>" not in result.stdout


def test_symlinked_continuity_cache_is_rejected(_hook_runtime: _HookRuntime) -> None:
    secret = _hook_runtime.root / "secret-continuity.md"
    secret.write_text(
        "<iai-mcp-agent-registry>\nplanted agent\n</iai-mcp-agent-registry>\n",
        encoding="utf-8",
    )
    (_hook_runtime.store / ".session-continuity.cached.md").symlink_to(secret)
    result = _run_hook(_hook_runtime, b'{"prompt":"hi"}')
    assert result.returncode == 0
    assert b"planted agent" not in result.stdout


def test_symlinked_foresight_pack_is_rejected(_hook_runtime: _HookRuntime) -> None:
    sid = "sess-fsym"
    secret = _hook_runtime.root / "secret-pack.md"
    secret.write_text("anticipated: attacker content\n", encoding="utf-8")
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").symlink_to(secret)
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    assert b"attacker content" not in result.stdout
    assert b"<iai-mcp-foresight>" not in result.stdout


def test_symlinked_foresight_state_is_rejected(_hook_runtime: _HookRuntime) -> None:
    # A symlinked .state.json must not let a spoofed session_id authorize a
    # pack that would otherwise be session-scope-blocked.
    sid = "sess-fssym"
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: real pack content\n", encoding="utf-8",
    )
    spoofed_state = _hook_runtime.root / "spoofed-state.json"
    spoofed_state.write_text(f'{{"session_id":"{sid}"}}', encoding="utf-8")
    (_hook_runtime.store / f".next-turn-pack.{sid}.state.json").symlink_to(spoofed_state)
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    # A rejected (absent) state file is "unknown" -> session_scope_blocks
    # fails open, so the real pack still renders; the point under test is
    # that the SYMLINK ITSELF was never opened/followed (no crash, no hang).
    assert b"anticipated: real pack content" in result.stdout


def test_symlink_rejection_does_not_block_other_emitters(
    _hook_runtime: _HookRuntime,
) -> None:
    secret = _hook_runtime.root / "secret.md"
    secret.write_text("goal: attacker\nnext action: leak\n", encoding="utf-8")
    (_hook_runtime.store / ".working-tier.sess-sym2.cached.md").symlink_to(secret)
    (_hook_runtime.store / ".directives.cached.md").write_text(
        "always answer in English\n", encoding="utf-8",
    )
    result = _run_hook(_hook_runtime, b'{"session_id":"sess-sym2","prompt":"hi"}')
    assert result.returncode == 0
    assert b"<iai-mcp-directives>" in result.stdout


# ---------------------------------------------------------------------------
# Unit-level coverage of the embedded safe-read helper: post-open fstat
# rejects a non-regular descriptor (the TOCTOU-class case O_NOFOLLOW alone
# does not cover), and O_NOFOLLOW being unavailable fails closed even for an
# otherwise legitimate regular file.
# ---------------------------------------------------------------------------

def _load_embedded_helpers(*names: str) -> dict:
    source = _HOOK.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", source, re.S)
    definitions = [
        node for block in blocks for node in ast.parse(block).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    namespace = dict(fcntl=fcntl, json=json, os=os, re=re,
                      stat=stat, sys=sys, time=time)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "<hook helpers>", "exec"), namespace)
    return namespace


def test_fifo_descriptor_is_rejected_by_post_open_fstat(tmp_path: Path) -> None:
    helpers = _load_embedded_helpers("_read_regular")
    fifo_path = tmp_path / "sneaky.md"
    os.mkfifo(fifo_path)
    data, mtime = helpers["_read_regular"](str(fifo_path), 4096)
    assert data == b""
    assert mtime == 0


def test_o_nofollow_unavailable_fails_closed_for_a_legitimate_file(
    tmp_path: Path,
) -> None:
    helpers = _load_embedded_helpers("_read_regular")
    target = tmp_path / "real.md"
    target.write_text("goal: real\n", encoding="utf-8")

    class _NoNofollowOS:
        def __getattr__(self, name: str) -> object:
            if name == "O_NOFOLLOW":
                raise AttributeError(name)
            return getattr(os, name)

    helpers["os"] = _NoNofollowOS()
    data, mtime = helpers["_read_regular"](str(target), 4096)
    assert data == b""
    assert mtime == 0


# ---------------------------------------------------------------------------
# Foresight-served ledger: flock-protected, size-capped, never symlink-followed.
# ---------------------------------------------------------------------------

def test_foresight_ledger_is_written_when_pack_served(
    _hook_runtime: _HookRuntime,
) -> None:
    sid = "sess-ledger"
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: something\n", encoding="utf-8",
    )
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    ledger = _hook_runtime.store / "logs" / "foresight-served.jsonl"
    assert ledger.is_file()
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["bytes"] > 0
    assert record["channel"] == "settings"


def test_foresight_ledger_rotates_past_4000_lines(_hook_runtime: _HookRuntime) -> None:
    sid = "sess-rotate"
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: something\n", encoding="utf-8",
    )
    logs_dir = _hook_runtime.store / "logs"
    logs_dir.mkdir()
    ledger = logs_dir / "foresight-served.jsonl"
    with ledger.open("w", encoding="utf-8") as fh:
        for i in range(4000):
            fh.write(json.dumps({"ts": "old", "bytes": i, "channel": "settings"}) + "\n")
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2000, "rotation must keep the newest 2000, including the new record"
    assert json.loads(lines[-1])["bytes"] > 0


def test_foresight_ledger_survives_concurrent_writers(
    _hook_runtime: _HookRuntime,
) -> None:
    packs = [f"sess-conc-{i}" for i in range(5)]
    for sid in packs:
        (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
            f"anticipated: task {sid}\n", encoding="utf-8",
        )

    results: list[subprocess.CompletedProcess[bytes]] = []
    lock = threading.Lock()

    def _fire(sid: str) -> None:
        result = _run_hook(
            _hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode(),
        )
        with lock:
            results.append(result)

    threads = [threading.Thread(target=_fire, args=(sid,)) for sid in packs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == len(packs)
    assert all(r.returncode == 0 for r in results)
    ledger = _hook_runtime.store / "logs" / "foresight-served.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(packs), "flock must serialize writes, not lose or merge any"
    for line in lines:
        json.loads(line)  # every line parses cleanly -- no interleaved partial writes


def test_held_ledger_lock_does_not_block_the_turn_past_the_deadline(
    _hook_runtime: _HookRuntime,
) -> None:
    sid = "sess-ledger-locked"
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: something\n", encoding="utf-8",
    )
    logs_dir = _hook_runtime.store / "logs"
    logs_dir.mkdir()
    ledger_path = logs_dir / "foresight-served.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    holder = os.open(str(ledger_path), os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        started = time.monotonic()
        result = _run_hook(
            _hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode(),
        )
        elapsed = time.monotonic() - started
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)

    assert result.returncode == 0
    assert elapsed < 3, "a lock held by another writer must not block the turn past the deadline"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert lines == [], "the ledger write must be skipped, not retried past the deadline"


def test_symlinked_ledger_path_is_not_followed(_hook_runtime: _HookRuntime) -> None:
    sid = "sess-ledger-sym"
    (_hook_runtime.store / f".next-turn-pack.{sid}.cached.md").write_text(
        "anticipated: something\n", encoding="utf-8",
    )
    logs_dir = _hook_runtime.store / "logs"
    logs_dir.mkdir()
    canary = _hook_runtime.root / "canary.jsonl"
    canary.write_text("untouched\n", encoding="utf-8")
    (logs_dir / "foresight-served.jsonl").symlink_to(canary)
    result = _run_hook(_hook_runtime, f'{{"session_id":"{sid}","prompt":"hi"}}'.encode())
    assert result.returncode == 0
    assert canary.read_text(encoding="utf-8") == "untouched\n"


# ---------------------------------------------------------------------------
# Socket-ownership check: only a real AF_UNIX socket owned by the current
# euid, with neither the socket nor its parent group/other-writable, is
# ever connected to. Every rejection degrades emit_socket_recall silently.
# ---------------------------------------------------------------------------

def test_safe_socket_path_rejects_missing_or_non_socket(tmp_path: Path) -> None:
    helpers = _load_embedded_helpers("_safe_socket_path")
    missing = tmp_path / "missing.sock"
    ordinary_file = tmp_path / "ordinary.sock"
    ordinary_file.write_text("not a socket", encoding="utf-8")
    assert not helpers["_safe_socket_path"](str(missing))
    assert not helpers["_safe_socket_path"](str(ordinary_file))


@pytest.mark.parametrize("foreign_target", ["socket", "parent"])
def test_safe_socket_path_rejects_foreign_owner(foreign_target: str) -> None:
    helpers = _load_embedded_helpers("_safe_socket_path")
    current_uid = 101
    foreign_uid = 202
    socket_stat = SimpleNamespace(
        st_mode=stat.S_IFSOCK | 0o600,
        st_uid=foreign_uid if foreign_target == "socket" else current_uid,
    )
    parent_stat = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_uid=foreign_uid if foreign_target == "parent" else current_uid,
    )
    helpers["os"] = SimpleNamespace(
        stat=lambda path: socket_stat if path == "alice.sock" else parent_stat,
        path=os.path,
        geteuid=lambda: current_uid,
    )
    assert not helpers["_safe_socket_path"]("alice.sock")


@pytest.mark.parametrize(("target", "write_bit"), [
    ("socket", stat.S_IWGRP), ("socket", stat.S_IWOTH),
    ("parent", stat.S_IWGRP), ("parent", stat.S_IWOTH),
])
def test_unsafe_write_bits_skip_live_socket_contact(
    _hook_runtime: _HookRuntime, target: str, write_bit: int,
) -> None:
    with _RecallPeer(_hook_runtime.socket, b'{"result":{"hits":[{"text":"alice"}]}}\n') as peer:
        if target == "socket":
            os.chmod(_hook_runtime.socket, 0o600 | write_bit)
        else:
            os.chmod(_hook_runtime.root, 0o700 | write_bit)
        try:
            env = _hook_env(
                _hook_runtime, IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
                IAI_DAEMON_SOCKET_PATH=str(_hook_runtime.socket),
            )
            result = _run_hook(_hook_runtime, b'{"prompt":"alice garden"}', env=env)
        finally:
            os.chmod(_hook_runtime.root, 0o700)
        assert result.returncode == 0
        assert not peer.requests, "an unsafe socket/parent must never be connected to"
    assert result.stdout == b""


def test_unsafe_socket_does_not_block_other_emitters(_hook_runtime: _HookRuntime) -> None:
    (_hook_runtime.store / ".directives.cached.md").write_text(
        "always answer in English\n", encoding="utf-8",
    )
    with _RecallPeer(_hook_runtime.socket, b'{"result":{"hits":[{"text":"alice"}]}}\n') as peer:
        os.chmod(_hook_runtime.socket, 0o666)
        try:
            env = _hook_env(
                _hook_runtime, IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
                IAI_DAEMON_SOCKET_PATH=str(_hook_runtime.socket),
            )
            result = _run_hook(_hook_runtime, b'{"prompt":"hi"}', env=env)
        finally:
            os.chmod(_hook_runtime.socket, 0o600)
        assert not peer.requests
    assert result.returncode == 0
    assert b"<iai-mcp-directives>" in result.stdout
    assert b"alice" not in result.stdout


# ---------------------------------------------------------------------------
# Single wall-clock deadline spans connect+send+recv: a peer that accepts
# the connection and reads the request but never replies must not hang the
# turn past the configured socket timeout.
# ---------------------------------------------------------------------------

class _StallPeer:
    """Accepts one connection, reads the request, then replies never."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "_StallPeer":
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.thread.join(timeout=5)
        self.server.close()

    def _serve(self) -> None:
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk or b"\n" in chunk:
                        break
            except OSError:
                pass
            time.sleep(3)  # never sends a reply


def test_ipc_deadline_covers_the_full_exchange_not_just_connect(
    _hook_runtime: _HookRuntime,
) -> None:
    with _StallPeer(_hook_runtime.socket):
        env = _hook_env(
            _hook_runtime, IAI_MCP_PER_TURN_SOCKET_ACCEL="1",
            IAI_DAEMON_SOCKET_PATH=str(_hook_runtime.socket),
            IAI_MCP_RECALL_SOCKET_TIMEOUT="0.3",
        )
        started = time.monotonic()
        result = _run_hook(_hook_runtime, b'{"prompt":"alice garden"}', env=env, timeout=8)
        elapsed = time.monotonic() - started
    assert result.returncode == 0
    assert elapsed < 3, "a stalled peer must not block past the socket deadline"
    assert result.stdout == b""


# ---------------------------------------------------------------------------
# Render timeout: a slow/huge renderer cannot block the turn, and the
# SIGALRM handler/timer are restored even after firing.
# ---------------------------------------------------------------------------

# This test asserts ITIMER_REAL is disarmed after the guard, so it must not run
# under pytest-timeout's signal method (which itself arms ITIMER_REAL).
@pytest.mark.timeout(method="thread")
def test_render_timeout_prevents_a_hanging_renderer_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal as signal_module

    helpers = _load_embedded_helpers("_render_bounded")
    helpers["signal"] = signal_module
    previous_handler = signal_module.getsignal(signal_module.SIGALRM)

    def _slow_render(result: object) -> str:
        time.sleep(5)
        return "unreachable"

    monkeypatch.setitem(
        sys.modules, "_recall_render", SimpleNamespace(render_recall_block=_slow_render),
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        helpers["_render_bounded"]({"hits": []}, "")
    elapsed = time.monotonic() - started

    assert elapsed < 2, "SIGALRM must preempt a hanging renderer well under its own sleep"
    assert signal_module.getsignal(signal_module.SIGALRM) == previous_handler
    remaining_delay, _ = signal_module.getitimer(signal_module.ITIMER_REAL)
    assert remaining_delay == 0.0, "the itimer must be disarmed after the guard exits"
