from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import json
import logging
import math
import os

# resource exists only on POSIX; used solely for the fd-limit raise, which is
# a no-op on Windows.
try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

from iai_mcp import s4
from iai_mcp.concurrency import serve_control_socket  # noqa: F401 -- re-exported here for the test suite; the function lives in concurrency.py
from iai_mcp.daemon_state import load_state, save_state, update_state
from iai_mcp.dream import run_rem_cycle
from iai_mcp.events import (
    CRISIS_MODE_AUTO_EXPIRED,
    DAEMON_MEMORY_PRESSURE_KILL,
    DAEMON_SLEEP_CYCLE_STALE,
    DAEMON_WATCHDOG_NEEDS_OPERATOR,
    DAEMON_WEDGE_KILL,
    write_event,
)
from iai_mcp.identity_audit import continuous_audit
from iai_mcp.quiet_window import (
    BUCKET_COUNT,
    BUCKET_MINUTES,
    effective_consolidation_window,
    learn_quiet_window_from_presence,
    prune_presence_masks,
    should_relearn,
    stamp_presence_mask,
    within_window,
)
from iai_mcp.hippo import AccessMode
from iai_mcp.lock_protocol import cleanup_stale_consolidation_intent
from iai_mcp.native_guard import _require_native
from iai_mcp.sleep_wal import SleepWAL
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore


STATE_WAKE: str = "WAKE"
STATE_TRANSITIONING: str = "TRANSITIONING"
STATE_SLEEP: str = "SLEEP"
STATE_DREAMING: str = "DREAMING"

VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_WAKE: {STATE_TRANSITIONING},
    STATE_TRANSITIONING: {STATE_SLEEP, STATE_WAKE},
    STATE_SLEEP: {STATE_DREAMING, STATE_WAKE},
    STATE_DREAMING: {STATE_SLEEP},
}

TICK_INTERVAL_SEC: int = 30

DEFAULT_CYCLE_COUNT: int = 4

S4_OFFLINE_INTERVAL_SEC: int = 60 * 60

S4_FIRST_ITER_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_S4_FIRST_ITER_GRACE_SEC", str(S4_OFFLINE_INTERVAL_SEC)),
)

SESSION_START_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"
from iai_mcp.session import SESSION_START_CACHE_MAX_CHARS  # noqa: E402 -- placed after PATH constant for readability

INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC: float = 30.0

#: Cooldown after a failed sleep-pipeline run before the next attempt. Each
#: run escalates the store lock to EXCLUSIVE; without a cooldown a persistent
#: step failure re-escalates every tick and starves SHARED clients all night.
#: An explicit force-rem bypasses the cooldown (operator intent wins).
SLEEP_FAIL_BACKOFF_SEC: float = float(
    os.environ.get("IAI_MCP_SLEEP_FAIL_BACKOFF_SEC", "600")
)


def _hippo_health_check_on_boot(store) -> dict[str, int | str]:
    try:
        db = store.db
        sqlite_count_row = db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
        sqlite_count = int(sqlite_count_row[0]) if sqlite_count_row else 0
    except Exception as exc:
        return {
            "sqlite_count": -1,
            "hnsw_active_count": -1,
            "hnsw_raw_count": -1,
            "action": "sqlite_count_failed",
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    try:
        active_label_count = int(len(db._label_map))
    except Exception:
        active_label_count = -1
    try:
        hnsw_raw_count = int(db._hnsw.get_current_count())
    except Exception:
        hnsw_raw_count = -1
    action = (
        "ok"
        if sqlite_count == active_label_count == hnsw_raw_count
        else "divergence_at_boot"
    )
    return {
        "sqlite_count": sqlite_count,
        "hnsw_active_count": active_label_count,
        "hnsw_raw_count": hnsw_raw_count,
        "action": action,
    }


_DAEMON_NOFILE_FLOOR_DEFAULT: int = 8192


def _raise_fd_limit() -> None:
    if resource is None:
        return
    try:
        floor = int(
            os.environ.get("IAI_MCP_DAEMON_NOFILE_FLOOR", _DAEMON_NOFILE_FLOOR_DEFAULT)
        )
    except (TypeError, ValueError):
        floor = _DAEMON_NOFILE_FLOOR_DEFAULT

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return

    effective_hard = hard if hard != resource.RLIM_INFINITY else floor

    target = min(max(soft, floor), effective_hard)
    if target <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        log.debug("daemon_fd_limit_raised soft=%d->%d hard=%d", soft, target, hard)
    except (OSError, ValueError) as exc:
        log.debug("daemon_fd_limit_raise failed (non-fatal): %s", exc)


def _should_drain_on_drowsy_edge(prev, current) -> bool:
    from iai_mcp.lifecycle_state import LifecycleState as _L
    return prev is _L.WAKE and current is _L.DROWSY


#: Post-read courtesy tail for the sleep pipeline: a foreground read that
#: finished this recently still defers the next chunk (a turn usually issues
#: several reads back to back). Kept short on purpose — pipeline steps now
#: yield between ~1-2s chunks, so a read arriving mid-grind waits at most one
#: chunk; a long tail here only starves consolidation on a busy machine.
SLEEP_FOREGROUND_RECENT_WINDOW_SEC: float = 5.0


def _sleep_interrupt_predicate() -> bool:
    # Module-level so it is directly unit-testable without assembling the
    # daemon. Sleep yields to the USER: a foreground read in flight, or one
    # stamped within the courtesy tail. Ambient socket traffic (capture
    # hooks, refresh polls, status probes) never defers a chunk.
    from iai_mcp.concurrency import foreground_recent

    return foreground_recent(SLEEP_FOREGROUND_RECENT_WINDOW_SEC)


def _sleep_pipeline_gate(daemon_state: dict | None) -> bool:
    # Module-level so it is directly unit-testable without assembling the
    # daemon. Returns False (pipeline must NOT run) when scheduler_paused is
    # truthy; True otherwise, including when daemon_state is None/missing the
    # key -- fail-open to the pre-existing always-run behavior.
    #
    # An explicit force-rem is an intentional operator request and overrides
    # the pause gate: a manual `iai-mcp daemon force-rem` must actually run
    # the consolidation pipeline, never be silently swallowed by a standing
    # pause. (Idle-driven auto-sleep, which sets no force_rem_request, still
    # obeys the pause.)
    ds = daemon_state or {}
    if bool((ds.get("force_rem_request") or {}).get("pending")):
        return True
    return not bool(ds.get("scheduler_paused"))


#: A live external request this recent counts as demand to leave
#: HIBERNATION: wide enough to bridge two lifecycle ticks, narrow enough
#: that a long-gone client does not keep resurrecting the engine.
HIBERNATION_DEMAND_WINDOW_SEC: float = 90.0


def _hibernation_demand_reason(
    wrapper_heartbeat_active: bool,
    socket_last_activity_mono: float,
    boot_activity_mono: float,
    now_mono: float,
    window_sec: float = HIBERNATION_DEMAND_WINDOW_SEC,
) -> str | None:
    # Module-level so it is directly unit-testable without assembling the
    # daemon. A booted process that finds persisted HIBERNATION must not
    # shut down while a live session demands the engine — the wake-signal
    # file is not the only admissible evidence of demand.
    # last_activity_ts initializes to construction time, so a bare recency
    # check would read every boot as demand: demand requires an external op
    # strictly after boot.
    if (
        socket_last_activity_mono > boot_activity_mono
        and (now_mono - socket_last_activity_mono) < window_sec
    ):
        return "wake_on_socket_request_in_hibernation"
    if wrapper_heartbeat_active:
        return "wake_on_live_wrapper_heartbeat"
    return None


def _sleep_backoff_active(
    daemon_state: dict | None, backoff_until: float, now: float
) -> bool:
    # Module-level so it is directly unit-testable without assembling the
    # daemon. force-rem is explicit operator intent and always wins over the
    # failure cooldown.
    if bool(((daemon_state or {}).get("force_rem_request") or {}).get("pending")):
        return False
    return now < backoff_until


def _sleep_cycle_cooldown_sec() -> float:
    try:
        return float(os.environ.get("IAI_MCP_SLEEP_CYCLE_COOLDOWN_SEC", "14400"))
    except ValueError:
        return 14400.0


def _sleep_cooldown_active(
    daemon_state: "dict | None",
    last_clean_cycle_mono: float,
    now: float,
    last_clean_cycle_wall: float = 0.0,
) -> bool:
    # Module-level so it is directly unit-testable without assembling the
    # daemon. A completed CLEAN cycle opens a refractory window: consolidation
    # already ran to completion, and re-running it back-to-back only taxes the
    # awake path — an active user blocks the hibernation exit, so without
    # this gate the pipeline loops continuously for as long as the user works
    # (observed: 117 cycles in one night, the awake socket starved to
    # timeout). force-rem is explicit operator intent and always wins — and
    # because the tick CONSUMES the pending flag (stamping honored_at) before
    # this gate runs, an honored force must keep the gate open until ITS
    # cycle completes (a clean completion newer than honored_at closes it, or
    # the forced cycle would re-open the continuous loop for the whole
    # honored-recency window).
    for req_key in ("force_rem_request", "user_sleep_request"):
        req = (daemon_state or {}).get(req_key) or {}
        if bool(req.get("pending")):
            return False
        honored_raw = req.get("honored_at")
        if not honored_raw:
            continue
        try:
            from datetime import datetime, timedelta, timezone as _tz

            honored = datetime.fromisoformat(str(honored_raw))
            if honored.tzinfo is None:
                honored = honored.replace(tzinfo=_tz.utc)
            recent = (datetime.now(_tz.utc) - honored) <= timedelta(hours=2)
            served = last_clean_cycle_wall > honored.timestamp()
            if recent and not served:
                return False
        except (ValueError, TypeError):
            pass
    if last_clean_cycle_mono <= 0.0:
        return False
    return (now - last_clean_cycle_mono) < _sleep_cycle_cooldown_sec()


#: Heartbeat-idle age past which a tick may consider SLEEP entry.
SLEEP_HEARTBEAT_IDLE_SEC: float = float(
    os.environ.get("LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC", "1800")
)

#: Minutes of continuous OS-level input idle (HID/logind) past which a
#: starved daemon may consolidate outside the window; <=0 disables.
DEEP_IDLE_OVERRIDE_ENV = "IAI_MCP_DEEP_IDLE_OVERRIDE_MIN"

#: The override arms only after this long without a clean sleep cycle —
#: it is a starvation backstop, not an everyday entry path.
DEEP_IDLE_STARVATION_SEC = 48 * 3600.0


def _deep_idle_override(
    os_idle_sec: "float | None", last_clean_cycle_wall: float
) -> bool:
    # os_idle_sec must be the OS-level input idle: heartbeat idle only says
    # no Claude session is running, which is true on most working machines
    # most of the day and must never put the pipeline on a machine in use.
    if os_idle_sec is None:
        return False
    raw = os.environ.get(DEEP_IDLE_OVERRIDE_ENV, "")
    try:
        minutes = float(raw) if raw else 90.0
    except (TypeError, ValueError):
        minutes = 90.0
    if not math.isfinite(minutes):
        minutes = 90.0
    if minutes <= 0:
        return False
    # Floor at the sleep_eligible HID threshold: an override armed below it
    # would suppress the wrapper scanner while sleep stays structurally
    # ineligible — a DROWSY dead end with every exit dropped.
    minutes = max(minutes, 30.0)
    # Unknown last-cycle time reads as NOT starved: the override must never
    # fire on ignorance, only on a demonstrated 48h gap.
    if last_clean_cycle_wall <= 0.0:
        return False
    if (time.time() - last_clean_cycle_wall) < DEEP_IDLE_STARVATION_SEC:
        return False
    return float(os_idle_sec) >= minutes * 60.0


#: Minutes of continuous OS-level input idle past which the nightly window
#: admits SLEEP while agent sessions keep the wrapper heartbeat fresh;
#: <=0 disables. Night-only: this arm never fires outside the effective
#: consolidation window.
WINDOW_DEEP_IDLE_ENV = "IAI_MCP_WINDOW_DEEP_IDLE_MIN"


def _window_membership_core(daemon_state: "dict | None") -> bool:
    """Membership in the effective consolidation window.

    The ONE holder of the window-key contract shared by the fail-open and
    fail-closed predicates; raises on a gate error so each caller applies
    its own failure contract.
    """
    if daemon_state is None:
        from iai_mcp.daemon_state import load_state as _gate_load
        daemon_state = _gate_load()
    if not isinstance(daemon_state, dict):
        raise TypeError("daemon_state must be a dict")
    window = effective_consolidation_window(
        daemon_state.get("quiet_window"),
        manual=daemon_state.get("quiet_window_manual_override"),
    )
    local = datetime.now().astimezone()
    return within_window(window, local, local.tzinfo)


def _window_membership(daemon_state: "dict | None") -> bool:
    """Window membership, fail-CLOSED.

    Unlike _in_consolidation_window this carries no force-request clauses
    and reads any gate error as "outside": its callers suppress the live
    wrapper scanner, and a broken gate must never suppress it in daytime.
    """
    try:
        return _window_membership_core(daemon_state)
    except Exception:  # noqa: BLE001 -- fail-closed, see docstring
        return False


def _force_consolidation_active(daemon_state: "dict | None") -> bool:
    """An explicit force admits consolidation: a pending request, or one
    honored within the last two hours so an in-flight forced cycle is
    never cut off mid-run. Malformed stamps read as no force."""
    if not isinstance(daemon_state, dict):
        return False
    now_utc = datetime.now(timezone.utc)
    for req_key in ("force_rem_request", "user_sleep_request"):
        req = daemon_state.get(req_key)
        if not isinstance(req, dict):
            continue
        if req.get("pending"):
            return True
        honored_raw = req.get("honored_at")
        if honored_raw:
            try:
                honored = datetime.fromisoformat(str(honored_raw))
                if honored.tzinfo is None:
                    honored = honored.replace(tzinfo=timezone.utc)
                if (now_utc - honored) <= timedelta(hours=2):
                    return True
            except (ValueError, TypeError):
                pass
    return False


def _window_deep_idle_arm(
    os_idle_sec: "float | None", daemon_state: "dict | None"
) -> bool:
    # Same source discipline as the starvation override: only OS-level
    # input idle proves the human is away — heartbeat idle merely says no
    # Claude session is open, and open sessions are exactly what this arm
    # exists to see past. Unknown idle reads as present.
    if os_idle_sec is None:
        return False
    raw = os.environ.get(WINDOW_DEEP_IDLE_ENV, "")
    try:
        minutes = float(raw) if raw else 30.0
    except (TypeError, ValueError):
        minutes = 30.0
    if not math.isfinite(minutes):
        minutes = 30.0
    if minutes <= 0:
        return False
    # Floor at the sleep_eligible HID threshold — armed below it would
    # suppress the scanner while sleep stays structurally ineligible.
    minutes = max(minutes, 30.0)
    if float(os_idle_sec) < minutes * 60.0:
        return False
    return _window_membership(daemon_state)


def _seed_starvation_clock(state: dict) -> "tuple[float, str | None]":
    """Wall-clock reference the deep-idle starvation gate measures from.

    Fallback chain, all real completion evidence: the last recorded clean
    cycle, else the last overnight digest shown, else a baseline stamped at
    the first boot that finds neither — the gate then arms only a full
    starvation period after that boot, never on pure ignorance. Returns
    (wall_ts, fresh_baseline_iso); the iso is non-None only when the caller
    must persist the newly minted baseline.
    """
    for key in (
        "last_clean_cycle_at",
        "last_digest_shown_at",
        "sleep_cycle_baseline_at",
    ):
        raw = state.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            continue
        # Older state files circulate tz-naive stamps; every sibling reader
        # of these keys normalizes to UTC before comparing.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), None
    now = datetime.now(timezone.utc)
    return now.timestamp(), now.isoformat()


#: Below this idle age a tick stamps the current half-hour bucket as busy.
PRESENCE_BUSY_IDLE_SEC = 300.0


def _effective_os_idle(
    os_idle_raw: "int | float | None", os_idle_source: "str | None"
) -> "float | None":
    # Tri-state: a reachable source that reports "not idle" (logind
    # IdleHint=false) means the user is active NOW — idle 0, never a
    # fallback to the wrapper heartbeat. Only a truly absent source
    # (both None) may fall back.
    if os_idle_raw is not None:
        return float(os_idle_raw)
    if os_idle_source is not None:
        return 0.0
    return None


def _stamp_activity_presence(
    state: dict,
    idle_elapsed_sec: float,
    *,
    os_idle_sec: "float | None",
) -> bool:
    """Stamp the current half-hour bucket for the quiet-window learner.

    Busy is judged by OS-level input idle (HID/logind) when the source is
    reachable; the wrapper-heartbeat idle is only the fallback. Returns
    True when the masks changed (bucket newly busy or old days pruned) —
    the caller persists only then, at most ~48 writes a day plus the
    daily prune.
    """
    masks = state.get("activity_presence")
    if not isinstance(masks, dict):
        masks = {}
        state["activity_presence"] = masks
    idle = (
        float(os_idle_sec) if os_idle_sec is not None else float(idle_elapsed_sec)
    )
    now_local = datetime.now().astimezone()
    changed = stamp_presence_mask(
        masks, now_local, busy=idle < PRESENCE_BUSY_IDLE_SEC
    )
    changed = prune_presence_masks(masks, now_local) or changed
    return changed


def _reset_unsourced_quiet_window(state: dict) -> bool:
    """Drop a stored window that the presence learner did not produce.

    The night default applies until the learner has enough observed days.
    Returns True when the window was reset.
    """
    if state.get("quiet_window") is None:
        return False
    if state.get("quiet_window_source") == "presence":
        return False
    state["quiet_window"] = None
    state["quiet_window_source"] = None
    return True


def _may_enter_sleep(
    idle_elapsed_sec: float,
    sleep_eligible: bool,
    daemon_state: "dict | None",
    *,
    os_idle_sec: "float | None",
    last_clean_cycle_wall: float,
) -> bool:
    # A starved daemon on a human-idle machine may consolidate even while
    # agent sessions keep the wrapper heartbeat fresh — those sessions are
    # exactly what starves the pipeline, readers are isolated from the
    # sleeping writer by the RO pool, and per-step interrupt checks yield
    # to foreground demand. Gating the override on heartbeat idle would
    # make the backstop unreachable on an always-driven machine.
    if _deep_idle_override(os_idle_sec, last_clean_cycle_wall):
        return sleep_eligible
    # Inside the nightly window a human-idle machine may consolidate even
    # while open sessions keep the wrapper heartbeat fresh — the wrapper
    # refreshes it on a timer, so heartbeat idle proves nothing about the
    # human. Same isolation argument as the starvation override, but
    # night-only and without the starvation precondition.
    if _window_deep_idle_arm(os_idle_sec, daemon_state):
        return sleep_eligible
    # Night-only consolidation: idle alone never enters SLEEP outside the
    # effective window.
    return (
        idle_elapsed_sec >= SLEEP_HEARTBEAT_IDLE_SEC
        and sleep_eligible
        and _in_consolidation_window(daemon_state)
    )


def _must_leave_sleep(
    daemon_state: "dict | None",
    *,
    os_idle_sec: "float | None",
    last_clean_cycle_wall: float,
) -> bool:
    # Symmetric with entry: a sleep entered under the starvation override
    # must not be kicked awake by the same window the override outranked.
    if _deep_idle_override(os_idle_sec, last_clean_cycle_wall):
        return False
    if not _in_consolidation_window(daemon_state):
        return True
    # In-window, the human back at the keyboard ends the nap: entry needed
    # the human away, so exit honors the same evidence — an interrupted
    # cycle otherwise resumes every tick until the window closes, paying
    # the lock-escalate dance against a live user. An explicit force is
    # the user's own request and keeps the machine; unknown idle reads as
    # away (never kick on ignorance). WAL resume finishes the cycle on
    # the next entry.
    if _force_consolidation_active(
        daemon_state if isinstance(daemon_state, dict) else None
    ):
        return False
    return os_idle_sec is not None and float(os_idle_sec) < PRESENCE_BUSY_IDLE_SEC


#: Label -> (LifecycleEvent name, dispatch reason, extra kwargs). One table
#: for the tick's idle transitions so the decision function's outputs and
#: the dispatched events cannot drift apart unpinned.
TRANSITION_DISPATCH: "dict[str, tuple[str, str, dict]]" = {
    "wake_refresh": ("HEARTBEAT_REFRESH", "heartbeat_refresh_active_wrapper", {}),
    "sleep": ("IDLE_30MIN", "sleep_on_idle_30min", {"sleep_eligible": True}),
    # The reason strings are persisted (s2_transition_attempt events) and are
    # the morning-after evidence of WHICH path admitted the entry — an armed
    # entry happens with the wrapper heartbeat FRESH, so labeling it
    # idle_30min would be a lie the events log repeats forever.
    "sleep_starved": (
        "IDLE_30MIN", "sleep_on_starvation_override", {"sleep_eligible": True},
    ),
    "sleep_window": (
        "IDLE_30MIN", "sleep_on_window_deep_idle", {"sleep_eligible": True},
    ),
    "drowsy": ("IDLE_5MIN", "drowsy_on_idle_5min", {}),
    "drowsy_armed": ("IDLE_5MIN", "drowsy_on_armed_entry", {}),
}


def _idle_transition_event(
    scanner_active: bool,
    idle_elapsed_sec: float,
    sleep_eligible: bool,
    daemon_state: "dict | None",
    *,
    os_idle_sec: "float | None",
    last_clean_cycle_wall: float,
    drowsy_after_sec: float,
    current_state: "Any | None" = None,
) -> "str | None":
    """The tick's idle-transition decision as one pure function.

    The tick maps the returned label to an FSM dispatch; keeping the whole
    branch here lets the truth table be tested without assembling a tick.
    """
    # A live wrapper heartbeat must not hold an ARMED entry hostage. Two
    # arms suppress it: the starvation backstop (48h without a clean cycle,
    # any hour), and the nightly-window arm (inside the effective window
    # with the human demonstrably away by OS input idle) — the wrapper
    # refreshes its heartbeat on a timer, so an open-but-idle session
    # would otherwise block sleep every night. Outside both arms the
    # scanner keeps its priority — and it is suppressed ONLY when the arm
    # can actually lead somewhere: armed with sleep_eligible False would
    # strand the daemon in DROWSY with every exit dropped.
    _override = _deep_idle_override(os_idle_sec, last_clean_cycle_wall)
    _window_armed = _window_deep_idle_arm(os_idle_sec, daemon_state)
    _armed = _override or _window_armed
    if scanner_active and not (_armed and sleep_eligible):
        return "wake_refresh"
    if _may_enter_sleep(
        idle_elapsed_sec,
        sleep_eligible,
        daemon_state,
        os_idle_sec=os_idle_sec,
        last_clean_cycle_wall=last_clean_cycle_wall,
    ):
        # The FSM accepts IDLE_30MIN only from DROWSY: a sleep decision
        # taken while still in WAKE must climb one rung first or the
        # dispatched event is dropped and the entry never happens — under
        # an armed entry the heartbeat may never age, so the ladder cannot
        # be left to the drowsy branch below. Compared by enum NAME:
        # identity breaks when a test reloads the lifecycle modules and
        # two class objects coexist. Label precedence mirrors
        # _may_enter_sleep's branch order so the persisted reason names
        # the path that actually admitted the entry.
        if getattr(current_state, "name", None) == "WAKE":
            return "drowsy_armed" if _armed else "drowsy"
        if _override:
            return "sleep_starved"
        if _window_armed:
            return "sleep_window"
        return "sleep"
    if idle_elapsed_sec >= drowsy_after_sec:
        return "drowsy"
    return None


def _in_consolidation_window(daemon_state: "dict | None") -> bool:
    """May consolidation run right now?

    True inside the effective window — the learned user-rhythm quiet window,
    an explicit env override, or the fixed night default — and ALWAYS for an
    explicit force (pending, or honored within the last two hours so an
    in-flight forced cycle is never cut off mid-run). Consolidation must
    never share the machine with an active user outside that window.
    Fail-open on any error: a broken gate must not stop consolidation
    forever.
    """
    try:
        if daemon_state is None:
            from iai_mcp.daemon_state import load_state as _gate_load
            daemon_state = _gate_load()
        if not isinstance(daemon_state, dict):
            return True
        if _force_consolidation_active(daemon_state):
            return True
        return _window_membership_core(daemon_state)
    except Exception:  # noqa: BLE001 -- gate errors must never stop consolidation
        log.debug("consolidation window gate failed; allowing", exc_info=True)
        return True


async def _consume_force_wake(ds: dict, state_machine: Any) -> dict:
    """Consume a pending dashboard/MCP force-wake request.

    Module-level so it is directly unit-testable without assembling the
    daemon. WAKE_SIGNAL wakes SLEEP/HIBERNATION; in WAKE/DROWSY there is
    nothing to wake, and the request is STILL marked honored — a stale
    pending flag must never re-fire every tick forever. Returns the updated
    (saved) state dict, or the input unchanged when nothing was pending.
    """
    if not bool((ds.get("force_wake_request") or {}).get("pending")):
        return ds

    from datetime import datetime as _dt, timezone as _tz

    from iai_mcp.daemon_state import update_state as _update
    from iai_mcp.lifecycle import LifecycleEvent as _Ev
    from iai_mcp.s2_coordinator import (
        S2OscillationBlocked as _Blocked,
        S2OscillationConflict as _Conflict,
    )

    try:
        await state_machine.dispatch(_Ev.WAKE_SIGNAL, reason="force_wake_request")
    except (_Conflict, _Blocked):
        # S2 refused (oscillation guard / min-interval). Marking honored here
        # would be the control lying about success again — leave the request
        # pending so the next tick retries after the interval clears.
        return ds

    def _honor(d: dict) -> None:
        req = dict(d.get("force_wake_request") or {})
        req["pending"] = False
        req["honored_at"] = _dt.now(_tz.utc).isoformat()
        d["force_wake_request"] = req

    return await asyncio.to_thread(_update, _honor)


_CAPTURE_DRAIN_MAX_RECORDS_DEFAULT: int = 500
_CAPTURE_DRAIN_BUDGET_SEC_DEFAULT: float = 10.0


def _capture_drain_max_records() -> int | None:
    raw = os.environ.get("IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS")
    if raw is None:
        return _CAPTURE_DRAIN_MAX_RECORDS_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _CAPTURE_DRAIN_MAX_RECORDS_DEFAULT
    return val if val > 0 else None


def _capture_drain_budget_sec() -> float | None:
    raw = os.environ.get("IAI_MCP_CAPTURE_DRAIN_BUDGET_SEC")
    if raw is None:
        return _CAPTURE_DRAIN_BUDGET_SEC_DEFAULT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _CAPTURE_DRAIN_BUDGET_SEC_DEFAULT
    return val if val > 0 else None


def _run_bounded_capture_queue_drain(store, *, write_event_fn) -> None:
    """One bounded pass over the persistent capture queue. Same bounds as the
    boot pass, so a backlog the boot pass left behind converges edge by edge."""
    try:
        from iai_mcp.capture import capture_turn as _capture_turn
        from iai_mcp.capture_queue import CaptureQueue

        queue = CaptureQueue()
        if queue.pending_count() == 0:
            return

        def _handler(record: dict) -> None:
            _capture_turn(
                store,
                cue=record.get("cue", ""),
                text=record.get("text", record.get("surface", "")),
                tier=record.get("tier", "episodic"),
                session_id=record.get("session_id", "-"),
                role=record.get("role", "user"),
                provenance_extra=(
                    {"cwd": record["cwd"]} if record.get("cwd") else None
                ),
            )

        ingested = queue.ingest_pending(
            _handler,
            max_records=_capture_drain_max_records(),
            budget_sec=_capture_drain_budget_sec(),
        )
        if ingested > 0:
            write_event_fn(
                store,
                "capture_queue_drained",
                {
                    "phase": "drowsy",
                    "ingested": ingested,
                    "remaining": queue.pending_count(),
                },
                severity="info",
            )
    except Exception as e:  # noqa: BLE001 -- lifecycle_tick MUST NOT crash
        log.warning("bounded capture-queue drain failed: %s", e, exc_info=True)


def _run_drowsy_drain(store, *, drain_fn, write_event_fn) -> None:
    try:
        result = drain_fn(store)
    except Exception as e:  # noqa: BLE001 -- lifecycle_tick MUST NOT crash
        log.warning("drowsy drain failed: %s", e, exc_info=True)
        try:
            write_event_fn(
                store,
                "deferred_drain_failed",
                {"error": str(e)[:200], "phase": "drowsy"},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write deferred_drain_failed event: %s", e)
        return
    if not isinstance(result, dict):
        return
    if result.get("files_drained") or result.get("files_failed"):
        try:
            write_event_fn(
                store,
                "deferred_drain_drowsy",
                result,
                severity="info",
            )
        except Exception:  # noqa: BLE001 -- event write non-critical
            log.debug("failed to write deferred_drain_drowsy event")


def _kick_drowsy_rgc_rebuild(store) -> None:
    import threading as _threading

    def _bg() -> None:
        try:
            import iai_mcp.runtime_graph_cache as _rgc
            _rgc._rebuild_and_save_rgc(store)
        except Exception:  # noqa: BLE001 -- best-effort; cache stays cold on failure
            log.debug("drowsy-edge graph-cache rebuild failed", exc_info=True)
        finally:
            try:
                import iai_mcp.runtime_graph_cache as _rgc
                _rgc.rebuild_ready.set()
            except Exception:  # noqa: BLE001
                log.debug("rebuild_ready.set() failed", exc_info=True)

    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _rgc.rebuild_ready.clear()
    except Exception:  # noqa: BLE001
        log.debug("rebuild_ready.clear() failed", exc_info=True)

    _threading.Thread(target=_bg, daemon=True).start()


def _wake_hook_rebuild_if_cold(store) -> None:
    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _, _, _, _src, _ = _rgc.load_recall_structural(store)
        if _src in ("cold_degrade", "last_good"):
            # This site only fires when the cache is already cold, so the gate's
            # own coldness term would rebuild here anyway; force makes the intent
            # explicit at the wake-if-cold edge.
            _rgc._rebuild_and_save_rgc(store, force=True)
    except Exception:  # noqa: BLE001 -- best-effort, never crash the wake hook
        log.debug("wake-hook graph-cache rebuild failed", exc_info=True)
    try:
        # Warm the topology snapshot off the awake path: the process-local
        # cache is empty after every restart, and status_light never computes
        # (by design), so without this `iai status` reports regime=unknown
        # until someone happens to call the deep topology verb.
        from iai_mcp import core as _core
        _core.dispatch(store, "topology", {})
    except Exception:  # noqa: BLE001 -- best-effort, never crash the wake hook
        log.debug("wake-hook topology warm failed", exc_info=True)


def _persist_keys(state: dict, *keys: str) -> None:
    # Copy only the named keys from the long-held in-memory dict to disk
    # under the state write lock. A whole-dict save_state of that dict
    # erases every key another writer persisted since it was loaded.
    def _apply(d: dict) -> None:
        for k in keys:
            if k in state:
                d[k] = state[k]
            else:
                d.pop(k, None)

    update_state(_apply)


def transition(state: dict, new_fsm: str) -> None:
    current = state.get("fsm_state", STATE_WAKE)
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_fsm not in allowed:
        raise ValueError(
            f"Illegal transition {current} -> {new_fsm}; allowed: {sorted(allowed)}"
        )
    state["fsm_state"] = new_fsm
    state["fsm_transition_at"] = datetime.now(timezone.utc).isoformat()
    _persist_keys(state, "fsm_state", "fsm_transition_at")


def _store_is_empty(store: MemoryStore) -> bool:
    try:
        return store.db.open_table("records").count_rows() == 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        log.debug("store empty check failed, assuming empty: %s", exc)
        return True


def _is_inside_window(
    window: tuple[int, int] | list | None,
    now: datetime,
    tz,
) -> bool:
    if not window:
        return False
    try:
        start, duration = int(window[0]), int(window[1])
    except (TypeError, ValueError, IndexError):
        return False
    if duration <= 0:
        return False
    now_local = now.astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // BUCKET_MINUTES
    end = (start + duration) % BUCKET_COUNT
    if start < end:
        return start <= cur_bucket < end
    return cur_bucket >= start or cur_bucket < end


def _update_pending_digest(state: dict, cycle_result: dict) -> None:
    digest = state.get("pending_digest") or {
        "rem_cycles_completed": 0,
        "episodes_processed": 0,
        "schemas_induced_tier0": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out_cycles": 0,
    }
    digest["rem_cycles_completed"] = int(digest.get("rem_cycles_completed", 0)) + 1
    digest["episodes_processed"] = int(digest.get("episodes_processed", 0)) + int(
        cycle_result.get("summaries_created", 0) or 0
    )
    digest["schemas_induced_tier0"] = int(digest.get("schemas_induced_tier0", 0)) + int(
        cycle_result.get("schema_candidates", 0) or 0
    )
    if cycle_result.get("claude_call_used"):
        digest["claude_call_used"] = True
        digest["main_insight_text"] = cycle_result.get("main_insight_text")
        # A success supersedes any earlier cycle's skip reason — leaving it
        # would present a used call and a skip explanation side by side.
        digest.pop("insight_skip_reason", None)
    elif cycle_result.get("insight_skip_reason"):
        # A digest that says only claude_call_used=false is indistinguishable
        # from a gated night — the WHY must survive the fold.
        digest["insight_skip_reason"] = str(
            cycle_result.get("insight_skip_reason")
        )[:200]
    if cycle_result.get("timed_out"):
        digest["timed_out_cycles"] = int(digest.get("timed_out_cycles", 0)) + 1
    state["pending_digest"] = digest


REM_MIN_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_REM_MIN_INTERVAL_SEC", str(20 * 3600))
)


def _recent_force_rem_honored(ds: dict, *, window_sec: float = 3600.0) -> bool:
    honored = (ds.get("force_rem_request") or {}).get("honored_at")
    if not honored:
        return False
    try:
        ts = datetime.fromisoformat(str(honored))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() < window_sec
    except (TypeError, ValueError):
        return False


def _last_rem_completed_age_sec(store) -> "float | None":
    try:
        from iai_mcp.events import query_events
        evs = query_events(store, kind="rem_cycle_completed", limit=1)
        if not evs:
            return None
        ts = evs[0].get("ts")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if not isinstance(ts, datetime):
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:  # noqa: BLE001 -- age probe must never block the pass
        return None


async def _maybe_run_rem(store, ds: dict) -> "dict | None":
    """One REM pass after a clean pipeline cycle. The claude insight call is
    subscription- and daily-budget-gated inside generate_overnight_insight;
    this gate only spaces routine cycles so REM keeps a nightly rhythm while
    a fresh force-rem always runs it. IAI_MCP_REM_DISABLED=1 disables."""
    if os.environ.get("IAI_MCP_REM_DISABLED") == "1":
        return None
    if not _recent_force_rem_honored(ds):
        age = await asyncio.to_thread(_last_rem_completed_age_sec, store)
        if age is not None and age < REM_MIN_INTERVAL_SEC:
            return None
    result = await run_rem_cycle(
        store, 1, 1, "rem", is_last=True, claude_enabled=True,
    )
    try:
        from iai_mcp.daemon_state import update_state as _update_ds

        def _fold_digest(d: dict) -> None:
            _update_pending_digest(d, result)

        await asyncio.to_thread(_update_ds, _fold_digest)
    except Exception:  # noqa: BLE001 -- digest fold is best-effort
        log.debug("pending_digest update after REM failed", exc_info=True)
    return result


def _write_session_start_cache(store, *, cache_path: Path = SESSION_START_CACHE_PATH) -> None:
    try:
        from iai_mcp import retrieve
        from iai_mcp.session import (
            _compose_session_start_payload,
            format_payload_as_markdown,
        )

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id="precache",
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if not rendered:
            return
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, cache_path)
    except Exception as exc:  # noqa: BLE001 -- cache write MUST NOT crash the REM loop
        log.warning("session start cache write failed: %s", exc, exc_info=True)
        try:
            write_event(
                store,
                "session_start_cache_write_failed",
                {"error": str(exc)[:200]},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write session_start_cache_write_failed event")


async def _tick_body(
    store: MemoryStore,
    state: dict,
    *,
    mcp_socket: SocketServer | None = None,
) -> None:
    try:
        from iai_mcp.daemon_state import (
            FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
            prune_first_turn_pending,
        )

        state, dropped = prune_first_turn_pending(
            state, now=datetime.now(timezone.utc),
        )
        if dropped:
            try:
                def _persist_prune(d: dict) -> None:
                    prune_first_turn_pending(d, now=datetime.now(timezone.utc))

                await asyncio.to_thread(update_state, _persist_prune)
            except (OSError, ValueError) as exc:  # noqa: BLE001 -- state save non-critical
                log.debug("save_state after prune failed: %s", exc)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "first_turn_pending_expired",
                    {
                        "dropped_count": len(dropped),
                        "session_ids": dropped,
                        "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                        "phase": "tick",
                    },
                    severity="info",
                )
            except (OSError, RuntimeError) as exc:  # noqa: BLE001 -- event write non-critical
                log.debug("first_turn_pending_expired event write failed: %s", exc)
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.warning("tick step 0.5 (prune first_turn_pending) failed", exc_info=True)

    try:
        _s4bg_ts = state.get("_last_s4bg_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_s4bg = not _s4bg_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_s4bg_ts)
        ).total_seconds() > 3600
        if _should_s4bg:
            from iai_mcp.s4 import s4_background_scan

            def _s4_body():
                from iai_mcp import retrieve as _retrieve_s4
                with _retrieve_s4.background_store_work("s4_background_scan") as _gate_ok:
                    if not _gate_ok:
                        return False
                    s4_background_scan(store, 50)
                    return True

            if await asyncio.to_thread(_s4_body):
                state["_last_s4bg_ts"] = _now_iso
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.6 (s4_background_scan) failed", exc_info=True)

    try:
        _forage_ts = state.get("_last_forage_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_forage = not _forage_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_forage_ts)
        ).total_seconds() > 3600
        if _should_forage:
            _skip_foraging_in_sleep = False
            try:
                from iai_mcp.lifecycle_state import lifecycle_state_path as _lifecycle_state_path, LifecycleState, load_state as _load_ls
                _ls_rec = await asyncio.to_thread(_load_ls, _lifecycle_state_path())
                _ls_current = _ls_rec.get("current_state", "")
                if _ls_current == LifecycleState.SLEEP.value:
                    _skip_foraging_in_sleep = True
            except Exception:
                _skip_foraging_in_sleep = True
            if not _skip_foraging_in_sleep:
                from iai_mcp.foraging import forage_for_connections

                def _forage_body():
                    from iai_mcp import retrieve as _retrieve_forage
                    with _retrieve_forage.background_store_work("forage_for_connections") as _gate_ok:
                        if not _gate_ok:
                            return None
                        return forage_for_connections(store, 3)

                _foraged = await asyncio.to_thread(_forage_body)
                if _foraged is None:
                    _foraged = 0
                else:
                    state["_last_forage_ts"] = _now_iso
                if _foraged > 0:
                    await asyncio.to_thread(
                        write_event, store, "self_foraging_pass",
                        {"edges_created": _foraged}, severity="info",
                    )
            else:
                log.debug("tick step 0.7 (foraging) skipped: canonical FSM in SLEEP")
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.7 (foraging) failed", exc_info=True)

    try:
        from iai_mcp.events import (
            _last_flush_at,
            flush_event_buffer,
            should_flush_by_time,
        )

        if should_flush_by_time(id(store), _last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_event_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("events buffer periodic flush skipped: %s", str(e)[:120])

    try:
        from iai_mcp.store import (
            _record_last_flush_at,
            flush_record_buffer,
            should_flush_record_buffer_by_time,
        )

        if should_flush_record_buffer_by_time(id(store), _record_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_record_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("records buffer periodic flush skipped: %s", str(e)[:120])

    try:
        from iai_mcp.store import (
            _edge_last_flush_at,
            flush_edge_buffer,
            should_flush_edge_buffer_by_time,
        )

        if should_flush_edge_buffer_by_time(id(store), _edge_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_edge_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("edges buffer periodic flush skipped: %s", str(e)[:120])


    if state.get("scheduler_paused") is True:
        try:
            await asyncio.to_thread(
                write_event,
                store,
                "daemon_tick_skipped",
                {"reason": "paused"},
                severity="info",
            )
        except (OSError, RuntimeError) as exc:
            log.debug("daemon_tick_skipped event write failed: %s", exc)
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "paused"
        try:
            await asyncio.to_thread(
                _persist_keys, state, "last_tick_at", "last_tick_skipped_reason",
            )
        except (OSError, ValueError) as exc:
            log.debug("save_state (paused) failed: %s", exc)
        return

    if await asyncio.to_thread(_store_is_empty, store):
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "empty_store"
        await asyncio.to_thread(
            _persist_keys, state, "last_tick_at", "last_tick_skipped_reason",
        )
        return

    now = datetime.now(timezone.utc)

    last_learned_raw = state.get("quiet_window_learned_at")
    last_learned_dt: datetime | None = None
    if last_learned_raw:
        try:
            last_learned_dt = datetime.fromisoformat(last_learned_raw)
        except (TypeError, ValueError):
            last_learned_dt = None
    if should_relearn(last_learned_dt, now):
        # Until the presence masks cover enough observed days, the
        # learner returns None and the fixed night default applies.
        try:
            window = learn_quiet_window_from_presence(
                state.get("activity_presence")
            )
        except (TypeError, ValueError) as exc:
            log.debug("learn_quiet_window_from_presence failed: %s", exc)
            window = None
        state["quiet_window"] = list(window) if window else None
        state["quiet_window_source"] = "presence" if window else None
        state["quiet_window_learned_at"] = now.isoformat()
        await asyncio.to_thread(
            _persist_keys, state,
            "quiet_window", "quiet_window_source", "quiet_window_learned_at",
        )


    state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
    state.pop("last_tick_skipped_reason", None)
    try:
        await asyncio.to_thread(
            _persist_keys, state, "last_tick_at", "last_tick_skipped_reason",
        )
    except (OSError, ValueError) as exc:
        log.debug("save_state after tick failed: %s", exc)


async def _scheduler_tick(
    store: MemoryStore,
    state: dict,
    *,
    tick_body: Callable[..., Awaitable[None]] | None = None,
    mcp_socket: SocketServer | None = None,
) -> None:
    body = tick_body or _tick_body
    while True:
        try:
            await body(store, state, mcp_socket=mcp_socket)
        except TypeError:
            try:
                await body(store, state)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 -- daemon tick boundary
                log.warning("tick failed (legacy body): %s", exc, exc_info=True)
                try:
                    write_event(
                        store,
                        "tick_error",
                        {"error": str(exc), "type": type(exc).__name__},
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- event write inside boundary guard
                    log.debug("tick_error event write failed")
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 -- daemon must never die mid-tick
            log.warning("tick failed: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "tick_error",
                    {"error": str(exc), "type": type(exc).__name__},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("tick_error event write failed")
        try:
            await asyncio.sleep(TICK_INTERVAL_SEC)
        except asyncio.CancelledError:
            break


async def _s4_offline_loop(store: MemoryStore, shutdown: asyncio.Event) -> None:
    if S4_FIRST_ITER_GRACE_SEC > 0:
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_FIRST_ITER_GRACE_SEC
            )
            return
        except asyncio.TimeoutError:
            pass
    while not shutdown.is_set():
        try:
            await asyncio.to_thread(s4.run_offline_pass, store)
        except Exception as exc:  # noqa: BLE001 -- never die on offline-pass failure
            log.warning("S4 offline pass failed: %s", exc, exc_info=True)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "s4_offline_pass_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("s4_offline_pass_error event write failed")
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_OFFLINE_INTERVAL_SEC
            )
            break
        except asyncio.TimeoutError:
            continue


from iai_mcp.daemon_config import (  # noqa: E402
    ErasureConfig,
    _load_erasure_config,
    PatSepConfig,
    _load_patsep_config,
    S2Config,
    _load_s2_config,
    SleepOverhaulConfig,
    _load_sleep_overhaul_config,
    ReconsolidationConfig,
    _load_reconsolidation_config,
    StcConfig,
    _load_stc_config,
    UserModelConfig,
    _load_user_model_config,
    SpatialConfig,
    _load_spatial_config,
    DmnConfig,
    _load_dmn_config,
    PaskConfig,
    _load_pask_config,
)


_USER_SHUTDOWN_FLAG = "user_requested_shutdown"


def _clear_user_shutdown_sentinel(state: dict) -> None:
    try:
        update_state(lambda d: d.pop(_USER_SHUTDOWN_FLAG, None))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.debug("clear_user_shutdown_sentinel disk op failed: %s", exc)
    state.pop(_USER_SHUTDOWN_FLAG, None)


def _new_efs_holder() -> dict:
    # The one owner of the holder shape shared between main() and the
    # install thread; construct it only here so a malformed hand-built
    # dict cannot silently disable the warm hold.
    return {
        "orig": None,
        "installed": False,
        "abandoned": False,
        "lock": threading.Lock(),
    }


#: Budget for the pre-bind WARM window (embedder construction + embed_identity
#: + warm_dispatch_surface), anchored to _daemon_started_monotonic. It bounds
#: the warm WAIT before the bind, not the preamble that precedes it (profile
#: hydration is itself unbounded). Must stay under the wrapper's 3.0s
#: single-probe timeout; a restart loop is separately prevented by the
#: pid-keyed liveness guards, not by this value.
_PRE_SERVE_WARM_DEADLINE_SEC = 2.5


def _install_warm_embedder_override(store, holder: dict | None = None) -> tuple[object, bool]:
    import iai_mcp.embed as _embed_mod

    orig_efs = _embed_mod.embedder_for_store
    # This runs on an executor thread and a cancelled await never receives
    # the return tuple — the caller's finally restores from the holder.
    # holder["orig"] must be set before the warm build; the swap happens
    # under holder["lock"] and is skipped once the caller has marked the
    # holder abandoned, so a cancel during the build can never leak an
    # unrestorable swap landing after the finally already ran.
    if holder is not None:
        holder["orig"] = orig_efs
        # Outside the try below: a holder without a lock is a programming
        # error and must fail loudly, never degrade into the prewarm path.
        _holder_lock = holder["lock"]
    try:
        warm = orig_efs(store)
        def _held_embedder_for_store(_store, **_kwargs):
            # The held embedder already passed the identity guard at boot;
            # kwargs (allow_identity_mismatch) are accepted for signature
            # parity with the real embedder_for_store.
            return warm

        if holder is not None:
            with _holder_lock:
                if holder.get("abandoned"):
                    return orig_efs, False
                holder["installed"] = True
                _embed_mod.embedder_for_store = _held_embedder_for_store
            return orig_efs, True
        _embed_mod.embedder_for_store = _held_embedder_for_store
        return orig_efs, True
    except _embed_mod.EmbedIdentityMismatch:
        # Identity-guard refusal: the store's vectors belong to a different
        # embedder. Serving would answer every recall with cross-generation
        # noise while looking healthy — refuse to boot instead. Every other
        # embed-layer ValueError keeps the prewarm degrade path below.
        raise
    except Exception as exc:  # noqa: BLE001 -- prewarm/hold failure is non-fatal
        log.warning("embedder prewarm/hold failed: %s", exc, exc_info=True)
        try:
            write_event(store, "prewarm_failed", {"error": str(exc)}, severity="warning")
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("prewarm_failed event write failed")
        return orig_efs, False


def _restore_embedder_funnel(orig_efs: object, installed: bool) -> None:
    if not installed:
        return
    try:
        import iai_mcp.embed as _embed_mod

        _embed_mod.embedder_for_store = orig_efs
    except Exception:  # noqa: BLE001 -- shutdown must never crash on restore
        log.debug("embedder funnel restore failed", exc_info=True)


def _set_process_title() -> None:
    # The store-scoped title is what the stop verb's orphan sweep matches
    # by equality; setproctitle absence leaves the sweep inert for this
    # process, never wrong.
    try:
        from setproctitle import setproctitle as _setproctitle

        from iai_mcp.lifecycle_lock import daemon_process_title
        from iai_mcp.tz import store_root

        _setproctitle(daemon_process_title(store_root()))
    except Exception:  # noqa: BLE001
        pass


STDERR_LOG_CAP_ENV = "IAI_MCP_STDERR_LOG_CAP_MB"


def _rotate_launchd_stderr(log_path: "Path | None" = None) -> bool:
    """Cap the append-only service stderr log at boot.

    launchd holds an O_APPEND fd to a fixed path, so rename-rotation would
    keep future writes going into the renamed file — the only rotation that
    works is copying the content aside and truncating the original in place.
    """
    # The WHOLE body is fail-soft: rotation is a hygiene step and no
    # failure in it — parsing, path resolution, copy, truncate, or the
    # notice write — may ever take the boot down.
    try:
        import math

        raw = os.environ.get(STDERR_LOG_CAP_ENV, "")
        try:
            cap_mb = float(raw) if raw else 5.0
        except (TypeError, ValueError):
            cap_mb = 5.0
        # Guard the PRODUCT, not just the input: a huge finite cap (1e308)
        # overflows to inf at the multiply and would crash at int().
        cap_product = cap_mb * 1024 * 1024
        if not math.isfinite(cap_product) or cap_product <= 0:
            return False
        cap_bytes = int(cap_product)
        if log_path is None:
            base = os.environ.get("IAI_MCP_STORE")
            root = Path(base) if base else Path.home() / ".iai-mcp"
            log_path = root / "logs" / "launchd-stderr.log"
        if not log_path.exists() or log_path.stat().st_size <= cap_bytes:
            return False
        rotated = log_path.parent / (log_path.name + ".1")
        shutil.copyfile(log_path, rotated)
        os.truncate(log_path, 0)
        sys.stderr.write(
            f"stderr log exceeded {cap_mb:g} MB; rotated to {rotated.name}\n"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        try:
            log.debug("stderr log rotation failed: %s", exc)
        except Exception:  # noqa: BLE001
            pass
        return False


def _install_boot_signal_trace() -> None:
    """Record a termination signal that lands before the graceful handlers.

    The handler restores the default disposition and re-raises, so the exit
    semantics are exactly what they were; the only change is that the death
    stops being anonymous.
    """
    from iai_mcp.daemon._watchdog import _watchdog_log_path

    # Resolved at install time: the handler should be syscalls only.
    try:
        path = _watchdog_log_path()
    except Exception:  # noqa: BLE001 -- no breadcrumb path, no trace
        return

    def _trace(signum, _frame):  # noqa: ANN001 -- signal handler signature
        try:
            line = (
                f"{datetime.now(timezone.utc).isoformat()} "
                f"daemon_boot_window_signal signal={signum} pid={os.getpid()}\n"
            ).encode()
            fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001 -- tracing must never mask the signal
            pass
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:  # noqa: BLE001 -- re-raise is best-effort
            os._exit(128 + int(signum))

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _trace)
        except (AttributeError, ValueError, OSError):
            pass


async def main() -> int:
    _set_process_title()
    _rotate_launchd_stderr()
    _require_native()
    _raise_fd_limit()
    # kill -USR2 <pid> dumps every thread's stack to stderr — the sanctioned
    # way to see where a live daemon is stuck without root or a debugger.
    try:
        faulthandler.register(signal.SIGUSR2, all_threads=True)
    except (AttributeError, ValueError, OSError):
        pass

    # Boot-window signal trace: the graceful handlers are installed only
    # after the store opens and the embedder warms, a window that spans
    # minutes at prod scale. A termination signal arriving in it would kill
    # the process under the default disposition and leave NO evidence — no
    # stamp, no breadcrumb, an unexplained exit code. This handler records
    # who asked and when, then restores the default so the semantics are
    # unchanged; the graceful handlers replace it once they exist.
    _install_boot_signal_trace()

    store = await _open_exclusive_store_with_backoff(
        lambda: MemoryStore(
            read_consistency_interval=timedelta(seconds=0),
            access_mode=AccessMode.EXCLUSIVE,
        )
    )

    try:
        hippo_lock_path = store.root / "hippo" / ".lock"
        cleanup_stale_consolidation_intent(hippo_lock_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        from iai_mcp.crypto_key_watch import check_crypto_key_file_rotation_event

        check_crypto_key_file_rotation_event(store)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        log.debug("crypto key rotation check skipped: %s", exc)

    from iai_mcp.migrate import detect_partial_migration
    _migration_state = detect_partial_migration(store.db)
    if _migration_state["state"] == "partial_swap_inconsistent":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback to restore from "
                        "records_old_<ts>, then iai-mcp daemon-start."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for partial_swap_inconsistent failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_rollback":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback (discard the partial "
                        "staging) OR iai-mcp migrate --resume (continue "
                        "from migration_progress.json checkpoint)."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for needs_rollback failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_cleanup":
        for _old_name in _migration_state.get("old_tables", []):
            try:
                store.db.drop_table(_old_name)
            except (OSError, RuntimeError, KeyError) as _exc:
                log.warning("migrate cleanup drop_table(%s) failed: %s", _old_name, _exc)
                try:
                    sys.stderr.write(
                        json.dumps({
                            "event": "migrate_cleanup_failed",
                            "table": _old_name,
                            "err": str(_exc)[:120],
                        }) + "\n"
                    )
                except (OSError, ValueError, TypeError):
                    pass

    _respawn_by = os.environ.pop("IAI_DAEMON_RESPAWN_BY", None)
    if _respawn_by:
        try:
            write_event(
                store,
                "doctor_action",
                {"action": "daemon_respawned_by_doctor", "respawned_by": _respawn_by},
            )
        except Exception:  # noqa: BLE001 -- audit write must not block boot
            log.debug("failed to write respawn audit event")

    _load_erasure_config()
    _load_patsep_config()
    _load_s2_config()
    _load_sleep_overhaul_config()
    _load_reconsolidation_config()
    _load_stc_config()
    _load_dmn_config()
    _load_pask_config()

    # Shared with the install thread: asyncio cannot stop a running
    # executor thread and a cancelled await loses the return tuple, so the
    # finally below restores from this holder — and first marks it
    # abandoned under the lock, which forbids a still-running build from
    # swapping the funnel after the restore already happened.
    _efs_holder = _new_efs_holder()

    from iai_mcp.lifecycle_lock import LifecycleLock, LifecycleLockConflict

    lifecycle_lock = LifecycleLock()
    try:
        lifecycle_lock.acquire()
    except LifecycleLockConflict as exc:
        sys.stderr.write(f"daemon already running: {exc}\n")
        return 1

    # The warm-embedder override installs AFTER the socket binds (below):
    # a cold model build must never make a healthy process read as dead.
    try:
        try:
            from iai_mcp.daemon_state import daemon_state_path as _drp_daemon_state_path
            from iai_mcp.fsm_reconcile import reconcile_fsm_state
            from iai_mcp.lifecycle_state import lifecycle_state_path as _drp_lifecycle_state_path

            _drift_report = reconcile_fsm_state(
                canonical_path=_drp_lifecycle_state_path(),
                legacy_path=_drp_daemon_state_path(),
                auto_correct=True,
            )
            if _drift_report.get("drift") is True:
                log.warning(
                    "fsm_drift_detected canonical=%s legacy=%s",
                    _drift_report.get("canonical"),
                    _drift_report.get("legacy"),
                )
                try:
                    write_event(
                        store,
                        "fsm_drift_detected",
                        _drift_report,
                        severity="warning",
                        domain="ops",
                    )
                except Exception:  # noqa: BLE001 -- fail-safe
                    log.debug("fsm_drift_detected event write failed")
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("fsm_reconcile failed", exc_info=True)

        try:
            from iai_mcp.archive_backups import archive_stuck_backups

            archive_stuck_backups()
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("archive_stuck_backups failed", exc_info=True)

        def _boot_stamp(d: dict) -> None:
            d.setdefault("fsm_state", STATE_WAKE)
            d["daemon_started_at"] = datetime.now(timezone.utc).isoformat()
            d["daemon_pid"] = os.getpid()
            if _reset_unsourced_quiet_window(d):
                log.info(
                    "quiet window reset to the night default: the stored one "
                    "was not learned from presence data",
                )

        state = await asyncio.to_thread(update_state, _boot_stamp)
        # Served by the status handler while the embedder is still building;
        # replaced with the real identity block right after the build.
        state["embed_identity"] = {"state": "warming"}
        global _daemon_started_monotonic
        _daemon_started_monotonic = time.monotonic()
        write_event(store, "daemon_started", {"state": state["fsm_state"]})

        _wake_was_pending = False
        try:
            from iai_mcp.wake_handler import WakeHandler, wake_signal_path

            _wake_signal_path = wake_signal_path()
            if WakeHandler(_wake_signal_path).consume_wake_signal():
                _wake_was_pending = True
                write_event(
                    store, "wake_signal_consumed", {"phase": "startup"}, severity="info"
                )
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake-handler
            log.debug("wake signal consume failed", exc_info=True)

        try:
            from iai_mcp.capture import capture_turn as _capture_turn
            from iai_mcp.capture_queue import CaptureQueue

            _capture_queue = CaptureQueue()
            def _capture_handler(record: dict) -> None:
                kwargs = {
                    "cue": record.get("cue", ""),
                    "text": record.get("text", record.get("surface", "")),
                    "tier": record.get("tier", "episodic"),
                    "session_id": record.get("session_id", "-"),
                    "role": record.get("role", "user"),
                    "provenance_extra": (
                        {"cwd": record["cwd"]} if record.get("cwd") else None
                    ),
                }
                _capture_turn(store, **kwargs)

            # Boot drain is bounded: an arbitrary backlog must never own the
            # startup path. The remainder stays pending and converges through
            # the same bounded pass on every drowsy edge.
            ingested = await asyncio.to_thread(
                lambda: _capture_queue.ingest_pending(
                    _capture_handler,
                    max_records=_capture_drain_max_records(),
                    budget_sec=_capture_drain_budget_sec(),
                )
            )
            if ingested > 0:
                remaining = await asyncio.to_thread(_capture_queue.pending_count)
                write_event(
                    store,
                    "capture_queue_drained",
                    {
                        "phase": "startup",
                        "ingested": ingested,
                        "remaining": remaining,
                    },
                    severity="info",
                )
        except Exception as exc:  # noqa: BLE001 -- never block boot on queue drain
            log.warning("capture queue drain failed at startup: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "capture_queue_drain_failed",
                    {"phase": "startup", "error": str(exc)[:200]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("capture_queue_drain_failed event write failed")

        try:
            from iai_mcp.daemon_state import (
                FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                prune_first_turn_pending,
            )

            state, dropped = prune_first_turn_pending(
                state, now=datetime.now(timezone.utc),
            )
            if dropped:
                def _persist_boot_prune(d: dict) -> None:
                    prune_first_turn_pending(d, now=datetime.now(timezone.utc))

                await asyncio.to_thread(update_state, _persist_boot_prune)
                try:
                    write_event(
                        store,
                        "first_turn_pending_expired",
                        {
                            "dropped_count": len(dropped),
                            "session_ids": dropped,
                            "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                            "phase": "startup",
                        },
                        severity="info",
                    )
                except (OSError, RuntimeError) as exc:
                    log.debug("first_turn_pending_expired (startup) event write failed: %s", exc)
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on startup prune
            log.debug("startup prune first_turn_pending failed", exc_info=True)

        try:
            _wal = SleepWAL()
            pending = _wal.pending_entries()
            if pending:
                log.warning(
                    "daemon startup: %d pending WAL entries found — prior process may have"
                    " died mid-sleep; entries logged but NOT re-executed",
                    len(pending),
                )
                write_event(
                    store,
                    "sleep_wal_pending_recovered",
                    {"count": len(pending), "phase": "startup"},
                    severity="info",
                )
        except Exception:  # noqa: BLE001 -- WAL check MUST NOT crash boot
            log.exception("daemon startup: sleep_wal pending check failed")


        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        # SIGHUP resolved lazily: absent on Windows, where the loop-signal
        # registration is a no-op anyway.
        _shutdown_signals = [signal.SIGTERM, signal.SIGINT]
        _sighup = getattr(signal, "SIGHUP", None)
        if _sighup is not None:
            _shutdown_signals.append(_sighup)
        for sig in _shutdown_signals:
            try:
                loop.add_signal_handler(sig, shutdown.set)
            except (NotImplementedError, RuntimeError):
                pass

        try:
            health = _hippo_health_check_on_boot(store)
            await asyncio.to_thread(
                write_event,
                store,
                "hippo_boot_health",
                health,
                severity=("info" if health.get("action") == "ok" else "warning"),
            )
        except Exception:  # noqa: BLE001
            log.debug("hippo boot health check failed", exc_info=True)

        if os.environ.get("IAI_MCP_ASYNC_WRITES_OFF", "").strip() not in (
            "1", "true", "TRUE", "yes",
        ):
            try:
                await store.enable_async_writes()
            except Exception:  # noqa: BLE001 -- boot must never fail on this optimization
                log.warning(
                    "enable_async_writes failed; using sync reinforce fallback",
                    exc_info=True,
                )

        try:
            from iai_mcp import core as _profile_core

            await asyncio.to_thread(_profile_core.ensure_profile_hydrated, store)
        except Exception:  # noqa: BLE001 -- boot must never fail on profile hydration
            log.debug("profile hydration failed at boot", exc_info=True)

        mcp_socket = SocketServer(store, state=state)
        # Demand baseline for the HIBERNATION exit check: captured BEFORE
        # serve() is scheduled — last_activity_ts starts at construction
        # time, and a snapshot taken any later would swallow real requests
        # served during the boot window.
        _boot_socket_activity_mono: list[float] = [mcp_socket.last_activity_ts]

        # Bind never waits past the total pre-bind deadline: on overflow
        # serve anyway; a late identity/config refusal still refuses boot
        # via the install task's done-callback.
        import iai_mcp.embed as _embed_mod

        _boot_refuse_exc: list[BaseException | None] = [None]
        _warm_deadline = _daemon_started_monotonic + _PRE_SERVE_WARM_DEADLINE_SEC

        async def _finish_prebind_warm(*, already_awaited: bool) -> None:
            """Warm the structural memo and compute embed_identity once the
            embedder construction resolves, whichever arm resolved it.
            Shared by the in-budget and serve-anyway arms so this work runs
            exactly once regardless of which one resolves the install."""
            if not already_awaited:
                try:
                    await install_task
                except (_embed_mod.EmbedIdentityMismatch, _embed_mod.EmbedderConfigError):
                    # Refuse-boot already recorded by the done-callback below;
                    # a surface built on a rejected/misconfigured embedder
                    # must not be warmed.
                    return
                except Exception:  # noqa: BLE001 -- non-fatal prewarm outcome
                    pass
            # Computed ONCE at boot and served from state by the status
            # handler: the status path is the liveness watchdog's signal and
            # must never block on _conn_lock behind a consolidating writer.
            try:
                from iai_mcp.concurrency import _status_embed_identity

                state["embed_identity"] = await asyncio.to_thread(
                    _status_embed_identity, store
                )
            except Exception:  # noqa: BLE001 -- diagnostic block must not fail boot
                log.debug("boot embed_identity compute failed", exc_info=True)
                state["embed_identity"] = None

            # Bounded so a pathological store cannot stall the boot tail.
            try:
                from iai_mcp.daemon._boot_warmup import warm_dispatch_surface

                # Shield keeps the worker alive: asyncio cannot cancel a
                # running thread, and retaining the task makes that
                # continuation explicit and observable until shutdown.
                _wds_task = asyncio.create_task(
                    asyncio.to_thread(warm_dispatch_surface, store)
                )
                _wds_summary = await asyncio.shield(_wds_task)
                log.info(
                    "dispatch surface warmed in %.0fms",
                    _wds_summary.get("elapsed_ms", -1.0),
                )
            except Exception:  # noqa: BLE001 -- warm-up must never block boot
                log.debug("dispatch surface warm-up failed", exc_info=True)

        install_task = asyncio.create_task(
            asyncio.to_thread(_install_warm_embedder_override, store, _efs_holder),
        )

        def _install_refuse_boot_cb(task: "asyncio.Task") -> None:
            # Retained-task done-callback: the only path that can still
            # observe an identity/config refusal once it surfaces AFTER the
            # bind (a refusal surfacing within the budget is re-raised
            # directly through the await below instead).
            if task.cancelled():
                return
            exc = task.exception()
            if exc is None:
                return
            if isinstance(
                exc, (_embed_mod.EmbedIdentityMismatch, _embed_mod.EmbedderConfigError)
            ):
                log.critical(
                    "embedder identity/config refusal surfaced post-bind; "
                    "forcing refuse-boot: %s", exc,
                )
                _boot_refuse_exc[0] = exc
                shutdown.set()
            else:
                log.warning(
                    "embedder prewarm/hold failed post-bind: %s", exc, exc_info=True,
                )

        install_task.add_done_callback(_install_refuse_boot_cb)

        def _finish_task_done_cb(task: "asyncio.Task") -> None:
            # Retrieves the exception so a timeout-orphaned finish_task never
            # emits an "exception was never retrieved" warning.
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                log.warning("post-embedder prebind warm failed: %s", exc, exc_info=exc)

        _remaining = _warm_deadline - time.monotonic()
        try:
            await asyncio.wait_for(
                asyncio.shield(install_task), timeout=max(0.0, _remaining),
            )
        except asyncio.TimeoutError:
            # SERVE-ANYWAY: the embedder construction alone already overflowed
            # the total deadline. finish_task still runs the rest of the warm
            # work post-bind, single-flight, in the background — never gated
            # on remaining budget, only its WAIT is.
            finish_task = asyncio.create_task(_finish_prebind_warm(already_awaited=False))
            finish_task.add_done_callback(_finish_task_done_cb)
        else:
            # Resolved within the budget: an identity/config refusal already
            # propagated out of this await (refuse boot, caught by the
            # caller's outer exception handling).
            finish_task = asyncio.create_task(_finish_prebind_warm(already_awaited=True))
            finish_task.add_done_callback(_finish_task_done_cb)
            _remaining = _warm_deadline - time.monotonic()
            if _remaining > 0:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(finish_task), timeout=_remaining,
                    )
                except asyncio.TimeoutError:
                    # finish_task keeps running post-bind, shielded above.
                    pass

        mcp_socket_task = asyncio.create_task(mcp_socket.serve())
        await asyncio.sleep(0.05)

        try:
            from iai_mcp.daemon._boot_warmup import run_boot_warmup

            async def _boot_warmup_task() -> None:
                try:
                    await asyncio.to_thread(run_boot_warmup, store, warm_dispatch=False)
                except Exception as _exc:  # noqa: BLE001 -- warm-up must never crash the daemon
                    log.debug("boot_warmup failed: %s", _exc, exc_info=True)

            asyncio.create_task(_boot_warmup_task())
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("boot_warmup scheduling failed", exc_info=True)

        try:
            from iai_mcp import runtime_graph_cache as _rgc_mod

            def _boot_preload_body() -> None:
                import time as _time

                from iai_mcp import retrieve as _retrieve_preload
                # Boot-once task: on a busy gate keep waiting in bounded
                # rounds instead of piling on; if the gate never frees, the
                # graph builds lazily on the first recall that needs it.
                for _ in range(20):
                    with _retrieve_preload.background_store_work("boot_preload") as _gate_ok:
                        if _gate_ok:
                            # Delta-only when the cached payload allows it;
                            # falls back to the full rebuild internally.
                            _retrieve_preload.build_runtime_graph_incremental(store)
                            return
                    _time.sleep(15)
                log.warning(
                    "boot_preload skipped: gate stayed busy; the runtime graph"
                    " will build lazily"
                )

            async def _boot_preload() -> None:
                try:
                    await asyncio.to_thread(_boot_preload_body)
                except Exception as _exc:  # noqa: BLE001 -- preload MUST NOT crash daemon
                    log.debug("boot_preload failed: %s", _exc, exc_info=True)
                finally:
                    _rgc_mod.preload_ready.set()

            asyncio.create_task(_boot_preload())
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("boot_preload scheduling failed", exc_info=True)
            try:
                import iai_mcp.runtime_graph_cache as _rgc_fallback
                _rgc_fallback.preload_ready.set()
            except Exception:  # noqa: BLE001
                pass

        try:
            from iai_mcp.capture import drain_capture_backlog as _drain

            def _drain_body():
                from iai_mcp import retrieve as _retrieve_drain
                with _retrieve_drain.background_store_work("drain_deferred_captures") as _gate_ok:
                    if not _gate_ok:
                        return {}
                    return _drain(store)

            async def _drain_and_report() -> None:
                try:
                    drain_counts = await asyncio.to_thread(_drain_body)
                    if drain_counts.get("files_drained") or drain_counts.get(
                        "files_failed"
                    ):
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_startup",
                            drain_counts,
                            severity="info",
                        )
                except Exception as e:  # noqa: BLE001 -- drain MUST NOT crash daemon
                    log.warning("startup deferred drain failed: %s", e, exc_info=True)
                    try:
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_failed",
                            {"error": str(e)[:200], "phase": "startup"},
                            severity="warning",
                        )
                    except Exception:  # noqa: BLE001 -- event write inside boundary guard
                        log.debug("deferred_drain_failed (startup) event write failed")

            _drain_task = asyncio.create_task(_drain_and_report())
            try:
                mcp_socket._test_drain_task = _drain_task  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                log.debug("test drain task attach failed: %s", exc)
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("startup drain scheduling failed", exc_info=True)


        from iai_mcp.heartbeat_scanner import HeartbeatScanner as _HeartbeatScanner
        from iai_mcp.idle_detector import IdleDetector as _IdleDetector
        from iai_mcp.lifecycle import (
            LifecycleEvent as _LifecycleEvent,
        )
        from iai_mcp.lifecycle import (
            LifecycleStateMachine as _LifecycleStateMachine,
        )
        from iai_mcp.lifecycle_state import LifecycleState as _LifecycleState
        from iai_mcp.s2_coordinator import (
            S2Coordinator,
            S2OscillationBlocked,
            S2OscillationConflict,
        )
        from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline as _SleepPipeline

        from pathlib import Path as _PathHere
        _store_root = os.environ.get("IAI_MCP_STORE")
        _wrappers_dir = (
            _PathHere(_store_root) if _store_root else _PathHere.home() / ".iai-mcp"
        ) / "wrappers"
        _heartbeat_scanner = _HeartbeatScanner(_wrappers_dir)
        _idle_detector = _IdleDetector()

        from iai_mcp.lifecycle_event_log import LifecycleEventLog as _LifecycleEventLog
        from iai_mcp.lifecycle_state import lifecycle_state_path as _lifecycle_state_path
        _lifecycle_log_dir = (
            _PathHere(_store_root) if _store_root else _PathHere.home() / ".iai-mcp"
        ) / "logs"
        _resolved_lifecycle_state_path = _lifecycle_state_path(
            _store_root if _store_root else None,
        )
        _resolved_lifecycle_event_log = _LifecycleEventLog(log_dir=_lifecycle_log_dir)

        _sleep_pipeline = _SleepPipeline(
            store=store,
            lifecycle_state_path=_resolved_lifecycle_state_path,
            lifecycle_event_log=_resolved_lifecycle_event_log,
        )

        _s2_config = _load_s2_config()
        _s2_coord = S2Coordinator(
            store=store,
            state_path=_resolved_lifecycle_state_path,
            min_interval_sec=_s2_config.min_interval_sec,
            dry_run=_s2_config.dry_run,
        )

        from iai_mcp.peri_event_buffer import PeriEventBuffer, set_buffer
        _stc_config = _load_stc_config()
        _peri_event_buffer = PeriEventBuffer(maxlen=_stc_config.peri_event_buffer_size)
        set_buffer(_peri_event_buffer)

        _state_machine = _LifecycleStateMachine(
            state_path=_resolved_lifecycle_state_path,
            event_log=_resolved_lifecycle_event_log,
            coordinator=_s2_coord,
        )

        _boot_wake_event = _LifecycleEvent.WAKE_SIGNAL
        _boot_wake_reason: str | None = None
        if _wake_was_pending:
            _boot_wake_reason = "wake_on_signal_consumed"
        else:
            # A boot that finds persisted HIBERNATION with a live wrapper
            # session attached is demand, signal file or not — without this
            # the daemon exits after one tick and every restart repeats it.
            try:
                if _state_machine.current_state is _LifecycleState.HIBERNATION and (
                    await asyncio.to_thread(_heartbeat_scanner.is_active)
                ):
                    _boot_wake_event = _LifecycleEvent.REQUEST_ARRIVED
                    _boot_wake_reason = "wake_on_live_wrapper_heartbeat"
            except Exception:  # noqa: BLE001 -- boot MUST NOT block on the demand probe
                log.debug("hibernation demand probe failed", exc_info=True)
        if _boot_wake_reason is not None:
            try:
                await _state_machine.dispatch(
                    _boot_wake_event,
                    reason=_boot_wake_reason,
                )
            except (S2OscillationConflict, S2OscillationBlocked):
                pass
            except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake dispatch
                log.debug("wake signal dispatch failed", exc_info=True)

        global _cascade_executor
        _cascade_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="iai-cascade"
        )

        tick_task = asyncio.create_task(
            _scheduler_tick(store, state, mcp_socket=mcp_socket)
        )
        audit_task = asyncio.create_task(
            continuous_audit(store, shutdown)
        )
        s4_task = asyncio.create_task(
            _s4_offline_loop(store, shutdown)
        )
        cascade_task = asyncio.create_task(
            _hippea_cascade_loop(store, shutdown)
        )

        cpu_watchdog_task = asyncio.create_task(
            _cpu_watchdog_loop(store, shutdown)
        )

        _watchdog_stop = threading.Event()
        watchdog_thread = threading.Thread(
            target=_liveness_watchdog,
            args=(store, _watchdog_stop),
            name="iai-liveness-watchdog",
            daemon=True,
        )
        watchdog_thread.start()


        LIFECYCLE_TICK_INTERVAL_SEC: float = 30.0
        DROWSY_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_DROWSY_AFTER_SEC", "300")
        )
        HIBERNATE_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_HIBERNATE_AFTER_SEC", "7200")
        )
        PENDING_EMBED_FLOOR_SEC: float = float(
            os.environ.get("IAI_MCP_PENDING_EMBED_FLOOR_SEC", "300")
        )

        _last_active_monotonic: list[float] = [time.monotonic()]
        # Below the 6h probe threshold at boot: the first version check
        # happens on the first tick, then every 6h.
        _last_version_probe_mono: list[float] = [
            time.monotonic() - 7 * 3600.0
        ]
        _prev_lifecycle_state: list = [_LifecycleState.WAKE]
        _lock_downgraded_to_shared: list[bool] = [False]
        # A failing pipeline must not re-escalate to EX every tick — that
        # starves every SHARED client (dashboard, CLI) for as long as the
        # failure persists. One failure buys a cooldown before the next try.
        _sleep_fail_backoff_until: list[float] = [0.0]
        _last_clean_cycle_mono: list[float] = [0.0]
        _last_clean_cycle_wall: list[float] = [0.0]
        # Seed the starvation clock from persisted state: process-local zero
        # would otherwise read as "never completed" after every restart —
        # and a store that never recorded a clean cycle would leave the
        # deep-idle backstop unarmed forever on exactly the machine that
        # starves it.
        # Two clocks with different meanings: _last_clean_cycle_wall is a
        # REAL completion only (the cooldown gate treats it as one), while
        # the starvation reference may be a digest timestamp or a minted
        # boot baseline and feeds ONLY the deep-idle gate.
        _seeded_wall, _new_baseline_iso = _seed_starvation_clock(state)
        _starvation_ref_wall: list[float] = [_seeded_wall]
        if _new_baseline_iso is not None:
            state["sleep_cycle_baseline_at"] = _new_baseline_iso
            try:
                from iai_mcp.daemon_state import update_state as _upd_baseline

                _upd_baseline(
                    lambda s: s.__setitem__(
                        "sleep_cycle_baseline_at", _new_baseline_iso
                    )
                )
            except Exception:  # noqa: BLE001 -- baseline persist is best-effort
                log.debug("sleep_cycle_baseline_at persist failed", exc_info=True)
        _last_pending_embed_mono: list[float] = [0.0]
        _pending_embed_inflight: list[bool] = [False]

        def _pending_embed_pass_sync() -> None:
            from iai_mcp import runtime_graph_cache as _rgc
            from iai_mcp.embed import embedder_for_store

            # warning level: the daemon runs with the root logger's WARNING
            # default, so anything quieter is invisible in launchd-stderr.
            log.warning("pending_embed_pass: start")
            try:
                _emb = embedder_for_store(store)
            except Exception as _emb_exc:  # noqa: BLE001 -- embed-less pass still heals sidecars
                _emb = None
                log.warning("pending_embed_pass: embedder unavailable: %s", _emb_exc)
            result = store.db.pending_embeddings_wake_sequence(embedder=_emb)
            log.warning("pending_embed_pass: result=%s", result)
            if _emb is not None:
                try:
                    from iai_mcp.foresight import refresh_from_anchor

                    refresh_from_anchor(store, _emb)
                except Exception:  # noqa: BLE001 -- anticipation is additive
                    log.debug("pending-embed pack refresh failed", exc_info=True)
            if isinstance(result, dict) and result.get("action") != "skip":
                try:
                    # Ambient churn, not an operator demand for freshness: drop
                    # only the persisted cache and let the kicked rebuild
                    # refresh behind the live bundle (stale-while-revalidate).
                    # The hard invalidate also nulled the in-process warm
                    # bundle, so every daytime heal degraded the recall hop
                    # stages to whole-table edge SQL until the rebuild landed.
                    _rgc.invalidate_at_root(store.root)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    _kick_drowsy_rgc_rebuild(store)
                except Exception:  # noqa: BLE001 -- best-effort
                    log.debug("pending-embed rgc kick failed", exc_info=True)

        async def _pending_embed_pass() -> None:
            await asyncio.to_thread(_pending_embed_pass_sync)

        async def lifecycle_tick() -> None:
            while not shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown.wait(),
                        timeout=LIFECYCLE_TICK_INTERVAL_SEC,
                    )
                    return
                except asyncio.TimeoutError:
                    pass

                try:
                    from iai_mcp.lifecycle_state import (
                        load_state as _load_lc,
                        save_state as _save_lc,
                    )
                    _lc_state = await asyncio.to_thread(_load_lc)
                    _now_utc = datetime.now(timezone.utc)
                    _expired, _ctx = _check_crisis_mode_expiry(_lc_state, _now_utc)
                    if _expired:
                        _lc_state["crisis_mode"] = False
                        _lc_state["crisis_mode_since_ts"] = None
                        await asyncio.to_thread(_save_lc, _lc_state)
                        try:
                            def _emit_expiry() -> None:
                                write_event(
                                    store,
                                    CRISIS_MODE_AUTO_EXPIRED,
                                    _ctx,
                                    severity="warning",
                                )
                            await asyncio.to_thread(_emit_expiry)
                        except Exception:  # noqa: BLE001 -- ledger emit failure non-fatal
                            log.debug(
                                "crisis_mode_auto_expired emit failed",
                                exc_info=True,
                            )
                    elif _ctx.get("backfilled_since_ts"):
                        _lc_state["crisis_mode_since_ts"] = _ctx["backfilled_since_ts"]
                        await asyncio.to_thread(_save_lc, _lc_state)
                except Exception:  # noqa: BLE001 -- expiry check MUST NOT crash lifecycle_tick
                    log.debug(
                        "lifecycle_tick crisis_mode expiry check failed",
                        exc_info=True,
                    )

                # Notify-only version probe, throttled tick-side to every 6h
                # (refresh_cache adds its own 24h TTL on the actual fetch).
                # wait_for bounds the tick's exposure: urlopen's timeout does
                # not cover a black-holed DNS resolve, and the tick must not
                # stall idle detection behind a PyPI lookup.
                if time.monotonic() - _last_version_probe_mono[0] > 6 * 3600.0:
                    _last_version_probe_mono[0] = time.monotonic()
                    try:
                        from iai_mcp.version_check import check_enabled, refresh_cache

                        if check_enabled():
                            await asyncio.wait_for(
                                asyncio.to_thread(refresh_cache), timeout=5.0,
                            )
                    except Exception:  # noqa: BLE001 — version probe must never crash the tick
                        log.debug("version check refresh failed", exc_info=True)

                try:
                    scanner_active = await asyncio.to_thread(
                        _heartbeat_scanner.is_active,
                    )
                    heartbeat_idle = await asyncio.to_thread(
                        _heartbeat_scanner.heartbeat_idle_30min,
                    )
                    # One OS-idle read per tick: sleep_eligible and the
                    # presence stamp must judge the same instant.
                    _os_idle_raw: "int | None" = None
                    _os_idle_source: "str | None" = None
                    try:
                        _os_idle_raw, _os_idle_source = await asyncio.to_thread(
                            _idle_detector.os_idle_time_sec,
                        )
                    except Exception:  # noqa: BLE001 -- idle source is best-effort
                        _os_idle_raw, _os_idle_source = None, None
                    os_idle_sec = _effective_os_idle(_os_idle_raw, _os_idle_source)
                    sleep_eligible = await asyncio.to_thread(
                        _idle_detector.sleep_eligible,
                        heartbeat_idle,
                        _os_idle_raw,
                    )

                    now_mono = time.monotonic()
                    idle_elapsed = now_mono - _last_active_monotonic[0]

                    try:
                        if _stamp_activity_presence(
                            state, idle_elapsed, os_idle_sec=os_idle_sec,
                        ):
                            # Snapshot: the live masks dict keeps mutating on
                            # the loop while the writer thread serialises.
                            await asyncio.to_thread(
                                _persist_keys,
                                {
                                    "activity_presence": dict(
                                        state["activity_presence"]
                                    )
                                },
                                "activity_presence",
                            )
                    except Exception:  # noqa: BLE001 -- presence is advisory
                        log.debug("activity presence stamp failed", exc_info=True)

                    _ds: dict = {}
                    try:
                        from iai_mcp.daemon_state import load_state as _load_ds
                        _ds = await asyncio.to_thread(_load_ds)
                        _force_rem = bool((_ds.get("force_rem_request") or {}).get("pending"))
                        _user_sleep = bool((_ds.get("user_sleep_request") or {}).get("pending"))
                        if _force_rem or _user_sleep:
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.FORCE_SLEEP,
                                    reason="force_sleep_request",
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass
                            if _state_machine.current_state is _LifecycleState.DROWSY:
                                try:
                                    await _state_machine.dispatch(
                                        _LifecycleEvent.FORCE_SLEEP,
                                        reason="force_sleep_drowsy_to_sleep",
                                    )
                                except (S2OscillationConflict, S2OscillationBlocked):
                                    pass
                            if _state_machine.current_state is _LifecycleState.SLEEP:
                                _now_iso = __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc,
                                ).isoformat()

                                def _honor_sleep_reqs(d: dict) -> None:
                                    if _force_rem:
                                        req = dict(d.get("force_rem_request") or {})
                                        req["pending"] = False
                                        req["honored_at"] = _now_iso
                                        d["force_rem_request"] = req
                                    if _user_sleep:
                                        req = dict(d.get("user_sleep_request") or {})
                                        req["pending"] = False
                                        req["honored_at"] = _now_iso
                                        d["user_sleep_request"] = req

                                from iai_mcp.daemon_state import update_state as _update_ds
                                # Later consumers in this tick must see the
                                # honored flags, not the stale pre-save dict.
                                _ds = await asyncio.to_thread(
                                    _update_ds, _honor_sleep_reqs,
                                )
                    except Exception:  # noqa: BLE001 -- FORCE_SLEEP dispatch is best-effort
                        log.debug("lifecycle_tick FORCE_SLEEP dispatch failed", exc_info=True)

                    try:
                        # Processed after the force-sleep block so the wake —
                        # the user's manual escape hatch — wins a same-tick tie.
                        _ds = await _consume_force_wake(_ds, _state_machine)
                    except Exception:  # noqa: BLE001 -- force-wake is best-effort
                        log.debug("lifecycle_tick force_wake dispatch failed", exc_info=True)

                    try:
                        from iai_mcp.daemon_state import daemon_state_path as _tick_daemon_state_path
                        from iai_mcp.fsm_reconcile import reconcile_fsm_state
                        from iai_mcp.lifecycle_state import (
                            lifecycle_state_path as _tick_lifecycle_state_path,
                        )
                        reconcile_fsm_state(
                            canonical_path=_tick_lifecycle_state_path(_store_root),
                            legacy_path=_tick_daemon_state_path(_store_root),
                            auto_correct=True,
                        )
                    except Exception:  # noqa: BLE001 -- reconcile is best-effort
                        pass

                    transition = _idle_transition_event(
                        scanner_active,
                        idle_elapsed,
                        sleep_eligible,
                        _ds,
                        os_idle_sec=os_idle_sec,
                        last_clean_cycle_wall=_starvation_ref_wall[0],
                        drowsy_after_sec=DROWSY_AFTER_SEC,
                        current_state=_state_machine.current_state,
                    )
                    if transition is not None:
                        if transition == "wake_refresh":
                            _last_active_monotonic[0] = now_mono
                        dispatch_row = TRANSITION_DISPATCH.get(transition)
                        if dispatch_row is None:
                            # An unlisted label must not crash the tick loop.
                            log.error(
                                "no dispatch entry for transition %r — "
                                "skipping this tick", transition,
                            )
                        else:
                            ev_name, reason, extra = dispatch_row
                            try:
                                await _state_machine.dispatch(
                                    getattr(_LifecycleEvent, ev_name),
                                    reason=reason,
                                    **extra,
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass

                    current = _state_machine.current_state
                    if _should_drain_on_drowsy_edge(_prev_lifecycle_state[0], current):
                        try:
                            from iai_mcp.capture import drain_capture_backlog

                            await asyncio.to_thread(
                                _run_drowsy_drain,
                                store,
                                drain_fn=drain_capture_backlog,
                                write_event_fn=write_event,
                            )
                            await asyncio.to_thread(
                                _run_bounded_capture_queue_drain,
                                store,
                                write_event_fn=write_event,
                            )
                        except Exception:  # noqa: BLE001 -- drowsy drain non-fatal
                            log.debug("lifecycle_tick drowsy drain failed", exc_info=True)

                        try:
                            _last_pending_embed_mono[0] = now_mono
                            await _pending_embed_pass()
                        except Exception:  # noqa: BLE001 -- wake sequence non-fatal
                            log.debug("lifecycle_tick pending_embeddings_wake_sequence failed", exc_info=True)

                    # A pending-embed backlog must never wait on the drowsy
                    # edge alone: a daemon that boots straight into SLEEP (or
                    # keeps getting restarted) never crosses that edge, and
                    # captured rows would stay invisible to semantic recall
                    # forever. The floor pass is windowed + RSS-bounded, and
                    # runs as a task guarded by an in-flight latch — a pass
                    # crawling on store-lock contention must not stall the
                    # lifecycle tick or stack a second pass on top of itself.
                    # FSM guard: no floor pass while the FSM is in SLEEP — the
                    # sleep pipeline owns the store then (its own steps run the
                    # wake sequence), and the rebuild serialization must not be
                    # contended by a background pass racing consolidation. The
                    # night-only window means daytime backlogs still drain on
                    # the 300s cadence.
                    if (
                        (now_mono - _last_pending_embed_mono[0]) >= PENDING_EMBED_FLOOR_SEC
                        and not _pending_embed_inflight[0]
                        and current is not _LifecycleState.SLEEP
                    ):
                        _last_pending_embed_mono[0] = now_mono
                        _pending_embed_inflight[0] = True

                        def _floor_pending_embed() -> None:
                            # Dedicated thread, NOT asyncio.to_thread: the
                            # shared default executor saturates under the
                            # daemon's recurring background work and a queued
                            # pass can wait there forever without running.
                            try:
                                if store.db.has_pending_rows():
                                    _pending_embed_pass_sync()
                            except Exception as _floor_exc:  # noqa: BLE001 -- floor pass non-fatal
                                log.warning(
                                    "pending-embed floor pass failed: %s",
                                    _floor_exc,
                                    exc_info=True,
                                )
                            finally:
                                _pending_embed_inflight[0] = False

                        import threading as _pe_threading
                        _pe_threading.Thread(
                            target=_floor_pending_embed,
                            name="pending-embed-floor",
                            daemon=True,
                        ).start()
                    if (
                        not _lock_downgraded_to_shared[0]
                        and current in (
                            _LifecycleState.WAKE,
                            _LifecycleState.DROWSY,
                        )
                    ):
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            _lock_downgraded_to_shared[0] = True
                            log.debug("daemon_lock_downgrade: EX→SH on first WAKE entry")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade failed", exc_info=True)

                    _prev_lifecycle_state[0] = current

                    # Publish the ANN/pool health counters into the live
                    # status surface every tick — a write-only counter
                    # observes nothing. Read via `iai-mcp daemon status`.
                    try:
                        _hp_pool = getattr(store.db, "_ro_pool", None)
                        state["ann_pool_health"] = {
                            "reuse_collisions": int(
                                getattr(store.db, "_reuse_collision_count", 0)
                            ),
                            "fence_reopens": int(
                                getattr(_hp_pool, "fence_reopen_count", 0) or 0
                            ),
                            "writer_fallbacks": int(
                                getattr(_hp_pool, "writer_fallback_count", 0) or 0
                            ),
                        }
                        await asyncio.to_thread(
                            _persist_keys, state, "ann_pool_health",
                        )
                    except Exception:  # noqa: BLE001 -- health publish is best-effort
                        pass

                    if current is _LifecycleState.SLEEP and not _sleep_pipeline_gate(_ds):
                        log.debug(
                            "lifecycle_tick: SLEEP pipeline skipped (scheduler_paused)",
                        )
                    elif current is _LifecycleState.SLEEP and _must_leave_sleep(
                        _ds,
                        os_idle_sec=os_idle_sec,
                        last_clean_cycle_wall=_starvation_ref_wall[0],
                    ):
                        # Night-only consolidation: outside the (learned) quiet
                        # window no cycle starts, whatever the idle state, and a
                        # daemon that finds itself in SLEEP out of hours (boot
                        # restored the pre-restart state, or the window closed
                        # mid-sleep) transitions back to WAKE so the awake path
                        # keeps the machine to itself. Explicit force bypasses
                        # via the gate's own force check.
                        log.debug(
                            "lifecycle_tick: SLEEP outside consolidation window — waking",
                        )
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.WAKE_SIGNAL,
                                reason="wake_outside_consolidation_window",
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass
                        except Exception:  # noqa: BLE001 -- wake-out must not crash the tick
                            log.debug("out-of-window wake dispatch failed", exc_info=True)
                    elif current is _LifecycleState.SLEEP and _sleep_backoff_active(
                        _ds, _sleep_fail_backoff_until[0], time.monotonic()
                    ):
                        log.debug(
                            "lifecycle_tick: SLEEP pipeline in failure backoff "
                            "(%.0fs left)",
                            _sleep_fail_backoff_until[0] - time.monotonic(),
                        )
                    elif current is _LifecycleState.SLEEP and _sleep_cooldown_active(
                        _ds,
                        _last_clean_cycle_mono[0],
                        time.monotonic(),
                        _last_clean_cycle_wall[0],
                    ):
                        log.debug(
                            "lifecycle_tick: SLEEP pipeline in post-cycle "
                            "cooldown (%.0fs left)",
                            _last_clean_cycle_mono[0]
                            + _sleep_cycle_cooldown_sec()
                            - time.monotonic(),
                        )
                    elif current is _LifecycleState.SLEEP:
                        # Consolidation is NEVER on the awake critical path:
                        # every step yields the moment a USER read is being
                        # served, even for a force-rem. The yield signal is the
                        # foreground beacon (stamped by recall/search dispatch),
                        # NOT raw socket traffic: ambient machinery — per-turn
                        # capture hooks, session-refresh polls, status probes
                        # from every open client session — keeps the socket
                        # warm around the clock, and a predicate keyed to it
                        # starves consolidation forever (a cycle that defers at
                        # its first chunk and never resumes, any hour of day).
                        def _interrupt_check() -> bool:
                            return _sleep_interrupt_predicate()

                        try:
                            await asyncio.to_thread(store.db.escalate_to_exclusive)
                            log.debug("daemon_lock_escalate: SH→EX for sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_escalate failed", exc_info=True)

                        try:
                            result = await asyncio.to_thread(
                                _sleep_pipeline.run, _interrupt_check,
                            )
                        except Exception:  # noqa: BLE001 -- a raising pipeline must not hold EX
                            # run() is designed not to raise; if it ever does,
                            # falling into the outer tick handler would skip
                            # the downgrade AND the backoff — the daemon would
                            # hold EXCLUSIVE while FSM=SLEEP and retry every
                            # tick, locking out every SHARED client. Synthesize
                            # a failed result so the normal downgrade + backoff
                            # path below runs.
                            log.warning(
                                "sleep pipeline raised — treating as failed step",
                                exc_info=True,
                            )
                            result = {
                                "failed_step": "pipeline_exception",
                                "interrupted": False,
                                "completed_steps": [],
                            }

                        # REM after a clean cycle: nightly insight synthesis
                        # (subscription/budget-gated inside the call) plus the
                        # rem-event/digest contract. Interval-gated so routine
                        # daytime cycles don't re-run it; a fresh force-rem
                        # always does. Runs BEFORE the session-start precache
                        # so the payload carries the new digest.
                        if (
                            result.get("failed_step") is None
                            and not result.get("interrupted")
                        ):
                            try:
                                await _maybe_run_rem(store, _ds)
                            except Exception:  # noqa: BLE001 -- REM MUST NOT block wake
                                log.debug("post-cycle REM failed", exc_info=True)

                        # --- WAKE hook (UNDER LOCK_EX, BEFORE downgrade) ---
                        try:
                            await asyncio.to_thread(_write_session_start_cache, store)
                        except Exception:  # noqa: BLE001 -- precache MUST NOT crash
                            log.debug("lifecycle_tick _write_session_start_cache failed", exc_info=True)
                        try:
                            from iai_mcp.memory_bank import write_processed_salience_top_n
                            await asyncio.to_thread(write_processed_salience_top_n, store)
                        except (ImportError, OSError, ValueError, RuntimeError) as exc:
                            log.debug("lifecycle_tick write_processed_salience_top_n failed: %s", exc)
                        try:
                            from iai_mcp.capture import drain_active_live_captures
                            _live_drain = await asyncio.to_thread(
                                drain_active_live_captures, store, exclude_session_id="-",
                            )
                            if _live_drain.get("events_inserted"):
                                await asyncio.to_thread(
                                    write_event, store, "active_live_drain_wake",
                                    _live_drain, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- drain MUST NOT crash
                            log.debug("lifecycle_tick active_live_drain failed: %s", _exc)
                        try:
                            from iai_mcp.provenance_buffer import flush_deferred_provenance
                            _prov_count = await asyncio.to_thread(
                                flush_deferred_provenance, store,
                            )
                            if _prov_count > 0:
                                await asyncio.to_thread(
                                    write_event, store, "deferred_provenance_flush_wake",
                                    {"count": _prov_count}, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- flush MUST NOT crash
                            log.debug("lifecycle_tick flush_deferred_provenance failed: %s", _exc)
                        try:
                            await asyncio.to_thread(_wake_hook_rebuild_if_cold, store)
                        except Exception as _exc:  # noqa: BLE001 -- best-effort
                            log.debug("lifecycle_tick wake-hook rebuild-if-cold failed: %s", _exc)

                        # Downgrade EX → SH after the consolidation window.
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            log.debug("daemon_lock_downgrade: EX→SH after sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade_post_sleep failed", exc_info=True)
                        if result.get("failed_step") is not None:
                            _sleep_fail_backoff_until[0] = (
                                time.monotonic() + SLEEP_FAIL_BACKOFF_SEC
                            )
                            log.warning(
                                "sleep pipeline failed at %s — next attempt "
                                "backed off %.0fs so awake clients are not "
                                "starved by the retry loop",
                                result.get("failed_step"),
                                SLEEP_FAIL_BACKOFF_SEC,
                            )
                        else:
                            _sleep_fail_backoff_until[0] = 0.0
                        if (
                            not result.get("interrupted", False)
                            and result.get("failed_step") is None
                            and not result.get("quarantine_triggered", False)
                            and len(result.get("completed_steps", [])) >= 5
                        ):
                            _last_clean_cycle_mono[0] = time.monotonic()
                            _last_clean_cycle_wall[0] = time.time()
                            _starvation_ref_wall[0] = _last_clean_cycle_wall[0]
                            try:
                                state["last_clean_cycle_at"] = datetime.now(
                                    timezone.utc
                                ).isoformat()
                                await asyncio.to_thread(
                                    _persist_keys, state, "last_clean_cycle_at",
                                )
                            except Exception:  # noqa: BLE001 -- clock persist is advisory
                                log.debug(
                                    "last_clean_cycle_at persist failed",
                                    exc_info=True,
                                )
                            still_idle_now = await asyncio.to_thread(
                                _heartbeat_scanner.heartbeat_idle_30min,
                            )
                            sleep_eligible_now = await asyncio.to_thread(
                                _idle_detector.sleep_eligible, still_idle_now,
                            )
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.SLEEP_CYCLE_DONE,
                                    reason="hibernate_on_sleep_cycle_done",
                                    still_idle=(still_idle_now and sleep_eligible_now),
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass

                    current = _state_machine.current_state
                    if (
                        current is _LifecycleState.HIBERNATION
                        and not _state_machine.shadow_run
                    ):
                        # Re-scan at the decision point: the tick-start
                        # snapshot is stale after a multi-minute pipeline
                        # run and can contradict the still_idle value that
                        # just entered HIBERNATION.
                        try:
                            _hb_active_now = await asyncio.to_thread(
                                _heartbeat_scanner.is_active,
                            )
                        except Exception:  # noqa: BLE001 -- probe failure is not demand
                            _hb_active_now = False
                        _demand_reason = _hibernation_demand_reason(
                            _hb_active_now,
                            mcp_socket.last_activity_ts,
                            _boot_socket_activity_mono[0],
                            time.monotonic(),
                        )
                        if _demand_reason is not None:
                            # Live demand: wake instead of exiting. On a
                            # blocked dispatch the state stays HIBERNATION
                            # and the next tick retries — never shut down
                            # while a live session is attached.
                            _last_active_monotonic[0] = time.monotonic()
                            try:
                                write_event(
                                    store,
                                    "lifecycle_hibernation_demand_wake",
                                    {"reason": _demand_reason},
                                    severity="info",
                                )
                            except (OSError, RuntimeError) as exc:
                                log.debug(
                                    "hibernation demand-wake event write failed: %s",
                                    exc,
                                )
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.REQUEST_ARRIVED,
                                    reason=_demand_reason,
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass
                            except Exception:  # noqa: BLE001 -- demand wake must not crash the tick
                                log.debug(
                                    "hibernation demand wake dispatch failed",
                                    exc_info=True,
                                )
                        else:
                            try:
                                write_event(
                                    store,
                                    "lifecycle_hibernation_exit",
                                    {
                                        "reason": "lifecycle_tick_hibernation",
                                        "shadow_run": False,
                                    },
                                    severity="info",
                                )
                            except (OSError, RuntimeError) as exc:
                                log.debug("lifecycle_hibernation_exit event write failed: %s", exc)
                            shutdown.set()
                            return
                except Exception:  # noqa: BLE001 -- lifecycle tick must NEVER crash
                    log.warning("lifecycle tick iteration failed", exc_info=True)

        lifecycle_tick_task = asyncio.create_task(lifecycle_tick())

        try:
            await shutdown.wait()
        finally:
            try:
                mcp_socket.shutdown_event.set()
            except (AttributeError, RuntimeError) as exc:
                log.debug("mcp_socket shutdown_event.set failed: %s", exc)
            try:
                _watchdog_stop.set()
            except (NameError, RuntimeError) as exc:
                log.debug("watchdog stop set failed: %s", exc)
            try:
                if _cascade_executor is not None:
                    _cascade_executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("cascade executor shutdown failed: %s", exc)
            _cancel_targets = [
                tick_task, audit_task, s4_task, cascade_task,
                mcp_socket_task,
                cpu_watchdog_task,
                lifecycle_tick_task,
            ]
            for t in _cancel_targets:
                t.cancel()
            await asyncio.gather(*_cancel_targets, return_exceptions=True)
            try:
                await store.disable_async_writes()
            except Exception as exc:  # noqa: BLE001 -- shutdown MUST complete
                log.debug("disable_async_writes on shutdown failed: %s", exc, exc_info=True)
            try:
                from iai_mcp.events import flush_event_buffer

                events_count = flush_event_buffer(store)
                if events_count > 0:
                    log.info("events buffer flushed on shutdown: count=%d", events_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("events buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                from iai_mcp.store import flush_record_buffer

                records_count = flush_record_buffer(store)
                if records_count > 0:
                    log.info("records buffer flushed on shutdown: count=%d", records_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("records buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                from iai_mcp.store import flush_edge_buffer

                edges_count = flush_edge_buffer(store)
                if edges_count > 0:
                    log.info("edges buffer flushed on shutdown: count=%d", edges_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("edges buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                write_event(store, "daemon_stopped", {"state": state.get("fsm_state")})
            except (OSError, RuntimeError) as exc:
                log.debug("daemon_stopped event write failed: %s", exc)
            _clear_user_shutdown_sentinel(state)
            try:
                state.pop("daemon_pid", None)
                state["daemon_stopped_at"] = datetime.now(timezone.utc).isoformat()

                def _stop_stamp(d: dict) -> None:
                    d.pop("daemon_pid", None)
                    d["daemon_stopped_at"] = state["daemon_stopped_at"]

                await asyncio.to_thread(update_state, _stop_stamp)
            except (OSError, ValueError) as exc:
                log.debug("final save_state failed: %s", exc)
            try:
                lifecycle_lock.release()
            except (OSError, RuntimeError) as exc:
                log.debug("lifecycle_lock release failed: %s", exc)
    finally:
        try:
            if getattr(store, "_write_queue", None) is not None:
                await store.disable_async_writes()
        except Exception as exc:  # noqa: BLE001 -- cancellation cleanup must finish
            log.debug("outer async-write cleanup failed: %s", exc, exc_info=True)
        try:
            if lifecycle_lock.is_held_by_self():
                lifecycle_lock.release()
        except (OSError, RuntimeError) as exc:
            log.debug("outer lifecycle-lock release failed: %s", exc)
        with _efs_holder["lock"]:
            _efs_holder["abandoned"] = True
            _efs_installed = _efs_holder["installed"]
        _restore_embedder_funnel(_efs_holder["orig"], _efs_installed)
    # A post-bind identity/config refusal (recorded by the serve-anyway
    # install's done-callback) refuses boot AFTER the graceful teardown
    # above completes, not instead of it — re-raising here still exits
    # non-zero, the same outcome an in-budget refusal produces directly.
    if _boot_refuse_exc[0] is not None:
        raise _boot_refuse_exc[0]
    return 0


from iai_mcp.daemon._watchdog import (  # noqa: E402 -- re-exported after main() so the package namespace is the single patchable source of truth
    HIPPEA_CASCADE_POLL_SEC,
    HIPPEA_CASCADE_MIN_INTERVAL_SEC,
    _last_cascade_completed_at,
    _cascade_executor,
    WATCHDOG_POLL_SEC,
    WATCHDOG_THRESHOLD_PERCENT,
    WATCHDOG_EVENT_COOLDOWN_SEC,
    WATCHDOG_SAMPLE_WINDOW,
    WATCHDOG_LIVENESS_POLL_SEC,
    WATCHDOG_WARN_POLL_SEC,
    WATCHDOG_PROBE_TIMEOUT_SEC,
    WATCHDOG_FAILURE_DEBOUNCE_N,
    WATCHDOG_RSS_HARD_CAP_BYTES,
    WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES,
    WATCHDOG_MAX_RECOVERIES,
    WATCHDOG_RECOVERY_WINDOW_SEC,
    WATCHDOG_COLD_START_GRACE_SEC,
    WATCHDOG_SLEEP_STALE_THRESHOLD_SEC,
    WATCHDOG_CRISIS_MODE_EXPIRY_SEC,
    _WATCHDOG_LOG_FD,
    _WATCHDOG_BLACKBOX_FD,
    _WATCHDOG_BLACKBOX_EPISODE_FIRED,
    _WATCHDOG_BLACKBOX_ENABLED,
    BOOT_LOCK_RETRY_ATTEMPTS,
    BOOT_LOCK_RETRY_BACKOFF_SEC,
    _last_overload_event_at,
    _last_rss_breadcrumb_at,
    _last_sleep_stale_started_at,
    _daemon_started_monotonic,
    _hippea_cascade_loop,
    _watchdog_active_task_names,
    _cpu_watchdog_loop,
    _next_poll_interval,
    _evaluate_watchdog,
    _check_sleep_cycle_staleness,
    _check_crisis_mode_expiry,
    _watchdog_state_dir,
    _watchdog_log_path,
    _watchdog_socket_path,
    _vm_pressure_level,
    _own_rss_bytes,
    _phys_footprint_bytes,
    _own_charged_bytes,
    _iso_now,
    _write_breadcrumb,
    _self_kill,
    _capture_blackbox,
    _open_exclusive_store_with_backoff,
    _load_recovery_timestamps,
    _probe_status_roundtrip,
    _watchdog_tick,
    _liveness_watchdog,
)

__all__ = [
    # lifecycle / main
    "main",
    "transition",
    "log",
    "serve_control_socket",
    "_hippo_health_check_on_boot",
    "_raise_fd_limit",
    "_run_drowsy_drain",
    "_should_drain_on_drowsy_edge",
    "_kick_drowsy_rgc_rebuild",
    "_wake_hook_rebuild_if_cold",
    "_store_is_empty",
    "_set_process_title",
    "_install_warm_embedder_override",
    "_restore_embedder_funnel",
    "_clear_user_shutdown_sentinel",
    "_USER_SHUTDOWN_FLAG",
    "_is_inside_window",
    "_update_pending_digest",
    "_write_session_start_cache",
    "_tick_body",
    "_scheduler_tick",
    "_s4_offline_loop",
    # FSM + tick constants
    "STATE_WAKE",
    "STATE_TRANSITIONING",
    "STATE_SLEEP",
    "STATE_DREAMING",
    "VALID_TRANSITIONS",
    "TICK_INTERVAL_SEC",
    "DEFAULT_CYCLE_COUNT",
    "S4_OFFLINE_INTERVAL_SEC",
    "S4_FIRST_ITER_GRACE_SEC",
    "SESSION_START_CACHE_PATH",
    "SESSION_START_CACHE_MAX_CHARS",
    "INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC",
    "_DAEMON_NOFILE_FLOOR_DEFAULT",
    # daemon_config
    "ErasureConfig",
    "_load_erasure_config",
    "PatSepConfig",
    "_load_patsep_config",
    "S2Config",
    "_load_s2_config",
    "SleepOverhaulConfig",
    "_load_sleep_overhaul_config",
    "ReconsolidationConfig",
    "_load_reconsolidation_config",
    "StcConfig",
    "_load_stc_config",
    "UserModelConfig",
    "_load_user_model_config",
    "SpatialConfig",
    "_load_spatial_config",
    "DmnConfig",
    "_load_dmn_config",
    "PaskConfig",
    "_load_pask_config",
    # watchdog
    "HIPPEA_CASCADE_POLL_SEC",
    "HIPPEA_CASCADE_MIN_INTERVAL_SEC",
    "_last_cascade_completed_at",
    "_cascade_executor",
    "WATCHDOG_POLL_SEC",
    "WATCHDOG_THRESHOLD_PERCENT",
    "WATCHDOG_EVENT_COOLDOWN_SEC",
    "WATCHDOG_SAMPLE_WINDOW",
    "WATCHDOG_LIVENESS_POLL_SEC",
    "WATCHDOG_WARN_POLL_SEC",
    "WATCHDOG_PROBE_TIMEOUT_SEC",
    "WATCHDOG_FAILURE_DEBOUNCE_N",
    "WATCHDOG_RSS_HARD_CAP_BYTES",
    "WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES",
    "WATCHDOG_MAX_RECOVERIES",
    "WATCHDOG_RECOVERY_WINDOW_SEC",
    "WATCHDOG_COLD_START_GRACE_SEC",
    "WATCHDOG_SLEEP_STALE_THRESHOLD_SEC",
    "WATCHDOG_CRISIS_MODE_EXPIRY_SEC",
    "_WATCHDOG_LOG_FD",
    "_WATCHDOG_BLACKBOX_FD",
    "_WATCHDOG_BLACKBOX_EPISODE_FIRED",
    "_WATCHDOG_BLACKBOX_ENABLED",
    "BOOT_LOCK_RETRY_ATTEMPTS",
    "BOOT_LOCK_RETRY_BACKOFF_SEC",
    "_last_overload_event_at",
    "_last_sleep_stale_started_at",
    "_daemon_started_monotonic",
    "_hippea_cascade_loop",
    "_watchdog_active_task_names",
    "_cpu_watchdog_loop",
    "_next_poll_interval",
    "_evaluate_watchdog",
    "_check_sleep_cycle_staleness",
    "_check_crisis_mode_expiry",
    "_watchdog_state_dir",
    "_watchdog_log_path",
    "_watchdog_socket_path",
    "_vm_pressure_level",
    "_own_rss_bytes",
    "_phys_footprint_bytes",
    "_own_charged_bytes",
    "_iso_now",
    "_write_breadcrumb",
    "_self_kill",
    "_capture_blackbox",
    "_open_exclusive_store_with_backoff",
    "_load_recovery_timestamps",
    "_probe_status_roundtrip",
    "_watchdog_tick",
    "_liveness_watchdog",
    "DAEMON_MEMORY_PRESSURE_KILL",
    "DAEMON_SLEEP_CYCLE_STALE",
    "DAEMON_WATCHDOG_NEEDS_OPERATOR",
    "DAEMON_WEDGE_KILL",
    "CRISIS_MODE_AUTO_EXPIRED",
]
