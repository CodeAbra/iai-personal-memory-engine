"""Declarative per-knob tuning specs consumed by the nightly step.

Pure: no store, no query fetch, no import from the sleep pipeline. Every
function here reads only already-fetched event `data` dicts and returns
plain values, so the whole module unit-tests in isolation. Adding a knob's
signal is adding a `TUNING_SPECS` entry -- the nightly step never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

WINDOW_DAYS = 7
MAX_EVENTS = 5000

WAKE_DEPTH_LADDER: tuple[str, ...] = ("minimal", "standard", "deep")
WAKE_DEPTH_AUTO_CEILING = "standard"

MIN_SESSIONS_FOR_WAKE_DEPTH = 8
MIN_SESSIONS_FOR_INERTIA = 6

# A session that already spent most of the 3000-token steady-state budget
# with no follow-up recall is evidence the payload was already thorough.
LARGE_CACHE_TOKENS_THRESHOLD = 2500

# A gap this long between two sessions reads as the user returning after a
# break; shorter gaps read as continuous same-day work.
INERTIA_GAP_THRESHOLD_S = 4 * 3600.0

# abs(alpha - beta) a bool posterior must clear before it flips.
BOOL_HYSTERESIS_BAND = 2.0

# One-time prior mass that defends the current enum value the first night a
# knob is evaluated. Decoupled from `min_samples` on purpose: seeding to
# `min_samples` (in the teens) would take 40+ nights of implicit accrual
# (0.3/night) for a challenger to overtake -- structurally near-never.
INCUMBENT_SEED_MASS = 3.0

# task_support re-exposure probe (Design A): a bounded, self-contained
# experiment that resets the frozen posterior on open and decides recovery
# from its own probe-local follow-through, rather than one more weak vote
# a locked incumbent mass can never be outvoted by.
PROBE_INTERVAL_NIGHTS = 14
PROBE_WINDOW_SESSIONS = 3
PROBE_MAX_HOURS = 48
MIN_PROBE_FOLLOW_SESSIONS = 2
RECOVER_THRESHOLD = 0.34

# Honest lower bound before a zero-follow-through window is trusted to
# propose the down move -- a couple of quiet sessions is not a pattern.
MIN_DISPLAYED_SESSIONS = 4

# monotropism_depth: cue -> pre-rank community-gate concentration, mapped to
# a per-topic depth. K below this floor makes a 1/K uniform baseline
# meaningless (a 2-community gate always looks "concentrated").
K_MIN = 4
# Strictly below the 0.7 hit-reorder / proactive-check cliff -- automatic
# tuning must never cross it; only an explicit user profile_set can.
MAX_AUTO_DEPTH = 0.65
# Bounded per-night step, in both directions, so one loud window cannot
# swing a key from floor to near-cap in a single cycle.
MAX_DEPTH_DELTA = 0.15
DEPTH_DECAY = 0.9
DEPTH_EPSILON = 0.05
# Consecutive untouched windows before a key is pruned from the value dict
# AND its per_key posterior, bounding per_key growth regardless of how
# slowly a high value decays.
MAX_UNOBSERVED_WINDOWS = 4
# A community below this many cue touches in the window is one record's
# fluke, not a concentration signal.
MIN_TOUCHES_PER_KEY = 5
# Over the WINDOW_DAYS window, roughly a dozen gated cue touches is the
# honest floor below which a per-community concentration fraction is noise.
# Also serves as this spec's min_samples.
MIN_TOTAL_TOUCHES = 12


@dataclass(frozen=True)
class TuningSpec:
    knob: str
    kinds: tuple[str, ...]
    min_samples: int
    observe: Callable[..., tuple[Any, int, str]]
    apply: Callable[..., Any]


def clamp_enum_step(current: str, proposed: str, ladder: tuple[str, ...]) -> str:
    """At most one rung of `ladder` toward `proposed`."""
    if proposed not in ladder or current not in ladder:
        return current
    cur_i = ladder.index(current)
    prop_i = ladder.index(proposed)
    if prop_i == cur_i:
        return current
    step = 1 if prop_i > cur_i else -1
    return ladder[cur_i + step]


def bool_flip_allowed(alpha: float, beta: float, band: float = BOOL_HYSTERESIS_BAND) -> bool:
    return abs(alpha - beta) >= band


def seed_incumbent_posterior(
    knob: str, current: Any, posterior: dict, *, mass: float = INCUMBENT_SEED_MASS,
) -> dict:
    """Defend `current` with `mass` the first time `knob` has no posterior entry.
    Idempotent -- a knob already carrying a posterior entry is returned unchanged, so
    a sustained per-night accrual can overtake the seed rather than being re-armed forever.
    """
    existing = posterior.get(knob)
    if existing:
        return existing
    if isinstance(current, (bool, dict)):
        # The bool defense is BOOL_HYSTERESIS_BAND, not incumbent seeding --
        # seeding alpha/beta here would double-defend against the same
        # signal. A dict-schema knob has no single incumbent value to seed
        # an enum mass for -- its per_key posteriors do the defending,
        # per key, in the dict branch of bayesian_update.
        return {}
    seeded = {"alphas": {current: mass}}
    posterior[knob] = seeded
    return seeded


def _parse_timestamp(raw: Any) -> "datetime | None":
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _observe_wake_depth(
    events_by_kind: dict[str, list[dict]], *, current: str,
) -> tuple["str | None", int, str]:
    started = events_by_kind.get("session_started") or []
    recalled = events_by_kind.get("first_turn_recall") or []

    sessions: dict[str, dict] = {}
    for row in started:
        sid = row.get("session_id")
        if not sid:
            continue
        sessions.setdefault(sid, row)

    if not sessions:
        return None, 0, "implicit"

    recalled_ids = {row.get("session_id") for row in recalled if row.get("session_id")}

    deeper_votes = 0
    shallower_votes = 0
    for sid, row in sessions.items():
        if sid in recalled_ids:
            deeper_votes += 1
        elif (row.get("total_cached_tokens") or 0) >= LARGE_CACHE_TOKENS_THRESHOLD:
            shallower_votes += 1

    if deeper_votes > shallower_votes:
        proposed = "standard"
    elif shallower_votes > deeper_votes:
        proposed = "minimal"
    else:
        return None, len(sessions), "implicit"

    ceiling_idx = WAKE_DEPTH_LADDER.index(WAKE_DEPTH_AUTO_CEILING)
    proposed_idx = min(WAKE_DEPTH_LADDER.index(proposed), ceiling_idx)
    return WAKE_DEPTH_LADDER[proposed_idx], len(sessions), "implicit"


def _observe_inertia(
    events_by_kind: dict[str, list[dict]], *, current: bool,
) -> tuple["bool | None", int, str]:
    started = events_by_kind.get("session_started") or []

    sessions: dict[str, Any] = {}
    for row in started:
        sid = row.get("session_id")
        ts = row.get("timestamp")
        if not sid or not ts:
            continue
        sessions.setdefault(sid, ts)

    if len(sessions) < 2:
        return None, len(sessions), "implicit"

    timestamps = sorted(
        t for t in (_parse_timestamp(ts) for ts in sessions.values()) if t is not None
    )
    if len(timestamps) < 2:
        return None, len(sessions), "implicit"

    gaps = [(b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:])]
    long_gaps = sum(1 for g in gaps if g >= INERTIA_GAP_THRESHOLD_S)
    short_gaps = sum(1 for g in gaps if g < INERTIA_GAP_THRESHOLD_S)

    if long_gaps > short_gaps:
        return True, len(sessions), "implicit"
    if short_gaps > long_gaps:
        return False, len(sessions), "implicit"
    return None, len(sessions), "implicit"


def _apply_wake_depth(current: str, observed: str, posterior: dict) -> str:
    stepped = clamp_enum_step(current, observed, WAKE_DEPTH_LADDER)
    ceiling_idx = WAKE_DEPTH_LADDER.index(WAKE_DEPTH_AUTO_CEILING)
    capped_idx = min(WAKE_DEPTH_LADDER.index(stepped), ceiling_idx)
    return WAKE_DEPTH_LADDER[capped_idx]


def _apply_inertia(current: bool, observed: bool, posterior: dict) -> bool:
    kp = (posterior or {}).get("inertia_awareness", {})
    alpha = float(kp.get("alpha", 1.0))
    beta = float(kp.get("beta", 1.0))
    if not bool_flip_allowed(alpha, beta):
        return current
    return alpha >= beta


# The four join keys a `retrieval_used` row must carry to be usable in the
# task_support follow-through join. The two fallback emitters (L0
# fast-path, baseline_recall) legitimately lack them; a row missing any one
# is a non-qualifying adjacency BOUNDARY, never scored as a survivor.
_TASK_SUPPORT_JOIN_KEYS: tuple[str, ...] = (
    "timestamp", "session_id", "suggestion_ids", "suggestions_visible", "probe",
)
_MISSING = object()


def _task_support_row_qualifies(row: dict) -> bool:
    for key in _TASK_SUPPORT_JOIN_KEYS:
        if row.get(key, _MISSING) is _MISSING:
            return False
    return True


def _task_support_pair_follows(a: dict, b: dict) -> bool:
    """True when `b`'s hits contain a suggestion `a` showed that `b` did not
    already carry as one of its own hits (re-surfacing the same record `a`
    already served is not follow-through on a suggestion)."""
    a_hits = {str(x) for x in (a.get("hit_ids") or [])}
    b_hits = {str(x) for x in (b.get("hit_ids") or [])}
    a_suggestions = {str(x) for x in (a.get("suggestion_ids") or [])}
    return bool((b_hits - a_hits) & a_suggestions)


def _task_support_session_votes(
    rows: list[dict],
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Walk `rows` (ts-DESC, as fetched) into per-session follow votes.

    Returns `(probe_sessions, ordinary_sessions)`, each insertion-ordered
    `session_id -> followed`. A session with multiple qualifying pairs
    counts once; a follow on ANY of its pairs counts as a follow for that
    session. A pair spanning a non-qualifying neighbor (L0 / baseline_recall
    / a different session's interposed turn) is never scored -- abstained,
    not a false non-follow.
    """
    # `query_events` returns ts-DESC; the step passes that order through
    # unchanged (only `row["data"]` survives the fetch). Walk chronologically
    # so consecutive pairs reflect real same-session turn order.
    ordered = list(reversed(rows))

    probe_sessions: dict[str, bool] = {}
    ordinary_sessions: dict[str, bool] = {}

    for a, b in zip(ordered, ordered[1:]):
        if not (_task_support_row_qualifies(a) and _task_support_row_qualifies(b)):
            continue
        if a.get("session_id") != b.get("session_id"):
            continue
        if a.get("suggestions_visible") is not True:
            continue

        sid = str(a["session_id"])
        followed = _task_support_pair_follows(a, b)
        target = probe_sessions if a.get("probe") is True else ordinary_sessions
        target[sid] = target.get(sid, False) or followed

    return probe_sessions, ordinary_sessions


def _observe_task_support(
    events_by_kind: dict[str, list[dict]], *, current: str,
) -> tuple["str | None", int, str]:
    rows = events_by_kind.get("retrieval_used") or []
    probe_sessions, ordinary_sessions = _task_support_session_votes(rows)

    # Bound the probe to the first PROBE_WINDOW_SESSIONS distinct sessions
    # seen chronologically -- the re-exposure window is a small, bounded
    # experiment, not an open-ended re-observation.
    capped_probe = list(probe_sessions.items())[:PROBE_WINDOW_SESSIONS]
    if len(capped_probe) >= MIN_PROBE_FOLLOW_SESSIONS:
        n_probe = len(capped_probe)
        follow_fraction = sum(1 for _, f in capped_probe if f) / n_probe
        if follow_fraction >= RECOVER_THRESHOLD:
            return "cued_recognition", n_probe, "implicit"
        return None, 0, "implicit"

    if len(ordinary_sessions) >= MIN_DISPLAYED_SESSIONS:
        if not any(ordinary_sessions.values()):
            return "blank_recall", len(ordinary_sessions), "implicit"
        return None, 0, "implicit"

    return None, 0, "implicit"


def _apply_task_support_tuning(current: str, observed: str, posterior: dict) -> str:
    kp = posterior or {}
    # A probe verdict is signalled through the marker the scheduler
    # set on open; it SETS the value directly, bypassing hysteresis -- a
    # plain nudge could never outvote the locked incumbent mass. Apply
    # reads the marker but never clears it: the scheduler owns its
    # lifecycle (open sets it, a post-loop expiry-clear removes it).
    if kp.get("probe_active_until") and observed == "cued_recognition":
        return "cued_recognition"
    # Every other path is the ordinary down direction only -- apply never
    # sets cued_recognition on its own; recovery is the probe's alone.
    if observed != "blank_recall":
        return current
    alphas = kp.get("alphas", {}) or {}
    down_mass = float(alphas.get("blank_recall", 0.0))
    up_mass = float(alphas.get(current, 0.0))
    if not bool_flip_allowed(down_mass, up_mass):
        return current
    return "blank_recall"


def _observe_monotropism(
    events_by_kind: dict[str, list[dict]], *, current: dict,
) -> tuple["dict[str, float] | None", int, str]:
    """Per-community share of pre-rank gated cue touches, mapped to a depth
    via excess over the 1/K uniform baseline. Keyed by GATE community id --
    the step remaps to a topic name before `bayesian_update` runs."""
    rows = events_by_kind.get("retrieval_used") or []

    k: "int | None" = None
    for row in rows:
        candidate = row.get("community_k")
        if candidate is not None:
            k = candidate
            break
    if k is None or k < K_MIN:
        return None, 0, "implicit"

    touches: dict[str, int] = {}
    total = 0
    for row in rows:
        cid = row.get("cue_community_id")
        if cid is None:
            continue
        touches[cid] = touches.get(cid, 0) + 1
        total += 1

    if total < MIN_TOTAL_TOUCHES:
        return None, 0, "implicit"

    baseline = 1.0 / float(k)
    observed: dict[str, float] = {}
    for cid, count in touches.items():
        if count < MIN_TOUCHES_PER_KEY:
            continue
        conc = count / total
        if conc <= baseline:
            continue
        depth = max(0.0, min(MAX_AUTO_DEPTH, (conc - baseline) / (1.0 - baseline)))
        if depth > 0.0:
            observed[cid] = depth

    # A merely-flat window (enough touches, none concentrated) moves nothing
    # -- it must not fall through and decay every already-learned key.
    if not observed:
        return None, 0, "implicit"
    return observed, total, "implicit"


def _apply_monotropism_depth(current: dict, observed: dict, posterior: dict) -> dict:
    """`current`/`observed` are name-keyed by the time this runs (the step's
    remap seam runs before `bayesian_update`). `observed` carries this
    window's touched keys; the smoothed per-key mean lives in
    `posterior["per_key"][name]["mean"]`, already updated by
    `bayesian_update`'s dict branch. Untouched keys decay and, past
    MAX_UNOBSERVED_WINDOWS or below DEPTH_EPSILON, are pruned from both the
    value dict and the per_key posterior -- never raises."""
    current = dict(current) if isinstance(current, dict) else {}
    observed = dict(observed) if isinstance(observed, dict) else {}
    per_key = dict((posterior or {}).get("per_key", {}) or {})

    new_value: dict[str, float] = {}
    new_per_key: dict[str, dict] = {}

    for name, prior_value in current.items():
        if name in observed:
            continue
        sub = dict(per_key.get(name, {}) or {})
        unobserved = int(sub.get("unobserved_windows", 0)) + 1
        try:
            decayed = float(prior_value) * DEPTH_DECAY
        except (TypeError, ValueError):
            continue
        if unobserved >= MAX_UNOBSERVED_WINDOWS or decayed < DEPTH_EPSILON:
            continue
        sub["unobserved_windows"] = unobserved
        new_per_key[name] = sub
        new_value[name] = decayed

    for name, raw_depth in observed.items():
        sub = dict(per_key.get(name, {}) or {})
        try:
            smoothed = float(sub.get("mean", raw_depth))
        except (TypeError, ValueError):
            try:
                smoothed = float(raw_depth)
            except (TypeError, ValueError):
                continue
        capped = max(0.0, min(MAX_AUTO_DEPTH, smoothed))
        prior_value = float(current.get(name, 0.0)) if isinstance(current.get(name), (int, float)) else 0.0
        delta = max(-MAX_DEPTH_DELTA, min(MAX_DEPTH_DELTA, capped - prior_value))
        sub["unobserved_windows"] = 0
        new_per_key[name] = sub
        new_value[name] = max(0.0, prior_value + delta)

    posterior["per_key"] = new_per_key
    return new_value


# A knob with no entry here is not auto-tunable: the nightly step never
# proposes a value for it, and only a user set changes it.
TUNING_SPECS: dict[str, TuningSpec] = {
    "wake_depth": TuningSpec(
        "wake_depth",
        ("session_started", "first_turn_recall"),
        MIN_SESSIONS_FOR_WAKE_DEPTH,
        _observe_wake_depth,
        _apply_wake_depth,
    ),
    "inertia_awareness": TuningSpec(
        "inertia_awareness",
        ("session_started",),
        MIN_SESSIONS_FOR_INERTIA,
        _observe_inertia,
        _apply_inertia,
    ),
    "task_support": TuningSpec(
        "task_support",
        ("retrieval_used", "task_support_probe"),
        MIN_PROBE_FOLLOW_SESSIONS,
        _observe_task_support,
        _apply_task_support_tuning,
    ),
    "monotropism_depth": TuningSpec(
        "monotropism_depth",
        ("retrieval_used",),
        MIN_TOTAL_TOUCHES,
        _observe_monotropism,
        _apply_monotropism_depth,
    ),
}
