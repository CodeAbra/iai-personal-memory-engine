from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def maybe_open_task_support_probe(
    store: Any, now: datetime, durable_blob: dict, work_post: dict,
) -> bool:
    """Open a bounded task_support re-exposure probe at most once every
    PROBE_INTERVAL_NIGHTS, only while task_support is blank_recall and
    unpinned. Resets the frozen posterior mass on open -- an accumulated
    lock a plain nudge could never outvote -- and boot-caches the expiry so
    THIS process re-shows suggestions without waiting for a restart.
    """
    from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
        PROBE_INTERVAL_NIGHTS,
        PROBE_MAX_HOURS,
        _parse_timestamp,
    )

    knobs = durable_blob.get("knobs", {}) or {}
    if knobs.get("task_support", "cued_recognition") != "blank_recall":
        return False
    pins = durable_blob.get("pins", {}) or {}
    if "task_support" in pins:
        return False

    from iai_mcp.events import query_events, write_event

    latest = query_events(store, kind="task_support_probe", limit=1)
    if latest:
        opened_at = _parse_timestamp((latest[0].get("data") or {}).get("opened_night"))
        if opened_at is not None and (now - opened_at) < timedelta(days=PROBE_INTERVAL_NIGHTS):
            return False

    active_until = now + timedelta(hours=PROBE_MAX_HOURS)
    write_event(
        store,
        kind="task_support_probe",
        data={
            "active_until": active_until.isoformat(),
            "opened_night": now.isoformat(),
        },
        severity="info",
    )
    work_post["task_support"] = {"probe_active_until": active_until.isoformat()}

    try:
        from iai_mcp import core
        core.set_task_support_probe_active_until(active_until)
    except Exception as exc:  # noqa: BLE001 -- a caller without live core state must not crash the step
        logger.debug("task_support_probe_boot_cache_set_failed: %s", exc)

    return True


def clear_expired_task_support_probe_marker(work_post: dict, now: datetime) -> bool:
    """Remove an expired probe marker after apply had its one chance this
    run to consume the verdict. Both a recovered and an empty probe end
    with the marker gone, re-arming `seed_incumbent_posterior` for the
    knob's next evaluation."""
    kp = work_post.get("task_support")
    if not isinstance(kp, dict):
        return False
    marker = kp.get("probe_active_until")
    if not marker:
        return False
    from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import _parse_timestamp

    active_until = _parse_timestamp(marker)
    if active_until is None or now < active_until:
        return False
    new_kp = dict(kp)
    del new_kp["probe_active_until"]
    work_post["task_support"] = new_kp
    return True


def step_knob_tune(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp import core
    from iai_mcp.events import query_events, write_event
    from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
        MAX_EVENTS,
        TUNING_SPECS,
        WINDOW_DAYS,
        seed_incumbent_posterior,
    )
    from iai_mcp.lilli.profile.community_names import load_community_names
    from iai_mcp.lilli.profile.knobs import PROFILE_KNOBS, bayesian_update
    from iai_mcp.lilli.profile.persistence import load_profile_state, save_profile_state

    store = self._store
    durable_blob = load_profile_state(store) or {}
    pins = dict(durable_blob.get("pins", {}))
    pinned_knob_values = dict(durable_blob.get("knobs", {}))

    # PREVIOUS night's names -- KNOB_TUNE runs before the naming step in
    # this cycle's own order, so a gate id from tonight's cue traffic
    # resolves through last night's map. A gate id absent from it
    # contributes no observation (honest abstention), never a fake move.
    community_names = load_community_names(store).get("reverse_index", {})

    cutoff = self._now() - timedelta(days=WINDOW_DAYS)
    kinds_needed: set[str] = set()
    for spec in TUNING_SPECS.values():
        kinds_needed.update(spec.kinds)
    fetched: dict[str, list[dict]] = {
        kind: [
            row["data"]
            for row in query_events(store, kind=kind, since=cutoff, limit=MAX_EVENTS)
        ]
        for kind in kinds_needed
    }

    # Working copies -- an interrupt discards these untouched, leaving the
    # process globals pristine; only an uninterrupted pass commits them.
    work_state = dict(core._profile_state)
    work_post = dict(core._posterior_state)
    knob_rows: list[dict[str, Any]] = []
    moved: list[str] = []
    skipped: dict[str, str] = {}
    any_evaluated = False

    # The spec observe/apply pair is pure and holds no store handle, so the
    # probe-open side effects (write the task_support_probe event; reset the
    # durable posterior) live here, in the one caller with store access.
    probe_opened = maybe_open_task_support_probe(store, self._now(), durable_blob, work_post)

    for idx, knob in enumerate(sorted(PROFILE_KNOBS)):
        if self._check_interrupt(SleepStep.KNOB_TUNE, idx, interrupt_check):
            return False, {}

        prev_value = work_state.get(knob)

        if knob in pins:
            # A pin is a promise about the DURABLE value, not the in-process
            # one; a divergent live global must never be the thing that gets
            # written back over the user's setting.
            pinned_value = pinned_knob_values.get(knob, prev_value)
            work_state[knob] = pinned_value
            reason = "skipped_pinned_by_user"
            skipped[knob] = reason
            knob_rows.append({
                "knob": knob, "from": prev_value, "to": pinned_value, "reason": reason,
            })
            continue

        spec = TUNING_SPECS.get(knob)
        if spec is None:
            reason = "skipped_not_tunable"
            skipped[knob] = reason
            knob_rows.append({
                "knob": knob, "from": prev_value, "to": prev_value, "reason": reason,
            })
            continue

        events_by_kind = {k: fetched.get(k, []) for k in spec.kinds}
        observed, n, signal = spec.observe(events_by_kind, current=prev_value)

        if knob == "monotropism_depth" and isinstance(observed, dict):
            remapped: dict[str, float] = {}
            for gate_cid, depth in observed.items():
                name = community_names.get(gate_cid)
                if name is None:
                    continue
                # Two gate ids resolving to one name (a namespace-bridge
                # case) keep the stronger signal rather than one silently
                # overwriting the other.
                remapped[name] = max(depth, remapped.get(name, 0.0))
            observed = remapped or None
            if observed is None:
                n = 0

        if observed is None or n == 0:
            reason = "skipped_no_signal"
            skipped[knob] = reason
            knob_rows.append({
                "knob": knob, "from": prev_value, "to": prev_value, "reason": reason,
                "sample_count": n,
            })
            continue

        if n < spec.min_samples:
            reason = "skipped_insufficient_samples"
            skipped[knob] = reason
            knob_rows.append({
                "knob": knob, "from": prev_value, "to": prev_value, "reason": reason,
                "sample_count": n, "min_samples": spec.min_samples,
            })
            continue

        any_evaluated = True
        seed_incumbent_posterior(knob, prev_value, work_post)
        new_raw, new_post = bayesian_update(knob, signal, observed, work_state, work_post)
        work_post[knob] = new_post.get(knob, work_post.get(knob, {}))
        # monotropism_depth's apply reads this window's touched-key set
        # directly (needed to decay/prune untouched keys); every other
        # knob's apply gets bayesian_update's own proposed value, unchanged.
        apply_input = observed if knob == "monotropism_depth" else new_raw
        new_value = spec.apply(prev_value, apply_input, work_post[knob])
        work_state[knob] = new_value

        if new_value != prev_value:
            moved.append(knob)
            reason = "moved"
        else:
            reason = "evaluated"
        knob_rows.append({
            "knob": knob, "from": prev_value, "to": new_value, "reason": reason,
            "sample_count": n, "min_samples": spec.min_samples, "signal": signal,
        })

    # The scheduler owns the marker lifecycle: apply reads it but never
    # clears it, so a recovered probe and an empty probe both need
    # this post-loop clear -- it runs after apply had its one chance this
    # run to consume the verdict.
    marker_cleared = clear_expired_task_support_probe_marker(work_post, self._now())

    # Commit point: apply the copies to the live globals IN PLACE (never
    # rebind -- core.LIVE_KNOBS stays aliased to core._profile_state), then
    # persist. Everything above this line is reachable only after a full,
    # uninterrupted pass over the registry.
    core._profile_state.update(work_state)
    core._posterior_state.update(work_post)

    persisted = True
    if any_evaluated or probe_opened or marker_cleared:
        persisted = save_profile_state(
            store, knobs=dict(work_state), posterior=dict(work_post), pins=dict(pins),
        )

    # A dict-schema knob's from/to carries user-content topic vocabulary
    # (monotropism_depth today, any future dict-schema knob tomorrow) --
    # every row, including a populated dict on a no-signal/pinned night,
    # is redacted to a key-count before this event is written.
    redacted_rows = [
        {
            **row,
            **({"from": {"keys": len(row["from"])}} if isinstance(row.get("from"), dict) else {}),
            **({"to": {"keys": len(row["to"])}} if isinstance(row.get("to"), dict) else {}),
        }
        for row in knob_rows
    ]

    severity = "info" if persisted else "warning"
    write_event(
        store,
        kind="profile_tuned",
        data={
            "window_days": WINDOW_DAYS,
            "moved_count": len(moved),
            "persisted": persisted,
            "knobs": redacted_rows,
            "timestamp": self._now().isoformat(),
        },
        severity=severity,
    )

    return True, {
        "knobs_moved": moved,
        "knobs_skipped": skipped,
        "persisted": persisted,
    }
