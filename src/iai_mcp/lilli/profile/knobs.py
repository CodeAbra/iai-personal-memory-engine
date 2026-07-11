from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KnobSpec:

    name: str
    phase: int
    default: Any
    description: str
    value_schema: str
    requirement_id: str


PROFILE_KNOBS: dict[str, KnobSpec] = {
    "monotropism_depth": KnobSpec(
        "monotropism_depth",
        1,
        {},
        "Monotropism depth per domain (voluntary tunnel; HIPPEA precision)",
        "dict:str:float_range:0.0..1.0",
        "AUTIST-01",
    ),
    "dunn_quadrant": KnobSpec(
        "dunn_quadrant",
        1,
        "neutral",
        "Sensory threshold x regulation posture (Dunn four-quadrant; "
        "drives HIPPEA precision weighting at runtime)",
        "enum:neutral|low-registration|seeking|sensitive|avoiding",
        "AUTIST-03",
    ),
    "literal_preservation": KnobSpec(
        "literal_preservation",
        1,
        "strong",
        "Verbatim vs semantic summary (raw always retained)",
        "enum:strong|medium|loose",
        "AUTIST-04",
    ),
    "demand_avoidance_tolerance": KnobSpec(
        "demand_avoidance_tolerance",
        1,
        "collaborative",
        "PDA-aware collaborative phrasing vs imperative",
        "enum:collaborative|neutral|imperative",
        "AUTIST-05",
    ),
    "masking_off": KnobSpec(
        "masking_off",
        1,
        True,
        "No small-talk, no performative empathy, literal pragmatics",
        "bool",
        "AUTIST-06",
    ),
    "task_support": KnobSpec(
        "task_support",
        1,
        "cued_recognition",
        "Blank-recall vs cued-recognition with adjacent suggestions (Bowler)",
        "enum:blank_recall|cued_recognition",
        "AUTIST-07",
    ),
    "interest_boost": KnobSpec(
        "interest_boost",
        1,
        0.0,
        "Salience amplification adjacent to monotropism domains",
        "float_range:0.0..1.0",
        "AUTIST-09",
    ),
    "inertia_awareness": KnobSpec(
        "inertia_awareness",
        1,
        False,
        "Ambient passive capture in high-inertia windows",
        "bool",
        "AUTIST-10",
    ),
    "camouflaging_relaxation": KnobSpec(
        "camouflaging_relaxation",
        1,
        0.0,
        "Detect over-formal writing and gradually relax communication formality",
        "float_range:0.0..1.0",
        "AUTIST-13",
    ),
    "scene_construction_scaffold": KnobSpec(
        "scene_construction_scaffold",
        1,
        True,
        "Scene-construction scaffold intensity for episodic encoding",
        "bool",
        "AUTIST-14",
    ),
    "wake_depth": KnobSpec(
        "wake_depth",
        1,
        "minimal",
        (
            "Session-start payload size: minimal=eager-30 (lazy default), "
            "standard=eager (full recent history), deep=full (<=2000 records)"
        ),
        "enum:minimal|standard|deep",
        "MCP-12",
    ),
}


PHASE_1_LIVE: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 1}
)
PHASE_2_DEFERRED: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 2}
)
PHASE_3_DEFERRED: frozenset[str] = frozenset(
    {name for name, spec in PROFILE_KNOBS.items() if spec.phase == 3}
)


assert len(PROFILE_KNOBS) == 11, (
    "10 autistic-kernel knobs + wake_depth = 11 sealed entries"
)
assert len(PHASE_1_LIVE) == 11, (
    "10 autistic-kernel knobs + MCP-12 wake_depth are live"
)
assert len(PHASE_2_DEFERRED) == 0, "PHASE_2_DEFERRED must be empty"
assert len(PHASE_3_DEFERRED) == 0, "PHASE_3_DEFERRED must be empty"


SIGNAL_WEIGHT: dict[str, float] = {
    "implicit": 0.3,
    "inferred": 0.5,
    "explicit": 1.0,
}


PROFILE_SENTINEL_UUID_STR = "00000000-0000-0000-0000-0000000000f1"


def default_state() -> dict[str, Any]:
    return {
        name: copy.deepcopy(spec.default)
        for name, spec in PROFILE_KNOBS.items()
        if spec.phase == 1
    }


def _validate(schema: str, value: Any) -> tuple[bool, str]:
    if schema == "bool":
        if isinstance(value, bool):
            return True, ""
        return False, f"value must be bool, got {type(value).__name__}"

    if schema.startswith("enum:"):
        allowed = schema[len("enum:"):].split("|")
        if value in allowed:
            return True, ""
        return False, f"value {value!r} not in enum {allowed}"

    if schema.startswith("int_range:"):
        bounds = schema[len("int_range:"):]
        try:
            lo_s, hi_s = bounds.split("..")
            lo, hi = int(lo_s), int(hi_s)
        except (ValueError, TypeError):
            return False, f"malformed int_range schema {schema!r}"
        if isinstance(value, bool):
            return False, "value must be int, got bool"
        if not isinstance(value, int):
            return False, f"value must be int, got {type(value).__name__}"
        if value < lo or value > hi:
            return False, f"value {value} out of range [{lo}, {hi}]"
        return True, ""

    if schema.startswith("float_range:"):
        bounds = schema[len("float_range:"):]
        try:
            lo_s, hi_s = bounds.split("..")
            lo, hi = float(lo_s), float(hi_s)
        except (ValueError, TypeError):
            return False, f"malformed float_range schema {schema!r}"
        if isinstance(value, bool):
            return False, "value must be float, got bool"
        if not isinstance(value, (int, float)):
            return False, f"value must be float, got {type(value).__name__}"
        v = float(value)
        if v < lo or v > hi:
            return False, f"value {v} out of range [{lo}, {hi}]"
        return True, ""

    if schema.startswith("dict:"):
        body = schema[len("dict:"):]
        key_type, _, val_type = body.partition(":")
        if not val_type:
            return False, f"malformed dict schema {schema!r}"
        if not isinstance(value, dict):
            return False, f"value must be dict, got {type(value).__name__}"
        for k, v in value.items():
            if key_type == "str" and not isinstance(k, str):
                return False, f"dict key must be str, got {type(k).__name__}"
            ok, reason = _validate(val_type, v)
            if not ok:
                return False, f"in key {k!r}: {reason}"
        return True, ""

    return False, f"unknown value_schema {schema!r}"


def profile_get(knob: str | None, state: dict[str, Any]) -> dict:
    if knob is None:
        live = {
            n: state.get(n, PROFILE_KNOBS[n].default)
            for n in sorted(PHASE_1_LIVE)
        }
        deferred = {}
        for n in sorted(PHASE_2_DEFERRED | PHASE_3_DEFERRED):
            spec = PROFILE_KNOBS[n]
            deferred[n] = {
                "status": "not-yet-implemented",
                "phase": spec.phase,
                "requirement_id": spec.requirement_id,
                "description": spec.description,
            }
        return {"live": live, "deferred": deferred, "total_knobs": 11}

    if knob in PHASE_1_LIVE:
        spec = PROFILE_KNOBS[knob]
        return {"knob": knob, "value": state.get(knob, spec.default)}

    if knob in PROFILE_KNOBS:
        spec = PROFILE_KNOBS[knob]
        return {
            "knob": knob,
            "status": "not-yet-implemented",
            "phase": spec.phase,
            "requirement_id": spec.requirement_id,
        }

    return {"knob": knob, "status": "unknown"}


def profile_set(
    knob: str,
    value: Any,
    state: dict[str, Any],
    *,
    store: "object | None" = None,
) -> dict:
    if knob not in PROFILE_KNOBS:
        return {"status": "error", "reason": "unknown knob", "knob": knob}

    spec = PROFILE_KNOBS[knob]
    if spec.phase == 2:
        return {
            "status": "error",
            "reason": "not yet activated",
            "knob": knob,
            "requirement_id": spec.requirement_id,
        }
    if spec.phase == 3:
        return {
            "status": "error",
            "reason": "not yet activated",
            "knob": knob,
            "requirement_id": spec.requirement_id,
        }

    ok, reason = _validate(spec.value_schema, value)
    if not ok:
        return {
            "status": "error",
            "reason": reason,
            "knob": knob,
            "schema": spec.value_schema,
        }

    old_value = state.get(knob, spec.default)
    state[knob] = value

    if store is not None and old_value != value:
        try:
            from datetime import datetime, timezone
            from iai_mcp.events import write_event
            write_event(
                store,
                kind="profile_updated",
                data={
                    "knob": knob,
                    "old": old_value,
                    "new": value,
                    "requirement_id": spec.requirement_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                severity="info",
            )
        except (OSError, RuntimeError, ValueError):
            pass

    return {"status": "ok", "knob": knob, "value": value}


def bayesian_update(
    knob: str,
    signal: str,
    observed: Any,
    state: dict,
    posterior: dict,
) -> tuple[Any, dict]:
    w = SIGNAL_WEIGHT.get(signal, 0.0)
    if w == 0.0:
        return state.get(knob, PROFILE_KNOBS[knob].default if knob in PROFILE_KNOBS else None), posterior

    spec = PROFILE_KNOBS.get(knob)
    if spec is None:
        return state.get(knob), posterior

    sch = spec.value_schema
    p = dict(posterior)
    kp = dict(p.get(knob, {}))

    current = state.get(knob, spec.default)

    if sch == "bool":
        alpha = float(kp.get("alpha", 1.0))
        beta = float(kp.get("beta", 1.0))
        if observed is True:
            alpha += w
        elif observed is False:
            beta += w
        else:
            return current, p
        kp["alpha"] = alpha
        kp["beta"] = beta
        new_value = alpha >= beta
    elif sch.startswith("enum:"):
        allowed = sch[len("enum:"):].split("|")
        alphas: dict[str, float] = dict(kp.get("alphas", {}))
        if observed not in allowed:
            return current, p
        alphas[observed] = alphas.get(observed, 1.0) + w
        kp["alphas"] = alphas
        if current in allowed and current not in alphas:
            alphas[current] = alphas.get(current, 1.0) + 0.001
        new_value = max(alphas.keys(), key=lambda k: alphas[k])
    elif sch.startswith("float_range:"):
        try:
            obs_f = float(observed)
        except (TypeError, ValueError):
            return current, p
        prev_sum = float(kp.get("weighted_sum", float(current) if isinstance(current, (int, float)) else 0.0))
        prev_wts = float(kp.get("total_weight", 0.0))
        new_sum = prev_sum + w * obs_f
        new_wts = prev_wts + w
        mean = new_sum / new_wts if new_wts > 0 else obs_f
        bounds = sch[len("float_range:"):]
        lo_s, hi_s = bounds.split("..")
        lo, hi = float(lo_s), float(hi_s)
        mean = max(lo, min(hi, mean))
        kp["weighted_sum"] = new_sum
        kp["total_weight"] = new_wts
        kp["mean"] = mean
        new_value = mean
    elif sch.startswith("int_range:"):
        try:
            obs_f = float(observed)
        except (TypeError, ValueError):
            return current, p
        prev_sum = float(kp.get("weighted_sum", float(current) if isinstance(current, (int, float)) else 0.0))
        prev_wts = float(kp.get("total_weight", 0.0))
        new_sum = prev_sum + w * obs_f
        new_wts = prev_wts + w
        mean = new_sum / new_wts if new_wts > 0 else obs_f
        bounds = sch[len("int_range:"):]
        lo_s, hi_s = bounds.split("..")
        lo, hi = int(lo_s), int(hi_s)
        new_value = max(lo, min(hi, int(round(mean))))
        kp["weighted_sum"] = new_sum
        kp["total_weight"] = new_wts
        kp["mean"] = mean
    elif sch.startswith("dict:"):
        if not isinstance(observed, dict):
            return current, p
        body = sch[len("dict:"):]
        _key_type, _, val_type = body.partition(":")
        per_key_posts: dict[str, dict] = dict(kp.get("per_key", {}))
        current_dict: dict = dict(current) if isinstance(current, dict) else {}
        for k, v in observed.items():
            sub_spec = val_type
            sub_kp = dict(per_key_posts.get(k, {}))
            if sub_spec.startswith("float_range:"):
                try:
                    obs_f = float(v)
                except (TypeError, ValueError):
                    continue
                prev_sum = float(sub_kp.get("weighted_sum", float(current_dict.get(k, 0.0))))
                prev_wts = float(sub_kp.get("total_weight", 0.0))
                new_sum = prev_sum + w * obs_f
                new_wts = prev_wts + w
                mean = new_sum / new_wts if new_wts > 0 else obs_f
                bounds = sub_spec[len("float_range:"):]
                lo_s, hi_s = bounds.split("..")
                lo, hi = float(lo_s), float(hi_s)
                mean = max(lo, min(hi, mean))
                sub_kp["weighted_sum"] = new_sum
                sub_kp["total_weight"] = new_wts
                sub_kp["mean"] = mean
                per_key_posts[k] = sub_kp
                current_dict[k] = mean
        kp["per_key"] = per_key_posts
        new_value = current_dict
    else:
        return current, p

    p[knob] = kp
    state[knob] = new_value
    return new_value, p


def profile_modulation_for_record(
    record,
    profile_state: dict,
    *,
    knobs_applied: dict | None = None,
) -> dict[str, float]:
    gains: dict[str, float] = {}

    md = profile_state.get("monotropism_depth", {})
    if isinstance(md, dict) and md:
        for tag in (record.tags or []):
            if tag.startswith("domain:"):
                dom = tag.split(":", 1)[1]
                if dom in md:
                    depth = md[dom]
                    try:
                        gains["monotropism_depth"] = 1.0 + float(depth)
                    except (TypeError, ValueError):
                        pass
                    if knobs_applied is not None:
                        knobs_applied["AUTIST-01"] = (
                            "profile.py:profile_modulation_for_record:monotropism_depth"
                        )
                    break

    ib = profile_state.get("interest_boost", 0.0)
    try:
        if float(ib) > 0:
            gains["interest_boost"] = 1.0 + float(ib)
            if knobs_applied is not None:
                knobs_applied["AUTIST-09"] = (
                    "profile.py:profile_modulation_for_record:interest_boost"
                )
    except (TypeError, ValueError):
        pass

    dq = profile_state.get("dunn_quadrant")
    if dq == "seeking":
        gains["dunn_quadrant"] = 1.2
        if knobs_applied is not None:
            knobs_applied["AUTIST-03"] = (
                "profile.py:profile_modulation_for_record:dunn_quadrant=seeking"
            )
    elif dq == "avoiding":
        gains["dunn_quadrant"] = 0.8
        if knobs_applied is not None:
            knobs_applied["AUTIST-03"] = (
                "profile.py:profile_modulation_for_record:dunn_quadrant=avoiding"
            )

    return gains
