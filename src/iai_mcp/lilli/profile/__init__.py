"""Autistic-cognition profile registry -- 10 sealed knobs (9 AUTIST + 1 wake_depth),
Bayesian tuner, double_empathy invariant.

Retrieval-policy RL and trust refinement are exposed from lilli.profile.tuner.

This package re-exports the public surface of ``lilli.profile.knobs`` so
consumers can write ``from iai_mcp.lilli.profile import PROFILE_KNOBS`` (or
any of the symbols listed in ``__all__``) WITHOUT having to know that the
implementation lives in the ``knobs`` submodule. The package boundary is
the supported public-API path; the submodule path is an implementation
detail that may be reshaped in a future extraction.
"""

from iai_mcp.lilli.profile.knobs import (
    KnobSpec,
    PROFILE_KNOBS,
    LIVE_KNOB_NAMES,
    DEFERRED_KNOB_NAMES,
    SIGNAL_WEIGHT,
    PROFILE_SENTINEL_UUID_STR,
    default_state,
    profile_get,
    profile_set,
    bayesian_update,
    profile_modulation_for_record,
)
from iai_mcp.lilli.profile.persistence import (
    PROFILE_META_KEY,
    PROFILE_BLOB_VERSION,
    save_profile_state,
    load_profile_state,
    hydrate_profile,
)

__all__ = [
    "KnobSpec",
    "PROFILE_KNOBS",
    "LIVE_KNOB_NAMES",
    "DEFERRED_KNOB_NAMES",
    "SIGNAL_WEIGHT",
    "PROFILE_SENTINEL_UUID_STR",
    "default_state",
    "profile_get",
    "profile_set",
    "bayesian_update",
    "profile_modulation_for_record",
    "PROFILE_META_KEY",
    "PROFILE_BLOB_VERSION",
    "save_profile_state",
    "load_profile_state",
    "hydrate_profile",
]
