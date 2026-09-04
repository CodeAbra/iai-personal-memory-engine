from __future__ import annotations

import dataclasses

from iai_mcp.core._serializers import _payload_to_json
from iai_mcp.session import SessionStartPayload

# Fields the payload dataclass intentionally leaves out of the serialized dict.
# Empty by design: every field must round-trip through the wire, or be added
# here with a stated reason.
INTENTIONALLY_UNSERIALIZED_FIELDS: frozenset[str] = frozenset()


def test_payload_to_json_round_trips_recent_thread_directives_and_live_state():
    payload = SessionStartPayload(
        recent_thread="most-recent-work continuity block",
        directives="standing orders block",
        live_state="session-continuity state block",
    )

    result = _payload_to_json(payload)

    assert result["recent_thread"] == "most-recent-work continuity block"
    assert result["directives"] == "standing orders block"
    assert result["live_state"] == "session-continuity state block"


def test_payload_to_json_serializes_every_dataclass_field():
    required = {f.name for f in dataclasses.fields(SessionStartPayload)} - INTENTIONALLY_UNSERIALIZED_FIELDS
    serialized_keys = set(_payload_to_json(SessionStartPayload()))

    missing = required - serialized_keys
    assert not missing, (
        f"Fields added to SessionStartPayload but not serialized: {missing}. "
        "A field added to the payload dataclass must either be serialized in "
        "_payload_to_json or added to INTENTIONALLY_UNSERIALIZED_FIELDS with a "
        "stated reason."
    )


def test_payload_to_json_tolerates_an_older_payload_object_missing_new_fields():
    class LegacyPayload:
        def __init__(self) -> None:
            self.l0 = "l0"
            self.l1 = "l1"
            self.l2 = ["a", "b"]
            self.rich_club = "rich_club"
            self.total_cached_tokens = 10
            self.total_dynamic_tokens = 20
            self.breakpoint_marker = "--<cache-breakpoint>--"

    legacy = LegacyPayload()

    result = _payload_to_json(legacy)

    assert result["recent_thread"] == ""
    assert result["directives"] == ""
    assert result["live_state"] == ""
