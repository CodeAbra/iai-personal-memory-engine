"""cmd_session_refresh_if_stale's first-seed branch must baseline the
per-turn delta from the source_watermark actually SERVED at session start
(the precache file), not the live store max at this later first-turn call
-- otherwise a gap opened between precache-write and this call is silently
absorbed into the baseline and can never surface as a delta.
"""
from __future__ import annotations

import argparse

from iai_mcp.cli import _capture


PRECACHE_REL = ".iai-mcp/.session-start-payload.cached.md"


def _run_seed(tmp_path, monkeypatch, *, live_max: str, precache_content: str | None):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".iai-mcp").mkdir(parents=True, exist_ok=True)
    if precache_content is not None:
        (tmp_path / PRECACHE_REL).write_text(precache_content, encoding="utf-8")

    monkeypatch.setattr(_capture, "get_max_created_at", lambda: live_max)

    rc = _capture.cmd_session_refresh_if_stale(argparse.Namespace(session_id="sid-baseline"))
    assert rc == 0
    return _capture.read_watermark("sid-baseline")


def test_seed_uses_embedded_precache_watermark_when_older_than_live(tmp_path, monkeypatch):
    embedded = "2026-09-01T00:00:00+00:00"
    live_max = "2026-09-06T12:00:00+00:00"
    seeded = _run_seed(
        tmp_path, monkeypatch,
        live_max=live_max,
        precache_content=(
            f"<!-- iai-mcp:source_watermark={embedded} -->\n\n## Identity\nhello"
        ),
    )
    assert seeded == _capture._utc_iso(embedded), (
        f"baseline must seed from the served pack watermark, got {seeded!r}"
    )
    assert seeded != _capture._utc_iso(live_max), (
        "baseline must NOT silently absorb the pre-session gap into the live max"
    )


def test_seed_falls_back_to_live_max_when_precache_absent(tmp_path, monkeypatch):
    live_max = "2026-09-06T12:00:00+00:00"
    seeded = _run_seed(tmp_path, monkeypatch, live_max=live_max, precache_content=None)
    assert seeded == _capture._utc_iso(live_max)


def test_seed_falls_back_to_live_max_when_precache_has_no_parseable_watermark(tmp_path, monkeypatch):
    live_max = "2026-09-06T12:00:00+00:00"
    seeded = _run_seed(
        tmp_path, monkeypatch,
        live_max=live_max,
        precache_content="# L0 identity\nfresh-cache-content-marker",
    )
    assert seeded == _capture._utc_iso(live_max)
