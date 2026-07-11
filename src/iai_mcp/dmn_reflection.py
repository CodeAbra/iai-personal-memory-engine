from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_V4


def _first_50_chars(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return s[:50].rstrip()


class ReflectionAgent:

    def synthesize(self, store, window_hours: int) -> MemoryRecord:
        import json as _json

        from iai_mcp.events import query_events

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        def _parse_created(raw) -> "datetime | None":
            # created_at arrives as a datetime on one driver and an ISO string
            # on the other; a parser that handles only one shape silently
            # skips EVERY record and reports an empty day.
            if isinstance(raw, datetime):
                dt = raw
            elif isinstance(raw, str):
                try:
                    dt = datetime.fromisoformat(raw)
                except ValueError:
                    return None
            else:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        # Columns-only stream: the window count needs no record decrypt — the
        # only surfaces decrypted are the five community-label records below.
        captured_count = 0
        community_counts: Counter = Counter()
        community_newest: dict[str, tuple[datetime, str]] = {}
        for row in store.iter_record_columns(
            ["id", "community_id", "created_at", "provenance_json", "tags_json"],
            batch_size=2048,
        ):
            created = _parse_created(row.get("created_at"))
            if created is None or created < cutoff:
                continue
            # Reflections must never re-ingest prior reflections. The tag is
            # the reliable plaintext marker (provenance_json is encrypted at
            # rest on a keyed store, so the provenance probe below only works
            # on plaintext stores — a secondary net, not the gate).
            tags_raw = row.get("tags_json") or ""
            if isinstance(tags_raw, str) and '"dmn_reflection"' in tags_raw:
                continue
            prov_raw = row.get("provenance_json")
            if prov_raw and isinstance(prov_raw, str):
                try:
                    prov_list = _json.loads(prov_raw)
                except (TypeError, ValueError):
                    prov_list = []
                if any(
                    isinstance(p, dict) and p.get("synthesized_by") == "dmn_reflection"
                    for p in prov_list
                ):
                    continue
            captured_count += 1
            cid = row.get("community_id")
            if cid:
                cid_s = str(cid)
                community_counts[cid_s] += 1
                prev = community_newest.get(cid_s)
                if prev is None or created > prev[0]:
                    community_newest[cid_s] = (created, str(row.get("id")))

        topics: list[str] = []
        for cid_s, _n in community_counts.most_common(5):
            newest = community_newest.get(cid_s)
            if newest is None:
                continue
            try:
                rec = store.get(UUID(newest[1]))
            except (ValueError, TypeError):
                continue
            label = _first_50_chars(getattr(rec, "literal_surface", "") if rec else "")
            # A reflection record identified by its surface prefix must not
            # become a topic.
            if label and not label.startswith("Daily reflection:"):
                topics.append(label)

        recall_events = query_events(
            store,
            kind="recall_dispatched",
            since=cutoff,
            limit=10000,
        )
        recalled_count = len(recall_events)

        topics_str = "[" + ", ".join(topics) + "]"
        literal_surface = (
            f"Daily reflection: top topics were {topics_str}; "
            f"captured {captured_count} turns; "
            f"recalled {recalled_count} times."
        )

        provenance_entry: dict[str, Any] = {
            "synthesized_by": "dmn_reflection",
            "window_hours": int(window_hours),
            "topics": list(topics),
            "captured_count": int(captured_count),
            "recalled_count": int(recalled_count),
            "ts": now.isoformat(),
        }

        # A real embedding, never a zero vector: zero-vector records score
        # 0.000 against every cue yet still occupy ANN slots — they surface as
        # junk hits on every degraded recall lane. Embed failure marks the
        # record so the pipeline step SKIPS the insert instead of storing junk.
        try:
            from iai_mcp.embed import embedder_for_store
            embedding = list(embedder_for_store(store).embed(literal_surface))
        except Exception:  # noqa: BLE001 -- reflection must not raise on embed
            embedding = [0.0] * int(store.embed_dim)
            provenance_entry["embed_failed"] = True

        return MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=literal_surface,
            aaak_index="",
            embedding=embedding,
            community_id=None,
            centrality=0.5,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[provenance_entry],
            created_at=now,
            updated_at=now,
            language="en",
            tags=["dmn_reflection"],
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_V4,
            structure_hv=b"",
        )


class MetaAnalyst:

    _QUERY_LIMIT: int = 10000

    def snapshot(self, store, window_hours: int) -> dict[str, Any]:
        from iai_mcp.events import query_events

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        events_list = query_events(
            store,
            since=cutoff,
            limit=self._QUERY_LIMIT,
        )

        recall_count = 0
        capture_count = 0
        sleep_cycles_count = 0
        breach_count = 0
        erasure_count = 0

        for ev in events_list:
            kind = ev.get("kind") or ""
            # The event vocabulary the store actually writes: every dispatch
            # emits recall_dispatched, every insert passes the
            # pattern-separation gate.
            if kind == "recall_dispatched":
                recall_count += 1
            elif kind == "pattern_separation_pass":
                capture_count += 1
            elif kind == "cls_consolidation_run":
                # The one per-completed-cycle event that lands in the store
                # events table (sleep_step_completed goes only to the
                # lifecycle-log sink and is invisible to query_events).
                sleep_cycles_count += 1
            elif kind == "essential_variable_breach":
                breach_count += 1
            elif kind == "erasure_agent_pass":
                erasure_count += 1

        if (capture_count + erasure_count) > 0:
            average_record_count_delta = float(
                capture_count - erasure_count
            )
        else:
            average_record_count_delta = 0.0

        return {
            "recall_count": recall_count,
            "capture_count": capture_count,
            "sleep_cycles_count": sleep_cycles_count,
            "breach_count": breach_count,
            "erasure_count": erasure_count,
            "average_record_count_delta": average_record_count_delta,
            "window_hours": int(window_hours),
            "generated_at": now.isoformat(),
        }
