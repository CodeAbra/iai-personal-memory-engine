from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID


def _resolve_path() -> Path:
    from iai_mcp.daemon_config import _load_user_model_config

    cfg = _load_user_model_config()
    return Path(os.path.expanduser(cfg.user_model_path))


@dataclass
class UserModel:

    top_recent_topics: list[str] = field(default_factory=list)
    tool_usage_freq: dict[str, int] = field(default_factory=dict)
    time_of_day_pattern: dict[int, int] = field(default_factory=dict)
    recent_projects: list[str] = field(default_factory=list)
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    aggregation_window_days: int = 30
    plasticity_gain: float = 1.0
    soft_knobs: dict[str, float] = field(default_factory=dict)


def default() -> UserModel:
    return UserModel()


def load() -> UserModel:
    path = _resolve_path()
    if not path.exists():
        return default()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default()

    if not isinstance(data, dict):
        return default()

    raw_ts = data.get("last_updated")
    try:
        last_updated = datetime.fromisoformat(raw_ts) if raw_ts else (
            datetime.now(timezone.utc)
        )
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return default()

    raw_tod = data.get("time_of_day_pattern", {}) or {}
    try:
        time_of_day_pattern = {int(k): int(v) for k, v in raw_tod.items()}
    except (TypeError, ValueError):
        time_of_day_pattern = {}

    return UserModel(
        top_recent_topics=list(data.get("top_recent_topics", []) or []),
        tool_usage_freq=dict(data.get("tool_usage_freq", {}) or {}),
        time_of_day_pattern=time_of_day_pattern,
        recent_projects=list(data.get("recent_projects", []) or []),
        last_updated=last_updated,
        aggregation_window_days=int(
            data.get("aggregation_window_days", 30)
        ),
    )


def save(model: UserModel) -> None:
    path = _resolve_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".user-model.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        payload = {
            "top_recent_topics": list(model.top_recent_topics),
            "tool_usage_freq": dict(model.tool_usage_freq),
            "time_of_day_pattern": dict(model.time_of_day_pattern),
            "recent_projects": list(model.recent_projects),
            "last_updated": model.last_updated.isoformat(),
            "aggregation_window_days": int(model.aggregation_window_days),
        }
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _first_50_chars(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return s[:50].rstrip()


class UserModelAggregator:

    def aggregate(
        self, store, window_days: int | None = None
    ) -> UserModel:
        from iai_mcp.daemon_config import _load_user_model_config
        from iai_mcp.events import query_events

        if window_days is None:
            window_days = _load_user_model_config().aggregation_window_days
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)

        recs = store.all_records()
        community_to_records: dict[UUID, list] = {}
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_counts: dict[UUID, int] = {
            cid: len(rlist) for cid, rlist in community_to_records.items()
        }

        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(
                rlist, key=lambda r: getattr(r, "created_at", now)
            )
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        surprises = query_events(
            store,
            kind="user_model_surprise",
            since=cutoff,
            limit=1000,
        )
        surprise_boost: Counter = Counter()
        for ev in surprises:
            actual = (ev.get("data") or {}).get("actual_topic")
            if isinstance(actual, str):
                surprise_boost[actual] += 1

        effective_counts: dict[UUID, int] = {}
        for cid, base_count in community_counts.items():
            label = community_labels.get(cid, "")
            effective_counts[cid] = base_count + surprise_boost.get(
                label, 0
            )
        sorted_cids = sorted(
            effective_counts.keys(),
            key=lambda c: effective_counts[c],
            reverse=True,
        )
        top_recent_topics = [
            community_labels[c]
            for c in sorted_cids[:10]
            if community_labels.get(c)
        ]

        events_list = query_events(store, since=cutoff, limit=10000)
        tool_counter: Counter = Counter()
        hour_counter: Counter = Counter()
        for ev in events_list:
            data = ev.get("data") or {}
            tool = data.get("tool")
            key = tool if isinstance(tool, str) else ev.get("kind")
            if isinstance(key, str):
                tool_counter[key] += 1

            ts = ev.get("ts")
            if ts is None:
                continue
            try:
                py = ts.to_pydatetime() if hasattr(
                    ts, "to_pydatetime"
                ) else ts
            except (TypeError, ValueError, AttributeError):
                continue
            if getattr(py, "tzinfo", None) is None:
                try:
                    py = py.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError, AttributeError):
                    continue
            try:
                hour_counter[int(py.hour)] += 1
            except (TypeError, ValueError, AttributeError):
                continue
        tool_usage_freq = dict(tool_counter.most_common(20))
        time_of_day_pattern = dict(hour_counter)

        project_events = query_events(
            store,
            kind="project_marker",
            since=cutoff,
            limit=1000,
        )
        projects: set[str] = set()
        for ev in project_events:
            p = (ev.get("data") or {}).get("project")
            if isinstance(p, str) and p:
                projects.add(p)
        recent_projects = sorted(projects)

        return UserModel(
            top_recent_topics=top_recent_topics,
            tool_usage_freq=tool_usage_freq,
            time_of_day_pattern=time_of_day_pattern,
            recent_projects=recent_projects,
            last_updated=now,
            aggregation_window_days=window_days,
        )


class UserModelPrefetcher:

    def prefetch(
        self,
        store,
        model: UserModel,
        top_k: int | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        from iai_mcp.daemon_config import _load_user_model_config

        if top_k is None:
            top_k = _load_user_model_config().prefetch_top_k
        if now is None:
            now = datetime.now(timezone.utc)

        if (
            not model.top_recent_topics
            and not model.time_of_day_pattern
        ):
            return []

        topic_rank: dict[str, int] = {
            t: i for i, t in enumerate(model.top_recent_topics)
        }

        top_hours: set[int] = set()
        if model.time_of_day_pattern:
            sorted_hours = sorted(
                model.time_of_day_pattern.keys(),
                key=lambda h: model.time_of_day_pattern[h],
                reverse=True,
            )
            top_hours = set(sorted_hours[:3])

        recs = store.all_records()
        community_to_records: dict[UUID, list] = {}
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(
                rlist, key=lambda r: getattr(r, "created_at", now)
            )
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        scored: list[tuple[float, datetime, str]] = []
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            label = community_labels.get(cid, "")
            score = 0.0
            if label and label in topic_rank:
                rank = topic_rank[label]
                score += (10 - rank) / 10.0
            created = getattr(r, "created_at", None)
            if created is not None:
                try:
                    if getattr(created, "tzinfo", None) is None:
                        created = created.replace(tzinfo=timezone.utc)
                    if created.hour in top_hours:
                        score += 0.3
                except (TypeError, ValueError, AttributeError):
                    pass
            if score > 0.0:
                scored.append((score, created or now, str(r.id)))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [rid for (_, _, rid) in scored[:top_k]]


def record_surprise(
    store,
    predicted_topic: str,
    actual_topic: str,
) -> None:
    from iai_mcp.daemon_config import _load_user_model_config
    from iai_mcp.events import write_event

    cfg = _load_user_model_config()
    write_event(
        store,
        "user_model_surprise",
        {
            "predicted_topic": str(predicted_topic),
            "actual_topic": str(actual_topic),
            "dry_run_mode": bool(cfg.dry_run),
        },
        severity="info",
    )
