from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
        "iai-mcp-bench-falsifiability-deterministic-2026"
    )

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


DEFAULT_SEED = 42

_LANG_SAMPLES: dict[str, list[str]] = {
    "en": [
        "authentication uses JWT with refresh rotation",
        "db migration scheduled for Friday evening",
        "web cache invalidation on deploy",
        "cli subcommand for trajectory aggregation",
    ],
    "ru": [
        "авторизация использует JWT с обновлением токена",
        "миграция базы данных запланирована на пятницу",
        "инвалидация кэша при деплое",
    ],
    "ja": [
        "認証はJWTとリフレッシュローテーションを使用",
        "データベース移行は金曜日の夕方に予定",
    ],
    "ar": [
        "المصادقة تستخدم JWT مع تدوير الرمز",
        "ترحيل قاعدة البيانات مجدول ليوم الجمعة",
    ],
    "de": [
        "Authentifizierung verwendet JWT mit Token-Rotation",
        "Datenbankmigration für Freitagabend geplant",
    ],
}


def generate_synthetic_corpus(
    n_sessions: int = 30,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    rng = random.Random(seed)
    languages = list(_LANG_SAMPLES.keys())
    corpus: list[dict] = []

    for i in range(n_sessions):
        session_id = f"synth-{i:03d}"
        if i < len(languages):
            lang = languages[i]
        else:
            lang = rng.choice(languages)
        samples = _LANG_SAMPLES[lang]

        n_records = rng.randint(3, 8)
        records: list[dict] = []
        for k in range(n_records):
            text = samples[k % len(samples)]
            records.append({
                "id": str(uuid4()),
                "literal_surface": text,
                "language": lang,
                "tags": [f"topic:t{k % 3}", f"session:{session_id}"],
            })

        n_curiosity = max(0, 6 - (i // 5))
        curiosity_events: list[dict] = []
        for _ in range(n_curiosity):
            curiosity_events.append({
                "question_id": str(uuid4()),
                "entropy": float(0.5 + rng.random() * 0.5),
            })

        progress = i / max(1, n_sessions - 1)
        m1 = max(0.5, 6.0 * (1.0 - progress))
        m2 = min(1.0, 0.4 + progress * 0.5)
        m3 = max(1000.0, 3000.0 * (1.0 - 0.6 * progress))
        m4 = max(0.05, 0.5 * (1.0 - progress))
        m5 = float(n_curiosity)
        m6 = min(1.0, 0.4 + progress * 0.55)

        corpus.append({
            "session_id": session_id,
            "records": records,
            "curiosity_events": curiosity_events,
            "trajectory_metrics": {
                "m1": m1, "m2": m2, "m3": m3,
                "m4": m4, "m5": m5, "m6": m6,
            },
        })
    return corpus


def run_trajectory_bench(
    corpus: list[dict],
    store_path: Path | str | None = None,
) -> dict:
    from iai_mcp.trajectory import record_session_metrics

    cleanup: tempfile.TemporaryDirectory | None = None
    if store_path is None:
        cleanup = tempfile.TemporaryDirectory(prefix="iai-bench-traj-")
        path = Path(cleanup.name)
    else:
        path = Path(store_path)

    try:
        store = MemoryStore(path=path)

        m1t: list[float] = []
        m2t: list[float] = []
        m3t: list[float] = []
        m4t: list[float] = []
        m5t: list[float] = []
        m6t: list[float] = []
        for session in corpus:
            sid = session["session_id"]
            for q in session["curiosity_events"]:
                write_event(
                    store,
                    kind="curiosity_question",
                    data={
                        "question_id": q["question_id"],
                        "text": "",
                        "tier": "question",
                        "entropy": q["entropy"],
                        "turn": 1,
                        "triggered_by": [],
                    },
                    severity="info",
                    session_id=sid,
                )
            metrics = dict(session["trajectory_metrics"])
            record_session_metrics(store, session_id=sid, metrics=metrics)
            m1t.append(metrics["m1"])
            m2t.append(metrics["m2"])
            m3t.append(metrics["m3"])
            m4t.append(metrics["m4"])
            m5t.append(metrics["m5"])
            m6t.append(metrics["m6"])

        def _down(trend: list[float]) -> bool:
            return bool(trend) and trend[-1] < trend[0]

        def _up(trend: list[float]) -> bool:
            return bool(trend) and trend[-1] > trend[0]

        passed = (
            _down(m1t) and _up(m2t) and _down(m3t)
            and _down(m4t) and _down(m5t) and _up(m6t)
        )
        return {
            "m1_trend": m1t,
            "m2_trend": m2t,
            "m3_trend": m3t,
            "m4_trend": m4t,
            "m5_trend": m5t,
            "m6_trend": m6t,
            "passed": passed,
        }
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main(
    n_sessions: int = 30,
    seed: int = DEFAULT_SEED,
    real_logs_path: str | None = None,
    store_path: Path | str | None = None,
) -> int:
    if real_logs_path and Path(real_logs_path).exists():
        corpus = generate_synthetic_corpus(n_sessions=n_sessions, seed=seed)
    else:
        corpus = generate_synthetic_corpus(n_sessions=n_sessions, seed=seed)

    out = run_trajectory_bench(corpus, store_path=store_path)
    print(json.dumps(out))
    return 0 if out["passed"] else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench.trajectory")
    parser.add_argument("--n-sessions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--real-logs", dest="real_logs", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(
        n_sessions=args.n_sessions,
        seed=args.seed,
        real_logs_path=args.real_logs,
    ))
