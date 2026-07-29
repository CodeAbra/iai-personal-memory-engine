#!/usr/bin/env python3
"""Export the longitudinal contradiction corpus as a portable JSON dataset.

The emitted file is self-contained: any memory system can be evaluated by
inserting `entries` in order and running `probes` against its own retrieval,
scoring the metrics named in `protocol`. No iai-mcp code is needed on the
consuming side.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

import argparse
import json

from bench.contradiction_longitudinal_claude import (
    SCALE_PRESETS,
    SYNTHETIC_FLIPS,
    generate_corpus,
)

PROTOCOL = {
    "name": "iai-longitudinal-v1",
    "insertion": (
        "insert entries strictly in list order (temporal order); one record "
        "per entry; timestamps must be monotonically increasing"
    ),
    "tiers": {
        "T1-ambient": (
            "plain inserts only — no update/supersede/contradict APIs; "
            "measures whether retrieval alone surfaces the CURRENT fact"
        ),
        "T2-native": (
            "each system may invoke its own update mechanism when an entry "
            "carries flip=correction; measures the system's designed "
            "supersede pathway"
        ),
    },
    "metrics": {
        "rescue_at_k": (
            "1 if any of the top-k retrieved texts contains `expects` as a "
            "substring; report k=1,5,10"
        ),
        "poison_at_k": (
            "for condition=post_flip only: 1 if a text containing "
            "`counterfact` ranks ABOVE the first text containing `expects` "
            "within top-k"
        ),
        "mrr": "reciprocal rank of the first text containing `expects`",
    },
    "conditions": {
        "post_flip": "cue asks for the current fact; expects = post-correction value",
        "historical_verbatim": (
            "cue explicitly asks what was said ORIGINALLY; expects = "
            "pre-correction value — verbatim history must remain reachable"
        ),
    },
}


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale", choices=sorted(SCALE_PRESETS), default="mvp",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    n_sessions, n_pre, n_post = SCALE_PRESETS[args.scale]
    entries, probes = generate_corpus(args.seed, n_sessions, n_pre, n_post)

    counterfact_by_topic = {
        topic: {"gold_after": gold_after, "gold_before": gold_before}
        for topic, _orig, _corr, gold_after, gold_before in SYNTHETIC_FLIPS
    }

    payload = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "scale": args.scale,
        "entries": [
            {
                "order": i,
                "session_id": e.session_id,
                "role": e.role,
                "text": e.text,
                "flip": (
                    "original" if e.is_flip_original
                    else "correction" if e.is_flip_correction
                    else None
                ),
                "topic": e.topic or None,
            }
            for i, e in enumerate(entries)
        ],
        "probes": [
            {
                "probe_id": p.probe_id,
                "cue": p.cue,
                "condition": p.condition,
                "topic": p.topic,
                "expects": p.expects,
                "counterfact": (
                    counterfact_by_topic[p.topic]["gold_before"]
                    if p.condition == "post_flip"
                    else counterfact_by_topic[p.topic]["gold_after"]
                ),
            }
            for p in probes
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(
        f"[longitudinal-export] seed={args.seed} scale={args.scale} "
        f"entries={len(payload['entries'])} probes={len(payload['probes'])} "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
