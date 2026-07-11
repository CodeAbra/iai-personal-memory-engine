#!/usr/bin/env python3
"""Contradiction-longitudinal falsifiability bench (pre-registered criteria)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "contradiction_longitudinal.jsonl",
    )
    args = parser.parse_args(argv)
    rows = load_rows(args.fixture)
    print(
        json.dumps(
            {
                "loaded_rows": len(rows),
                "fixture": str(args.fixture),
                "status": "harness_stub",
                "criteria": [
                    "B: post-flip cues — pipeline beats flat cosine",
                    "A: historical verbatim probes — superseded text still retrievable",
                    "No regression: B gain without A collapse",
                ],
            },
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
