#!/usr/bin/env python3
"""
Measure zstd (level 3) compression ratio on the four text/JSON columns the
at-rest codec actually targets — literal_surface, provenance_json,
profile_modulation_gain_json, and events.data_json — over a deterministic
synthetic corpus shaped to match realistic capture sizes.

The corpus is generated with a fixed RNG seed so the script is byte-exact
reproducible. The column set is derived from the runtime constants in
iai_mcp.hippo rather than hardcoded so future schema changes propagate
without a separate edit here.

Output: bench/results/zstd_ratio.json — per-column and total byte sums plus
ratio reduction. The script exits 0 iff total reduction meets the 30% gate.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import zstandard as zstd

# Make `src/` importable when running the script directly out of the repo.
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from iai_mcp.hippo import (  # noqa: E402  (path setup must run first)
    _ENCRYPTED_EVENTS_COLUMNS,
    _ENCRYPTED_RECORD_COLUMNS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

SEED: int = 20260618
N_RECORDS: int = 500
ZSTD_LEVEL: int = 3
RATIO_GATE: float = 0.30
RESULTS_PATH: Path = _REPO_ROOT / "bench" / "results" / "zstd_ratio.json"


# Word-list shaped to mirror real captures: a mix of English conversational
# prose, code/JSON fragments, technical terms, and short Cyrillic snippets.
_WORDS: tuple[str, ...] = (
    "the", "user", "asked", "about", "memory", "recall", "and", "how", "it",
    "actually", "works", "with", "the", "ambient", "capture", "pipeline", "we",
    "discussed", "yesterday", "morning", "in", "the", "context", "of", "the",
    "session", "where", "she", "corrected", "my", "assumption", "about", "the",
    "tokeniser", "boundary", "between", "the", "wrapper", "and", "the", "core",
    "module", "the", "behaviour", "is", "consistent", "across", "macOS", "and",
    "Linux", "platforms", "but", "fails", "loudly", "on", "Windows", "because",
    "the", "filesystem", "lock", "semantics", "differ", "see", "the", "issue",
    "tracker", "for", "the", "full", "reproduction", "steps", "and", "the",
    "follow", "up", "patch", "that", "tightens", "the", "thread", "safety",
    "story", "in", "the", "shared", "connection", "wrapper", "which", "used",
    "to", "drop", "rows", "silently", "under", "concurrent", "fan", "out",
    "from", "the", "asyncio", "event", "loop", "during", "the", "sleep",
    "cycle", "consolidation", "pass", "and", "during", "the", "long",
    "running", "embedding", "warm", "up", "where", "the", "BERT", "model",
    "loads", "weights", "from", "disk", "into", "the", "Rust", "native",
    "extension", "via", "PyO3", "bindings", "compiled", "with", "maturin",
    "in", "release", "mode", "and", "exported", "back", "to", "Python",
    "as", "a", "single", "flat", "ndarray", "of", "float32", "values",
    "we", "then", "feed", "the", "vector", "into", "the", "hnswlib", "ANN",
    "index", "which", "owns", "the", "approximate", "nearest", "neighbour",
    "graph", "and", "returns", "labels", "with", "cosine", "distances",
    "ranked", "and", "trimmed", "to", "the", "k", "the", "caller", "asked",
    "for", "originally", "and", "we", "score", "the", "candidates", "by",
    "blending", "centrality", "and", "similarity", "with", "an", "optional",
    "rich", "club", "bonus", "that", "fires", "only", "when", "the", "node",
    "sits", "in", "the", "top", "five", "percent", "of", "the", "centrality",
    "distribution", "for", "the", "community", "the", "user", "is", "actively",
    "querying", "into", "during", "the", "live", "session", "right", "now",
    "this", "is", "exactly", "the", "shape", "of", "prose", "real",
    "literal", "surface", "captures", "tend", "to", "carry", "in", "this",
    "system", "когда", "пользователь", "пишет", "по-русски", "тоже", "тут",
    "встречается", "regular", "non", "ascii", "content", "and", "JSON",
    "fragments", "like", "structured", "objects", "with", "nested", "arrays",
)


def _word_burst(rng: random.Random, low: int, high: int) -> str:
    """Generate a span of {low..high} word tokens drawn from the corpus."""
    n = rng.randint(low, high)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _gen_literal_surface(rng: random.Random) -> str:
    """Realistic ambient-capture content. Length distribution is bimodal:
    most captures fall in the 300-800 byte range, with a long tail out
    to a few KB for longer turns or pasted code snippets."""
    if rng.random() < 0.85:
        text = _word_burst(rng, 40, 130)
    else:
        text = _word_burst(rng, 200, 500)
    # Sprinkle small structured fragments to mirror real prose+code mixes.
    if rng.random() < 0.4:
        text += " example payload: {\"role\": \"user\", \"text\": \""
        text += _word_burst(rng, 8, 25)
        text += "\"}"
    return text


def _gen_provenance_json(rng: random.Random) -> str:
    """Provenance list with 1-5 entries. Each entry pins ts/cue/session_id/
    role, matching the live shape captures actually carry."""
    n_entries = rng.randint(1, 5)
    entries = []
    for i in range(n_entries):
        entries.append(
            {
                "ts": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                      f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
                "cue": _word_burst(rng, 3, 12),
                "session_id": f"s-{rng.randint(100, 9999)}",
                "role": rng.choice(("user", "assistant")),
            }
        )
    return json.dumps(entries)


def _gen_profile_modulation_gain_json(rng: random.Random) -> str:
    """Profile modulation gain dict — 0 to 11 knob keys, mirroring the
    sealed knob registry. Empty {} when no knobs are active for this
    record."""
    n_keys = rng.randint(0, 11)
    knob_names = (
        "monotropism_pin", "epf_boost", "pattern_separation_gate",
        "hippea_arousal", "rem_pressure", "wake_depth", "schema_mine_rate",
        "centrality_blend", "rich_club_bonus", "active_forgetting",
        "stc_decay",
    )
    chosen = rng.sample(knob_names, n_keys)
    gain = {name: round(rng.uniform(-0.5, 0.5), 4) for name in chosen}
    return json.dumps(gain)


def _gen_events_data_json(rng: random.Random) -> str:
    """events.data_json — typically smaller than record columns. Mixed
    event kinds (capture, insert, reconsolidation) produce varying
    payload shapes."""
    kind = rng.choice(("capture", "insert", "reconsolidation", "recall"))
    if kind == "capture":
        return json.dumps(
            {"kind": kind, "session_id": f"s-{rng.randint(100, 9999)}",
             "tokens": rng.randint(50, 800)}
        )
    if kind == "insert":
        return json.dumps(
            {"kind": kind, "record_id": f"rec-{rng.randint(1000, 99999)}",
             "tier": rng.choice(("episodic", "semantic", "procedural"))}
        )
    if kind == "reconsolidation":
        return json.dumps(
            {"kind": kind, "prediction_error": round(rng.uniform(0, 1), 4),
             "reconsolidated_at": f"2026-{rng.randint(1, 12):02d}-"
             f"{rng.randint(1, 28):02d}T00:00:00Z"}
        )
    return json.dumps(
        {"kind": kind, "cue": _word_burst(rng, 2, 8),
         "k": rng.randint(1, 20), "hits": rng.randint(0, 20)}
    )


_COLUMN_GENERATORS = {
    "literal_surface": _gen_literal_surface,
    "provenance_json": _gen_provenance_json,
    "profile_modulation_gain_json": _gen_profile_modulation_gain_json,
    "data_json": _gen_events_data_json,
}


def _measure_column(
    cctx: zstd.ZstdCompressor,
    generator,
    rng: random.Random,
    n_values: int,
) -> dict:
    bytes_uncompressed = 0
    bytes_compressed = 0
    for _ in range(n_values):
        plaintext = generator(rng).encode("utf-8")
        bytes_uncompressed += len(plaintext)
        bytes_compressed += len(cctx.compress(plaintext))
    reduction = (
        1.0 - (bytes_compressed / bytes_uncompressed)
        if bytes_uncompressed
        else 0.0
    )
    return {
        "n_values": n_values,
        "bytes_uncompressed": bytes_uncompressed,
        "bytes_compressed": bytes_compressed,
        "ratio_reduction": round(reduction, 6),
    }


def main() -> int:
    columns = tuple(_ENCRYPTED_RECORD_COLUMNS) + tuple(_ENCRYPTED_EVENTS_COLUMNS)
    assert set(columns) == set(_COLUMN_GENERATORS.keys()), (
        f"column set drifted: runtime={set(columns)}, "
        f"generator={set(_COLUMN_GENERATORS.keys())}"
    )

    rng = random.Random(SEED)
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)

    per_column: dict[str, dict] = {}
    for column in columns:
        per_column[column] = _measure_column(
            cctx, _COLUMN_GENERATORS[column], rng, N_RECORDS
        )

    total_uncompressed = sum(c["bytes_uncompressed"] for c in per_column.values())
    total_compressed = sum(c["bytes_compressed"] for c in per_column.values())
    total_reduction = (
        1.0 - (total_compressed / total_uncompressed)
        if total_uncompressed
        else 0.0
    )
    total_n = sum(c["n_values"] for c in per_column.values())
    total = {
        "n_values": total_n,
        "bytes_uncompressed": total_uncompressed,
        "bytes_compressed": total_compressed,
        "ratio_reduction": round(total_reduction, 6),
    }

    passed = total_reduction >= RATIO_GATE
    report = {
        "n_records": N_RECORDS,
        "zstd_level": ZSTD_LEVEL,
        "seed": SEED,
        "per_column": per_column,
        "total": total,
        "pass": passed,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + trailing newline so two runs produce byte-identical files.
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verdict = "PASS" if passed else "FAIL"
    pct = total_reduction * 100
    print(
        f"total uncompressed={total_uncompressed} bytes  "
        f"compressed={total_compressed} bytes  "
        f"reduction={pct:.1f}%  {verdict}"
    )
    for column in columns:
        c = per_column[column]
        cpct = c["ratio_reduction"] * 100
        print(
            f"  {column:32s} n={c['n_values']:4d}  "
            f"uncompressed={c['bytes_uncompressed']:8d}  "
            f"compressed={c['bytes_compressed']:8d}  "
            f"reduction={cpct:.1f}%"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
