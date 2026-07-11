#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")

REPO = Path(__file__).resolve().parent.parent


_PAYLOAD_CONSTRUCT = r"""
import sys, time
sys.path.insert(0, {src_path!r})
# Outer timer starts before import (interpreter + import + construction)
t_outer_start = time.monotonic()
from iai_mcp.embed import Embedder
# Inner timer: Embedder() construction only
t_inner_start = time.monotonic()
e = Embedder()
t_inner_end = time.monotonic()
# First encode — measures lazy weight load if any (no prior encode call)
text = {text!r}
t_first_enc_start = time.monotonic()
_ = e.embed(text)
t_first_enc_end = time.monotonic()
t_outer_end = time.monotonic()
print(f"construction_ms={{(t_inner_end - t_inner_start) * 1000:.3f}}")
print(f"first_encode_ms={{(t_first_enc_end - t_first_enc_start) * 1000:.3f}}")
print(f"import_plus_construction_ms={{(t_inner_end - t_outer_start) * 1000:.3f}}")
print(f"subprocess_total_ms={{(t_outer_end - t_outer_start) * 1000:.3f}}")
"""

_PAYLOAD_ENCODE = r"""
import sys, time
sys.path.insert(0, {src_path!r})
from iai_mcp.embed import Embedder
e = Embedder()
text = {text!r}
# Prime: one encode before measurement (ensures any lazy load is done)
_ = e.embed(text)
# Measure n_measure warm encodes
n_measure = {n_measure}
samples = []
for _ in range(n_measure):
    t0 = time.monotonic()
    e.embed(text)
    t1 = time.monotonic()
    samples.append((t1 - t0) * 1000)
print("encode_ms=" + ",".join(f"{{x:.4f}}" for x in samples))
"""

_PAYLOAD_RSS = r"""
import sys, resource
sys.path.insert(0, {src_path!r})
from iai_mcp.embed import Embedder
e = Embedder()
rss_post_construct_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
text = {text!r}
_ = e.embed(text)
rss_post_encode_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
import platform as _plat
is_mac = (_plat.system() == "Darwin")
def to_mb(raw):
    return raw / 1048576 if is_mac else raw / 1024
print(f"rss_post_construct_mb={{to_mb(rss_post_construct_raw):.1f}}")
print(f"rss_post_encode_mb={{to_mb(rss_post_encode_raw):.1f}}")
print(f"rss_post_construct_raw={{rss_post_construct_raw}}")
print(f"rss_post_encode_raw={{rss_post_encode_raw}}")
print(f"unit_is_bytes={{is_mac}}")
"""


def _python() -> str:
    return sys.executable


def _run_subprocess(code: str, label: str) -> tuple[str, float]:
    t0 = time.monotonic()
    result = subprocess.run(
        [_python(), "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    wall = time.monotonic() - t0
    if result.returncode != 0:
        print(f"[ERROR] {label} stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Subprocess failed for {label}: rc={result.returncode}")
    return result.stdout.strip(), wall


def _parse_kv(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    n = len(samples)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    sorted_s = sorted(samples)
    return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac


def measure_construction(n_warm: int, src_path: str, text: str) -> dict:
    payload = _PAYLOAD_CONSTRUCT.format(src_path=src_path, text=text)
    samples_construction_ms: list[float] = []
    samples_first_encode_ms: list[float] = []
    samples_subprocess_wall_ms: list[float] = []
    raw_samples: list[dict] = []

    for i in range(n_warm):
        label = "cold_attempt" if i == 0 else f"warm_{i}"
        stdout, wall = _run_subprocess(payload, label)
        kv = _parse_kv(stdout)
        construction_ms = float(kv["construction_ms"])
        first_encode_ms = float(kv["first_encode_ms"])
        import_plus_construction_ms = float(kv["import_plus_construction_ms"])
        subprocess_wall_ms = wall * 1000.0
        raw_samples.append(
            {
                "sample": i,
                "label": label,
                "construction_only_ms": construction_ms,
                "first_encode_ms": first_encode_ms,
                "import_plus_construction_ms": import_plus_construction_ms,
                "subprocess_wall_ms": subprocess_wall_ms,
            }
        )
        if i > 0:
            samples_construction_ms.append(construction_ms)
            samples_first_encode_ms.append(first_encode_ms)
            samples_subprocess_wall_ms.append(subprocess_wall_ms)
        print(
            f"  [{label}] construction={construction_ms:.1f}ms  "
            f"first_encode={first_encode_ms:.1f}ms  "
            f"import+construct={import_plus_construction_ms:.1f}ms  "
            f"subprocess_wall={subprocess_wall_ms:.0f}ms"
        )

    result: dict = {
        "samples": raw_samples,
        "cold_attempt_construction_ms": raw_samples[0]["construction_only_ms"],
        "cold_attempt_first_encode_ms": raw_samples[0]["first_encode_ms"],
        "cold_attempt_subprocess_wall_ms": raw_samples[0]["subprocess_wall_ms"],
    }
    if samples_construction_ms:
        result["warm_construction_median_ms"] = statistics.median(
            samples_construction_ms
        )
        result["warm_first_encode_median_ms"] = statistics.median(
            samples_first_encode_ms
        )
        result["warm_construction_samples_ms"] = samples_construction_ms
        result["warm_first_encode_samples_ms"] = samples_first_encode_ms
        result["warm_subprocess_wall_median_ms"] = statistics.median(
            samples_subprocess_wall_ms
        )
    return result


def measure_encode(n_measure: int, src_path: str, text: str) -> dict:
    payload = _PAYLOAD_ENCODE.format(
        src_path=src_path,
        text=text,
        n_measure=n_measure,
    )
    print(f"  Running {n_measure} warm encodes (1 prior encode discarded) in subprocess...")
    stdout, wall = _run_subprocess(payload, "encode_latency")
    kv = _parse_kv(stdout)
    samples = [float(x) for x in kv["encode_ms"].split(",") if x]
    median_ms = statistics.median(samples)
    p95_ms = _pct(samples, 95)
    p99_ms = _pct(samples, 99)
    print(
        f"  encode median={median_ms:.3f}ms  p95={p95_ms:.3f}ms  p99={p99_ms:.3f}ms"
    )
    return {
        "text": text,
        "n_prime_discarded": 1,
        "n_measured": n_measure,
        "samples_ms": samples,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
    }


def measure_rss(src_path: str, text: str) -> dict:
    payload = _PAYLOAD_RSS.format(src_path=src_path, text=text)
    print("  Running RSS measurement subprocess...")
    stdout, wall = _run_subprocess(payload, "rss")
    kv = _parse_kv(stdout)
    rss_post_construct_mb = float(kv["rss_post_construct_mb"])
    rss_post_encode_mb = float(kv["rss_post_encode_mb"])
    unit_is_bytes = kv["unit_is_bytes"] == "True"
    print(
        f"  RSS post-construct={rss_post_construct_mb:.1f}MB  "
        f"post-first-encode={rss_post_encode_mb:.1f}MB  "
        f"unit={'bytes (macOS)' if unit_is_bytes else 'KB (Linux)'}"
    )
    return {
        "rss_post_construct_mb": rss_post_construct_mb,
        "rss_post_encode_mb": rss_post_encode_mb,
        "rss_post_construct_raw": int(kv["rss_post_construct_raw"]),
        "rss_post_encode_raw": int(kv["rss_post_encode_raw"]),
        "unit_is_bytes_macos": unit_is_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedder warm construction cost benchmark")
    parser.add_argument(
        "--n-warm",
        type=int,
        default=5,
        help="Number of fresh-subprocess construction+first-encode samples (first = cold attempt)",
    )
    parser.add_argument(
        "--n-encode",
        type=int,
        default=50,
        help="Number of warm encode() calls to measure",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON output path for results",
    )
    args = parser.parse_args()

    src_path = str(REPO / "src")
    text = "what does alice remember about her daily routine"
    print(f"\n=== Embedder construction + encode cost benchmark ===")
    print(f"Python: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"n_warm={args.n_warm}  n_encode={args.n_encode}")
    print()

    print(f"[1] Construction + first-encode ({args.n_warm} fresh-subprocess samples):")
    construction = measure_construction(args.n_warm, src_path, text)
    print()

    print(f"[2] Warm per-encode latency (inside one subprocess):")
    encode = measure_encode(args.n_encode, src_path, text)
    print()

    print("[3] Standalone RSS (post-construct and post-first-encode):")
    rss = measure_rss(src_path, text)
    print()

    cold_note = (
        "NOTE: 'cold_attempt' is NOT a true cold-disk sample — "
        "macOS page cache was NOT purged (no sudo). "
        "Model is likely already cached from daemon/tests. "
        "Documented cold figure: ~8s."
    )
    warm_med = construction.get("warm_construction_median_ms", float("nan"))
    warm_fe_med = construction.get("warm_first_encode_median_ms", float("nan"))
    warm_wall_med = construction.get("warm_subprocess_wall_median_ms", float("nan"))

    print("=== SUMMARY ===")
    print(f"  {cold_note}")
    print(
        f"  cold_attempt construction:      {construction['cold_attempt_construction_ms']:.1f} ms  "
        f"(subprocess wall: {construction['cold_attempt_subprocess_wall_ms']:.0f} ms)"
    )
    print(
        f"  cold_attempt first_encode:      {construction['cold_attempt_first_encode_ms']:.1f} ms"
    )
    print(
        f"  warm construction median:       {warm_med:.1f} ms  "
        f"(subprocess wall median: {warm_wall_med:.0f} ms)"
    )
    print(
        f"  warm first_encode median:       {warm_fe_med:.1f} ms"
    )
    print(
        f"  warm encode median (hot):       {encode['median_ms']:.3f} ms"
    )
    print(
        f"  warm encode p95 (hot):          {encode['p95_ms']:.3f} ms"
    )
    print(
        f"  RSS post-construct:             {rss['rss_post_construct_mb']:.1f} MB"
    )
    print(
        f"  RSS post-first-encode (serving):{rss['rss_post_encode_mb']:.1f} MB"
    )

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.executable,
        "n_warm": args.n_warm,
        "construction": construction,
        "encode": encode,
        "rss": rss,
        "cold_page_cache_note": cold_note,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
