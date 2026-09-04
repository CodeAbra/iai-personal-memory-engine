"""Opt-in latency evidence for the epistemic classifier on the capture
path. The classifier is pure capped regex running before store.insert()
-- driver-agnostic by construction, off the recall read path entirely.
Gated on a loose wall-clock ceiling (opt-in `--perf` lane); reported for
the record beyond pass/fail.
"""

from __future__ import annotations

import statistics
import time as _time

import pytest

from iai_mcp.epistemic_classify import (
    EPISTEMIC_TEXT_SCAN_CAP,
    classify_epistemic_status,
)

_N_CALLS = 500
_P95_CEILING_MS = 1.0


@pytest.mark.perf
def test_classify_epistemic_status_capture_path_latency_report():
    text = (
        "the response latency was roughly confirmed to be measured at "
        "around 400ms depending on load conditions, maybe a bit slower "
    ) * 40
    assert len(text) > EPISTEMIC_TEXT_SCAN_CAP

    # Warm call.
    classify_epistemic_status(text)

    samples_ms: list[float] = []
    for _ in range(_N_CALLS):
        t0 = _time.perf_counter()
        classify_epistemic_status(text)
        samples_ms.append((_time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    p95_ms = samples_ms[int(len(samples_ms) * 0.95) - 1]
    mean_ms = statistics.mean(samples_ms)

    print(
        f"classify_epistemic_status latency over {_N_CALLS} calls "
        f"(scan-cap-length input): mean={mean_ms:.4f}ms p95={p95_ms:.4f}ms"
    )
    assert p95_ms <= _P95_CEILING_MS, (
        f"p95 {p95_ms:.4f}ms exceeds the loose {_P95_CEILING_MS}ms ceiling"
    )
