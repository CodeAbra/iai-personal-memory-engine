"""Dump a synthetic 16-byte rails golden that exercises the loader end to end.

Run under the project virtualenv:

    source .venv/bin/activate
    python scripts/golden/dump_rails.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "golden"))

from _common import write_golden  # noqa: E402

GENERATED_BY = "scripts/golden/dump_rails.py"
OUT_STEM = _REPO_ROOT / "fixtures" / "golden" / "_rails" / "round_trip"


def main() -> None:
    buf = bytes(range(16))
    written = write_golden(
        path_stem=OUT_STEM,
        arr_bytes=buf,
        dtype="u8",
        shape=[16],
        byte_order="le",
        subsystem="_rails",
        # source describes where the DATA came from; this fixture has no upstream
        # origin (the bytes are a synthetic 0..16 ramp), so "synthetic" is honest.
        # The generator script is recorded separately in generated_by.
        source="synthetic",
        generated_by=GENERATED_BY,
    )
    print(f"wrote {OUT_STEM}.bin (16 bytes) sha256={written}")


if __name__ == "__main__":
    main()
