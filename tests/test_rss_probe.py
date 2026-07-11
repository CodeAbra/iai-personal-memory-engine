"""Unit coverage for the best-effort resident-set snapshot probe.

The probe samples the process's own RSS, a vmmap region/VM_ALLOCATE summary,
and the pyarrow + numba allocator counters. Every path is best-effort: a parse
failure yields a None/-1 field, never an exception, so the probe can be called
from the sleep-pipeline emit hooks without ever aborting a step.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# A captured `/usr/bin/vmmap -summary <pid>` tail. The region count is the LAST
# field of the all-caps TOTAL line; the mixed-case "Total" line carries no
# region count and must not be mistaken for it.
_VMMAP_SUMMARY_FIXTURE = textwrap.dedent(
    """\
    ==== Summary for process 12345
    ReadOnly portion of Libraries: Total=512.0M resident=128.0M
    Writable regions: Total=300.1M written=64.0M

    REGION TYPE                    VIRTUAL RESIDENT  DIRTY   SWAP VOLATILE NONVOL  EMPTY  REGION
    ===========                    ======= ========  =====   ==== ======== ======  =====  ======
    VM_ALLOCATE                       16K      16K     16K    0K       0K     0K     0K       1
    MALLOC guard page                  32K       0K      0K    0K       0K     0K     0K       8
    __TEXT                           120.0M    40.0M     0K    0K       0K     0K     0K     289

    TOTAL                            812.1M   114.2M  1600K    0K       0K    16K     0K     289
    TOTAL, minus reserved VM space   812.1M   114.2M  1600K    0K       0K    16K     0K     289
    Total                            812.1M
    """
)


def test_vmmap_summary_parse() -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import _parse_vmmap_summary

    region_count, vm_allocate_kib = _parse_vmmap_summary(_VMMAP_SUMMARY_FIXTURE)

    # Region count is the last field of the all-caps TOTAL line.
    assert region_count == 289
    # VM_ALLOCATE virtual column "16K" -> 16 KiB.
    assert vm_allocate_kib == 16


def test_vmmap_parse_ignores_mixed_case_total_decoy() -> None:
    # A line that starts with the mixed-case word "Total" (no region count) must
    # not be read as the region-count line. The all-caps TOTAL line is the only
    # source.
    from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import _parse_vmmap_summary

    decoy = "Total                            812.1M\n"
    region_count, vm_allocate_kib = _parse_vmmap_summary(decoy)
    assert region_count is None
    assert vm_allocate_kib is None


def test_vmmap_parse_tolerates_garbage() -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import _parse_vmmap_summary

    for text in (
        "",
        "TOTAL\n",  # no trailing numeric field
        "TOTAL not_a_number\n",  # non-numeric last field
        "VM_ALLOCATE\n",  # no size column
        "VM_ALLOCATE bogus\n",  # unparseable size token
        "garbage line with no markers\n",
        "TOTAL 1 2 3 abc\n",  # last field non-numeric
    ):
        region_count, vm_allocate_kib = _parse_vmmap_summary(text)
        # Never raises; missing fields come back None.
        assert region_count is None or isinstance(region_count, int)
        assert vm_allocate_kib is None or isinstance(vm_allocate_kib, int)


def test_humansize_to_kib() -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import _humansize_to_kib

    assert _humansize_to_kib("16K") == 16
    assert _humansize_to_kib("812.1M") == int(812.1 * 1024)
    assert _humansize_to_kib("1.2G") == int(1.2 * 1024 * 1024)


def test_capture_snapshot_keys_present() -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import capture_step_snapshot

    snap = capture_step_snapshot()
    assert isinstance(snap, dict)
    for key in (
        "rss_kib",
        "vmmap_region_count",
        "vm_allocate_kib",
        "pa_pool_bytes",
        "pa_pool_max",
        "numba_nrt_alloc_count",
        "numba_nrt_free_count",
    ):
        assert key in snap, f"snapshot missing key {key!r}: {sorted(snap)}"
    # A monotonic timestamp is always present.
    assert "t" in snap


def test_capture_snapshot_never_raises_when_vmmap_unavailable(monkeypatch) -> None:
    # Force the vmmap path to blow up; the snapshot must still return a dict.
    import iai_mcp.lilli.cycle.sleep_pipeline._rss_probe as probe

    def _boom(*_a, **_k):
        raise RuntimeError("vmmap exploded")

    monkeypatch.setattr(probe, "_vmmap_summary_counts", _boom)
    snap = probe.capture_step_snapshot()
    assert isinstance(snap, dict)
    assert "rss_kib" in snap


# The probe is loaded in isolation by file path (not via its parent package) so
# the guard measures the probe's OWN import surface. The parent package imports
# the events ledger, which transitively pulls in the storage stack — that is
# unrelated to the probe and would mask the very thing this fence verifies.
_PROBE_PATH = (
    REPO_ROOT
    / "src"
    / "iai_mcp"
    / "lilli"
    / "cycle"
    / "sleep_pipeline"
    / "_rss_probe.py"
)

_FENCE_SCRIPT = textwrap.dedent(
    f"""
    import sys
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rss_probe_fence", {str(_PROBE_PATH)!r}
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.capture_step_snapshot()

    forbidden = ("iai_mcp.hippo", "iai_mcp.store", "iai_mcp.crypto")
    offenders = [m for m in sys.modules if any(f in m for f in forbidden)]
    if offenders:
        print("OFFENDERS:" + ",".join(sorted(offenders)))
        sys.exit(2)
    sys.exit(0)
    """
).strip()


def test_snapshot_imports_no_crypto_or_storage() -> None:
    # The snapshot reads psutil + pyarrow pool + numba + a public vmmap summary
    # only. The probe's own module must never pull in the storage or crypto
    # surface (no key material ever crosses into the telemetry path).
    proc = subprocess.run(
        [sys.executable, "-c", _FENCE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"the probe module pulled in a fenced module. exit={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]!r}"
    )
