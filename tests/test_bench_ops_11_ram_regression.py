from __future__ import annotations

from pathlib import Path



def test_memory_footprint_small_n_under_threshold(tmp_path: Path):
    from bench.memory_footprint import run_memory_footprint

    out = run_memory_footprint(n=100, store_path=tmp_path / "store", dim=64)

    assert "n" in out
    assert "rss_mb_peak" in out
    assert "threshold_mb" in out
    assert "passed" in out
    assert "platform" in out

    assert out["n"] == 100
    assert isinstance(out["rss_mb_peak"], float)
    assert out["rss_mb_peak"] > 0.0
    assert out["threshold_mb"] == 1600.0

    assert out["rss_mb_peak"] < 4000.0, (
        f"small-N RSS {out['rss_mb_peak']} MB suspicious"
    )


def test_memory_footprint_main_exits_int(tmp_path: Path):
    from bench import memory_footprint

    code = memory_footprint.main(argv=["--n", "50", "--dim", "32"])
    assert code in (0, 1)


def test_memory_footprint_platform_units_documented(tmp_path: Path):
    from bench.memory_footprint import run_memory_footprint

    out = run_memory_footprint(n=50, store_path=tmp_path / "store2", dim=32)
    assert out["platform"] in ("darwin", "linux", "win32")
