from __future__ import annotations

import pytest


def test_check_z_pass_when_avx2_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf, "has_avx2", lambda: True)

    from iai_mcp.doctor import check_z_avx2_support

    result = check_z_avx2_support()

    assert result.passed is True, f"expected passed=True; got {result.passed}"
    assert result.status == "PASS", f"expected status='PASS'; got {result.status!r}"
    assert result.name == "(z) AVX2 CPU support", (
        f"expected name='(z) AVX2 CPU support'; got {result.name!r}"
    )


def test_check_z_fail_when_avx2_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf, "has_avx2", lambda: False)

    from iai_mcp.doctor import check_z_avx2_support

    result = check_z_avx2_support()

    assert result.passed is False, f"expected passed=False; got {result.passed}"
    assert result.status == "FAIL", f"expected status='FAIL'; got {result.status!r}"
    assert "AVX2" in result.detail, (
        f"detail must name AVX2; got {result.detail!r}"
    )
    assert "native memory store cannot load" in result.detail, (
        f"detail must explain the native memory store cannot load; got {result.detail!r}"
    )
    assert "iai-mcp memory store is unavailable" in result.detail, (
        f"detail must say store is unavailable; got {result.detail!r}"
    )


def test_run_diagnosis_includes_z_row() -> None:
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    assert len(results) == 30, (
        f"expected 30 rows after hippo rows (r/s/t) + (u) centrality + (v) native embedder + (ii) embed identity + (w) permanent-failed + (x) collapsed-timestamp + (aa) capture-state hygiene + (y) RSS plateau + (bb) nightly insight mint + (+) update-available check added; got {len(results)}: {names}"
    )
    assert results[-2].name.startswith("(z)"), (
        f"(z) must be last among the lettered rows; got {results[-2].name!r}"
    )
    assert results[-1].name.startswith("(+)"), (
        f"the (+) update row is appended after the lettered rows; got last name {results[-1].name!r}"
    )
