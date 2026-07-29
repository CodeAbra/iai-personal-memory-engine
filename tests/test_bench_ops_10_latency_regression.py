from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.perf


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password",
        lambda s, u, p: fake_store.__setitem__((s, u), p),
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None),
    )
    yield fake_store


def test_neural_map_small_n_p95_under_regression_ceiling(tmp_path: Path):
    from bench.neural_map import run_neural_map_bench

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        run = run_neural_map_bench(
            n=100, iterations=10, store_path=tmp_path / f"store{i}",
        )
        return float(run["latency_ms_p95"])

    min_p95 = best_of_n(_one_p95, n=3)

    assert min_p95 < 200.0, (
        f"OPS-10 regression: best-of-3 p95 {min_p95:.2f}ms > 200ms at N=100 "
        f"(2x D-SPEED ceiling — likely a real regression, not concurrency noise)"
    )
    assert min_p95 > 0.0


def test_neural_map_main_with_matrix_returns_int(tmp_path: Path):
    from bench import neural_map

    code = neural_map.main(ns=[50], iterations=3, store_path=tmp_path)
    assert code in (0, 1)


def test_neural_map_argparse_has_reference_flags():
    from bench import neural_map

    ns = neural_map._parse_args([
        "--n", "100",
        "--ref-p95-ms", "external=42.5",
        "--ref-p95-ms", "other=61.0",
    ])
    assert ns.refs_p95_ms == ["external=42.5", "other=61.0"]


def test_neural_map_comparative_gate_flips_passed_false_when_above_ref(tmp_path: Path):
    from bench import neural_map

    code = neural_map.main(
        ns=[50],
        iterations=3,
        store_path=tmp_path,
        refs_p95_ms={"external": 0.0001},
    )
    assert code == 1
