from __future__ import annotations



def test_synthetic_corpus_generates_30_sessions():
    from bench.trajectory import generate_synthetic_corpus

    corpus = generate_synthetic_corpus(n_sessions=30, seed=42)
    assert len(corpus) == 30
    for s in corpus:
        assert "session_id" in s
        assert "records" in s
        assert "curiosity_events" in s
        assert "trajectory_metrics" in s


def test_synthetic_corpus_deterministic_from_seed():
    from bench.trajectory import generate_synthetic_corpus

    a = generate_synthetic_corpus(n_sessions=5, seed=42)
    b = generate_synthetic_corpus(n_sessions=5, seed=42)
    assert [s["session_id"] for s in a] == [s["session_id"] for s in b]


def test_synthetic_corpus_multilingual():
    from bench.trajectory import generate_synthetic_corpus

    corpus = generate_synthetic_corpus(n_sessions=30, seed=42)
    languages: set[str] = set()
    for s in corpus:
        for r in s["records"]:
            languages.add(r.get("language", "en"))
    assert "en" in languages
    non_english = languages - {"en"}
    assert len(non_english) >= 1, (
        f"diverse-language fixture has only languages={languages}"
    )
    assert len(languages) >= 4


def test_synthetic_corpus_covers_six_metrics():
    from bench.trajectory import generate_synthetic_corpus

    corpus = generate_synthetic_corpus(n_sessions=30, seed=42)
    metric_keys: set[str] = set()
    for s in corpus:
        for k in s["trajectory_metrics"]:
            metric_keys.add(k)
    assert metric_keys >= {"m1", "m2", "m3", "m4", "m5", "m6"}


def test_trajectory_bench_runs_over_corpus(tmp_path):
    from bench.trajectory import (
        generate_synthetic_corpus,
        run_trajectory_bench,
    )

    corpus = generate_synthetic_corpus(n_sessions=6, seed=42)
    out = run_trajectory_bench(corpus, store_path=tmp_path)
    assert "m1_trend" in out
    assert "m2_trend" in out
    assert "m3_trend" in out
    assert "m4_trend" in out
    assert "m5_trend" in out
    assert "m6_trend" in out
    assert "passed" in out


def test_trajectory_bench_main_runs(tmp_path, capsys):
    from bench.trajectory import main

    code = main(n_sessions=5, store_path=tmp_path)
    assert code in (0, 1)


def test_trajectory_bench_accepts_real_logs_flag(tmp_path):
    from bench.trajectory import main

    code = main(
        n_sessions=3, real_logs_path=None, store_path=tmp_path,
    )
    assert code in (0, 1)
