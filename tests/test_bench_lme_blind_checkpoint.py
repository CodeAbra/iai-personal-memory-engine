from __future__ import annotations

import json

import pytest


class _StubLMESession:

    def __init__(self, qid: str, question_type: str = "test") -> None:
        self.queries = [
            {
                "question_id": qid,
                "query": f"q for {qid}",
                "question_type": question_type,
                "relevant_turn_ids": [f"sess-{qid}"],
            }
        ]
        self.session_id = f"sess-{qid}"
        self.turns = []


def _patch_adapter(monkeypatch, qids: list[str] | None = None) -> None:
    sessions = [_StubLMESession(qid) for qid in (qids or [])]

    def _stub_load_dataset(self, split="S"):
        yield from sessions

    from bench.adapters.longmemeval import LongMemEvalAdapter
    from bench.adapters.longmemeval_cleaned import CleanedLongMemEvalAdapter

    monkeypatch.setattr(
        LongMemEvalAdapter, "load_dataset", _stub_load_dataset, raising=True
    )
    monkeypatch.setattr(
        CleanedLongMemEvalAdapter,
        "load_dataset",
        _stub_load_dataset,
        raising=True,
    )

    # The cleaned adapter (the default dataset) resolves a HuggingFace Hub
    # revision in __init__ via a live ``repo_info`` network call, BEFORE
    # load_dataset runs. Stubbing only load_dataset leaves that network call
    # live, so the test flaked on any HF unreachability (offline, rate-limit,
    # slow network under load). Pin __init__ to a fixed revision so the whole
    # adapter is hermetic — no network on any path.
    def _stub_init(self, revision=None):
        self.revision = revision or "test-pinned-revision"

    monkeypatch.setattr(
        CleanedLongMemEvalAdapter, "__init__", _stub_init, raising=True
    )


def _success_row(qid: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "test",
        "classification": "SUCCESS",
        "r_at_5_retrieve": 0.0,
        "r_at_10_retrieve": 0.0,
        "r_at_5_pipeline": 0.0,
        "r_at_10_pipeline": 0.0,
        "pipeline_error": None,
        "query_tokens": 0,
        "inserted_text_tokens": 0,
        "n_haystack_sessions": 0,
        "n_turns_inserted": 0,
        "timing_seconds": {
            "insert": 0.0,
            "graph": 0.0,
            "recall_retrieve": 0.0,
            "recall_pipeline": 0.0,
            "total": 0.0,
        },
    }


def _error_row(qid: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "test",
        "classification": "ERROR",
        "error": {
            "error_class": "RuntimeError",
            "error": "synthetic",
        },
    }


def test_checkpoint_auto_cleans_when_prior_errors(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_success_row("q-ok-1")) + "\n")
        f.write(json.dumps(_error_row("q-err-1")) + "\n")
    cp_size_before = cp_path.stat().st_size
    assert cp_size_before > 0, "pre-flight: precondition checkpoint nonempty"

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    expected = (
        "Resuming from prior run with 1 errors; starting fresh. "
        "Pass --resume to keep checkpoint."
    )
    assert expected in err, (
        "verbatim auto-clean phrase missing from stderr; got:\n" + err
    )

    if cp_path.exists():
        assert cp_path.stat().st_size == 0, (
            "checkpoint must be empty after auto-clean; size="
            f"{cp_path.stat().st_size}"
        )

    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


def test_resume_flag_keeps_checkpoint_with_errors(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    payload = (
        json.dumps(_success_row("q-ok-1"))
        + "\n"
        + json.dumps(_error_row("q-err-1"))
        + "\n"
    )
    cp_path.write_text(payload, encoding="utf-8")

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(
        ["--limit", "0", "--out", str(out_path), "--resume"]
    )
    assert rc == 0

    err = capsys.readouterr().err
    assert "starting fresh" not in err, (
        "auto-clean must NOT fire when --resume is passed; stderr was:\n" + err
    )
    assert "resume:" in err, (
        "expected the 'resume: N rows already in checkpoint' log line: "
        + err
    )

    assert cp_path.exists()
    assert cp_path.read_text(encoding="utf-8") == payload, (
        "--resume must not mutate the checkpoint"
    )


def test_fresh_flag_force_cleans_clean_checkpoint(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_success_row("q-ok-1")) + "\n")
    assert cp_path.stat().st_size > 0

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(
        ["--limit", "0", "--out", str(out_path), "--fresh"]
    )
    assert rc == 0

    err = capsys.readouterr().err
    assert "--fresh" in err and "discarding" in err, (
        "expected --fresh force-clean log line; got:\n" + err
    )

    if cp_path.exists():
        assert cp_path.stat().st_size == 0

    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


def test_fresh_and_resume_mutually_exclusive(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    with pytest.raises(SystemExit) as ei:
        mod.main(
            [
                "--limit",
                "0",
                "--out",
                str(out_path),
                "--fresh",
                "--resume",
            ]
        )
    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "--fresh" in err and "--resume" in err, (
        "mutual-exclusion error must name both flags: " + err
    )


def test_default_behavior_clean_checkpoint_no_errors_keeps_it(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    payload = json.dumps(_success_row("q-ok-1")) + "\n"
    cp_path.write_text(payload, encoding="utf-8")

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    assert "starting fresh" not in err, (
        "auto-clean must NOT fire on a clean SUCCESS-only checkpoint: " + err
    )
    assert "resume:" in err, (
        "expected the existing resume log line on clean checkpoint: " + err
    )

    assert cp_path.exists()
    assert cp_path.read_text(encoding="utf-8") == payload
