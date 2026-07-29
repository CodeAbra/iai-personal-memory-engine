from __future__ import annotations

import json
import os
import secrets



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


def _patch_run_one_row(
    monkeypatch,
    raise_on_indices: set[int],
    success_template: dict | None = None,
) -> list[int]:
    counter = [0]
    default_success = success_template or {
        "question_id": None,
        "question_type": "test",
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

    def _wrapped(
        *,
        row_id,
        question,
        question_type,
        answer_session_ids,
        sessions,
        tmp_root,
        granularity,
        embedder_key,
        run_hybrid: bool = False,
        warm_lexical: bool = False,
        **_future_kwargs,
    ):
        idx = counter[0]
        counter[0] += 1
        if idx in raise_on_indices:
            raise RuntimeError("synthetic")
        out = dict(default_success)
        out["question_id"] = row_id
        out["question_type"] = question_type
        return out

    import bench.longmemeval_blind as mod

    monkeypatch.setattr(mod, "_run_one_row", _wrapped, raising=True)
    return counter


def test_preflight_exits_when_no_passphrase(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    _patch_adapter(monkeypatch)
    _patch_run_one_row(monkeypatch, raise_on_indices=set())

    rc = mod.main(["--limit", "1", "--out", str(out_path)])

    assert rc == 0, f"expected rc=0 (passphrase auto-filled); got {rc}"
    assert out_path.exists(), "output JSON must be written when pre-flight passes"


def test_preflight_passes_with_passphrase(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=[])

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists(), "happy path must write the output JSON"
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


def test_preflight_rejects_keyfile_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    _patch_adapter(monkeypatch)
    _patch_run_one_row(monkeypatch, raise_on_indices=set())

    rc = mod.main(["--limit", "1", "--out", str(out_path)])

    assert rc == 0, f"expected rc=0 (passphrase auto-filled); got {rc}"
    assert out_path.exists(), "output JSON must be written when pre-flight passes"


def test_error_row_classified_as_error_not_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=["q1", "q2"])
    _patch_run_one_row(monkeypatch, raise_on_indices={0, 1})

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "2", "--out", str(out_path)])
    assert rc == 0

    cp_path = tmp_path / "o.json.jsonl"
    assert cp_path.exists(), "checkpoint JSONL must be written"
    lines = [
        json.loads(line)
        for line in cp_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    for rec in lines:
        assert rec.get("classification") == "ERROR", (
            "every errored row must carry classification=ERROR: " + repr(rec)
        )
        assert isinstance(rec.get("error"), dict)
        assert "error_class" in rec["error"]

    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert len(out["errors"]) == 2
    assert out["n_errors"] == 2
    assert out["n_misses"] == 0
    assert out["n_hits"] == 0


def test_summary_line_separates_errors_from_misses(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=["q1", "q2", "q3"])
    _patch_run_one_row(monkeypatch, raise_on_indices={0, 1})

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "3", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    assert "hits=0" in err, "hits count missing from DONE line: " + err
    assert "misses=1" in err, "misses count missing from DONE line: " + err
    assert "errors=2" in err, "errors count missing from DONE line: " + err
    hi = err.index("hits=0")
    mi = err.index("misses=1")
    ei_ = err.index("errors=2")
    assert hi < mi < ei_, (
        "DONE summary must list hits / misses / errors in that order; "
        f"saw indices {hi}/{mi}/{ei_} in: {err}"
    )
