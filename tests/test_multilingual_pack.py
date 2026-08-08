"""The multilingual opt-in: registry, config resolution, identity, lang verb.

The pack switches the whole store to one new vector generation — every
seam here exists to make a half-switched state impossible: the model is
chosen by the store config (never the service environment), the identity
stamp distinguishes pooling and prefixes, and the CLI refuses languages
outside the benchmarked set.
"""
from __future__ import annotations

import json
import sys

import pytest

from iai_mcp import embed as embed_mod
from iai_mcp.embed import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    MULTILINGUAL_MODEL_KEY,
    SUPPORTED_OPTIN_LANGS,
    _configured_model_key,
    _resolve_model_key,
    effective_model_identity,
)


def test_registry_shapes() -> None:
    for key, spec in MODEL_REGISTRY.items():
        assert spec["dim"] == 384, key
        assert spec["pool"] in {"cls", "mean"}, key
    ml = MODEL_REGISTRY[MULTILINGUAL_MODEL_KEY]
    assert ml["pool"] == "mean"
    assert ml["query_prefix"] == "query: "
    assert ml["document_prefix"] == "passage: "
    assert ml["revision"] != "main"
    bge = MODEL_REGISTRY[DEFAULT_MODEL_KEY]
    assert bge["pool"] == "cls"
    assert bge["query_prefix"] == "" and bge["document_prefix"] == ""


def test_uk_not_in_supported_langs() -> None:
    assert "uk" not in SUPPORTED_OPTIN_LANGS
    assert len(SUPPORTED_OPTIN_LANGS) == 13


def _write_config(root, model_key) -> None:
    (root / "config.json").write_text(
        json.dumps({"embed": {"model_key": model_key}}), encoding="utf-8"
    )


def test_config_selects_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert _configured_model_key() is None
    assert _resolve_model_key() == DEFAULT_MODEL_KEY
    _write_config(tmp_path, MULTILINGUAL_MODEL_KEY)
    assert _configured_model_key() == MULTILINGUAL_MODEL_KEY
    assert _resolve_model_key() == MULTILINGUAL_MODEL_KEY
    # An explicit key always wins over config.
    assert _resolve_model_key(DEFAULT_MODEL_KEY) == DEFAULT_MODEL_KEY


def test_config_rejects_unknown_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    _write_config(tmp_path, "no-such-model")
    with pytest.raises(ValueError, match="no-such-model"):
        _configured_model_key()


class _StubEmbedder:
    def __init__(self, **kw) -> None:
        self.model_name = kw.get("model_name", "BAAI/bge-small-en-v1.5")
        self.DIM = 384
        self._backend = "rust"
        for k, v in kw.items():
            setattr(self, k, v)


def test_default_identity_stays_byte_stable(monkeypatch) -> None:
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL_ID", raising=False)
    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    ident = effective_model_identity(_StubEmbedder())
    assert ident == "BAAI/bge-small-en-v1.5@pinned|pool=cls|dim=384"


def test_multilingual_identity_carries_pool_and_prefixes(monkeypatch) -> None:
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL_ID", raising=False)
    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    ident = effective_model_identity(_StubEmbedder(
        model_name="intfloat/multilingual-e5-small",
        _pool="mean",
        _query_prefix="query: ",
        _document_prefix="passage: ",
    ))
    assert "pool=mean" in ident
    assert "qprefix=query: " in ident
    assert "dprefix=passage: " in ident
    assert ident != effective_model_identity(_StubEmbedder())


def _run_lang(argv: list[str]) -> int:
    from iai_mcp.iai_cli import main

    old = sys.argv
    sys.argv = ["iai"] + argv
    try:
        return main()
    finally:
        sys.argv = old


def test_lang_verb_roundtrip(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert _run_lang(["lang", "status"]) == 0
    assert "english-only" in capsys.readouterr().out

    assert _run_lang(["lang", "add", "ru"]) == 0
    out = capsys.readouterr().out
    assert MULTILINGUAL_MODEL_KEY in out
    assert "re-embedded" in out
    # The runbook must name the one migration flag that honors the config;
    # the from-text flag resolves its own embedder and must never appear.
    assert "--reembed-to-configured-provider" in out
    assert "--reembed-from-text" not in out
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["embed"]["model_key"] == MULTILINGUAL_MODEL_KEY
    assert cfg["embed"]["languages"] == ["ru"]

    assert _run_lang(["lang", "remove", "ru"]) == 0
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["embed"]["model_key"] == DEFAULT_MODEL_KEY
    assert cfg["embed"]["languages"] == []


def test_lang_add_refuses_unsupported(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert _run_lang(["lang", "add", "uk"]) == 2
    assert not (tmp_path / "config.json").exists()


def test_lang_add_preserves_other_config(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"user": {"timezone": "UTC"}}), encoding="utf-8"
    )
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert _run_lang(["lang", "add", "de"]) == 0
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["user"]["timezone"] == "UTC"
    assert cfg["embed"]["model_key"] == MULTILINGUAL_MODEL_KEY


def _install_rust_stub(monkeypatch, recorded: list[dict]) -> None:
    import types

    import iai_mcp_native

    mod = types.ModuleType("iai_mcp_native.embed")

    class Embedder:
        def __init__(self, **kw) -> None:
            recorded.append(dict(kw))

        def encode(self, text: str) -> list[float]:
            return [0.0] * 384

    mod.Embedder = Embedder
    monkeypatch.setattr(iai_mcp_native, "embed", mod)
    monkeypatch.setitem(sys.modules, "iai_mcp_native.embed", mod)


def test_configured_e5_store_gets_e5_embedder(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from iai_mcp.embed import embedder_for_store

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    _write_config(tmp_path, MULTILINGUAL_MODEL_KEY)
    recorded: list[dict] = []
    _install_rust_stub(monkeypatch, recorded)
    e = embedder_for_store(SimpleNamespace(embed_dim=384))
    assert e.model_key == MULTILINGUAL_MODEL_KEY
    spec = MODEL_REGISTRY[MULTILINGUAL_MODEL_KEY]
    assert recorded == [
        {"model_id": spec["hf"], "revision": spec["revision"], "pool": "mean"}
    ]


def test_configured_model_dim_mismatch_raises(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from iai_mcp.embed import embedder_for_store

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    _write_config(tmp_path, MULTILINGUAL_MODEL_KEY)
    recorded: list[dict] = []
    _install_rust_stub(monkeypatch, recorded)
    with pytest.raises(ValueError, match="re-embedding migration"):
        embedder_for_store(SimpleNamespace(embed_dim=512))
    assert recorded == []


def test_identity_env_override_only_applies_to_default_model(monkeypatch) -> None:
    from iai_mcp.embed import DEFAULT_MODEL_KEY as _default_key

    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "intfloat/multilingual-e5-large")
    monkeypatch.delenv("IAI_MCP_EMBED_REVISION", raising=False)
    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    # A spec-loaded model ignores the env at load time — the stamp must too,
    # or it attests to a model the vectors did not come from.
    e5_ident = effective_model_identity(_StubEmbedder(
        model_name="intfloat/multilingual-e5-small",
        model_key="multilingual-e5-small",
        _pool="mean",
    ))
    assert "multilingual-e5-large" not in e5_ident
    assert e5_ident.startswith("intfloat/multilingual-e5-small@pinned")
    # The argless default arm DOES honor the env in the native loader, so
    # its stamp keeps reporting the override truthfully.
    bge_ident = effective_model_identity(_StubEmbedder(model_key=_default_key))
    assert bge_ident.startswith("intfloat/multilingual-e5-large@main")


def test_migrate_reembed_flags_are_mutually_exclusive(capsys) -> None:
    import argparse

    from iai_mcp.cli._analytics import cmd_migrate

    args = argparse.Namespace(
        reembed_from_text=True, reembed_to_configured_provider=True
    )
    assert cmd_migrate(args) == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_cue_sites_stay_on_embed_query() -> None:
    """Reverting any converted cue site to plain .embed() would silently
    drop the e5 query prefix; pin the exact call shapes."""
    import inspect

    from iai_mcp import concurrency, curiosity, foresight

    assert "_embed_query, embedder, cue" in inspect.getsource(concurrency)
    assert "embed_query(embedder_for_store(store), cue)" in inspect.getsource(
        curiosity
    )
    assert "_embed_query(embedder, q_cue)" in inspect.getsource(foresight)


def test_native_prefixing_is_input_type_aware() -> None:
    e = object.__new__(embed_mod.Embedder)
    e._backend = "rust"
    e._quantize_mode = None
    e._query_prefix = "query: "
    e._document_prefix = "passage: "
    seen: list[str] = []

    class _Rust:
        def encode(self, text: str) -> list[float]:
            seen.append(text)
            return [0.0] * 384

    e._model = _Rust()
    e.supports_batch = False
    e._encode_batch(["hello"], input_type="query")
    e._encode_batch(["hello"], input_type="document")
    assert seen == ["query: hello", "passage: hello"]


def _run_recall_cmd(monkeypatch, capsys, daemon_resp, warm_raises=None):
    import argparse

    import iai_mcp.cli as cli_mod
    import iai_mcp.iai_cli as iai_cli
    import iai_mcp.semantic_recall as sr

    monkeypatch.setattr(
        cli_mod, "_send_jsonrpc_request", lambda *a, **k: daemon_resp
    )
    if warm_raises is not None:
        def _raise(*a, **k):
            raise warm_raises

        monkeypatch.setattr(sr, "recall_semantic_warm", _raise)
    rc = iai_cli.cmd_recall(
        argparse.Namespace(cue="probe", limit=3, json=False)
    )
    return rc, capsys.readouterr()


def test_cli_surfaces_daemon_embedder_refusal(monkeypatch, capsys) -> None:
    from iai_mcp.errors import ERR_EMBEDDER_REFUSAL

    # The typed wire code alone must trigger the refusal exit — no
    # message prose required.
    rc, out = _run_recall_cmd(monkeypatch, capsys, {
        "error": {
            "code": ERR_EMBEDDER_REFUSAL,
            "message": "store vector space refused",
        }
    })
    assert rc == 2
    assert "embedder refused" in out.err
    assert "daemon unreachable" not in out.err


def test_cli_daemon_refusal_hint_fallback(monkeypatch, capsys) -> None:
    from iai_mcp.embed import REEMBED_RUNBOOK_HINT

    # A daemon predating the wire code still reaches the refusal exit
    # through the message hint.
    rc, out = _run_recall_cmd(monkeypatch, capsys, {
        "error": {"message": f"boom; {REEMBED_RUNBOOK_HINT} now"}
    })
    assert rc == 2
    assert "embedder refused" in out.err


def test_cli_surfaces_direct_store_embedder_refusal(
    tmp_path, monkeypatch, capsys
) -> None:
    from iai_mcp.embed import EmbedderConfigError

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    (hippo / "brain.sqlite3").write_bytes(b"")
    rc, out = _run_recall_cmd(
        monkeypatch, capsys, None,
        warm_raises=EmbedderConfigError("vector space mismatch"),
    )
    assert rc == 2
    assert "embedder refused" in out.err
    assert "bank-fallback" not in out.err


def test_crisis_tier_surfaces_embedder_refusal_through_dispatch(
    tmp_path, monkeypatch
) -> None:
    """The refusal must escape the FULL dispatch under crisis_mode — a
    helper-level pin once passed while the dispatch guard swallowed the
    raise one frame up."""
    import traceback

    import iai_mcp.embed as embed_mod
    from iai_mcp import core
    from iai_mcp.embed import EmbedderConfigError
    from iai_mcp.store import MemoryStore

    # EMPTY store: the warm path would answer without ever touching the
    # embedder, so only the crisis tier can raise — a seeded store would
    # let the warm path raise the same type and mask a reverted hoist.
    store = MemoryStore(path=tmp_path)

    def _refuse(store, **kwargs):
        raise EmbedderConfigError("vector space mismatch")

    monkeypatch.setattr(embed_mod, "embedder_for_store", _refuse)
    # A plain dict, not the production 1s-TTL cache — an expiring entry
    # would flake this test toward a false pass through the warm path.
    monkeypatch.setattr(
        core, "_CRISIS_STATE_CACHE", {"crisis": {"crisis_mode": True}}
    )
    with pytest.raises(EmbedderConfigError) as ei:
        core.dispatch(store, "memory_recall", {
            "cue": "probe", "session_id": "s",
        })
    frames = "".join(traceback.format_tb(ei.value.__traceback__))
    assert "_crisis_degraded_recall" in frames, (
        "refusal must originate in the crisis tier, not the warm path"
    )


def test_cli_json_mode_refusal_emits_json(monkeypatch, capsys) -> None:
    import argparse

    import iai_mcp.cli as cli_mod
    import iai_mcp.iai_cli as iai_cli
    from iai_mcp.embed import REEMBED_RUNBOOK_HINT

    monkeypatch.setattr(
        cli_mod, "_send_jsonrpc_request",
        lambda *a, **k: {"error": {"message": f"boom; {REEMBED_RUNBOOK_HINT} now"}},
    )
    rc = iai_cli.cmd_recall(argparse.Namespace(cue="probe", limit=3, json=True))
    out = capsys.readouterr()
    assert rc == 2
    doc = json.loads(out.out)
    assert doc["hits"] == [] and "embedder refused" in doc["error"]


def test_refusal_messages_carry_the_runbook_hint(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from iai_mcp.embed import REEMBED_RUNBOOK_HINT, embedder_for_store

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    _write_config(tmp_path, MULTILINGUAL_MODEL_KEY)
    _install_rust_stub(monkeypatch, [])
    with pytest.raises(ValueError) as ei:
        embedder_for_store(SimpleNamespace(embed_dim=512))
    assert REEMBED_RUNBOOK_HINT in str(ei.value)


def test_identity_mismatch_message_carries_hint(tmp_path) -> None:
    from iai_mcp.embed import (
        EMBED_IDENTITY_META_KEY,
        REEMBED_RUNBOOK_HINT,
        EmbedIdentityMismatch,
        embedder_for_store,
    )
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    db = store.db
    with db._conn_lock:
        db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (EMBED_IDENTITY_META_KEY,)
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (EMBED_IDENTITY_META_KEY, "foreign/model@main|pool=cls|dim=384"),
        )
        db._conn.commit()
    with pytest.raises(EmbedIdentityMismatch) as ei:
        embedder_for_store(store)
    assert REEMBED_RUNBOOK_HINT in str(ei.value)
    assert "--reembed-to-configured-provider" in str(ei.value)


def test_cli_bank_fallback_survives_unimportable_embed(
    tmp_path, monkeypatch, capsys
) -> None:
    """Refusal detection is best-effort: with the embed stack unimportable
    (numpy-less host), the recall CLI must still reach bank-fallback
    instead of dying on the hoisted import."""
    import argparse

    import iai_mcp.cli as cli_mod
    import iai_mcp.iai_cli as iai_cli

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "iai_mcp.embed", None)
    monkeypatch.setattr(cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    class _BankDone:
        returncode = 0
        stdout = "  0.500  bank hit line\n"
        stderr = ""

    monkeypatch.setattr(
        iai_cli.subprocess, "run", lambda *a, **k: _BankDone()
    )
    rc = iai_cli.cmd_recall(
        argparse.Namespace(cue="probe", limit=3, json=False)
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "bank-fallback" in out.err
    assert "bank hit line" in out.out


def test_sleep_path_cue_reembed_surfaces_refusal(tmp_path, monkeypatch) -> None:
    import iai_mcp.embed as embed_mod
    from iai_mcp import retrieve
    from iai_mcp.embed import EmbedderConfigError
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    def _refuse(store, **kwargs):
        raise EmbedderConfigError("vector space mismatch")

    monkeypatch.setattr(embed_mod, "embedder_for_store", _refuse)
    with pytest.raises(EmbedderConfigError):
        retrieve.recall(
            store=store, cue_embedding=None, cue_text="probe cue",
            session_id="s", budget_tokens=500,
        )


def test_unknown_config_model_is_a_selection_refusal(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp.embed import EmbedderConfigError

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    _write_config(tmp_path, "no-such-model")
    with pytest.raises(EmbedderConfigError):
        _configured_model_key()


def test_sleep_branch_surfaces_refusal_through_dispatch(
    tmp_path, monkeypatch
) -> None:
    """Under SLEEP the refusal must escape from the retrieve tier itself —
    with the dispatch folded back into the detection guard it would be
    swallowed and re-raised later by the warm path instead."""
    import traceback
    from datetime import datetime, timezone
    from uuid import uuid4

    import iai_mcp.embed as embed_mod
    from iai_mcp import core
    from iai_mcp.embed import EmbedderConfigError
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    store.insert(MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface="sleep probe record",
        aaak_index="", embedding=[0.1] * EMBED_DIM, community_id=None,
        centrality=0.5, detail_level=3, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False,
        never_merge=False, provenance=[], created_at=now, updated_at=now,
        tags=[], language="en",
    ))
    flush_record_buffer(store)

    def _refuse(store, **kwargs):
        raise EmbedderConfigError("vector space mismatch")

    monkeypatch.setattr(embed_mod, "embedder_for_store", _refuse)
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"current_state": "SLEEP"},
    )
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", lambda s: None)
    monkeypatch.setattr(
        core, "_CRISIS_STATE_CACHE", {"crisis": {"crisis_mode": False}}
    )

    with pytest.raises(EmbedderConfigError) as ei:
        core.dispatch(store, "memory_recall", {
            "cue": "probe", "session_id": "s",
        })
    frames = "".join(traceback.format_tb(ei.value.__traceback__))
    assert "retrieve.py" in frames, (
        "refusal must originate in the retrieve tier, not the warm path"
    )


def test_sleep_branch_degrades_on_non_refusal_failure(
    tmp_path, monkeypatch
) -> None:
    """The fail-safe half of the sleep guard split: a genuine tier failure
    (not a refusal) must still degrade to the warm path, never raise."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp import core, retrieve
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    store.insert(MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface="sleep degrade record",
        aaak_index="", embedding=[0.1] * EMBED_DIM, community_id=None,
        centrality=0.5, detail_level=3, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False,
        never_merge=False, provenance=[], created_at=now, updated_at=now,
        tags=[], language="en",
    ))
    flush_record_buffer(store)

    def _tier_broken(*args, **kwargs):
        raise RuntimeError("simulated tier failure")

    monkeypatch.setattr(retrieve, "recall", _tier_broken)
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"current_state": "SLEEP"},
    )
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", lambda s: None)
    monkeypatch.setattr(
        core, "_CRISIS_STATE_CACHE", {"crisis": {"crisis_mode": False}}
    )

    resp = core.dispatch(store, "memory_recall", {
        "cue": "sleep degrade record", "session_id": "s",
    })
    assert isinstance(resp, dict)
    assert "hits" in resp


def test_socket_classifies_refusal_under_wire_code(tmp_path, monkeypatch) -> None:
    """Deleting the socket-side classification must fail THIS test — the
    whole typed contract pivots on it."""
    import asyncio

    from test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    from iai_mcp import core
    from iai_mcp.embed import EmbedderConfigError
    from iai_mcp.errors import ERR_EMBEDDER_REFUSAL
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "sock-store")

    def _refuse(store_, method, params):
        raise EmbedderConfigError("vector space mismatch")

    monkeypatch.setattr(core, "dispatch", _refuse)
    # AF_UNIX paths are length-limited; tmp_path exceeds it on macOS. The
    # short dir gets per-run entropy and a teardown so /tmp never accretes
    # predictable world-adoptable directories.
    import os as _os
    import shutil as _shutil
    from pathlib import Path as _P

    _sock_dir = _P(f"/tmp/iai-refusal-{_os.getpid()}-{id(tmp_path)}")
    _sock_dir.mkdir(parents=True, exist_ok=False)
    sock_path = _sock_dir / "r.sock"

    async def _runner(sp, st):
        return await _send_jsonrpc(sp, "memory_recall", {"cue": "x"}, req_id=7)

    try:
        resp = asyncio.run(_with_socket_server(sock_path, store, _runner))
    finally:
        _shutil.rmtree(_sock_dir, ignore_errors=True)
    assert isinstance(resp.get("error"), dict), resp
    assert resp["error"]["code"] == ERR_EMBEDDER_REFUSAL
    assert "vector space mismatch" in str(resp["error"].get("message"))


def test_wire_code_constant_matches_across_languages() -> None:
    """The TypeScript wrapper carries the wire code as a literal — drifting
    it from the Python constant silently breaks the whole contract."""
    from pathlib import Path

    from iai_mcp.errors import ERR_EMBEDDER_REFUSAL

    import re

    repo = Path(__file__).resolve().parents[1]
    tools_src = (repo / "mcp-wrapper" / "src" / "tools.ts").read_text()
    m = re.search(r"ERR_EMBEDDER_REFUSAL\s*=\s*(-?\d+)\s*;", tools_src)
    assert m, "tools.ts lost the shared wire-code declaration"
    assert int(m.group(1)) == ERR_EMBEDDER_REFUSAL, (
        f"tools.ts declares {m.group(1)}, Python declares {ERR_EMBEDDER_REFUSAL}"
    )
    bundle = repo / "mcp-wrapper" / "dist-bundle" / "index.js"
    if bundle.exists():
        bundle_text = bundle.read_text()
        assert re.search(
            rf"(?<![\d.]){ERR_EMBEDDER_REFUSAL}(?![\d.])", bundle_text
        ), "shipped bundle is stale: rebuild with npm run bundle"
        # Value presence alone cannot see LOGIC drift — a bundle can keep
        # the declaration while missing every gate. esbuild preserves
        # identifiers, so each source reference to the constant must
        # survive into the bundle, and the child-exit parse must too.
        # Comment mentions are stripped by esbuild — count code lines only.
        src_refs = sum(
            line.count("ERR_EMBEDDER_REFUSAL")
            for line in tools_src.splitlines()
            if not line.lstrip().startswith("//")
        )
        bundle_refs = bundle_text.count("ERR_EMBEDDER_REFUSAL")
        assert bundle_refs >= src_refs, (
            f"shipped bundle carries {bundle_refs} ERR_EMBEDDER_REFUSAL refs "
            f"but src/tools.ts has {src_refs}: rebuild with npm run bundle"
        )
        assert 'doc["error_code"]' in bundle_text, (
            "shipped bundle lost the child-exit error_code parse: "
            "rebuild with npm run bundle"
        )


def test_ann_rail_refusal_reaches_caller(tmp_path, monkeypatch) -> None:
    """The ANN-only rail's identity guard must RAISE to the caller — a
    swallow-to-[] would relabel a refusal as recency junk with rc 0."""
    from types import SimpleNamespace

    import iai_mcp.hippo as hippo_mod
    import iai_mcp.semantic_recall as _sr
    from iai_mcp.embed import EmbedIdentityMismatch, stamp_store_embed_identity
    from iai_mcp.store import MemoryStore

    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    stamp_store_embed_identity(
        store, SimpleNamespace(model_name="foreign-model", DIM=384, model_key="f")
    )
    # The rail opens its own SHARED handle; the writer's EXCLUSIVE lock
    # must be released first (daemon-down means no other holder).
    store.close()
    runtime = SimpleNamespace(model_name="runtime-model", DIM=384, model_key="r")

    with pytest.raises(EmbedIdentityMismatch):
        _sr._fetch_records_by_labels(store_root, [1], n=5, embedder=runtime)

    monkeypatch.setattr(hippo_mod, "_ann_lookup_client", lambda *a, **k: [1])
    with pytest.raises(EmbedIdentityMismatch):
        _sr._ann_only_daemon_down(
            store_root, [0.0] * 384, 5, "cue", None, embedder=runtime
        )


def test_ann_rail_guard_fault_serves_nothing_confident(
    tmp_path, monkeypatch
) -> None:
    """A guard FAULT (not refusal) on the ANN-only rail must not serve
    score-1.0 direct-store rows: the rail cannot verify the vector space."""
    from types import SimpleNamespace

    import numpy as np

    import iai_mcp.embed as _embed_mod
    import iai_mcp.semantic_recall as _sr
    from iai_mcp.store import MemoryStore
    from test_store import _make

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))
    rng = np.random.default_rng(11)
    vec = rng.standard_normal(store.embed_dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    store.insert(_make(text="guard fault probe record", vec=vec.tolist()))
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT vec_label FROM records WHERE vec_label IS NOT NULL LIMIT 1"
        ).fetchone()
    assert row is not None, "store insert produced no vec_label"
    label = int(row["vec_label"])
    store.close()

    control = _sr._fetch_records_by_labels(store_root, [label], n=5, embedder=None)
    assert control, "control without a guard must serve the row"

    def _faulting_guard(*_a, **_k):
        raise RuntimeError("simulated meta read fault")

    monkeypatch.setattr(
        _embed_mod, "_enforce_store_embed_identity", _faulting_guard
    )
    runtime = SimpleNamespace(model_name="runtime-model", DIM=384, model_key="r")
    rows = _sr._fetch_records_by_labels(
        store_root, [label], n=5, embedder=runtime
    )
    assert rows == [], (
        f"guard fault must serve NOTHING from this rail, got {rows!r}"
    )


def test_dimless_embedder_identity_contract(tmp_path, monkeypatch) -> None:
    """An embedder without DIM cannot have its identity computed. On an
    UNSTAMPED store there is nothing to verify — pass (duck-typed test
    embedders are legitimate). On a STAMPED store a skip would fail OPEN
    and serve cross-generation rows confidently — it must REFUSE, unless
    the caller sanctioned the mismatch."""
    from types import SimpleNamespace

    from iai_mcp.embed import (
        REEMBED_RUNBOOK_HINT,
        EmbedderConfigError,
        EmbedIdentityMismatch,
        _enforce_store_embed_identity,
        stamp_store_embed_identity,
    )
    from iai_mcp.store import MemoryStore

    monkeypatch.delenv("IAI_MCP_EMBED_TEXT_PREFIX", raising=False)
    dimless = SimpleNamespace(model_name="duck")

    unstamped = MemoryStore(str(tmp_path / "unstamped"))
    _enforce_store_embed_identity(unstamped, dimless, allow_mismatch=False)

    stamped = MemoryStore(str(tmp_path / "stamped"))
    stamp_store_embed_identity(
        stamped, SimpleNamespace(model_name="stamped-model", DIM=384, model_key="s")
    )
    with pytest.raises(EmbedIdentityMismatch, match="declares no dimension") as exc:
        _enforce_store_embed_identity(stamped, dimless, allow_mismatch=False)
    assert REEMBED_RUNBOOK_HINT in str(exc.value)
    _enforce_store_embed_identity(stamped, dimless, allow_mismatch=True)

    # The sanctioned override must not dead-end at stamp time with an
    # untyped crash — stamping refuses the unverifiable embedder loudly.
    with pytest.raises(EmbedderConfigError, match="cannot stamp") as stamp_exc:
        stamp_store_embed_identity(stamped, dimless)
    assert REEMBED_RUNBOOK_HINT in str(stamp_exc.value)

    unstamped.close()
    stamped.close()


def test_migration_stamp_refusal_is_not_swallowed(tmp_path, monkeypatch) -> None:
    """Both re-embed migrations wrap the stamp in a fault-tolerant except;
    a TYPED stamp refusal must escape it — swallowing would report success
    over a stale stamp that now attests falsely. The dim carrier must also
    clear its checkpoint before raising: the swap is complete, and a resume
    must not re-stage a finished migration."""
    from types import SimpleNamespace

    import iai_mcp.embed as _embed_mod
    from iai_mcp.embed import EmbedderConfigError
    from iai_mcp.migrate._reembed import (
        _progress_path,
        migrate_reembed_to_current_dim,
    )
    from iai_mcp.migrate._reembed_from_text import migrate_reembed_from_text
    from iai_mcp.store import MemoryStore

    def _refusing_stamp(*_a, **_k):
        raise EmbedderConfigError("cannot stamp: unverifiable embedder")

    monkeypatch.setattr(
        _embed_mod, "stamp_store_embed_identity", _refusing_stamp
    )

    store = MemoryStore(str(tmp_path / "store"))
    try:
        with pytest.raises(EmbedderConfigError, match="cannot stamp"):
            migrate_reembed_from_text(store)
    finally:
        store.close()

    dim_store = MemoryStore(str(tmp_path / "dim-store"))
    try:
        # A seeded row makes the staging loop write a real checkpoint —
        # on an empty store the checkpoint assertion below is vacuous.
        import numpy as np

        from test_store import _make

        rng = np.random.default_rng(3)
        vec = rng.standard_normal(dim_store.embed_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        dim_store.insert(_make(text="checkpoint probe record", vec=vec.tolist()))

        dim = dim_store.embed_dim
        target = SimpleNamespace(
            DIM=dim,
            model_key="stub",
            embed=lambda text: [0.0] * dim,
            embed_batch=lambda texts: [[0.0] * dim for _ in texts],
        )
        with pytest.raises(EmbedderConfigError, match="cannot stamp"):
            migrate_reembed_to_current_dim(dim_store, target, force=True)
        assert not _progress_path(dim_store).exists(), (
            "dim carrier must clear its checkpoint before the refusal raise"
        )
    finally:
        dim_store.close()
