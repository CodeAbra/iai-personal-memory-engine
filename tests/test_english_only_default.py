from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    yield


def test_default_model_key_is_bge_small():
    from iai_mcp.embed import DEFAULT_MODEL_KEY

    assert DEFAULT_MODEL_KEY == "bge-small-en-v1.5"


def test_embedder_defaults_to_384d_small():
    from iai_mcp.embed import Embedder

    assert Embedder.DEFAULT_MODEL_KEY == "bge-small-en-v1.5"
    assert Embedder.DEFAULT_DIM == 384
    assert Embedder.DIM == 384


def test_types_embed_dim_defaults_to_384():
    from iai_mcp.types import DEFAULT_EMBED_DIM, EMBED_DIM

    assert DEFAULT_EMBED_DIM == 384
    assert EMBED_DIM == 384


def test_model_registry_is_english_only_single_entry():
    from iai_mcp.embed import MODEL_REGISTRY

    assert set(MODEL_REGISTRY.keys()) == {"bge-small-en-v1.5"}
    assert MODEL_REGISTRY["bge-small-en-v1.5"] == {
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    }


def test_embedder_for_store_picks_bge_small_for_384d_store():
    from iai_mcp.embed import embedder_for_store

    store = SimpleNamespace(embed_dim=384)
    e = embedder_for_store(store)
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384


def test_import_embed_does_not_pull_sentence_transformers():
    from unittest import mock

    with mock.patch.dict(sys.modules):
        for name in [k for k in sys.modules if k.startswith("iai_mcp")]:
            sys.modules.pop(name, None)
        import iai_mcp.embed  # noqa: F401
        import iai_mcp.types  # noqa: F401
        assert "sentence_transformers" not in sys.modules, (
            "iai_mcp.embed must not pull sentence_transformers into sys.modules; "
            "sentence-transformers is not a dependency"
        )
