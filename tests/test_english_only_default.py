from __future__ import annotations

import sys
from pathlib import Path
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


def test_model_registry_default_is_english_and_optin_is_explicit():
    """English-only stays the default; the ONE multilingual entry is an
    explicit opt-in that only a config-file switch can select — never the
    environment, never an implicit fallback."""
    from iai_mcp.embed import (
        DEFAULT_MODEL_KEY,
        MODEL_REGISTRY,
        MULTILINGUAL_MODEL_KEY,
        _resolve_model_key,
    )

    assert set(MODEL_REGISTRY.keys()) == {
        "bge-small-en-v1.5",
        "multilingual-e5-small",
    }
    assert DEFAULT_MODEL_KEY == "bge-small-en-v1.5"
    bge = MODEL_REGISTRY[DEFAULT_MODEL_KEY]
    assert bge["hf"] == "BAAI/bge-small-en-v1.5"
    assert bge["dim"] == 384
    assert MODEL_REGISTRY[MULTILINGUAL_MODEL_KEY]["dim"] == 384
    # With no config present, resolution lands on the English default.
    assert _resolve_model_key() == DEFAULT_MODEL_KEY


def test_embedder_for_store_picks_bge_small_for_384d_store():
    from iai_mcp.embed import embedder_for_store

    store = SimpleNamespace(embed_dim=384)
    e = embedder_for_store(store)
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384


def test_project_md_still_pins_bge_small_constraint():
    p = Path(__file__).resolve().parents[1] / ".planning" / "PROJECT.md"
    if not p.exists():
        pytest.skip(".planning is gitignored; PROJECT.md not present in this checkout")
    content = p.read_text()
    assert "bge-small-en-v1.5" in content
    assert "384d embeddings" in content or "384d" in content


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
