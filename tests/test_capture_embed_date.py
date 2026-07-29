from __future__ import annotations

import re

import pytest

import iai_mcp.embed as embed_mod
from iai_mcp.types import EMBED_DIM

_DATE_PREFIX_RE = re.compile(r"^On \d{1,2} [A-Z][a-z]+ \d{4}: ")


class _RecordingEmbedder:
    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        v = [0.0] * EMBED_DIM
        v[hash(text) % EMBED_DIM] = 1.0
        return v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_result_for(self, text: str) -> list[float]:
        """Recompute the deterministic vector without recording the call."""
        v = [0.0] * EMBED_DIM
        v[hash(text) % EMBED_DIM] = 1.0
        return v


@pytest.fixture
def recording_embedder(monkeypatch):
    rec = _RecordingEmbedder()
    monkeypatch.setattr(embed_mod, "embedder_for_store", lambda _store: rec)
    return rec


def _capture(store, text: str) -> dict:
    from iai_mcp.capture import capture_turn
    return capture_turn(
        store, cue="test", text=text, tier="episodic",
        session_id="s-embed-date", role="user",
    )


def test_capture_embeds_with_date_prefix_surface_verbatim(
    tmp_path, monkeypatch, recording_embedder,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-embed-date")
    monkeypatch.setenv("IAI_MCP_EMBED_DATE", "true")
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path / "lancedb")

    text = "the deploy target moved to the staging cluster"
    result = _capture(store, text)
    assert result["status"] == "inserted"

    assert recording_embedder.inputs, "embedder was never called"
    embed_input = recording_embedder.inputs[-1]
    assert _DATE_PREFIX_RE.match(embed_input), (
        f"embed input must carry the date prefix, got: {embed_input[:60]!r}"
    )
    assert embed_input.endswith(text)

    from iai_mcp.store import flush_record_buffer
    flush_record_buffer(store)
    surfaces = [r.literal_surface for r in store.all_records()]
    assert text in surfaces, "stored surface must stay verbatim"
    assert all(not _DATE_PREFIX_RE.match(s) for s in surfaces), (
        "no stored surface may carry the embed-only date prefix"
    )


def test_capture_embed_date_default_off(
    tmp_path, monkeypatch, recording_embedder,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-embed-date")
    monkeypatch.delenv("IAI_MCP_EMBED_DATE", raising=False)
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path / "lancedb")

    text = "the deploy target moved back to the primary cluster"
    result = _capture(store, text)
    assert result["status"] == "inserted"
    assert recording_embedder.inputs[-1] == text


_DATE_PHRASE_RE = re.compile(r"^On \d{1,2} [A-Z][a-z]+ \d{4}\.$")


def test_capture_embed_date_blend_bounded_component(
    tmp_path, monkeypatch, recording_embedder,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-embed-date")
    monkeypatch.setenv("IAI_MCP_EMBED_DATE", "blend")
    monkeypatch.setenv("IAI_MCP_EMBED_DATE_ALPHA", "0.15")
    from iai_mcp.store import MemoryStore, flush_record_buffer
    store = MemoryStore(path=tmp_path / "lancedb")

    text = "the deploy target moved to the violet cluster"
    result = _capture(store, text)
    assert result["status"] == "inserted"

    # Two encoder calls: the verbatim text and the bare date phrase.
    assert text in recording_embedder.inputs
    date_inputs = [
        s for s in recording_embedder.inputs if _DATE_PHRASE_RE.match(s)
    ]
    assert date_inputs, recording_embedder.inputs

    flush_record_buffer(store)
    rec = next(r for r in store.all_records() if r.literal_surface == text)
    v_text = recording_embedder.embed_result_for(text)
    v_date = recording_embedder.embed_result_for(date_inputs[0])
    t_norm = sum(x * x for x in v_text) ** 0.5
    t_hat = [x / t_norm for x in v_text]
    dot = sum(t * d for t, d in zip(t_hat, v_date))
    perp = [d - dot * t for t, d in zip(t_hat, v_date)]
    expected = [t + 0.15 * p for t, p in zip(v_text, perp)]
    norm = sum(x * x for x in expected) ** 0.5
    expected = [x / norm for x in expected]
    # float32 storage quantization bounds the round-trip error.
    assert rec.embedding == pytest.approx(expected, abs=1e-6)
    assert rec.literal_surface == text


def test_capture_embed_date_blend_alpha_zero_is_pure_text(
    tmp_path, monkeypatch, recording_embedder,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-embed-date")
    monkeypatch.setenv("IAI_MCP_EMBED_DATE", "blend")
    monkeypatch.setenv("IAI_MCP_EMBED_DATE_ALPHA", "0")
    from iai_mcp.store import MemoryStore, flush_record_buffer
    store = MemoryStore(path=tmp_path / "lancedb")

    text = "the deploy target moved to the amber cluster"
    assert _capture(store, text)["status"] == "inserted"
    flush_record_buffer(store)
    rec = next(r for r in store.all_records() if r.literal_surface == text)
    assert rec.embedding == pytest.approx(
        recording_embedder.embed_result_for(text)
    )
