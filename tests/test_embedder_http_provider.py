from __future__ import annotations

import builtins
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture
def embed_server():
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size))
            requests.append(payload)
            vectors = [[float(len(text)), 1.0, 0.0] for text in payload["texts"]]
            body = json.dumps(
                {
                    "model": "test-multilingual",
                    "dimensions": 3,
                    "vectors": vectors,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _configure(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_PROVIDER", "http")
    monkeypatch.setenv("IAI_MCP_EMBED_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "3")
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "test-multilingual")


def test_http_provider_replaces_native_and_preserves_input_type(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import Embedder, embed_query

    port, requests = embed_server
    _configure(monkeypatch, port)

    embedder = Embedder()
    assert embedder._backend == "http"
    assert embedder.model_key == "test-multilingual"
    assert embedder.DIM == 3
    assert embedder.embed("memory") == [6.0, 1.0, 0.0]
    assert embed_query(embedder, "question") == [8.0, 1.0, 0.0]
    assert embedder.embed_batch(["a", "bb"]) == [
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    assert [request["input_type"] for request in requests] == [
        "document",
        "query",
        "document",
    ]


def test_http_provider_preserves_empty_batch_without_request(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import Embedder

    port, requests = embed_server
    _configure(monkeypatch, port)
    assert Embedder().embed_batch([]) == []
    assert requests == []


def test_http_provider_caps_response_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp import embed as embed_module

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size: int) -> bytes:
            return b"x" * size

    _configure(monkeypatch, 4488)
    monkeypatch.setattr(embed_module, "urlopen", lambda *_a, **_kw: OversizedResponse())

    with pytest.raises(RuntimeError, match="too large"):
        embed_module.Embedder().embed("memory")


def test_http_provider_does_not_import_native_backend(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import Embedder

    port, _requests = embed_server
    _configure(monkeypatch, port)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("iai_mcp_native"):
            raise AssertionError("native backend must not load for the http provider")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert Embedder()._backend == "http"


def test_http_provider_rejects_non_loopback_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.embed import Embedder

    monkeypatch.setenv("IAI_MCP_EMBED_PROVIDER", "http")
    monkeypatch.setenv("IAI_MCP_EMBED_URL", "https://example.com/embed")
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "3")
    with pytest.raises(ValueError, match="loopback"):
        Embedder()


@pytest.mark.parametrize(
    "url",
    [
        "http://user:pass@127.0.0.1:4488/embed",
        "http://127.0.0.1:4488/embed?model=other",
        "http://127.0.0.1:4488/embed#fragment",
        "http://127.0.0.1:4488/other",
    ],
)
def test_http_provider_rejects_ambiguous_loopback_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    from iai_mcp.embed import Embedder

    monkeypatch.setenv("IAI_MCP_EMBED_PROVIDER", "http")
    monkeypatch.setenv("IAI_MCP_EMBED_URL", url)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "3")
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "test-multilingual")
    with pytest.raises(ValueError):
        Embedder()


def test_http_provider_requires_model_id(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import Embedder

    port, _requests = embed_server
    _configure(monkeypatch, port)
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL_ID")
    with pytest.raises(ValueError, match="MODEL_ID is required"):
        Embedder()


def test_http_provider_refuses_store_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import embedder_for_store

    port, _requests = embed_server
    _configure(monkeypatch, port)
    with pytest.raises(ValueError, match="re-embedding migration"):
        embedder_for_store(SimpleNamespace(embed_dim=384))


def test_http_provider_refuses_unexpected_model(
    monkeypatch: pytest.MonkeyPatch, embed_server
) -> None:
    from iai_mcp.embed import Embedder

    port, _requests = embed_server
    _configure(monkeypatch, port)
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL_ID", "another-model")
    with pytest.raises(ValueError, match="returned model"):
        Embedder().embed("memory")


def test_legacy_test_double_still_embeds_queries() -> None:
    from iai_mcp.embed import embed_query

    class LegacyEmbedder:
        def embed(self, text: str) -> list[float]:
            return [float(len(text))]

    assert embed_query(LegacyEmbedder(), "cue") == [3.0]


def test_foresight_anchor_is_embedded_as_a_query(monkeypatch) -> None:
    from iai_mcp import foresight

    class RecordingEmbedder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def embed_query(self, text: str) -> list[float]:
            self.calls.append((text, "query"))
            return [1.0]

    store = SimpleNamespace(
        _foresight_anchor=("now", "the next-turn retrieval cue", "session")
    )
    embedder = RecordingEmbedder()
    monkeypatch.setattr(foresight, "refresh_pack", lambda *_a, **_kw: None)

    assert foresight.refresh_from_anchor(store, embedder)
    assert embedder.calls == [("the next-turn retrieval cue", "query")]
