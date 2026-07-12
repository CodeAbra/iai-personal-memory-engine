from __future__ import annotations

import logging
import os
import json
import math
from dataclasses import dataclass
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, dict] = {
    "bge-small-en-v1.5": {"hf": "BAAI/bge-small-en-v1.5", "dim": 384},
}
DEFAULT_MODEL_KEY = "bge-small-en-v1.5"

VALID_QUANTIZE_MODES: set[str] = {"int8"}

embed_failure_total: int = 0

InputType = Literal["query", "document"]
VALID_PROVIDERS: set[str] = {"native", "http"}
MAX_HTTP_RESPONSE_BYTES = 32 * 1024 * 1024


def _resolve_model_key(model_key: str | None = None) -> str:
    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown embed model key {model_key!r}; valid: {sorted(MODEL_REGISTRY)}"
            )
        return model_key
    return DEFAULT_MODEL_KEY


def _resolve_quantize_mode() -> str | None:
    raw = os.environ.get("IAI_MCP_EMBED_QUANTIZE", "")
    if not raw:
        return None
    if raw not in VALID_QUANTIZE_MODES:
        raise ValueError(
            f"IAI_MCP_EMBED_QUANTIZE={raw!r} is not a valid quantization mode; "
            f"valid: {sorted(VALID_QUANTIZE_MODES)} or unset for fp32 default"
        )
    return raw


def _resolve_provider() -> str:
    provider = os.environ.get("IAI_MCP_EMBED_PROVIDER", "native").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"IAI_MCP_EMBED_PROVIDER={provider!r} is not valid; "
            f"valid: {sorted(VALID_PROVIDERS)}"
        )
    return provider


def _resolve_http_config() -> tuple[str, int, float, str]:
    raw_url = os.environ.get("IAI_MCP_EMBED_URL", "").strip()
    if not raw_url:
        raise ValueError("IAI_MCP_EMBED_URL is required for the http provider")
    parsed = urlparse(raw_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "IAI_MCP_EMBED_URL must be an unauthenticated loopback http URL"
        )
    if parsed.path in {"", "/"}:
        raw_url = raw_url.rstrip("/") + "/embed"
    elif parsed.path != "/embed":
        raise ValueError("IAI_MCP_EMBED_URL path must be /embed or empty")

    raw_dim = os.environ.get("IAI_MCP_EMBED_DIM", "").strip()
    try:
        dim = int(raw_dim)
    except ValueError as exc:
        raise ValueError(
            "IAI_MCP_EMBED_DIM must be a positive integer for the http provider"
        ) from exc
    if dim <= 0:
        raise ValueError(
            "IAI_MCP_EMBED_DIM must be a positive integer for the http provider"
        )

    raw_timeout = os.environ.get("IAI_MCP_EMBED_TIMEOUT_SEC", "30").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError("IAI_MCP_EMBED_TIMEOUT_SEC must be positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("IAI_MCP_EMBED_TIMEOUT_SEC must be positive")

    model_id = os.environ.get("IAI_MCP_EMBED_MODEL_ID", "").strip()
    if not model_id:
        raise ValueError("IAI_MCP_EMBED_MODEL_ID is required for the http provider")
    return raw_url, dim, timeout, model_id


class _HttpEmbeddingBackend:
    def __init__(self, url: str, dim: int, timeout: float, model_id: str) -> None:
        self.url = url
        self.dim = dim
        self.timeout = timeout
        self.model_id = model_id

    def encode_batch(
        self, texts: list[str], *, input_type: InputType
    ) -> list[list[float]]:
        request = Request(
            self.url,
            data=json.dumps({"texts": texts, "input_type": input_type}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
                if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                    raise RuntimeError("http embed provider response is too large")
                payload = json.loads(raw)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(f"http embed provider failed: {exc}") from exc

        vectors = payload.get("vectors") if isinstance(payload, dict) else None
        dimensions = payload.get("dimensions") if isinstance(payload, dict) else None
        model = payload.get("model") if isinstance(payload, dict) else None
        if model != self.model_id:
            raise ValueError(
                f"http embed provider returned model {model!r}; "
                f"expected {self.model_id!r}"
            )
        if dimensions != self.dim:
            raise ValueError(
                f"http embed provider returned dimension {dimensions!r}; "
                f"expected {self.dim}"
            )
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError(
                "http embed provider returned an unexpected number of vectors"
            )

        checked: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dim:
                raise ValueError(f"http embed provider vector must be {self.dim}d")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError("http embed provider returned a non-finite vector")
            checked.append([float(value) for value in vector])
        return checked


@dataclass(frozen=True)
class QuantizedVector:
    values: list[int]
    scale: float
    zero_point: int
    dim: int


def _quantize_int8(vec: list[float]) -> QuantizedVector:
    arr = np.asarray(vec, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax == vmin:
        return QuantizedVector(
            values=[0] * len(vec), scale=1.0, zero_point=0, dim=len(vec)
        )
    scale = (vmax - vmin) / 255.0
    zero_point = int(round(-vmin / scale)) - 128
    quantized = np.round(arr / scale).astype(np.int32) + zero_point
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return QuantizedVector(
        values=[int(x) for x in quantized.tolist()],
        scale=float(scale),
        zero_point=int(zero_point),
        dim=len(vec),
    )


class Embedder:
    DEFAULT_MODEL_KEY: str = DEFAULT_MODEL_KEY
    DEFAULT_DIM: int = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["dim"]
    DEFAULT_MODEL: str = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf"]
    DIM: int = DEFAULT_DIM

    def __init__(
        self,
        model_key: str | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        self._quantize_mode: str | None = _resolve_quantize_mode()
        provider = _resolve_provider()
        if provider == "http":
            if model_key is not None or model_name is not None:
                raise ValueError(
                    "model_key and model_name only apply to the native provider"
                )
            url, dim, timeout, model_id = _resolve_http_config()
            self.model_key = model_id
            self.model_name = model_id
            self.DIM = dim
            self._model = _HttpEmbeddingBackend(url, dim, timeout, model_id)
            self._backend = "http"
            self.supports_batch = True
            return

        if model_key is None and model_name is not None:
            match = next(
                (k for k, v in MODEL_REGISTRY.items() if v["hf"] == model_name),
                None,
            )
            if match is None:
                raise ValueError(
                    f"model_name {model_name!r} is not in MODEL_REGISTRY; "
                    f"valid hf ids: {[v['hf'] for v in MODEL_REGISTRY.values()]}"
                )
            key = match
        else:
            key = _resolve_model_key(model_key)
        self.model_key: str = key
        spec = MODEL_REGISTRY[key]
        self.model_name: str = spec["hf"]
        self.DIM: int = int(spec["dim"])

        from iai_mcp_native import embed as rust

        self._model = rust.Embedder()
        self._backend: str = "rust"
        self.supports_batch = False

    def _encode_batch(
        self, texts: list[str], *, input_type: InputType
    ) -> list[list[float]]:
        global embed_failure_total
        if input_type not in {"query", "document"}:
            raise ValueError("input_type must be 'query' or 'document'")
        if not texts:
            return []
        try:
            if self._backend == "http":
                return self._model.encode_batch(texts, input_type=input_type)
            return [self._model.encode(text) for text in texts]
        except Exception as exc:
            embed_failure_total += 1
            logger.error(
                "%s embed encode failed: %s: %s",
                "native" if self._backend == "rust" else self._backend,
                type(exc).__name__,
                exc,
            )
            raise

    def embed(self, text: str, *, input_type: InputType = "document") -> list[float]:
        return self._encode_batch([text], input_type=input_type)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text, input_type="query")

    def embed_batch(
        self, texts: list[str], *, input_type: InputType = "document"
    ) -> list[list[float]]:
        return self._encode_batch(texts, input_type=input_type)

    def embed_quantized(
        self, text: str, *, input_type: InputType = "document"
    ) -> QuantizedVector:
        fp32 = self.embed(text, input_type=input_type)
        return _quantize_int8(fp32)


def embed_query(embedder, text: str) -> list[float]:
    """Embed a retrieval cue while preserving compatibility with test doubles."""
    method = getattr(embedder, "embed_query", None)
    if callable(method):
        return method(text)
    return embedder.embed(text)


def embedder_for_store(store) -> "Embedder":
    target_dim = getattr(store, "embed_dim", None)
    if _resolve_provider() == "http":
        embedder = Embedder()
        if target_dim is not None and int(target_dim) != embedder.DIM:
            raise ValueError(
                f"store uses {target_dim}d embeddings but the configured http "
                f"provider uses {embedder.DIM}d; run the re-embedding migration "
                "before starting IAE with this provider"
            )
        return embedder
    if target_dim is None:
        return Embedder()
    preferred = {384: "bge-small-en-v1.5"}
    key = preferred.get(int(target_dim))
    try:
        if key is not None and key in MODEL_REGISTRY:
            return Embedder(model_key=key)
        for reg_key, spec in MODEL_REGISTRY.items():
            if int(spec["dim"]) == int(target_dim):
                return Embedder(model_key=reg_key)
    except TypeError:
        pass
    return Embedder()
