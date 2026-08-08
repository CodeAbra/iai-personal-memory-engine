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
    "bge-small-en-v1.5": {
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "pool": "cls",
        "query_prefix": "",
        "document_prefix": "",
    },
    # Opt-in multilingual pack. Mean-pooled with asymmetric E5 prefixes —
    # serving it CLS-pooled or unprefixed silently degrades every vector.
    "multilingual-e5-small": {
        "hf": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "dim": 384,
        "pool": "mean",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}
DEFAULT_MODEL_KEY = "bge-small-en-v1.5"
MULTILINGUAL_MODEL_KEY = "multilingual-e5-small"

#: Languages the multilingual opt-in is benchmarked and gated for.
SUPPORTED_OPTIN_LANGS: frozenset[str] = frozenset({
    "cs", "de", "es", "fr", "hi", "id", "it",
    "ja", "pt", "ru", "th", "vi", "zh",
})

VALID_QUANTIZE_MODES: set[str] = {"int8"}

embed_failure_total: int = 0

InputType = Literal["query", "document"]
VALID_PROVIDERS: set[str] = {"native", "http"}
MAX_HTTP_RESPONSE_BYTES = 32 * 1024 * 1024


def _configured_model_key() -> str | None:
    """Model selection from the store config file, never from env.

    The daemon's service environment can carry stale overrides invisibly
    (a poisoned launchd plist served wrong-generation vectors for weeks);
    the config file is the one deliberate, inspectable switch. Written by
    `iai lang` alongside the full re-embed that makes it safe.
    """
    from iai_mcp.tz import store_config_path

    path = store_config_path()
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    key = ((cfg.get("embed") or {}).get("model_key") or "").strip()
    if not key:
        return None
    if key not in MODEL_REGISTRY:
        raise EmbedderConfigError(
            f"config.json embed.model_key {key!r} is not a known model; "
            f"valid: {sorted(MODEL_REGISTRY)}"
        )
    return key


def _resolve_model_key(model_key: str | None = None) -> str:
    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown embed model key {model_key!r}; valid: {sorted(MODEL_REGISTRY)}"
            )
        return model_key
    configured = _configured_model_key()
    if configured is not None:
        return configured
    return DEFAULT_MODEL_KEY


def _resolve_quantize_mode() -> str | None:
    raw = os.environ.get("IAI_MCP_EMBED_QUANTIZE", "")
    if not raw:
        return None
    if raw not in VALID_QUANTIZE_MODES:
        raise EmbedderConfigError(
            f"IAI_MCP_EMBED_QUANTIZE={raw!r} is not a valid quantization mode; "
            f"valid: {sorted(VALID_QUANTIZE_MODES)} or unset for fp32 default"
        )
    return raw


def _resolve_provider() -> str:
    provider = os.environ.get("IAI_MCP_EMBED_PROVIDER", "native").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise EmbedderConfigError(
            f"IAI_MCP_EMBED_PROVIDER={provider!r} is not valid; "
            f"valid: {sorted(VALID_PROVIDERS)}"
        )
    return provider


def _resolve_http_config() -> tuple[str, int, float, str]:
    raw_url = os.environ.get("IAI_MCP_EMBED_URL", "").strip()
    if not raw_url:
        raise EmbedderConfigError("IAI_MCP_EMBED_URL is required for the http provider")
    parsed = urlparse(raw_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EmbedderConfigError(
            "IAI_MCP_EMBED_URL must be an unauthenticated loopback http URL"
        )
    if parsed.path in {"", "/"}:
        raw_url = raw_url.rstrip("/") + "/embed"
    elif parsed.path != "/embed":
        raise EmbedderConfigError("IAI_MCP_EMBED_URL path must be /embed or empty")

    raw_dim = os.environ.get("IAI_MCP_EMBED_DIM", "").strip()
    try:
        dim = int(raw_dim)
    except ValueError as exc:
        raise EmbedderConfigError(
            "IAI_MCP_EMBED_DIM must be a positive integer for the http provider"
        ) from exc
    if dim <= 0:
        raise EmbedderConfigError(
            "IAI_MCP_EMBED_DIM must be a positive integer for the http provider"
        )

    raw_timeout = os.environ.get("IAI_MCP_EMBED_TIMEOUT_SEC", "30").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise EmbedderConfigError("IAI_MCP_EMBED_TIMEOUT_SEC must be positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise EmbedderConfigError("IAI_MCP_EMBED_TIMEOUT_SEC must be positive")

    model_id = os.environ.get("IAI_MCP_EMBED_MODEL_ID", "").strip()
    if not model_id:
        raise EmbedderConfigError("IAI_MCP_EMBED_MODEL_ID is required for the http provider")
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
        self._pool: str = str(spec.get("pool", "cls"))
        self._query_prefix: str = str(spec.get("query_prefix", ""))
        self._document_prefix: str = str(spec.get("document_prefix", ""))

        from iai_mcp_native import embed as rust

        if key == DEFAULT_MODEL_KEY:
            # Argless construction keeps the env-override resolution of the
            # pinned default byte-identical; the identity stamp guards mixing.
            self._model = rust.Embedder()
        else:
            self._model = rust.Embedder(
                model_id=spec["hf"],
                revision=str(spec.get("revision", "main")),
                pool=self._pool,
            )
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
            pfx = (
                self._query_prefix
                if input_type == "query"
                else self._document_prefix
            )
            if pfx:
                texts = [pfx + t for t in texts]
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


def _valid_cue_vec(vec, dim) -> "list[float] | None":
    """A caller-supplied cue vector is honored only when it can rank:
    numeric, the store's embed dim, all-finite, nonzero norm. Anything else
    must fall back to the server-side embed -- degraded dispatchers pad an
    absent cue with zeros, and a NaN/inf vector sanitizes to zero downstream,
    which flattens the score head."""
    if vec is None:
        return None
    try:
        if len(vec) != int(dim):
            return None
        out = [float(x) for x in vec]
    except (TypeError, ValueError):
        return None
    norm_sq = 0.0
    for x in out:
        if not math.isfinite(x):
            return None
        norm_sq += x * x
    if norm_sq <= 0.0:
        return None
    return out


EMBED_IDENTITY_META_KEY = "embed_model_identity"


def effective_model_identity(embedder: "Embedder") -> str:
    """The full identity of the vectors this embedder produces.

    Dimension alone cannot distinguish models: two different 384d models
    produce mutually meaningless vectors, and the native backend honors
    ``IAI_MCP_EMBED_MODEL_ID``/``IAI_MCP_EMBED_REVISION`` overrides that the
    Python layer would otherwise never see. The identity string therefore
    folds in the effective model id, revision, pooling scheme, and dimension.
    """
    prefix = os.environ.get("IAI_MCP_EMBED_TEXT_PREFIX", "")
    prefix_part = f"|prefix={prefix}" if prefix else ""
    if getattr(embedder, "_backend", "") == "http":
        return f"http:{embedder.model_name}|dim={embedder.DIM}{prefix_part}"
    # The env override changes what the native backend loads ONLY on the
    # argless default construction; a registry-selected model is loaded by
    # explicit spec and ignores the env. The stamp must mirror that exactly,
    # or it attests to a model the vectors did not come from.
    override = ""
    if getattr(embedder, "model_key", "") == DEFAULT_MODEL_KEY:
        override = os.environ.get("IAI_MCP_EMBED_MODEL_ID", "").strip()
    if override:
        # An overridden model with no revision env loads the moving HF "main"
        # ref — record that truthfully rather than pretending it is pinned.
        model_id = override
        revision = os.environ.get("IAI_MCP_EMBED_REVISION", "").strip() or "main"
    else:
        model_id = embedder.model_name
        revision = "pinned"
    pool = getattr(embedder, "_pool", "cls") or "cls"
    qp = getattr(embedder, "_query_prefix", "")
    dp = getattr(embedder, "_document_prefix", "")
    # Appended only when set, so the pinned default's stamp stays
    # byte-identical to every identity already written to stores.
    asym = (f"|qprefix={qp}" if qp else "") + (f"|dprefix={dp}" if dp else "")
    return (
        f"{model_id}@{revision}|pool={pool}|dim={embedder.DIM}"
        f"{prefix_part}{asym}"
    )


#: Every embedder-selection refusal message carries this phrase as the
#: human-facing runbook pointer; the wire contract clients key on is
#: errors.ERR_EMBEDDER_REFUSAL, with this phrase only as the legacy
#: fallback — rewording must still move every carrier together.
REEMBED_RUNBOOK_HINT = "run the re-embedding migration"


class EmbedderConfigError(ValueError):
    """Embedder selection refused: no embedder can serve this store's
    vector space (foreign dimension or misconfigured model).

    A dedicated type so the recall surface can fail loudly on THIS error
    alone — every other pipeline exception keeps its soft degrade path.
    """


class EmbedIdentityMismatch(ValueError):
    """The store's vectors were produced by a different embedder identity.

    A dedicated type so the daemon can refuse boot on THIS error alone —
    every other config ValueError from the embed layer keeps its normal
    degrade path instead of crash-looping under the service manager.
    """


def _enforce_store_embed_identity(store, embedder: "Embedder", *, allow_mismatch: bool) -> None:
    # One store, one vector space. The identity is written ONLY by a completed
    # re-embed migration — never adopted here: on a store whose vectors are
    # already foreign, an implicit adopt would stamp a false attestation the
    # guard could never see past. An unstamped store therefore passes
    # unguarded (legacy tolerance); protection begins at the first stamp.
    # This is also a pure read — the guard must work on read-only opens.
    db = getattr(store, "db", None)
    if db is None:
        return
    try:
        from iai_mcp.hippo import HippoDB
        if not isinstance(db, HippoDB):
            return
    except ImportError:
        return
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (EMBED_IDENTITY_META_KEY,),
        ).fetchone()
        stored = row["value"] if row is not None else None
    if stored is None:
        # Unstamped store: nothing to compare against — and the runtime
        # identity must not even be computed here, because duck-typed test
        # embedders legitimately lack the attributes it reads.
        return
    if getattr(embedder, "DIM", None) is None:
        # A stamped store demands verification; an embedder without DIM
        # cannot have its identity computed (effective_model_identity reads
        # it unconditionally). Skipping here would fail OPEN — serving
        # cross-generation rows confidently — so refuse unless the caller
        # explicitly sanctioned a mismatch.
        if allow_mismatch:
            return
        # An unverifiable identity on a stamped store is an identity
        # failure — the type routes it to the daemon's refuse-boot exactly
        # like a concrete mismatch.
        raise EmbedIdentityMismatch(
            f"store vectors are identity-stamped as {stored!r} but the "
            "runtime embedder declares no dimension, so its identity cannot "
            f"be verified; {REEMBED_RUNBOOK_HINT} or use an embedder with a "
            "declared DIM"
        )
    identity = effective_model_identity(embedder)
    if stored != identity and not allow_mismatch:
        raise EmbedIdentityMismatch(
            f"store vectors were produced by {stored!r} but the runtime "
            f"embedder is {identity!r}; refusing to mix vector generations — "
            f"{REEMBED_RUNBOOK_HINT} "
            "(iai-mcp migrate --reembed-to-configured-provider) to move the "
            "store to the new model, or restore the original embedder "
            "configuration."
        )


def stamp_store_embed_identity(store, embedder: "Embedder") -> None:
    """Overwrite the store's recorded vector identity.

    Only a completed re-embed migration may call this — it is the single
    sanctioned way the identity changes.
    """
    db = getattr(store, "db", None)
    if db is None:
        return
    from iai_mcp.hippo import HippoDB
    if not isinstance(db, HippoDB):
        return
    if getattr(embedder, "DIM", None) is None:
        # Refuse BEFORE the identity computation would crash untyped —
        # a stamp written mid-migration from an unverifiable embedder
        # would attest to nothing.
        raise EmbedderConfigError(
            "cannot stamp a store from an embedder that declares no "
            f"dimension; {REEMBED_RUNBOOK_HINT} with an embedder whose "
            "DIM is set"
        )
    identity = effective_model_identity(embedder)
    with db._conn_lock:
        db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (EMBED_IDENTITY_META_KEY,)
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (EMBED_IDENTITY_META_KEY, identity),
        )
        db._conn.commit()


def embedder_for_store(store, *, allow_identity_mismatch: bool = False) -> "Embedder":
    target_dim = getattr(store, "embed_dim", None)
    if _resolve_provider() == "http":
        embedder = Embedder()
        if target_dim is not None and int(target_dim) != embedder.DIM:
            raise EmbedderConfigError(
                f"store uses {target_dim}d embeddings but the configured http "
                f"provider uses {embedder.DIM}d; {REEMBED_RUNBOOK_HINT} "
                "before starting IAE with this provider"
            )
        _enforce_store_embed_identity(
            store, embedder, allow_mismatch=allow_identity_mismatch
        )
        return embedder
    if target_dim is None:
        embedder = Embedder()
        _enforce_store_embed_identity(
            store, embedder, allow_mismatch=allow_identity_mismatch
        )
        return embedder
    # The configured model wins whenever its dimension fits the store —
    # otherwise a dim-keyed table with two 384d entries would silently hand
    # an e5-stamped store a bge embedder and brick boot on the identity
    # guard. The English default is the fallback, never the override.
    configured = _configured_model_key()
    if configured is not None:
        configured_dim = int(MODEL_REGISTRY[configured]["dim"])
        if configured_dim != int(target_dim):
            raise EmbedderConfigError(
                f"store uses {target_dim}d embeddings but the configured model "
                f"'{configured}' produces {configured_dim}d; "
                f"{REEMBED_RUNBOOK_HINT} before starting IAE with this model"
            )
        key = configured
    else:
        preferred = {384: DEFAULT_MODEL_KEY}
        key = preferred.get(int(target_dim))
    if key == DEFAULT_MODEL_KEY:
        # With no config the resolver lands on the default anyway; argless
        # construction keeps this arm byte-identical to every other default
        # call site (and to the native env-honoring arm).
        embedder = Embedder()
    elif key is not None and key in MODEL_REGISTRY:
        embedder = Embedder(model_key=key)
    else:
        embedder = None
        for reg_key, spec in MODEL_REGISTRY.items():
            if int(spec["dim"]) == int(target_dim):
                embedder = Embedder(model_key=reg_key)
                break
        if embedder is None:
            # Same contract as the http branch: a store whose vectors no
            # registered model can reproduce is refused, never silently
            # written to in a foreign vector space. (The synthetic-dim
            # bench harness catches this and hosts its shim embedderless.)
            raise EmbedderConfigError(
                f"store uses {target_dim}d embeddings but no registered "
                f"model produces {target_dim}d vectors; "
                f"{REEMBED_RUNBOOK_HINT} before starting IAE with this store"
            )
    _enforce_store_embed_identity(
        store, embedder, allow_mismatch=allow_identity_mismatch
    )
    return embedder
