"""Backend selector for the hypervector kernels.

The pure-Python tier implementations are the always-available reference. When
``IAI_MCP_HD_BACKEND`` is set to ``rust``, the inner numeric kernels delegate to
the native module instead; the Python path stays the parity reference and any
host-side policy (capacity caps, telemetry, key handling) remains in the host.

Hybrid routing under ``rust`` — NOT every op goes native:

* similarity / hamming / cosine — route to native (a single hardware popcount,
  far faster than the numpy unpack+count path).
* BSC majority-vote bundle — routes to native (a SWAR bit-sliced popcount vote,
  faster than vectorised numpy).
* projection (``embed @ P``) — STAYS on numpy in both backends. The matmul
  dispatches to the platform BLAS sgemv, which is memory-bandwidth-bound and
  already at the physical ceiling; a scalar native matmul is slower and it is
  also the bit-parity reference. The native projection kernel is kept and proven
  byte-identical on 100 frozen embeddings, but production projection is never
  routed through it. See ``crossmodal.embed_to_hv.from_embedding``.

The native module is imported lazily at the call site so a checkout without the
built extension still imports and runs the pure-Python path.
"""

from __future__ import annotations

import os

_BACKEND_ENV: str = "IAI_MCP_HD_BACKEND"
_RUST: str = "rust"
_PYTHON: str = "python"

_native_ok: bool | None = None


def _native_available() -> bool:
    """Whether the native ``hd`` module can be imported (cached)."""
    global _native_ok
    if _native_ok is None:
        try:
            from iai_mcp_native import hd as _hd  # noqa: F401

            _native_ok = True
        except Exception:  # noqa: BLE001 — absence is a normal dev/test state
            _native_ok = False
    return _native_ok


def backend() -> str:
    """Return the active backend, ``"rust"`` or ``"python"``.

    Read from the environment on every call so tests can flip it per-case. When
    the variable is unset, the native kernels serve by default whenever the
    extension is built (the output is byte-identical to the Python reference and
    faster on the hot path); a checkout without the built extension transparently
    uses the pure-Python path. Set ``IAI_MCP_HD_BACKEND`` to force either side.
    """
    raw = os.environ.get(_BACKEND_ENV)
    if raw is None:
        return _RUST if _native_available() else _PYTHON
    value = raw.strip().lower()
    return _RUST if value == _RUST else _PYTHON


def use_rust() -> bool:
    """True when the native kernels should serve the inner numeric op."""
    return backend() == _RUST


def native():
    """Lazily import and return the native ``hd`` sub-module.

    Raises a clear error if the extension is not built, so a misconfigured
    ``IAI_MCP_HD_BACKEND=rust`` fails fast with an actionable message instead of
    silently falling back.
    """
    try:
        from iai_mcp_native import hd as _hd
    except Exception as exc:  # noqa: BLE001 — surface a single actionable error
        raise RuntimeError(
            f"{_BACKEND_ENV}={_RUST} requires the native hypervector module, "
            "but iai_mcp_native.hd is not available. Rebuild the extension "
            "(iai-mcp build-native) or unset the variable to use the "
            "pure-Python path."
        ) from exc
    return _hd
