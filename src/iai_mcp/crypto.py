from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

import zstandard as zstd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


CIPHERTEXT_PREFIX: str = "iai:enc:v1:"
NONCE_BYTES: int = 12
KEY_BYTES: int = 32
PBKDF2_ITERATIONS: int = 600_000
SERVICE_NAME_DEFAULT: str = "iai-mcp"

# One-byte version tag prefixed to the AEAD plaintext, BEFORE encryption. The
# version byte rides inside the AEAD envelope so it is authenticated for free
# and the outer wire format (iai:enc:v1:...) is unchanged. The codec routes on
# this byte AFTER the AEAD tag validates; legacy 0x00 rows (raw utf-8) decode
# identically to fresh 0x01 rows (zstd-compressed utf-8) — mixed corpus is the
# permanent steady-state.
ENVELOPE_VERSION_UNCOMPRESSED: int = 0x00
ENVELOPE_VERSION_ZSTD: int = 0x01
ZSTD_LEVEL: int = 3

# python-zstandard 0.25 documents that ZstdCompressor / ZstdDecompressor
# instances are NOT thread-safe (backend_cffi.py:1774, verbatim: "assume
# instances are not thread safe"). Sharing a single instance across threads
# corrupts the internal libzstd context state and crashes with SIGBUS under
# concurrent .compress() callers. Construction is cheap — a single libzstd
# ZSTD_createCCtx() call (~microseconds with no dict + no custom params) —
# so each encrypt / decrypt call constructs a fresh instance. The hot path
# stays lock-free + parallel-safe, and the per-call overhead is well below
# the AES-256-GCM cost that already dominates the per-field budget.

_DEFAULT_STORE_ROOT: Path = Path.home() / ".iai-mcp"
_KEY_FILE_NAME: str = ".crypto.key"


class CryptoKeyError(RuntimeError):
    pass


def is_encrypted(field: Optional[str]) -> bool:
    if not field or not isinstance(field, str):
        return False
    return field.startswith(CIPHERTEXT_PREFIX)


def encrypt_field(
    plaintext: str,
    key: bytes,
    associated_data: bytes = b"",
) -> str:
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(NONCE_BYTES)
    # Compose envelope: version byte + zstd-compressed utf-8 plaintext. The
    # AEAD then encrypts the full envelope, so the version byte is part of the
    # authenticated content.
    envelope = bytes([ENVELOPE_VERSION_ZSTD]) + zstd.ZstdCompressor(
        level=ZSTD_LEVEL
    ).compress(plaintext.encode("utf-8"))
    ct_with_tag = aesgcm.encrypt(nonce, envelope, associated_data or None)
    payload = nonce + ct_with_tag
    return CIPHERTEXT_PREFIX + base64.b64encode(payload).decode("ascii")


def decrypt_field(
    ciphertext_b64: str,
    key: bytes,
    associated_data: bytes = b"",
) -> str:
    if not is_encrypted(ciphertext_b64):
        raise ValueError("field is not iai:enc:v1:-prefixed ciphertext")
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
    payload_b64 = ciphertext_b64[len(CIPHERTEXT_PREFIX):]
    payload = base64.b64decode(payload_b64)
    if len(payload) < NONCE_BYTES + 16:
        raise ValueError("ciphertext payload too short")
    nonce = payload[:NONCE_BYTES]
    ct_with_tag = payload[NONCE_BYTES:]
    aesgcm = AESGCM(key)
    # AEAD authenticates the envelope FIRST. Any tamper raises InvalidTag here,
    # before the codec sees a single byte.
    envelope = aesgcm.decrypt(nonce, ct_with_tag, associated_data or None)
    if not envelope:
        raise ValueError("envelope empty after AEAD validation")
    # Route on the version byte AFTER AEAD validates and BEFORE utf-8 decode.
    # Decoding before the route would crash on the binary version byte (or on
    # the binary compressed body) for fresh 0x01 envelopes.
    #
    # Legacy detection is content-validated, not byte-0-heuristic.  Pre-97
    # rows have no version byte — envelope[0] is the first byte of the user's
    # verbatim content, which may legally be 0x00 (U+0000) or 0x01 (U+0001).
    # Trusting byte-0 alone would silently truncate a NUL-leading memory
    # (ENVELOPE_VERSION_UNCOMPRESSED strip) or permanently corrupt a
    # SOH-leading memory (spurious zstd.decompress on plain text).
    #
    # Disambiguation rules:
    #   0x01 → genuine zstd frame ONLY when the 4-byte zstd magic
    #           (\x28\xb5\x2f\xfd) immediately follows; otherwise treat the
    #           whole envelope as legacy plaintext (SOH-leading user content).
    #   0x00 → whole-envelope legacy.  The writer NEVER emits a bare 0x00
    #           envelope (every new write is 0x01), so any on-disk leading
    #           0x00 byte is legacy NUL-leading user content — never strip it.
    #   anything else → pre-envelope legacy passthrough (unchanged).
    _ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
    version = envelope[0]
    body = envelope[1:]
    if version == ENVELOPE_VERSION_ZSTD:
        if body[:4] == _ZSTD_MAGIC:
            # Genuine zstd frame written by this codec.
            try:
                plaintext_bytes = zstd.ZstdDecompressor().decompress(body)
            except zstd.ZstdError as exc:
                # Surface compressed-body corruption (after AEAD has
                # authenticated the wire) as a plain ValueError. Call sites
                # decide policy: records re-raise as HippoIntegrityError
                # (fail-loud); events absorb it and degrade to "{}"
                # (fail-soft).
                raise ValueError(
                    f"compressed envelope corrupt after AEAD validation: {exc}"
                ) from exc
        else:
            # 0x01 but no zstd magic — legacy content whose first byte
            # happens to be SOH (U+0001).  Return the whole envelope as
            # plaintext bytes; do not strip the version byte.
            plaintext_bytes = envelope
    elif version == ENVELOPE_VERSION_UNCOMPRESSED:
        # The writer never emits a bare 0x00 envelope, so any on-disk 0x00
        # leading byte is legacy NUL-leading user content.  Return the whole
        # envelope unchanged (do not strip the 0x00).
        plaintext_bytes = envelope
    else:
        # Pre-envelope legacy: rows written before the envelope was introduced
        # have no version byte — the entire AEAD plaintext is the raw utf-8
        # payload.  Any byte other than 0x00 / 0x01 is a printable or
        # high-byte lead char (e.g. 0x7b = `{`).  Use the whole envelope.
        plaintext_bytes = envelope
    return plaintext_bytes.decode("utf-8")


def derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    if len(salt) < 16:
        raise ValueError(f"salt must be at least 16 bytes (got {len(salt)})")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


class CryptoKey:

    SERVICE_NAME: str = SERVICE_NAME_DEFAULT

    def __init__(
        self,
        user_id: str = "default",
        store_root: Path | None = None,
    ) -> None:
        self.user_id = user_id
        self.store_root: Path | None = store_root
        self._cached_key: Optional[bytes] = None


    def _passphrase_salt(self) -> bytes:
        return hashlib.sha256(self.user_id.encode("utf-8")).digest()[:16]

    def _key_file_path(self) -> Path:
        if self.store_root is not None:
            root = Path(self.store_root)
        else:
            env_path = os.environ.get("IAI_MCP_STORE")
            root = Path(env_path) if env_path else _DEFAULT_STORE_ROOT
        return root / _KEY_FILE_NAME

    def _try_file_get(self) -> Optional[bytes]:
        path = self._key_file_path()
        if not path.exists():
            return None
        st = os.stat(path)
        # Unix permission-mode check. On Windows os.stat returns a default
        # mode that isn't meaningful (there's no chmod), and file access is
        # gated by ACLs — the setter handles that via icacls.
        if sys.platform != "win32" and st.st_mode & 0o077 != 0:
            raise CryptoKeyError(
                f"crypto key file at {path} has insecure mode "
                f"0o{st.st_mode & 0o777:03o}; expected 0o600 "
                f"(run: chmod 0o600 {path})"
            )
        # Unix uid check; on Windows there is no numeric uid concept and
        # os.geteuid does not exist. Access is enforced via ACLs (icacls) at
        # write time; skip the check on Windows.
        if sys.platform != "win32":
            euid = os.geteuid()  # type: ignore[attr-defined]
            if st.st_uid != euid:
                raise CryptoKeyError(
                    f"crypto key file at {path} is owned by uid={st.st_uid}; "
                    f"current process runs as uid={euid} (refusing to read)"
                )
        raw = path.read_bytes()
        if len(raw) != KEY_BYTES:
            raise CryptoKeyError(
                f"crypto key file at {path} has wrong length {len(raw)} "
                f"(expected {KEY_BYTES})"
            )
        return raw

    def _try_file_set(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
        final = self._key_file_path()
        final.parent.mkdir(parents=True, exist_ok=True)
        for stale in final.parent.glob(f"{final.name}.tmp.*"):
            try:
                stale.unlink()
            except OSError:
                pass
        tmp = final.parent / f"{final.name}.tmp.{os.getpid()}"
        # os.open defaults to text mode on Windows, which translates raw 0x0A
        # bytes in the AES key to 0x0D0A and silently corrupts it. os.O_BINARY
        # forces byte-exact writes; POSIX defines the flag as a no-op.
        _o_binary = getattr(os, "O_BINARY", 0)
        fd = os.open(
            str(tmp),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | _o_binary,
            0o600,
        )
        try:
            # os.fchmod does not exist on Windows; the initial mode bits already
            # ran through the umask, and Windows access control is handled via
            # icacls elsewhere.
            _fchmod = getattr(os, "fchmod", None)
            if _fchmod is not None:
                _fchmod(fd, 0o600)
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        # On Windows os.rename fails if the destination already exists; use
        # os.replace which is atomic on POSIX and does replace-if-exists on Win.
        os.replace(str(tmp), str(final))


    def get_or_create(self) -> bytes:
        if self._cached_key is not None:
            return self._cached_key

        existing = self._try_file_get()
        if existing is not None:
            self._cached_key = existing
            return existing

        passphrase = os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        if passphrase:
            derived = derive_key_from_passphrase(passphrase, self._passphrase_salt())
            self._cached_key = derived
            return derived

        path = self._key_file_path()
        raise CryptoKeyError(
            f"crypto key file not found at {path} and IAI_MCP_CRYPTO_PASSPHRASE "
            f"is not set.\n"
            f"\n"
            f"To fix:\n"
            f"  - Existing install (key already in macOS Keychain): "
            f"run `iai-mcp crypto migrate-to-file` from a Terminal where the "
            f"Keychain prompt can appear, then click \"Always Allow\".\n"
            f"  - Fresh install: run `iai-mcp crypto init` to generate a new key "
            f"file, OR set IAI_MCP_CRYPTO_PASSPHRASE to a strong passphrase "
            f"(suitable for CI or non-interactive environments)."
        )

    def rotate(self) -> bytes:
        fresh = secrets.token_bytes(KEY_BYTES)
        self._try_file_set(fresh)
        self._cached_key = fresh
        return fresh

    def delete(self) -> None:
        self._cached_key = None
        path = self._key_file_path()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
