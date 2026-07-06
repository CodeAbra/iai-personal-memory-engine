from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


import platform as _platform
import subprocess as _subprocess

CIPHERTEXT_PREFIX: str = "iai:enc:v1:"
NONCE_BYTES: int = 12
KEY_BYTES: int = 32
PBKDF2_ITERATIONS: int = 600_000
SERVICE_NAME_DEFAULT: str = "iai-mcp"


def _secure_key_file(path: Path) -> None:
    """Restrict file permissions to owner-only. On POSIX uses chmod; on Windows uses icacls."""
    if _platform.system() == "Windows":
        user = os.environ.get("USERNAME", "")
        if user:
            _subprocess.run(
                ["icacls", str(path), "/inheritance:d", "/grant:r", f"{user}:F"],
                check=False, capture_output=True,
            )
    else:
        path.chmod(0o600)

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
    ct_with_tag = aesgcm.encrypt(
        nonce, plaintext.encode("utf-8"), associated_data or None
    )
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
    plaintext_bytes = aesgcm.decrypt(
        nonce, ct_with_tag, associated_data or None
    )
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
        if hasattr(os, "geteuid") and st.st_mode & 0o077 != 0:
            raise CryptoKeyError(
                f"crypto key file at {path} has insecure mode "
                f"0o{st.st_mode & 0o777:03o}; expected 0o600 "
                f"(run: chmod 0o600 {path})"
            )
        if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            raise CryptoKeyError(
                f"crypto key file at {path} is owned by uid={st.st_uid}; "
                f"current process runs as uid={os.geteuid()} (refusing to read)"
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
        # O_BINARY: on Windows os.open defaults to text mode, which translates
        # 0x0A bytes in the raw key to 0x0D0A on write and silently corrupts it
        # (a random 32-byte key almost always contains a 0x0A). No-op on POSIX.
        fd = os.open(
            str(tmp),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        if not hasattr(os, "fchmod"):
            _secure_key_file(tmp)
        # os.replace (not os.rename): on Windows rename raises if the
        # destination exists, which it always does during key rotation. POSIX
        # rename already replaces, so this is behaviour-preserving there.
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
