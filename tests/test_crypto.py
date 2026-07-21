from __future__ import annotations

import pytest


def test_crypto_module_exports() -> None:
    from iai_mcp import crypto
    assert hasattr(crypto, "encrypt_field")
    assert hasattr(crypto, "decrypt_field")
    assert hasattr(crypto, "is_encrypted")
    assert hasattr(crypto, "CryptoKey")
    assert hasattr(crypto, "derive_key_from_passphrase")


def test_crypto_roundtrip_basic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x00" * 32
    plaintext = "hello world"
    ciphertext = encrypt_field(plaintext, key)
    assert isinstance(ciphertext, str)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext


def test_crypto_roundtrip_cyrillic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x01" * 32
    plaintext = "こんにちは世界！これは暗号化のテストです。"
    ciphertext = encrypt_field(plaintext, key)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext
    assert recovered.encode("utf-8") == plaintext.encode("utf-8")


def test_crypto_roundtrip_cjk() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x02" * 32
    plaintext = "こんにちは世界。これは暗号化テストです。"
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_roundtrip_arabic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x03" * 32
    plaintext = "مرحبا بالعالم. هذا اختبار تشفير."
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_empty_string_roundtrip() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x04" * 32
    assert decrypt_field(encrypt_field("", key), key) == ""


def test_crypto_associated_data_binding() -> None:
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x05" * 32
    ciphertext = encrypt_field("secret", key, associated_data=b"record_id_A")
    with pytest.raises(InvalidTag):
        decrypt_field(ciphertext, key, associated_data=b"record_id_B")


def test_crypto_associated_data_roundtrip_when_matching() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x06" * 32
    ad = b"record_id_matching"
    ct = encrypt_field("secret", key, associated_data=ad)
    assert decrypt_field(ct, key, associated_data=ad) == "secret"


def test_crypto_tamper_detection() -> None:
    import base64
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x07" * 32
    ct = encrypt_field("secret", key)
    prefix = "iai:enc:v1:"
    assert ct.startswith(prefix)
    payload_b64 = ct[len(prefix):]
    raw = bytearray(base64.b64decode(payload_b64))
    raw[15] ^= 0x01
    tampered = prefix + base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(InvalidTag):
        decrypt_field(tampered, key)


def test_crypto_wrong_key_fails() -> None:
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key_a = b"\x08" * 32
    key_b = b"\x09" * 32
    ct = encrypt_field("secret", key_a)
    with pytest.raises(InvalidTag):
        decrypt_field(ct, key_b)


def test_is_encrypted_prefix_true() -> None:
    from iai_mcp.crypto import encrypt_field, is_encrypted
    key = b"\x0a" * 32
    ct = encrypt_field("hello", key)
    assert is_encrypted(ct) is True


def test_is_encrypted_prefix_false() -> None:
    from iai_mcp.crypto import is_encrypted
    assert is_encrypted("plaintext") is False
    assert is_encrypted("") is False
    assert is_encrypted("iai:enc:v0:abc") is False
    assert is_encrypted("foo:bar") is False


def test_crypto_unique_nonce_per_encrypt() -> None:
    from iai_mcp.crypto import encrypt_field
    key = b"\x0b" * 32
    ct1 = encrypt_field("repeat", key)
    ct2 = encrypt_field("repeat", key)
    assert ct1 != ct2


def test_derive_key_from_passphrase_deterministic() -> None:
    from iai_mcp.crypto import derive_key_from_passphrase
    salt = b"saltsaltsaltsalt"
    k1 = derive_key_from_passphrase("hunter2", salt)
    k2 = derive_key_from_passphrase("hunter2", salt)
    assert k1 == k2
    assert len(k1) == 32


def test_derive_key_from_passphrase_different_salts() -> None:
    from iai_mcp.crypto import derive_key_from_passphrase
    salt_a = b"A" * 16
    salt_b = b"B" * 16
    assert derive_key_from_passphrase("same", salt_a) != derive_key_from_passphrase("same", salt_b)


def test_derive_key_uses_600k_iterations() -> None:
    from iai_mcp import crypto
    assert crypto.PBKDF2_ITERATIONS >= 600_000


def test_crypto_key_passphrase_fallback_when_file_missing(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp import crypto

    assert not (tmp_path / ".crypto.key").exists()

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2-fallback")

    ck = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key1 = ck.get_or_create()
    assert isinstance(key1, bytes)
    assert len(key1) == 32

    ck2 = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key2 = ck2.get_or_create()
    assert key1 == key2


# --------------------------------------------------------------------------- #
# Envelope contract tests (compression codec inside the AES-GCM envelope).    #
#                                                                             #
# The codec composes as  compress -> prefix-version-byte -> AEAD-encrypt  on  #
# the write side, and  AEAD-decrypt -> route-on-version -> decompress (if     #
# compressed) -> utf-8-decode  on the read side. The version byte lives       #
# INSIDE the AEAD plaintext, so it is authenticated for free and the outer    #
# wire prefix "iai:enc:v1:" never changes.                                    #
#                                                                             #
# These tests fail on the current tree (no codec yet) and turn green once the #
# codec lands.                                                                #
# --------------------------------------------------------------------------- #

ENVELOPE_VERSION_UNCOMPRESSED: int = 0x00
ENVELOPE_VERSION_ZSTD: int = 0x01


def _aead_plaintext_of(ciphertext_b64: str, key: bytes, ad: bytes) -> bytes:
    """Decrypt the AEAD layer directly and return the raw envelope payload
    (version byte + body). Used to inspect what the codec wrote without
    delegating to the codec read path itself."""
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    prefix = "iai:enc:v1:"
    assert ciphertext_b64.startswith(prefix)
    raw = base64.b64decode(ciphertext_b64[len(prefix):])
    nonce, ct_with_tag = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ct_with_tag, ad or None)


def _wrap_payload_as_ciphertext(payload: bytes, key: bytes, ad: bytes) -> str:
    """AEAD-encrypt a hand-built envelope payload (version byte + body) and
    return the wire form. Used for hand-crafting legacy 0x00 rows and for
    constructing AEAD-valid but body-corrupt 0x01 rows."""
    import base64
    import secrets

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ct_with_tag = AESGCM(key).encrypt(nonce, payload, ad or None)
    return "iai:enc:v1:" + base64.b64encode(nonce + ct_with_tag).decode("ascii")


def test_envelope_default_is_zstd_version_byte() -> None:
    from iai_mcp.crypto import encrypt_field

    key = b"\x10" * 32
    # Highly compressible plaintext makes the failure mode unambiguous if the
    # codec misroutes (the body must not equal the raw utf-8 bytes).
    plaintext = "the quick brown fox " * 30
    wire = encrypt_field(plaintext, key)
    payload = _aead_plaintext_of(wire, key, ad=b"")
    assert payload, "AEAD plaintext unexpectedly empty"
    assert payload[0] == ENVELOPE_VERSION_ZSTD, (
        f"new writes must default to compressed envelope, got version "
        f"byte {payload[0]:#x}"
    )


def test_envelope_legacy_0x00_decrypts_to_identity() -> None:
    """The writer never emits a bare 0x00 envelope.  Any on-disk leading 0x00
    byte is therefore legacy NUL-leading user content.  The whole envelope
    must be returned verbatim (no byte stripped).

    This test seeds the row the way the old (pre-97) writer actually wrote it:
    raw utf-8, no version prefix.  The plaintext happens to start with U+0000
    (NUL), which is the exact collision that the naive byte-0 heuristic turns
    into a silent one-character loss.
    """
    from iai_mcp.crypto import decrypt_field

    key = b"\x11" * 32
    # Old-writer encoding: raw utf-8, no version prefix at all.
    nul_leading = "\x00legacy value"
    raw_utf8 = nul_leading.encode("utf-8")
    legacy_wire = _wrap_payload_as_ciphertext(raw_utf8, key, ad=b"")
    recovered = decrypt_field(legacy_wire, key)
    assert recovered == nul_leading, (
        "legacy NUL-leading content must round-trip byte-identical (no char dropped)"
    )


def test_envelope_unknown_first_byte_is_treated_as_pre_envelope_legacy() -> None:
    """Pre-Phase-97 records were written without a version byte — the entire
    AEAD plaintext was the raw utf-8 payload. The decrypt path must accept
    these or every legacy row on disk raises HippoIntegrityError and crashes
    the daemon (the regression that surfaced this contract). Any byte other
    than 0x00 / 0x01 is treated as a pre-envelope legacy record and the whole
    envelope is utf-8 decoded. Picking a printable ascii first byte here
    (0x7b = `{`, the leading char of every JSON object) mirrors the real
    on-disk shape that crashed the daemon.
    """
    from iai_mcp.crypto import decrypt_field

    key = b"\x12" * 32
    payload = b'{"channels": ["trust"]}'
    wire = _wrap_payload_as_ciphertext(payload, key, ad=b"")
    out = decrypt_field(wire, key)
    assert out == payload.decode("utf-8")


def test_envelope_lm_utf_decode_after_decompress_not_before() -> None:
    from iai_mcp.crypto import decrypt_field, encrypt_field

    key = b"\x13" * 32
    # Pick a plaintext whose first utf-8 byte is 0x28 (the first byte of the
    # zstd magic 0x28 0xb5 0x2f 0xfd). If anyone reorders the codec to do
    # utf-8 decode BEFORE the version-byte route, the test crashes with
    # UnicodeDecodeError on the binary version byte or the binary compressed
    # body. Mixing high-byte multi-byte sequences guarantees the round-trip
    # would also tear under premature decoding.
    plaintext = "(µ/ý trailing — non-ascii mix to keep utf-8 boundaries honest"
    wire = encrypt_field(plaintext, key)
    recovered = decrypt_field(wire, key)
    assert recovered == plaintext
    assert recovered.encode("utf-8") == plaintext.encode("utf-8")


def test_envelope_byte_identity_roundtrip() -> None:
    from iai_mcp.crypto import decrypt_field, encrypt_field

    key = b"\x14" * 32
    plaintexts = [
        "ok",
        "the quick brown fox jumps over the lazy dog " * 50,
        '[{"a":1,"b":2,"c":[1,2,3,4,5]}]',
        "こんにちは世界 — CJK mix",
        "",
        # Adversarial: new-write round-trip must survive leading control bytes
        # that collide with envelope version tags (these are the exact bytes
        # that the CR-01 fix guards against for legacy records).
        "\x00leading nul char",
        "\x01leading soh char",
        "\x01\x28\xb5\x2f\xfd looks like zstd magic but is not our envelope",
    ]
    for pt in plaintexts:
        wire = encrypt_field(pt, key)
        assert decrypt_field(wire, key) == pt, (
            f"byte-identity round-trip failed for plaintext of length {len(pt)}: "
            f"first 4 bytes = {pt[:4]!r}"
        )


def test_decrypt_field_handles_pre_envelope_legacy_payload() -> None:
    """Records written before the envelope version byte was introduced have
    no leading 0x00 / 0x01 — the entire AEAD plaintext is the raw utf-8
    payload. decrypt_field must read them as-is, not interpret byte[0] as a
    version tag (or it would crash on legitimate JSON `{` = 0x7b, ASCII
    letters, etc.).
    """
    import base64
    import secrets

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from iai_mcp.crypto import CIPHERTEXT_PREFIX, decrypt_field

    key = bytes([0xAB] * 32)
    raw_payload = b'{"channels": [{"name": "trust", "gain": 0.42}]}'
    nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(key)
    aad = b"legacy-record-id"
    # Encrypt the raw payload directly — no envelope version byte. This
    # mirrors how pre-Phase-97 records were written to disk.
    ct = aesgcm.encrypt(nonce, raw_payload, aad)
    wire = CIPHERTEXT_PREFIX + base64.b64encode(nonce + ct).decode("ascii")

    out = decrypt_field(wire, key, associated_data=aad)
    assert out == raw_payload.decode("utf-8")


# --------------------------------------------------------------------------- #
# Adversarial legacy detection — the CR-01 regression guard set.              #
#                                                                             #
# These four cases are the exact shapes that a naive byte-0 heuristic fails   #
# on.  All four must return the FULL raw utf-8 payload byte-identical (no     #
# byte stripped, no crash) because they are seeded as legacy rows (no version #
# prefix) the way the old writer actually wrote them.                         #
# --------------------------------------------------------------------------- #

def _legacy_wire(raw_utf8: bytes, key: bytes) -> str:
    """Encrypt raw utf-8 bytes with no version prefix, mirroring the old
    pre-97 writer.  The resulting wire token is a legitimate AEAD-authenticated
    field that the codec must decode as-is."""
    return _wrap_payload_as_ciphertext(raw_utf8, key, ad=b"")


def test_legacy_nul_leading_roundtrip_byte_identical() -> None:
    """Legacy literal_surface beginning with U+0000 (NUL).
    Old heuristic: routes as ENVELOPE_VERSION_UNCOMPRESSED → strips first
    byte → silent one-char loss.  Fixed: whole envelope returned verbatim."""
    from iai_mcp.crypto import decrypt_field

    key = b"\x20" * 32
    content = "\x00 leading nul — this character must survive"
    wire = _legacy_wire(content.encode("utf-8"), key)
    recovered = decrypt_field(wire, key)
    assert recovered == content, (
        f"legacy NUL-leading payload corrupted: got {recovered!r}"
    )
    assert recovered.encode("utf-8") == content.encode("utf-8")


def test_legacy_soh_leading_roundtrip_byte_identical() -> None:
    """Legacy literal_surface beginning with U+0001 (SOH).
    Old heuristic: routes as ENVELOPE_VERSION_ZSTD → ZstdError →
    HippoIntegrityError → record permanently unreadable.
    Fixed: no zstd magic found → whole envelope returned verbatim."""
    from iai_mcp.crypto import decrypt_field

    key = b"\x21" * 32
    content = "\x01 leading soh — this must not be fed to zstd"
    wire = _legacy_wire(content.encode("utf-8"), key)
    recovered = decrypt_field(wire, key)
    assert recovered == content, (
        f"legacy SOH-leading payload corrupted: got {recovered!r}"
    )
    assert recovered.encode("utf-8") == content.encode("utf-8")


def test_legacy_soh_plus_zstd_magic_lookalike_roundtrip_byte_identical() -> None:
    """Legacy content that begins with 0x01 followed by the exact 4-byte zstd
    magic sequence (0x28 0xb5 0x2f 0xfd).  This is the most adversarial case:
    the byte pattern looks like a genuine zstd frame, but it is raw user text
    that was written before the codec existed — feeding it to zstd.decompress
    would either silently corrupt it (wrong output) or raise ZstdError.
    Fixed: the old writer never prefixed 0x01, so this row was written as a
    raw-utf8 legacy row; the zstd branch requires the actual compressed frame
    to produce valid output; on failure the whole envelope is returned.
    NOTE: this case is particularly subtle.  The zstd decompressor may
    successfully produce *some* output for certain byte patterns, or may raise.
    Either way the output would NOT equal the original content.  The fix
    prevents reaching zstd at all for this row because a real zstd frame built
    from user text shorter than ~10 bytes would embed a proper content-size
    header and frame checksum; a raw-text lookalike has neither.  We test that
    the round-trip returns the exact original bytes."""
    from iai_mcp.crypto import decrypt_field

    key = b"\x22" * 32
    # Start with SOH, then the 4-byte zstd magic literal, then normal text.
    content = "\x01\x28\xb5\x2f\xfd this looks like a zstd frame but it is not"
    wire = _legacy_wire(content.encode("utf-8"), key)
    recovered = decrypt_field(wire, key)
    assert recovered == content, (
        f"legacy SOH+magic-lookalike payload corrupted: got {recovered!r}"
    )


def test_legacy_printable_ascii_leading_roundtrip_byte_identical() -> None:
    """Baseline: legacy content beginning with a printable ASCII byte (the
    common case — JSON, plain text).  Must continue to work after the fix."""
    from iai_mcp.crypto import decrypt_field

    key = b"\x23" * 32
    content = '{"channels": ["trust"], "note": "baseline legacy"}'
    wire = _legacy_wire(content.encode("utf-8"), key)
    recovered = decrypt_field(wire, key)
    assert recovered == content
