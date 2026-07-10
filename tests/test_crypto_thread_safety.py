"""Concurrent-burst hardening for the at-rest field codec.

The Phase 97 zstd envelope ships short-lived `ZstdCompressor` / `ZstdDecompressor`
instances inside `encrypt_field` / `decrypt_field` precisely because the
python-zstandard library documents that those instances are not thread-safe.
These tests pin that contract: 16 threads hit the codec under a synchronized
barrier so any future regression that re-introduces a shared mutable codec
state surfaces here, not in production.
"""

from __future__ import annotations

import secrets
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from iai_mcp.crypto import CIPHERTEXT_PREFIX, decrypt_field, encrypt_field


_KEY = bytes([0xAB] * 32)
_THREADS = 16
_ENCRYPT_ROUNDS_PER_THREAD = 50
_DECRYPT_REPEATS_PER_THREAD = 10
_CORPUS_SIZE = 100


def test_concurrent_encrypt_field_under_barrier() -> None:
    barrier = threading.Barrier(_THREADS)
    results: list[tuple[str, str]] = []
    results_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        local_pairs: list[tuple[str, str]] = []
        try:
            barrier.wait(timeout=10)
            for r in range(_ENCRYPT_ROUNDS_PER_THREAD):
                plaintext = f"thread-{tid}-round-{r}-payload-{secrets.token_hex(8)}"
                ciphertext = encrypt_field(plaintext, _KEY, associated_data=b"crypto-ts-test")
                local_pairs.append((plaintext, ciphertext))
        except BaseException as exc:
            errors.append(exc)
            return
        with results_lock:
            results.extend(local_pairs)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        futures = [pool.submit(worker, tid) for tid in range(_THREADS)]
        for f in futures:
            f.result()

    assert errors == [], f"unexpected exceptions under concurrent encrypt: {errors!r}"
    assert len(results) == _THREADS * _ENCRYPT_ROUNDS_PER_THREAD
    for plaintext, ciphertext in results:
        assert ciphertext.startswith(CIPHERTEXT_PREFIX)
        decrypted = decrypt_field(ciphertext, _KEY, associated_data=b"crypto-ts-test")
        assert decrypted == plaintext


def test_concurrent_decrypt_field_under_barrier() -> None:
    corpus: list[tuple[str, str]] = []
    for i in range(_CORPUS_SIZE):
        plaintext = f"corpus-{i}-{secrets.token_hex(16)}"
        corpus.append((plaintext, encrypt_field(plaintext, _KEY, associated_data=b"crypto-ts-test")))

    barrier = threading.Barrier(_THREADS)
    errors: list[BaseException] = []
    mismatches: list[tuple[str, str]] = []
    mismatches_lock = threading.Lock()

    def worker(tid: int) -> None:
        local_mismatches: list[tuple[str, str]] = []
        try:
            barrier.wait(timeout=10)
            for _ in range(_DECRYPT_REPEATS_PER_THREAD):
                for plaintext, ciphertext in corpus:
                    decrypted = decrypt_field(ciphertext, _KEY, associated_data=b"crypto-ts-test")
                    if decrypted != plaintext:
                        local_mismatches.append((plaintext, decrypted))
        except BaseException as exc:
            errors.append(exc)
            return
        if local_mismatches:
            with mismatches_lock:
                mismatches.extend(local_mismatches)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        futures = [pool.submit(worker, tid) for tid in range(_THREADS)]
        for f in futures:
            f.result()

    assert errors == [], f"unexpected exceptions under concurrent decrypt: {errors!r}"
    assert mismatches == [], f"plaintext mismatches under concurrent decrypt: {mismatches[:5]!r}"
