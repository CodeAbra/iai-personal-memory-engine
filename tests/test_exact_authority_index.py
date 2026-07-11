"""Unit contract for ExactCosineIndex — the resident exact-cosine matrix.

This class is pure host-side numpy + stdlib threading: it never imports
hippo, sqlite, crypto, hnswlib, or the frozen projection. All tests in this
file construct the index directly with synthetic float32 blobs; no
MemoryStore, no store directory, no embedder.

All tests are deterministic (seeded numpy RNG), single-process.
"""

from __future__ import annotations

import struct
import sys
import threading

import numpy as np
import pytest

from iai_mcp.store._exact_index import ExactCosineIndex

EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec_to_blob(vec: np.ndarray) -> bytes:
    """Pack a float32 vector into a raw little-endian blob, mirroring the
    store's on-disk embedding encoding."""
    return struct.pack(f"<{len(vec)}f", *vec.astype(np.float32).tolist())


def _random_rows(n: int, seed: int, dim: int = EMBED_DIM) -> list[tuple[str, bytes]]:
    """Build n (record_id, blob) rows from a seeded RNG, unnormalized."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        v = rng.random(dim).astype(np.float32)
        rows.append((f"id-{i}", _vec_to_blob(v)))
    return rows


def _decode_matrix(rows: list[tuple[str, bytes]], dim: int = EMBED_DIM) -> np.ndarray:
    """Decode a row list into a raw (unnormalized) float32 matrix, in row order."""
    return np.stack(
        [np.frombuffer(blob, dtype=np.float32) for _, blob in rows]
    ).reshape(len(rows), dim)


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero-norm rows stay zero."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    out = m / safe
    out[norms.squeeze(-1) == 0] = 0.0
    return out.astype(np.float32)


def _brute_force_top_k(
    rows: list[tuple[str, bytes]], cue: np.ndarray, k: int
) -> list[tuple[str, float]]:
    """Reference brute-force cosine top-k: normalize all rows, dot with unit cue,
    argsort descending, truncate to k."""
    ids = [rid for rid, _ in rows]
    m = _decode_matrix(rows)
    m_norm = _normalize_rows(m)
    cue_norm = cue.astype(np.float32)
    cue_norm_len = np.linalg.norm(cue_norm)
    if cue_norm_len != 0:
        cue_norm = cue_norm / cue_norm_len
    sims = m_norm @ cue_norm
    order = np.argsort(-sims, kind="stable")
    order = order[:k]
    return [(ids[i], float(sims[i])) for i in order]


def _random_cue(seed: int, dim: int = EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Build + top_k exactness
# ---------------------------------------------------------------------------


def test_build_top_k_matches_brute_force_reference():
    """build 500 rows, query 50 cues at k in {1, 5, 10, 200}; ids and scores
    match the brute-force reference exactly in order, scores within 1e-6."""
    rows = _random_rows(500, seed=1)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)

    assert idx.is_warm

    for k in (1, 5, 10, 200):
        for cue_seed in range(50):
            cue = _random_cue(seed=10_000 + cue_seed)
            expected = _brute_force_top_k(rows, cue, k)
            got = idx.top_k(cue, k)
            assert got is not None
            assert len(got) == len(expected)
            for (exp_id, exp_score), (got_id, got_score) in zip(expected, got):
                assert got_id == exp_id, (
                    f"k={k} cue_seed={cue_seed}: id mismatch "
                    f"expected={exp_id} got={got_id}"
                )
                assert abs(exp_score - got_score) < 1e-6, (
                    f"k={k} cue_seed={cue_seed}: score mismatch for {exp_id} "
                    f"expected={exp_score} got={got_score}"
                )


def test_top_k_boundary_tie_matches_stable_argsort_reference_exactly():
    """When the score at the k-boundary ties with scores just outside the
    partition (duplicate-capture embeddings, or several zero-norm rows),
    top_k must select and order EXACTLY what a full stable argsort over the
    whole array would: the lowest original indices among the tied group,
    never a partition-dependent subset.

    Regression guard for the dropped argpartition fast path: an
    argpartition-based selection gives no guarantee about which
    boundary-tied elements land inside the partition, so this test
    constructs a corpus with an intentional tie straddling k exactly and
    repeats the query many times to catch any run-to-run nondeterminism.
    """
    rng = np.random.default_rng(500)
    base = rng.random(EMBED_DIM).astype(np.float32)
    blob = _vec_to_blob(base)

    # 12 rows share the IDENTICAL embedding (exact cosine tie against any
    # cue), interleaved with distinct rows so original-index order matters.
    rows: list[tuple[str, bytes]] = []
    for i in range(6):
        rows.append((f"distinct-{i}", _vec_to_blob(rng.random(EMBED_DIM).astype(np.float32))))
    for i in range(12):
        rows.append((f"tied-{i}", blob))
    for i in range(6, 12):
        rows.append((f"distinct-{i}", _vec_to_blob(rng.random(EMBED_DIM).astype(np.float32))))

    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)

    cue = _random_cue(seed=501)
    k = 10  # cuts straight through the 12-row tied group

    expected = _brute_force_top_k(rows, cue, k)

    # Repeat the query many times: an argpartition-based selection could
    # legally vary run to run for the same warm matrix; a full stable
    # argsort cannot.
    for _ in range(25):
        got = idx.top_k(cue, k)
        assert got is not None
        assert got == expected, (
            f"boundary-tie top_k result diverged from the stable-argsort "
            f"reference: expected={expected} got={got}"
        )

    # Also exercise the all-zero-norm tie case: many zero vectors, cue cuts
    # through the tied-at-0.0 group.
    zero_rows: list[tuple[str, bytes]] = [
        (f"zero-{i}", _vec_to_blob(np.zeros(EMBED_DIM, dtype=np.float32)))
        for i in range(12)
    ]
    zero_rows += [
        (f"pos-{i}", _vec_to_blob(rng.random(EMBED_DIM).astype(np.float32)))
        for i in range(3)
    ]
    zidx = ExactCosineIndex(embed_dim=EMBED_DIM)
    zidx.build(zero_rows)
    zcue = _random_cue(seed=502)
    zexpected = _brute_force_top_k(zero_rows, zcue, 8)
    for _ in range(25):
        zgot = zidx.top_k(zcue, 8)
        assert zgot == zexpected


def test_top_k_descending_order():
    """Result scores are strictly non-increasing (descending cos order)."""
    rows = _random_rows(200, seed=2)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)

    cue = _random_cue(seed=99)
    got = idx.top_k(cue, 50)
    assert got is not None
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True), (
        f"top_k result not in descending order: {scores}"
    )


def test_top_k_k_greater_than_n_returns_all_rows():
    """top_k(k=10) on a 3-row index returns 3 pairs, descending."""
    rows = _random_rows(3, seed=3)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)

    cue = _random_cue(seed=7)
    got = idx.top_k(cue, 10)
    assert got is not None
    assert len(got) == 3
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Cold contract
# ---------------------------------------------------------------------------


def test_cold_index_is_warm_false():
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    assert idx.is_warm is False


def test_cold_index_top_k_returns_none():
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    cue = _random_cue(seed=1)
    assert idx.top_k(cue, 10) is None


def test_cold_index_upsert_is_noop():
    """upsert on a cold index is a no-op; len stays 0."""
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    assert len(idx) == 0

    vec = _random_cue(seed=5)
    idx.upsert("some-id", vec)

    assert len(idx) == 0
    assert idx.is_warm is False
    assert idx.top_k(vec, 1) is None


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------


def test_invalidate_marks_cold_and_subsequent_build_restores():
    rows = _random_rows(50, seed=4)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)
    assert idx.is_warm

    idx.invalidate()
    assert idx.is_warm is False
    cue = _random_cue(seed=11)
    assert idx.top_k(cue, 5) is None

    # A subsequent build restores exact answers.
    idx.build(rows)
    assert idx.is_warm
    expected = _brute_force_top_k(rows, cue, 5)
    got = idx.top_k(cue, 5)
    assert got is not None
    assert [i for i, _ in got] == [i for i, _ in expected]


# ---------------------------------------------------------------------------
# upsert-new == rebuild
# ---------------------------------------------------------------------------


def test_upsert_new_rows_matches_fresh_rebuild():
    """build N rows, upsert M new rows, assert top_k over any cue equals a
    fresh build of the N+M row set (byte-consistency with the rebuild path)."""
    base_rows = _random_rows(100, seed=20)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(base_rows)

    new_rows = []
    rng = np.random.default_rng(21)
    for i in range(30):
        v = rng.random(EMBED_DIM).astype(np.float32)
        rid = f"new-id-{i}"
        new_rows.append((rid, _vec_to_blob(v)))
        idx.upsert(rid, v)

    combined_rows = base_rows + new_rows

    fresh = ExactCosineIndex(embed_dim=EMBED_DIM)
    fresh.build(combined_rows)

    for cue_seed in range(10):
        cue = _random_cue(seed=30_000 + cue_seed)
        expected = fresh.top_k(cue, 20)
        got = idx.top_k(cue, 20)
        assert got == expected, (
            f"cue_seed={cue_seed}: upsert-new result diverges from fresh rebuild\n"
            f"  expected={expected}\n  got={got}"
        )


# ---------------------------------------------------------------------------
# upsert-replace
# ---------------------------------------------------------------------------


def test_upsert_replace_existing_id_overwrites_vector():
    """upsert an EXISTING id with a different vector; the old vector is gone
    (a cue equal to the old vector no longer scores ~1.0 for that id; a cue
    equal to the new vector does); len unchanged."""
    rows = _random_rows(10, seed=40)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)
    before_len = len(idx)

    target_id, old_blob = rows[3]
    old_vec = np.frombuffer(old_blob, dtype=np.float32)

    new_vec = _random_cue(seed=999)
    idx.upsert(target_id, new_vec)

    assert len(idx) == before_len

    # Cue equal to the OLD vector must no longer rank target_id at ~1.0.
    old_top = idx.top_k(old_vec, 1)
    assert old_top is not None
    top_id, top_score = old_top[0]
    if top_id == target_id:
        assert top_score < 0.999, (
            "old vector still scores ~1.0 against the replaced id "
            f"(score={top_score}) — upsert did not overwrite in place"
        )

    # Cue equal to the NEW vector must rank target_id at ~1.0.
    new_top = idx.top_k(new_vec, 1)
    assert new_top is not None
    new_top_id, new_top_score = new_top[0]
    assert new_top_id == target_id
    assert new_top_score > 0.999, (
        f"new vector does not score ~1.0 against the replaced id "
        f"(score={new_top_score})"
    )


# ---------------------------------------------------------------------------
# Capacity growth
# ---------------------------------------------------------------------------


def test_capacity_growth_all_rows_answered_exactly():
    """build 4 rows then upsert 100 more one at a time; all 104 answered
    exactly (exercises amortized doubling, no strict perf gate)."""
    base_rows = _random_rows(4, seed=50)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(base_rows)

    rng = np.random.default_rng(51)
    grown_rows = list(base_rows)
    for i in range(100):
        v = rng.random(EMBED_DIM).astype(np.float32)
        rid = f"grown-{i}"
        grown_rows.append((rid, _vec_to_blob(v)))
        idx.upsert(rid, v)

    assert len(idx) == 104

    fresh = ExactCosineIndex(embed_dim=EMBED_DIM)
    fresh.build(grown_rows)

    cue = _random_cue(seed=52)
    expected = fresh.top_k(cue, 104)
    got = idx.top_k(cue, 104)
    assert got == expected


# ---------------------------------------------------------------------------
# Zero-norm guard
# ---------------------------------------------------------------------------


def test_zero_norm_row_accepted_and_never_raises_or_outranks():
    """A zero-vector row is accepted, normalized form stays zeros, never
    appears in top-k above any positive-cos row, and never raises."""
    rows = _random_rows(20, seed=60)
    zero_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    rows.append(("zero-id", _vec_to_blob(zero_vec)))

    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(rows)  # must not raise

    cue = _random_cue(seed=61)
    got = idx.top_k(cue, len(rows))
    assert got is not None

    # zero-id's score must be 0.0 (never positive, never NaN).
    zero_score = dict(got)["zero-id"]
    assert zero_score == 0.0

    # zero-id must not rank above any row with a positive score.
    scores_by_id = dict(got)
    positive_scores = [s for rid, s in scores_by_id.items() if rid != "zero-id" and s > 0]
    if positive_scores:
        zero_rank = [rid for rid, _ in got].index("zero-id")
        best_positive_rank = min(
            [rid for rid, _ in got].index(rid)
            for rid in scores_by_id
            if rid != "zero-id" and scores_by_id[rid] > 0
        )
        assert zero_rank > best_positive_rank, (
            "zero-norm row ranked above a positive-cos row"
        )

    # upsert a zero vector too — must not raise.
    idx.upsert("zero-id-2", zero_vec)
    got2 = idx.top_k(cue, len(idx))
    assert got2 is not None
    assert dict(got2)["zero-id-2"] == 0.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_writer_upsert_reader_top_k():
    """One writer thread upserting 200 rows while a reader thread calls top_k
    in a loop — no exception, every returned list is internally consistent
    (descending scores, no duplicate ids)."""
    base_rows = _random_rows(50, seed=70)
    idx = ExactCosineIndex(embed_dim=EMBED_DIM)
    idx.build(base_rows)

    errors: list[BaseException] = []
    stop = threading.Event()

    def _writer():
        try:
            rng = np.random.default_rng(71)
            for i in range(200):
                v = rng.random(EMBED_DIM).astype(np.float32)
                idx.upsert(f"writer-{i}", v)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def _reader():
        try:
            cue = _random_cue(seed=72)
            while not stop.is_set():
                got = idx.top_k(cue, 20)
                if got is None:
                    continue
                scores = [s for _, s in got]
                assert scores == sorted(scores, reverse=True), (
                    f"reader observed non-descending scores: {scores}"
                )
                ids = [i for i, _ in got]
                assert len(ids) == len(set(ids)), (
                    f"reader observed duplicate ids: {ids}"
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer_thread = threading.Thread(target=_writer)
    reader_thread = threading.Thread(target=_reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join(timeout=30)
    stop.set()
    reader_thread.join(timeout=30)

    assert not errors, f"thread-safety test raised: {errors}"
    assert len(idx) == 250


# ---------------------------------------------------------------------------
# Purity: no hippo/sqlite/crypto/hnswlib import
# ---------------------------------------------------------------------------


def test_module_imports_no_hippo_or_store_machinery():
    """Statically assert the module's import list contains no hippo/sqlite/
    crypto import — the class is pure host-side numpy + stdlib threading."""
    import ast
    import iai_mcp.store._exact_index as mod

    src = ast.parse(open(mod.__file__).read())
    imported_names: set[str] = set()
    for node in ast.walk(src):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.split(".")[0])

    forbidden = {"hippo", "sqlite3", "hnswlib"}
    for name in imported_names:
        assert "hippo" not in name.lower(), f"module imports hippo-adjacent name: {name}"
        assert name not in forbidden, f"module imports forbidden dependency: {name}"

    # No AES/crypto machinery either.
    for name in imported_names:
        assert "crypto" not in name.lower(), f"module imports crypto machinery: {name}"

    # Allowed: numpy + stdlib only.
    allowed_stdlib = {
        "__future__",
        "threading",
        "typing",
        "struct",
        "dataclasses",
        "math",
    }
    for name in imported_names:
        assert name == "numpy" or name in allowed_stdlib, (
            f"module imports unexpected dependency: {name} "
            "(only numpy + stdlib are permitted)"
        )


def test_module_import_does_not_pull_in_hippo_subprocess():
    """Loading the module FILE directly (bypassing the ``iai_mcp.store``
    parent package's own ``__init__.py``, which legitimately imports hippo
    for unrelated store machinery) never touches iai_mcp.hippo or sqlite3.

    This isolates the module's OWN import list from its parent package's —
    a submodule import always runs the parent package's ``__init__.py``
    first as an unavoidable consequence of Python import mechanics, so the
    purity claim under test is about ``_exact_index.py`` itself, loaded via
    ``importlib.util`` under a throwaway module name with no package context.
    """
    import subprocess

    code = (
        "import sys\n"
        "import importlib.util\n"
        "import iai_mcp.store as _store_pkg\n"
        "before = set(sys.modules)\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    '_exact_index_standalone',\n"
        "    _store_pkg.__path__[0] + '/_exact_index.py',\n"
        ")\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "after = set(sys.modules) - before\n"
        "bad = [m for m in after if 'hippo' in m.lower()]\n"
        "assert not bad, f'hippo module(s) newly loaded by the module body: {bad}'\n"
        "assert 'sqlite3' not in after, 'sqlite3 newly loaded by the module body'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess import check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
