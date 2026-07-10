from __future__ import annotations

from iai_mcp.lilli.tiers.sparse_vsa import (
    LILLI_SPARSE_DIM,
    SPARSE_K,
    TIER_INFO,
    bind,
    bundle,
    pack_indices,
    permute,
    role_hv,
    similarity,
    unpack_indices,
)

def test_role_hv_length_K() -> None:
    result = role_hv("WHEN")
    assert len(result) == SPARSE_K == 20

def test_role_hv_sorted_and_in_range() -> None:
    result = role_hv("WHO")
    assert result == sorted(result), "Indices must be sorted ascending"
    assert all(0 <= i < LILLI_SPARSE_DIM for i in result), "All indices must be in [0, D)"

def test_role_hv_deterministic() -> None:
    assert role_hv("WHEN") == role_hv("WHEN")
    assert role_hv("WHERE") == role_hv("WHERE")

def test_role_hv_no_duplicates() -> None:
    result = role_hv("WHEN")
    assert len(result) == len(set(result)), "role_hv must return deduplicated indices"

def test_pack_unpack_round_trip() -> None:
    indices = [5, 100, 1000]
    assert unpack_indices(pack_indices(indices)) == indices

def test_pack_length_40_bytes() -> None:
    assert len(pack_indices(role_hv("WHEN"))) == 40

def test_pack_full_hv_round_trip() -> None:
    hv = role_hv("ENTITY")
    assert unpack_indices(pack_indices(hv)) == hv

def test_bind_cdt_basic() -> None:
    a = role_hv("a")
    b = role_hv("b")
    result = bind(a, b)
    assert len(result) == SPARSE_K
    assert result == sorted(result), "bind result must be sorted"
    assert all(0 <= i < LILLI_SPARSE_DIM for i in result), "All bound indices must be in range"

def test_bind_commutative() -> None:
    a = role_hv("apple")
    b = role_hv("orange")
    assert bind(a, b) == bind(b, a)

def test_bind_length_always_sparse_k() -> None:
    hv = role_hv("overlap_test")
    result = bind(hv, hv)
    assert len(result) == SPARSE_K

def test_bundle_empty() -> None:
    assert bundle([]) == []

def test_bundle_single_preserves_K() -> None:
    hv = role_hv("SINGLE")
    assert bundle([hv]) == hv

def test_bundle_two_identical_preserves_input() -> None:
    hv = role_hv("CLONE")
    assert bundle([hv, hv]) == hv

def test_bundle_three_distinct_keeps_K() -> None:
    result = bundle([role_hv("a"), role_hv("b"), role_hv("c")])
    assert len(result) == SPARSE_K

def test_permute_round_trip() -> None:
    hv = role_hv("WHEN")
    assert permute(permute(hv, 7), -7) == hv

def test_permute_preserves_length() -> None:
    hv = role_hv("ENTITY")
    for k in (0, 1, -1, 100, LILLI_SPARSE_DIM - 1):
        result = permute(hv, k)
        assert len(result) == SPARSE_K, f"permute({k}) changed length"

def test_similarity_identical() -> None:
    hv = role_hv("WHEN")
    assert similarity(hv, hv) == 1.0

def test_similarity_disjoint() -> None:
    a = list(range(0, 20))
    b = list(range(100, 120))
    assert similarity(a, b) == 0.0

def test_similarity_half_overlap() -> None:
    shared = list(range(10))
    a = shared + list(range(10, 20))
    b = shared + list(range(100, 110))
    expected = 10 / 30
    assert abs(similarity(a, b) - expected) < 1e-9

def test_tier_info_metadata() -> None:
    assert TIER_INFO == {
        "backend": "sparse_vsa",
        "D": 2048,
        "bytes_per_hv": 40,
        "use_case": "procedural",
    }
