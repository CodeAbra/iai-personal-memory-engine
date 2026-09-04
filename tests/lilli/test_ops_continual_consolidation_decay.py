from __future__ import annotations

import pytest

from iai_mcp.lilli.ops.continual import add_pair, empty_hv, update_role
from iai_mcp.lilli.ops.consolidation import consolidate
from iai_mcp.lilli.tiers import bsc, sparse_vsa

def test_continual_empty_hv_default_D():
    hv = empty_hv()
    assert hv == bytes(512)
    assert len(hv) == 512

def test_continual_add_pair_recovers_filler():
    hv = add_pair(empty_hv(), "WHEN", "today")
    recovered = bsc.unbind(hv, bsc.role_hv("WHEN"))
    expected = bsc.filler_hv("today")
    assert recovered == expected

def test_continual_add_pair_at_D_10000():
    hv = add_pair(empty_hv(D=10000), "WHERE", "mars", D=10000)
    assert len(hv) == 10000 // 8
    recovered = bsc.unbind(hv, bsc.role_hv("WHERE", D=10000))
    expected = bsc.filler_hv("mars", D=10000)
    assert recovered == expected

def test_continual_update_role_replaces_filler():
    hv0 = empty_hv()
    hv1 = add_pair(hv0, "ROLE", "user")
    hv2 = update_role(hv1, "user", "ROLE", "admin")
    recovered = bsc.unbind(hv2, bsc.role_hv("ROLE"))
    assert recovered == bsc.filler_hv("admin")

def test_continual_add_two_pairs_xor_merge():
    hv0 = empty_hv()
    hv1 = add_pair(hv0, "WHEN", "today")
    hv2 = add_pair(hv1, "WHERE", "home")

    recovered_when = bsc.unbind(hv2, bsc.role_hv("WHEN"))
    assert recovered_when != bsc.filler_hv("today"), (
        "With two XOR-overlaid pairs, unbind(WHEN) must be contaminated by the WHERE pair "
        "-- clean recovery only works when exactly one pair is in the hv"
    )

    bundle_result = bsc.bundle([("WHEN", bsc.filler_hv("today")), ("WHERE", bsc.filler_hv("home"))])
    assert hv2 != bundle_result, (
        "XOR-merge of two pairs must differ from majority-vote bundle "
        "-- this confirms add_pair is an approximation, not a drop-in for bundle"
    )

def test_continual_add_pair_deterministic():
    hv_base = empty_hv()
    out1 = add_pair(hv_base, "TOPIC", "cognition")
    out2 = add_pair(hv_base, "TOPIC", "cognition")
    assert out1 == out2

def test_consolidate_bsc_empty():
    result = consolidate([], "bsc")
    assert result == bytes(512)
    assert len(result) == 512

def test_consolidate_bsc_two_identical():
    hv = bsc.bundle([("WHEN", bsc.filler_hv("yesterday"))])
    result = consolidate([hv, hv], "bsc")
    assert result == hv

def test_consolidate_bsc_at_D_10000():
    from iai_mcp.lilli.core.seed import hv_from_seed
    hv1 = hv_from_seed(1, 10000)
    hv2 = hv_from_seed(2, 10000)
    result = consolidate([hv1, hv2], "bsc")
    assert len(result) == 1250

def test_consolidate_fhrr_empty_returns_10000_zeros():
    result = consolidate([], "fhrr")
    assert result == bytes(10000)
    assert len(result) == 10000

def test_consolidate_sparse_vsa_returns_list():
    a = sparse_vsa.role_hv("WHEN")
    b = sparse_vsa.role_hv("WHERE")
    result = consolidate([a, b], "sparse_vsa")
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

def test_consolidate_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        consolidate([bytes(512)], "garbage")
