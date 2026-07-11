from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.lilli.tiers.bsc import (
    BSC_ROLE_VOCABULARY,
    LILLI_BSC_DEFAULT_DIM,
    TIER_INFO,
    bind,
    bundle,
    filler_hv,
    permute,
    role_hv,
    similarity,
    unbind,
)

GOLDEN_PATH = Path(__file__).parent / "golden_tem_pre_shim.json"

def test_role_vocabulary_18_items() -> None:
    assert len(BSC_ROLE_VOCABULARY) == 18
    assert BSC_ROLE_VOCABULARY[0] == "WHEN"
    assert BSC_ROLE_VOCABULARY[-1] == "PARENT_ID"

def test_role_hv_default_dim_4096() -> None:
    hv = role_hv("WHEN")
    assert len(hv) == LILLI_BSC_DEFAULT_DIM // 8 == 512

def test_role_hv_dim_10000() -> None:
    hv = role_hv("WHEN", D=10000)
    assert len(hv) == 1250

def test_role_hv_dim_2048() -> None:
    hv = role_hv("WHEN", D=2048)
    assert len(hv) == 256

def test_role_hv_deterministic() -> None:
    assert role_hv("WHEN") == role_hv("WHEN")

def test_filler_hv_default_dim_4096() -> None:
    hv = filler_hv("today")
    assert len(hv) == 512

def test_bind_xor_self_inverse() -> None:
    a = filler_hv("alpha")
    b = filler_hv("beta")
    assert bind(bind(a, b), b) == a

def test_bind_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bind(b"\x00" * 5, b"\x00" * 6)

def test_unbind_equals_bind() -> None:
    x = role_hv("WHEN")
    y = filler_hv("today")
    assert unbind(x, y) == bind(x, y)

def test_bundle_empty_returns_zero_bytes() -> None:
    result = bundle([])
    assert result == bytes(LILLI_BSC_DEFAULT_DIM // 8)
    assert len(result) == 512

def test_bundle_majority_vote_two_pairs() -> None:
    f1 = filler_hv("x")
    f2 = filler_hv("y")
    result = bundle([("WHEN", f1), ("WHERE", f2)])
    assert isinstance(result, bytes)
    assert len(result) == 512

def test_bundle_single_pair_round_trip() -> None:
    fv = filler_hv("today")
    bundled = bundle([("WHEN", fv)])
    recovered = unbind(bundled, role_hv("WHEN"))
    assert recovered == fv

def test_permute_round_trip() -> None:
    hv = role_hv("WHEN")
    assert permute(permute(hv, 7), -7) == hv

def test_similarity_identical() -> None:
    hv = role_hv("WHEN")
    assert similarity(hv, hv) == 1.0

def test_similarity_random_pair_near_half() -> None:
    rng = np.random.default_rng(1234)
    a = rng.integers(0, 256, size=512, dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=512, dtype=np.uint8).tobytes()
    s = similarity(a, b)
    assert 0.4 < s < 0.6, f"expected ~0.5 for random BSC pair, got {s}"

def _build_golden():
    import iai_mcp.tem as tem
    from iai_mcp.lilli.tiers.bsc import BSC_ROLE_VOCABULARY

    return {
        "schema_version": 1,
        "tem_role_vocabulary": list(BSC_ROLE_VOCABULARY),
        "roles": {role: tem.role_hv(role).hex() for role in BSC_ROLE_VOCABULARY},
        "fillers": {
            v: tem.filler_hv(v).hex()
            for v in [
                "alice",
                "today",
                "iai-mcp",
                "user",
                "neutral",
                "trust_0.5",
                "pinned",
                "episodic",
                "text",
                "root",
            ]
        },
        "bind_pairs": {
            f"{role}|{filler}": bind(tem.role_hv(role), tem.filler_hv(filler)).hex()
            for role, filler in [
                ("WHEN", "today"),
                ("WHERE", "home"),
                ("ROLE", "user"),
                ("PROJECT", "iai-mcp"),
                ("TIER", "episodic"),
            ]
        },
        "bundle_outputs": [
            {
                "pairs": [["WHEN", "today"], ["WHERE", "home"]],
                "output_hex": tem.pack_pairs(
                    [
                        ("WHEN", tem.filler_hv("today")),
                        ("WHERE", tem.filler_hv("home")),
                    ]
                ).hex(),
            },
        ],
    }

def test_bit_for_bit_with_tem() -> None:
    import iai_mcp.tem as tem
    from iai_mcp.lilli.tiers.bsc import BSC_ROLE_VOCABULARY

    for role in BSC_ROLE_VOCABULARY:
        lilli_hv = role_hv(role, D=10000)
        tem_hv = tem.role_hv(role)
        assert lilli_hv == tem_hv, (
            f"lilli.tiers.bsc.role_hv({role!r}, D=10000) != tem.role_hv({role!r})"
        )

    sample_fillers = [
        "alice", "today", "iai-mcp", "user", "neutral",
        "trust_0.5", "pinned", "episodic", "text", "root",
    ]
    for v in sample_fillers:
        assert filler_hv(v, D=10000) == tem.filler_hv(v), (
            f"filler_hv({v!r}, D=10000) mismatch"
        )

    role_a_10k = role_hv("WHEN", D=10000)
    filler_b_10k = filler_hv("today", D=10000)
    assert bind(role_a_10k, filler_b_10k) == bind(tem.role_hv("WHEN"), tem.filler_hv("today"))

    lilli_bundle = bundle(
        [("WHEN", tem.filler_hv("today")), ("WHERE", tem.filler_hv("home"))],
        D=10000,
    )
    tem_bundle = tem.pack_pairs(
        [("WHEN", tem.filler_hv("today")), ("WHERE", tem.filler_hv("home"))]
    )
    assert lilli_bundle == tem_bundle, "bundle fidelity mismatch at D=10000"

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(_build_golden(), indent=2, sort_keys=True))

    golden = json.loads(GOLDEN_PATH.read_text())

    for role_key, expected_hex in golden["roles"].items():
        actual = role_hv(role_key, D=10000).hex()
        assert actual == expected_hex, (
            f"lilli.tiers.bsc.role_hv({role_key!r}, D=10000) byte-drift from frozen golden"
        )

    for filler_val, expected_hex in golden["fillers"].items():
        actual = filler_hv(filler_val, D=10000).hex()
        assert actual == expected_hex, (
            f"filler_hv({filler_val!r}, D=10000) byte-drift from frozen golden"
        )

    for key, expected_hex in golden["bind_pairs"].items():
        r, f = key.split("|")
        actual = bind(role_hv(r, D=10000), filler_hv(f, D=10000)).hex()
        assert actual == expected_hex, f"bind_pair {key!r} byte-drift from frozen golden"

    for entry in golden["bundle_outputs"]:
        pairs_hv = [(r, filler_hv(f, D=10000)) for r, f in entry["pairs"]]
        actual = bundle(pairs_hv, D=10000).hex()
        assert actual == entry["output_hex"], (
            f"bundle_output byte-drift from frozen golden (pairs={entry['pairs']})"
        )

def test_tier_info_metadata() -> None:
    assert TIER_INFO == {
        "backend": "bsc",
        "D": 4096,
        "bytes_per_hv": 512,
        "use_case": "episodic",
        "max_bundle_pairs": 10,
    }
