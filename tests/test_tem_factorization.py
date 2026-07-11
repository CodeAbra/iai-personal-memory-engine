from __future__ import annotations


def test_role_vocabulary_has_18_entries() -> None:
    from iai_mcp.lilli.tiers.bsc import BSC_ROLE_VOCABULARY as ROLE_VOCABULARY

    assert isinstance(ROLE_VOCABULARY, tuple)
    assert len(ROLE_VOCABULARY) == 18
    for required in ("WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID", "TEMPORAL_POSITION"):
        assert required in ROLE_VOCABULARY, f"missing required role {required!r}"

def test_role_hv_is_deterministic_and_correct_length() -> None:
    from iai_mcp.tem import role_hv
    from iai_mcp.types import STRUCTURE_HV_BYTES

    a = role_hv("WHEN")
    b = role_hv("WHEN")
    assert isinstance(a, bytes)
    assert len(a) == STRUCTURE_HV_BYTES
    assert a == b

    c = role_hv("WHERE")
    assert c != a

def test_filler_hv_is_deterministic_and_correct_length() -> None:
    from iai_mcp.tem import filler_hv
    from iai_mcp.types import STRUCTURE_HV_BYTES

    a = filler_hv("2026-04-17")
    b = filler_hv("2026-04-17")
    assert isinstance(a, bytes)
    assert len(a) == STRUCTURE_HV_BYTES
    assert a == b

def test_bind_is_xor_reversible() -> None:
    from iai_mcp.lilli.tiers.bsc import bind
    from iai_mcp.tem import role_hv

    a = role_hv("WHEN")
    b = role_hv("PROJECT")
    bound = bind(a, b)
    assert isinstance(bound, bytes)
    assert len(bound) == len(a)
    assert bind(bound, b) == a
    assert bind(bound, a) == b

def test_unbind_inverts_bind() -> None:
    from iai_mcp.lilli.tiers.bsc import bind, unbind
    from iai_mcp.tem import role_hv

    a = role_hv("ROLE")
    b = role_hv("LANG")
    bound = bind(a, b)
    recovered = unbind(bound, a)
    assert recovered == b

def _fidelity_at(n_pairs: int) -> float:
    from iai_mcp.lilli.tiers.bsc import BSC_ROLE_VOCABULARY as ROLE_VOCABULARY
    from iai_mcp.lilli.tiers.bsc import unbind
    from iai_mcp.tem import (
        filler_hv,
        pack_pairs,
        role_hv,
    )

    fillers = [filler_hv(f"filler-seed42-{i}") for i in range(len(ROLE_VOCABULARY))]
    roles = list(ROLE_VOCABULARY[:n_pairs])
    pairs = [(roles[i], fillers[i]) for i in range(n_pairs)]
    packed = pack_pairs(pairs)
    assert isinstance(packed, bytes)

    def hamming(x: bytes, y: bytes) -> int:
        return sum(bin(a ^ b).count("1") for a, b in zip(x, y))

    correct = 0
    for i, role in enumerate(roles):
        unbound = unbind(packed, role_hv(role))
        best = min(range(len(fillers)), key=lambda j: hamming(unbound, fillers[j]))
        if best == i:
            correct += 1
    return correct / n_pairs

def test_unbind_fidelity_15_pairs() -> None:
    fidelity = _fidelity_at(15)
    assert fidelity >= 0.95, f"unbind fidelity at 15 pairs = {fidelity:.3f} < 0.95"

def test_unbind_fidelity_17_pairs() -> None:
    fidelity = _fidelity_at(17)
    assert fidelity >= 0.92, f"unbind fidelity at 17 pairs = {fidelity:.3f} < 0.92"

def test_unbind_fidelity_18_pairs() -> None:
    fidelity = _fidelity_at(18)
    assert fidelity >= 0.90, f"unbind fidelity at 18 pairs = {fidelity:.3f} < 0.90"
