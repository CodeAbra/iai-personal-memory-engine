from __future__ import annotations

import inspect
import os

def test_brain_cognitive_mode_is_autistic():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    assert b.cognitive_mode == "autistic"

def test_brain_cognitive_mode_no_init_kwarg():
    from iai_mcp.lilli.brain import Brain

    sig = inspect.signature(Brain.__init__)
    assert "cognitive_mode" not in sig.parameters, (
        "cognitive_mode kwarg must not exist -- anti-sycophancy is an architecture "
        "invariant, not a runtime knob"
    )

def test_brain_has_three_tier_backends():
    from iai_mcp.lilli.brain import Brain
    from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa

    b = Brain()
    assert b.bsc is bsc
    assert b.fhrr is fhrr
    assert b.sparse_vsa is sparse_vsa

def test_brain_ops_bundle_has_seven_modules():
    from iai_mcp.lilli.brain import Brain
    from iai_mcp.lilli.ops import (
        cleanup,
        consolidation,
        continual,
        delta,
        orthogonalize,
        replay,
        separation,
    )

    b = Brain()
    assert b.ops.continual is continual
    assert b.ops.consolidation is consolidation
    assert b.ops.replay is replay
    assert b.ops.orthogonalize is orthogonalize
    assert b.ops.cleanup is cleanup
    assert b.ops.delta is delta
    assert b.ops.separation is separation

def test_brain_crossmodal_bundle():
    from iai_mcp.lilli.brain import Brain
    from iai_mcp.lilli.crossmodal import embed_to_hv

    b = Brain()
    assert b.crossmodal.embed_to_hv is embed_to_hv
    assert callable(b.crossmodal.hv_to_neighbors)

def test_brain_profile_placeholder():
    from types import SimpleNamespace

    from iai_mcp.lilli.brain import Brain

    b = Brain()
    assert isinstance(b.profile, SimpleNamespace)
    assert len(vars(b.profile)) == 0

def test_brain_emit_telemetry_no_op_when_none_conn():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    b.emit_rank_deficiency_warning({"x": 1})

def test_brain_emit_methods_exist():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    assert hasattr(b, "emit_telemetry") and callable(b.emit_telemetry)
    assert hasattr(b, "emit_rank_deficiency_warning") and callable(
        b.emit_rank_deficiency_warning
    )
    assert hasattr(b, "emit_role_saturation_warning") and callable(
        b.emit_role_saturation_warning
    )
    assert hasattr(b, "emit_codec_marker_missing") and callable(
        b.emit_codec_marker_missing
    )

def test_brain_with_hippo_conn_stored():
    from iai_mcp.lilli.brain import Brain

    sentinel = object()
    b = Brain(hippo_conn=sentinel)
    assert b.hippo_conn is sentinel

def test_brain_repr_includes_autistic():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    assert b.cognitive_mode == "autistic"
    _ = repr(b)

def test_brain_recall_signature():
    from iai_mcp.lilli.brain import Brain

    assert callable(Brain.recall)
    sig = inspect.signature(Brain.recall)
    assert "cue" in sig.parameters
    assert "limit" in sig.parameters
    assert "session_id" in sig.parameters
    limit_param = sig.parameters["limit"]
    session_param = sig.parameters["session_id"]
    assert limit_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert session_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert limit_param.default == 5
    assert session_param.default == "brain-recall"

def test_brain_recall_raises_on_none_hippo_conn():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    try:
        b.recall("test")
        raise AssertionError("recall should have raised RuntimeError")
    except RuntimeError as e:
        assert "hippo_conn" in str(e), f"Expected 'hippo_conn' in error message, got: {e}"

def test_tier_info_bsc():
    from iai_mcp.lilli.tier_info import tier_info

    info = tier_info("bsc")
    assert info["backend"] == "bsc"
    assert info["D"] == 4096
    assert info["bytes_per_hv"] == 512
    assert info["use_case"] == "episodic"

def test_tier_info_fhrr():
    from iai_mcp.lilli.tier_info import tier_info

    info = tier_info("fhrr")
    assert info == {"backend": "fhrr", "D": 10000, "bytes_per_hv": 10000, "use_case": "semantic"}

def test_tier_info_sparse_vsa():
    from iai_mcp.lilli.tier_info import tier_info

    info = tier_info("sparse_vsa")
    assert info == {
        "backend": "sparse_vsa",
        "D": 2048,
        "bytes_per_hv": 40,
        "use_case": "procedural",
    }

def test_tier_info_unknown_raises():
    from iai_mcp.lilli.tier_info import tier_info

    try:
        tier_info("garbage")
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "garbage" in str(e)

def test_laws_active_false():
    from iai_mcp.lilli.laws import LAWS_ACTIVE

    assert LAWS_ACTIVE is False

def test_laws_readme_documents_L0_L3():
    laws_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "iai_mcp", "lilli", "laws"
    )
    readme_path = os.path.join(laws_dir, "README.md")
    assert os.path.isfile(readme_path), f"README.md not found at {readme_path}"
    content = open(readme_path).read()
    for label in ("L0", "L1", "L2", "L3"):
        assert label in content, f"{label} not found in laws/README.md"

def test_laws_no_runtime_hooks_in_init():
    laws_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "iai_mcp", "lilli", "laws"
    )
    init_path = os.path.join(laws_dir, "__init__.py")
    assert os.path.isfile(init_path)
    content = open(init_path).read()
    assert "def " not in content, (
        "laws/__init__.py must not define any functions -- empty slot only"
    )

def test_lilli_public_api_imports_clean():
    from iai_mcp.lilli import (  # noqa: F401
        Brain,
        from_embedding,
        list_tiers,
        tier_info,
        to_embedding_neighbors,
    )

def test_lilli_all_attribute_complete():
    import iai_mcp.lilli as lilli

    expected = {"Brain", "tier_info", "list_tiers", "from_embedding", "to_embedding_neighbors"}
    assert set(lilli.__all__) == expected

def test_no_forbidden_tokens_in_lilli_brain_laws():
    import re

    pattern = re.compile(
        r"Plan\s+\d+|Phase\s+\d+|D-\d+|LILLIHD-|OPS-\d+|TOK-\d+"
    )
    lilli_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "iai_mcp", "lilli"
    )
    files_to_check = [
        os.path.join(lilli_dir, "brain.py"),
        os.path.join(lilli_dir, "tier_info.py"),
        os.path.join(lilli_dir, "laws", "__init__.py"),
    ]
    violations = []
    for fpath in files_to_check:
        if not os.path.isfile(fpath):
            violations.append(f"MISSING: {fpath}")
            continue
        for lineno, line in enumerate(open(fpath), start=1):
            if pattern.search(line):
                violations.append(f"{fpath}:{lineno}: {line.rstrip()}")
    assert not violations, "Forbidden tokens found:\n" + "\n".join(violations)

def test_telemetry_constants_defined():
    from iai_mcp.events import (
        TELEMETRY_CODEC_MARKER_MISSING,
        TELEMETRY_RANK_DEFICIENCY,
        TELEMETRY_ROLE_SATURATION,
    )

    assert TELEMETRY_RANK_DEFICIENCY == "rank_deficiency_warning"
    assert TELEMETRY_ROLE_SATURATION == "role_saturation_warning"
    assert TELEMETRY_CODEC_MARKER_MISSING == "codec_marker_missing"

def test_brain_emit_rank_deficiency_no_op():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    b.emit_rank_deficiency_warning({"reason": "test", "batch_size": 4})

def test_brain_emit_codec_marker_missing_no_op():
    from iai_mcp.lilli.brain import Brain

    b = Brain()
    b.emit_codec_marker_missing({"record_id": "abc123"})
