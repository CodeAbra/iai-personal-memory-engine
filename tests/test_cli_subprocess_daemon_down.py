from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recall_helpers import (  # noqa: E402
    UUID_TWO_HOP_SURFACE,
    _populate_store,
    _prime_structural_cache,
)

_TEST_CRYPTO_PASSPHRASE = "iai-mcp-test-passphrase"


def _child_env(store_root: Path, tmp_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_home / "no-such-daemon.sock")
    return env


def _seed_store_with_drained_turn(store_root: Path, text: str) -> None:
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(store_root)
    try:
        rng = np.random.RandomState(seed=88)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "c3h1-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()


def test_subprocess_iai_last_daemon_down_returns_drained_store_turn(
    hermetic_store: Path, tmp_path: Path
) -> None:
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    drained_text = "c3h1 last drained distinctive store turn text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "last", "--n", "10"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai last` failed (rc={result.returncode}):\n{result.stderr}"
    )
    assert drained_text in result.stdout, (
        f"C3-H1: drained store turn not in `iai last` stdout;\n"
        f"stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}\n"
        "The live-layer fallback cannot produce this turn — must be store-backed."
    )


def test_subprocess_iai_capture_daemon_down_writes_to_store(
    hermetic_store: Path, tmp_path: Path
) -> None:
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    capture_text = "c3h1 capture distinctive write probe text"
    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "capture", capture_text],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai capture` failed (rc={result.returncode}):\n{result.stderr}\n"
        "C3-H1: capture must succeed (exit 0) even with daemon down; "
        "the direct-write fallback is not yet wired."
    )

    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        records = store.all_records()
        surfaces = [r.literal_surface or "" for r in records]
        assert any(capture_text in s for s in surfaces), (
            f"C3-H1: captured text not found in tmp Hippo store after subprocess capture;\n"
            f"surfaces={surfaces!r}"
        )
    finally:
        store.close()


def test_subprocess_iai_recall_daemon_down_returns_store_backed_degraded(
    hermetic_store: Path, tmp_path: Path
) -> None:
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    drained_text = "c3h1 recall store backed degraded distinctive probe text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "recall", "c3h1 recall store backed"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai recall` failed (rc={result.returncode}):\n{result.stderr}"
    )
    assert drained_text in result.stdout, (
        f"C3-H1: drained store turn not in `iai recall` stdout;\n"
        f"stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}\n"
        "The bank-recall subprocess cannot produce this turn — must be store-backed."
    )


def _hf_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _live_gate_child_env(store_root: Path, tmp_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_home / "no-such-daemon.sock")
    env["IAI_MCP_EMBED_OFFLINE"] = "1"
    env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    hf_root = _hf_cache_root()
    env["HF_HOME"] = str(hf_root)
    env["HF_HUB_CACHE"] = str(hf_root / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")
    return env


def test_subprocess_iai_recall_daemon_down_returns_daemon_down_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_cache = _hf_cache_root()
    weights_dir = hf_cache / "hub" / "models--BAAI--bge-small-en-v1.5"
    if not weights_dir.exists():
        pytest.skip(
            f"bge-small weight cache absent ({weights_dir}); the offline LIVE-gate "
            "construct cannot run. The authoritative real-hibernated-daemon proof "
            "is the human-live checkpoint (orchestrator), which this gate approximates."
        )

    store_root = tmp_path / "store"
    tmp_home = tmp_path / "home"
    tmp_home.mkdir(parents=True, exist_ok=True)


    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_CRYPTO_PASSPHRASE)
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
    monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")
    from iai_mcp.embed import Embedder
    from iai_mcp.store import MemoryStore

    cue = "User reference gold document semantic recall probe cue"
    cue_vec = Embedder().embed(cue)

    from uuid import UUID

    from iai_mcp.pipeline import K_CANDIDATES

    store = MemoryStore(str(store_root))
    try:
        _populate_store(store, cue_vec=cue_vec, n_filler=700)
        _prime_structural_cache(store)

        ann_top_k = {r.id for r, _ in store.query_similar(cue_vec, k=K_CANDIDATES)}
        assert UUID(int=5) not in ann_top_k, (
            f"PRECONDITION FAILED: the structural-only gold UUID(5) is a DIRECT ANN "
            f"top-{K_CANDIDATES} hit — the 2-hop spread would not be load-bearing and "
            f"the gate would be hollow. store size={store.active_records_count()}."
        )
    finally:
        store.close()

    env = _live_gate_child_env(store_root, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "recall", "--json", "--limit", "50", cue],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"subprocess `iai recall --json` failed (rc={result.returncode}):\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )

    stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert stdout_lines, f"no JSON on stdout; stderr={result.stderr!r}"
    payload = json.loads(stdout_lines[-1])

    source = payload.get("_source")
    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}

    assert UUID_TWO_HOP_SURFACE in surfaces, (
        "VERIFICATION-INTEGRITY FAILURE: the STRUCTURAL-ONLY 2-hop gold "
        f"({UUID_TWO_HOP_SURFACE!r}) is MISSING from the real-subprocess "
        "daemon-down recall. It is reachable ONLY via the 2-hop / rich-club "
        "spread (cosine ~0.02, outside ANN top-K), so its absence means the "
        "construct did NOT feed the full structural pipeline — the "
        f"daemon-down-full label would be hollow.\n_source={source!r}\n"
        f"gold surfaces present={sorted(s for s in surfaces if 'gold doc' in s)}\n"
        f"stderr={result.stderr!r}"
    )

    assert source == "daemon-down-full", (
        f"expected EXACT _source == 'daemon-down-full', got {source!r}.\n"
        f"stderr={result.stderr!r}"
    )

    assert source != "daemon", "a daemon answered — impossible with a dead socket"
    assert source != "direct-store", "ANN-only fall-through (structural pipeline skipped)"
    assert source != "daemon-down-degrade", "recency degrade (no structural pipeline ran)"

    assert int(payload.get("count", 0)) > 0, f"empty recall; payload={payload!r}"
    top_score = hits[0].get("score") if hits else None
    assert top_score is not None and float(top_score) != 0.0, (
        f"top hit must have a non-zero score; got {top_score!r}"
    )
