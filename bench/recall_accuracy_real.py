"""Real-cue recall accuracy harness.

Measures recall@k on real, owner-labelled cue text against an isolated,
read-only COPY of the operator's own Hippo store. Every cue is dispatched
through ``core.dispatch`` — the same production entry point the daemon
serves recall through — so the D-A exact-authority merge, confidence-gated
escalation, multi-seed widening, and the structural cleanup attractor are
all exercised exactly as they run in production.

Reads the labelled fixture at ``IAI_MCP_EVAL_FIXTURE_PATH`` (or the default
local path below); never mines or writes it. The companion labelling
script (``scripts/label_real_thread_cues.py``) produces that fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from iai_mcp import core
from iai_mcp.hippo import AccessMode, _operator_home
from iai_mcp.store import MemoryStore

_DEFAULT_FIXTURE_PATH = Path("~/.iai-mcp/eval-fixtures/labelled_real_thread_cues.json").expanduser()
_RECALL_K = 10
_BUDGET_TOKENS = 2000
_SESSION_ID_HARNESS = "harness-real-thread"

_BASELINE_STRUCTURAL_WEIGHT = 0.0
_AFTER_STRUCTURAL_WEIGHT = 0.8


def fixture_path() -> Path:
    """Resolve the labelled fixture path: env override, else the local default."""
    env_val = os.environ.get("IAI_MCP_EVAL_FIXTURE_PATH")
    if env_val:
        return Path(env_val).expanduser()
    return _DEFAULT_FIXTURE_PATH


def _copy_real_store(dest: Path) -> Path:
    """Copy the real Hippo store (db + WAL + key + graph cache) into ``dest``.

    Read-only against the source: every step is ``shutil.copy2``, never an
    open-for-write. Mirrors ``tests/test_migrate_to_lilli_dryrun.py:57-75``
    (db + WAL sidecar) extended with the ``.crypto.key`` +
    ``runtime_graph_cache.json`` pair from
    ``bench/isolated_daemon_boot_proof.py:411-441`` — both live at the store
    root (sibling of ``hippo/``), and both must travel with the copy: the
    key so the copy decrypts against the same ciphertext, the graph cache so
    the copy's community/soft-gate mechanism is structure-informed rather
    than cold-degraded.
    """
    real_root = _operator_home() / ".iai-mcp"
    real_db = real_root / "hippo" / "brain.sqlite3"
    if not real_db.exists():
        raise FileNotFoundError(f"real store not found: {real_db}")

    mtime_before = real_db.stat().st_mtime
    size_before = real_db.stat().st_size
    print(
        f"[recall_accuracy_real] source brain.sqlite3 mtime={mtime_before} "
        f"size={size_before}",
        flush=True,
    )

    (dest / "hippo").mkdir(parents=True, exist_ok=True)
    copy_db = dest / "hippo" / "brain.sqlite3"
    shutil.copy2(real_db, copy_db)
    real_wal = real_db.with_name(real_db.name + "-wal")
    if real_wal.exists():
        shutil.copy2(real_wal, copy_db.with_name(copy_db.name + "-wal"))
    for aux in (".crypto.key", "runtime_graph_cache.json"):
        src = real_root / aux
        if src.exists():
            shutil.copy2(src, dest / aux)

    mtime_after = real_db.stat().st_mtime
    size_after = real_db.stat().st_size
    print(
        f"[recall_accuracy_real] source brain.sqlite3 mtime={mtime_after} "
        f"size={size_after} (unchanged: {mtime_after == mtime_before and size_after == size_before})",
        flush=True,
    )
    return dest


def _open_copy_store_shared(copy_root: Path) -> MemoryStore:
    """Open the copy in SHARED mode (builds the HNSW ANN index) and shim reinforce.

    Reinforce-defer shim: dispatch's default reinforce path writes metadata
    even on a copy; a no-op protects byte-reproducibility across repeated
    runs (source: ``bench/recall_accuracy.py:1461``).
    """
    os.environ["IAI_MCP_STORE"] = str(copy_root)
    store = MemoryStore(path=copy_root, access_mode=AccessMode.SHARED)
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]
    return store


@contextmanager
def open_eval_copy_store(*, driver: str | None = None) -> Iterator[MemoryStore]:
    """Open a fresh read-only-sourced copy of the real store for the duration
    of the ``with`` block, closing it on exit. The temp copy directory is
    removed on exit too (``TemporaryDirectory`` cleanup)."""
    if driver == "lilli":
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    elif driver == "stdlib":
        os.environ.pop("LILLI_STORAGE_DRIVER", None)

    with tempfile.TemporaryDirectory(prefix="iai-mcp-recall-real-") as td:
        copy_root = _copy_real_store(Path(td) / "copy")
        store = _open_copy_store_shared(copy_root)
        try:
            yield store
        finally:
            store.close()


@contextmanager
def _toggles(**envs: str) -> Iterator[None]:
    """Set env vars on enter, restore (or delete) the originals on exit."""
    originals: dict[str, str | None] = {}
    try:
        for key, value in envs.items():
            originals[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, original in originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


@contextmanager
def _structural(weight: float) -> Iterator[None]:
    """Scope a ``structural_weight`` override onto the live profile_state global.

    ``core._profile_state`` is the SAME dict object ``dispatch`` reads (and
    ``LIVE_KNOBS`` aliases) — mutate the key in place and restore it on exit,
    mirroring the direct-mutation pattern used across the test suite (e.g.
    ``tests/test_hebbian.py:81``, ``tests/test_recall_verbatim_mode.py:417``).
    """
    original = core._profile_state.get("structural_weight")
    core._profile_state["structural_weight"] = weight
    try:
        yield
    finally:
        if original is None:
            core._profile_state.pop("structural_weight", None)
        else:
            core._profile_state["structural_weight"] = original


def _dispatch_real_cue(store: MemoryStore, cue_text: str, structural_weight: float) -> list[str]:
    with _structural(structural_weight):
        response = core.dispatch(
            store,
            "memory_recall",
            {
                "cue": cue_text,
                "session_id": _SESSION_ID_HARNESS,
                "budget_tokens": _BUDGET_TOKENS,
            },
        )
    return [str(h["record_id"]) for h in response.get("hits", [])]


def dispatch_cue_traced(
    cue: dict,
    *,
    driver: str,
    structural_weight: float | None = None,
    store: MemoryStore | None = None,
) -> tuple[list[str], set[str]]:
    """Dispatch ONE cue with ``IAI_MCP_RECALL_TRACE=1`` enabled.

    If ``store`` is supplied, dispatches against it directly (the caller owns
    the store's lifecycle — used when a test wants byte-identity checks
    around the SAME open copy). Otherwise opens a fresh copy of the real
    store for this one call and closes it before returning.

    Returns ``(returned_ids, marks_set)`` where ``marks_set`` is the set of
    ``_recall_trace_ms`` mark names that fired during the dispatch.
    """
    weight = structural_weight if structural_weight is not None else (
        _AFTER_STRUCTURAL_WEIGHT if cue.get("expects_structural") else 0.0
    )

    def _run(active_store: MemoryStore) -> tuple[list[str], set[str]]:
        with _toggles(IAI_MCP_RECALL_TRACE="1"):
            with _structural(weight):
                response = core.dispatch(
                    active_store,
                    "memory_recall",
                    {
                        "cue": cue["cue_text"],
                        "session_id": _SESSION_ID_HARNESS,
                        "budget_tokens": _BUDGET_TOKENS,
                    },
                )
        returned_ids = [str(h["record_id"]) for h in response.get("hits", [])]
        marks_set = {name for name, _ms in response.get("_recall_trace_ms", [])}
        return returned_ids, marks_set

    if store is not None:
        return _run(store)

    with open_eval_copy_store(driver=driver) as fresh_store:
        return _run(fresh_store)


def _recall_at_k_multi(returned_ids: list[str], relevant_ids: set[str], k: int = _RECALL_K) -> float:
    return 1.0 if set(returned_ids[:k]) & relevant_ids else 0.0


def _false_negative_rate(per_cue: list[dict]) -> float:
    if not per_cue:
        return 0.0
    misses = sum(1 for c in per_cue if not c["hit"])
    return misses / len(per_cue)


def _run_one_pass(
    store: MemoryStore,
    cues: list[dict],
    *,
    structural_weight_for: Callable[[dict], float],
) -> dict:
    per_cue: list[dict] = []
    for cue in cues:
        weight = structural_weight_for(cue)
        returned_ids = _dispatch_real_cue(store, cue["cue_text"], weight)
        relevant_ids = set(cue["relevant_record_ids"])
        hit_score = _recall_at_k_multi(returned_ids, relevant_ids)
        per_cue.append(
            {
                "cue_id": cue["cue_id"],
                "recall_at_k": hit_score,
                "hit": hit_score > 0.0,
                "returned_ids": returned_ids,
            }
        )
    return {
        "false_negative_rate": _false_negative_rate(per_cue),
        "per_cue": per_cue,
    }


def _validate_fixture_dict(raw: dict) -> list[dict]:
    if "cues" not in raw or not isinstance(raw["cues"], list):
        raise ValueError("fixture missing 'cues' array")
    cues = raw["cues"]
    for entry in cues:
        for required_key in ("cue_id", "cue_text", "relevant_record_ids"):
            if required_key not in entry:
                raise ValueError(f"fixture cue missing required key {required_key!r}: {entry}")
    return cues


def run_both_passes(fixture: "dict | Path | None" = None, *, driver: str | None = None) -> dict:
    """Run the baseline (mechanisms off) and after (shipped defaults) passes.

    ``fixture`` accepts either an already-loaded fixture dict (``{"cues": [...]}``)
    or a ``Path``/``None`` to a fixture JSON file (resolved via ``fixture_path()``
    when ``None``). Both passes share the same store copy; reinforce is
    shimmed to no-op before either pass runs so neither mutates the copy.
    """
    if isinstance(fixture, dict):
        resolved_path = fixture.get("_source_path", "in-memory")
        cues = _validate_fixture_dict(fixture)
    else:
        resolved_path = fixture if fixture is not None else fixture_path()
        if not resolved_path.exists():
            raise FileNotFoundError(f"labelled fixture not found: {resolved_path}")
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
        cues = _validate_fixture_dict(raw)
        resolved_path = str(resolved_path)

    resolved_driver = driver if driver is not None else os.environ.get("LILLI_STORAGE_DRIVER", "stdlib")

    with open_eval_copy_store(driver=driver) as store:
        with _toggles(
            IAI_MCP_EXACT_AUTHORITY_OFF="1",
            IAI_MCP_CONF_ESCALATE_OFF="true",
            IAI_MCP_MULTI_SEED_OFF="true",
        ):
            baseline_result = _run_one_pass(
                store, cues, structural_weight_for=lambda _c: _BASELINE_STRUCTURAL_WEIGHT
            )

        after_result = _run_one_pass(
            store,
            cues,
            structural_weight_for=(
                lambda c: _AFTER_STRUCTURAL_WEIGHT if c.get("expects_structural") else 0.0
            ),
        )

    return {
        "driver": resolved_driver,
        "fixture_path": str(resolved_path),
        "cue_count": len(cues),
        "baseline": baseline_result,
        "after_184": after_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure real-cue recall accuracy against a read-only copy of the owner's Hippo store."
    )
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default="stdlib")
    args = parser.parse_args()
    if args.driver == "lilli":
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    result = run_both_passes(args.fixture, driver=args.driver)
    print(json.dumps(result, indent=2, default=str))
