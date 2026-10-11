"""Pure dict-merge logic for the baseline-evidence read-merge-write fixture
in test_baseline_knowledge_vs_chatter.py. Zero iai_mcp imports -- no
embedder, no store, no bench artifacts -- so it can be exercised by a plain,
un-flagged pytest run without loading anything heavy.
"""
from __future__ import annotations

_REQUIRED_EVIDENCE_FIELDS = ("reproduced", "shape", "miss_rate", "recall_served_rank")


class _IncompleteEvidenceError(Exception):
    """This run's own contribution to a driver is missing a required field."""


def _merge_and_validate_evidence(
    existing: dict, new: dict, required_fields: tuple = _REQUIRED_EVIDENCE_FIELDS,
) -> dict:
    """Merge new per-driver evidence into the existing committed dict,
    preserving unrelated drivers and unrelated fields within a touched
    driver. Completeness is checked against THIS RUN's own contribution
    (`fields`), never the merged result -- checking the merged result lets
    a full run that freshly fails calibration inherit stale numeric fields
    (e.g. miss_rate, recall_served_rank) from an earlier, unrelated commit
    and pass the guard silently. `required_fields` defaults to this
    module's own field set so every existing caller is unchanged; a caller
    with a different evidence shape passes its own tuple."""
    merged = dict(existing)
    incomplete: dict = {}
    for driver, fields in new.items():
        merged[driver] = {**merged.get(driver, {}), **fields}
        missing = [f for f in required_fields if f not in fields]
        if missing:
            incomplete[driver] = missing
    if incomplete:
        raise _IncompleteEvidenceError(f"incomplete field set written this run for {incomplete}")
    return merged
