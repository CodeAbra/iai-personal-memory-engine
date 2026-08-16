from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog, _LIVENESS_SPEC_VERSION
from iai_mcp.lilli.cycle.sleep_pipeline import (
    _LIVENESS_SPEC,
    SleepPipeline,
    SleepStep,
)


@pytest.fixture
def event_log(tmp_path: Path) -> LifecycleEventLog:
    return LifecycleEventLog(log_dir=tmp_path / "logs")


@pytest.fixture
def pipeline(tmp_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    return SleepPipeline(
        store=None,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=event_log,
    )


def _last_row(event_log: LifecycleEventLog) -> dict:
    rows = [
        r for r in event_log.read_all() if r.get("event") == "sleep_step_completed"
    ]
    assert rows, "no sleep_step_completed row was written"
    return rows[-1]


class TestExhaustiveness:
    def test_spec_covers_every_sleep_step(self) -> None:
        assert set(_LIVENESS_SPEC) == set(SleepStep)

    def test_spec_rejects_no_extras(self) -> None:
        assert len(_LIVENESS_SPEC) == len(SleepStep)


class TestInjectionOrder:
    def test_normalized_pair_wins_over_shadowing_payload_keys(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.RECONSOLIDATION,
            1.0,
            records_scanned=5,
            records_reconsolidated=2,
            liveness_candidates="shadow-attempt",
            liveness_processed="shadow-attempt",
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 5
        assert row["liveness_processed"] == 2
        assert row["liveness_spec_version"] == _LIVENESS_SPEC_VERSION


class TestCleanMapping:
    def test_reconsolidation_shaped_payload(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.RECONSOLIDATION,
            1.0,
            records_scanned=5,
            records_reconsolidated=2,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 5
        assert row["liveness_processed"] == 2
        assert row["liveness_spec_version"] == 1


class TestEmptyVsDead:
    def test_zero_of_positive_candidates_is_fail_eligible(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.RECONSOLIDATION,
            1.0,
            records_scanned=7,
            records_reconsolidated=0,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 7
        assert row["liveness_processed"] == 0

    def test_zero_of_zero_candidates_is_legitimately_idle(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.RECONSOLIDATION,
            1.0,
            records_scanned=0,
            records_reconsolidated=0,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 0
        assert row["liveness_processed"] == 0


class TestMissingFieldDegradesToNone:
    def test_missing_candidate_field_yields_none_not_zero(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.RECONSOLIDATION,
            1.0,
            records_reconsolidated=2,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] is None
        assert row["liveness_processed"] == 2


class TestKnobTuneHonestUnknown:
    def test_knob_tune_maps_to_none_none(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.KNOB_TUNE,
            1.0,
            knobs_tuned=11,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] is None
        assert row["liveness_processed"] is None


# step -> the module whose source text must contain each non-None spec
# field's literal key, so a field rename in the step source (not just a
# drift between two test-local literals) fails this guard.
_STEP_SOURCE_FILE: dict[SleepStep, str] = {
    SleepStep.SCHEMA_MINE: "_schema_mine.py",
    SleepStep.HIPPO_CLEANUP: "_compact.py",
    SleepStep.ERASURE_AGENT: "_erasure.py",
    SleepStep.CLUSTER_REPLAY: "_cluster_replay.py",
    SleepStep.CRISIS_RECLUSTER: "_crisis.py",
    SleepStep.RECONSOLIDATION: "_reconsolidation.py",
    SleepStep.COMMUNITY_NAMING: "_topic_naming.py",
}


class TestDriftGuard:
    def test_every_non_none_spec_field_is_a_real_payload_key(self) -> None:
        import iai_mcp.lilli.cycle.sleep_pipeline as sleep_pipeline_pkg

        pkg_dir = Path(sleep_pipeline_pkg.__file__).parent
        for step, (cand_field, proc_field) in _LIVENESS_SPEC.items():
            for field in (cand_field, proc_field):
                if field is None:
                    continue
                src_file = _STEP_SOURCE_FILE.get(step)
                assert src_file is not None, (
                    f"{step.name}: has a non-None spec field {field!r} but "
                    f"no entry in _STEP_SOURCE_FILE to check it against"
                )
                text = (pkg_dir / src_file).read_text(encoding="utf-8")
                assert f'"{field}"' in text, (
                    f"{step.name}: spec field {field!r} not found as a "
                    f"literal key in {src_file}"
                )


class TestCrisisReclusterZeroOfPositive:
    def test_crisis_recluster_zero_dropped_of_positive_communities(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.CRISIS_RECLUSTER,
            1.0,
            total_communities=12,
            communities_dropped=0,
            dry_run=False,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 12
        assert row["liveness_processed"] == 0


class TestErasureAgentEffectOnly:
    def test_erasure_agent_row_carries_none_candidate_and_real_processed(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.ERASURE_AGENT,
            1.0,
            count_quarantined=2,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] is None
        assert row["liveness_processed"] == 2


class TestClusterReplayCleanMapping:
    def test_cluster_replay_shaped_payload(
        self, pipeline: SleepPipeline, event_log: LifecycleEventLog,
    ) -> None:
        pipeline._emit_step_completed(
            SleepStep.CLUSTER_REPLAY,
            1.0,
            clusters_replayed=6,
            sequential_pairs=9,
        )
        row = _last_row(event_log)
        assert row["liveness_candidates"] == 6
        assert row["liveness_processed"] == 9
