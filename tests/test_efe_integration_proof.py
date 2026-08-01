from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore

class TestEFEIntegration:

    def test_pipeline_scoring_includes_stability_bonus(self, tmp_path):
        import iai_mcp.pipeline as pipeline_mod
        import inspect
        source = inspect.getsource(pipeline_mod)
        assert "_stability" in source, "EFE stability variable not in pipeline.py"
        assert "_ig" in source, "EFE information-gain variable not in pipeline.py"
        assert "1.0 - min(float(_stability)" in source, "EFE formula not in pipeline.py"

class TestWALIntegration:

    def test_erasure_agent_imports_wal(self):
        import iai_mcp.lilli.cycle.sleep_pipeline._erasure as sp_mod
        import inspect
        source = inspect.getsource(sp_mod)
        assert "from iai_mcp.sleep_wal import SleepWAL" in source
        assert "_wal = SleepWAL()" in source

    def test_wal_writes_to_file(self, tmp_path):
        from iai_mcp.sleep_wal import SleepWAL
        wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")
        entry = wal.begin("tombstone", ["rec-1", "rec-2"])
        assert (tmp_path / ".sleep-wal.jsonl").exists()
        content = (tmp_path / ".sleep-wal.jsonl").read_text()
        data = json.loads(content.strip())
        assert data["operation"] == "tombstone"
        assert data["target_ids"] == ["rec-1", "rec-2"]
        assert data["status"] == "pending"
