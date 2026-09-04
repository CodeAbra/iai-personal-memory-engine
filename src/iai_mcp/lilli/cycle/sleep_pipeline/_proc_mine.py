from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_proc_mine(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.lilli.cycle.chunk import decay_proc_chunks, persist_proc_chunk
    from iai_mcp.lilli.cycle.proc_mine import load_cofired_events, mine_cofired_pairs
    from iai_mcp.lilli.cycle.toolseq_mine import load_assistant_turns, mine_tool_ngrams

    if self._check_interrupt(SleepStep.PROC_MINE, 0, interrupt_check):
        return False, {}

    events = load_cofired_events(self._store)
    a_candidates = mine_cofired_pairs(events)

    turns = load_assistant_turns(self._store)
    b_candidates = mine_tool_ngrams(turns)

    candidates = [*a_candidates, *b_candidates]

    persisted = 0
    for candidate in candidates:
        if persist_proc_chunk(self._store, candidate) is not None:
            persisted += 1

    decay_proc_chunks(self._store, now=self._now())

    from iai_mcp import prime_cache

    prime_cache.save(self._store, prime_cache.build(self._store))
    prime_cache.invalidate(self._store)

    return True, {"candidates_gated": len(candidates), "chunks_persisted": persisted}
