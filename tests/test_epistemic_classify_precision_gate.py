"""Precision gate for the classifier's ``fact`` label.

A wrong ``fact`` tag misleads every future recall of that record (the field
is read back on every hit), while a conservative ``unknown`` costs nothing.
"""

from __future__ import annotations

from iai_mcp.epistemic_classify import classify_epistemic_status

# Pre-committed BEFORE measurement. Do not adjust to make a run pass --
# narrow the classifier's fact patterns instead.
FACT_PRECISION_BAR = 0.95

MIN_TOTAL_ITEMS = 60
MIN_TRUE_FACT_ITEMS = 20
MIN_ADVERSARIAL_NON_FACT_ITEMS = 15
MIN_PREDICTED_FACT_ITEMS = 12
MIN_PREDICTED_FACT_RECALL_ON_TRUE_FACT = 0.5

# (text, true_label, adversarial)
# adversarial=True marks items engineered to bait a keyword-only classifier
# into a wrong "fact": numeric measurements phrased with hedge wording the
# lexicon does not literally cover, and comparative/preference opinions
# phrased as bare declaratives ("X is Y") with no explicit opinion word.
# Composed independently of the classifier's own pattern strings -- these
# sentences were written from general phrasing knowledge, not copied from
# epistemic_classify.py.
LABELLED_SAMPLE: "list[tuple[str, str, bool]]" = [
    # -- fact (22): explicit assertion/measurement marker, no hedge word --
    ("the team confirmed the outage started at noon", "fact", False),
    ("we measured the p95 latency at 230ms", "fact", False),
    ("the fix was verified against the full suite", "fact", False),
    ("the build definitely failed on the arm runner", "fact", False),
    ("it turns out the config file was never read", "fact", False),
    ("in fact the migration finished before the deadline", "fact", False),
    ("the fact is the index rebuild took four hours", "fact", False),
    ("the vendor confirmed the shipment left the warehouse", "fact", False),
    ("the audit verified every transaction against the ledger", "fact", False),
    ("the profiler measured a 40 percent reduction in allocations", "fact", False),
    ("the release notes confirmed the patch shipped Tuesday", "fact", False),
    ("the postmortem verified the root cause was a stale lock", "fact", False),
    ("the benchmark measured throughput at nine thousand requests per second", "fact", False),
    ("the security review confirmed no credentials were exposed", "fact", False),
    ("I confirmed the field is called epistemic_status", "fact", False),
    ("I verified the migration against the staging database", "fact", False),
    ("the compliance team confirmed the retention policy is enforced", "fact", False),
    ("the incident report verified the timeline against the logs", "fact", False),
    ("engineering confirmed the rollback completed cleanly", "fact", False),
    ("the test run measured zero flaky failures this week", "fact", False),
    ("the changelog confirmed the deprecated flag was removed", "fact", False),
    ("in fact the cache hit rate rose after the change", "fact", False),
    # -- estimate, clean (6) --
    ("the fix should land around Friday", "estimate", False),
    ("it seems the cache is cold on first boot", "estimate", False),
    ("the deploy will probably finish within the hour", "estimate", False),
    ("I think the release ships next week", "estimate", False),
    ("the error rate is more or less flat this week", "estimate", False),
    ("I guess the retry budget is close to exhausted", "estimate", False),
    # -- estimate, adversarial (8): numeric measurement + non-obvious hedge --
    ("the response latency was roughly 400ms on that run", "estimate", True),
    ("the queue depth is approximately two hundred", "estimate", True),
    ("the p95 is ~400ms based on the last sample", "estimate", True),
    ("somewhere around 400ms based on early numbers", "estimate", True),
    ("give or take 400ms depending on load", "estimate", True),
    ("in the ballpark of 400ms", "estimate", True),
    ("close to 400ms, not exact yet", "estimate", True),
    ("about 40 percent faster in early testing", "estimate", True),
    # -- opinion, adversarial (7): declarative comparative; 6 of 7 use no
    # lexicon word, one ("better") is covered by an existing opinion
    # pattern and included as a sanity check --
    ("this design is better than the previous one", "opinion", True),
    ("the new dashboard layout is cleaner", "opinion", True),
    ("the refactor is simpler to follow", "opinion", True),
    ("the second draft is stronger overall", "opinion", True),
    ("the old version feels clunkier", "opinion", True),
    ("the updated flow is much smoother", "opinion", True),
    ("the new schema is tidier than before", "opinion", True),
    # -- opinion, clean (4) --
    ("I prefer the shorter migration path", "opinion", False),
    ("I like the new dashboard layout better", "opinion", False),
    ("I'd rather ship the smaller change first", "opinion", False),
    ("the old renderer was worse for large batches", "opinion", False),
    # -- hypothesis (8) --
    ("maybe the regression is in the embedder", "hypothesis", False),
    ("the bug might be a race condition", "hypothesis", False),
    ("I suspect the index is stale", "hypothesis", False),
    ("what if the cache never warms", "hypothesis", False),
    ("the fix could be a lock reorder", "hypothesis", False),
    ("possibly the retry storm caused the spike", "hypothesis", False),
    ("perhaps the timeout is too aggressive", "hypothesis", False),
    ("it could be that the DNS cache is stale", "hypothesis", False),
    # -- unknown (6): genuinely ambiguous, no signal --
    ("the meeting is scheduled for next Monday", "unknown", False),
    ("the document lists several unrelated topics", "unknown", False),
    ("she mentioned the roadmap during the call", "unknown", False),
    ("the folder contains a handful of scripts", "unknown", False),
    ("the team reviewed the proposal yesterday", "unknown", False),
    ("the report was shared with stakeholders", "unknown", False),
]


def test_labelled_sample_composition_is_non_vacuous():
    """Fixture must offer real false-positive opportunities before the
    precision number below means anything."""
    assert len(LABELLED_SAMPLE) >= MIN_TOTAL_ITEMS, len(LABELLED_SAMPLE)

    true_fact_count = sum(1 for _, label, _ in LABELLED_SAMPLE if label == "fact")
    assert true_fact_count >= MIN_TRUE_FACT_ITEMS, true_fact_count

    adversarial_non_fact_count = sum(
        1 for _, label, adversarial in LABELLED_SAMPLE
        if adversarial and label != "fact"
    )
    assert adversarial_non_fact_count >= MIN_ADVERSARIAL_NON_FACT_ITEMS, (
        adversarial_non_fact_count
    )


def test_fact_precision_clears_the_pre_committed_bar():
    predictions = [
        (text, true_label, classify_epistemic_status(text))
        for text, true_label, _ in LABELLED_SAMPLE
    ]

    predicted_fact = [(t, tl, pl) for t, tl, pl in predictions if pl == "fact"]
    true_positive_fact = sum(1 for _, tl, _ in predicted_fact if tl == "fact")
    false_positive_fact = len(predicted_fact) - true_positive_fact

    denominator = true_positive_fact + false_positive_fact
    assert denominator > 0, "classifier predicted fact on zero items -- undefined precision"

    precision = true_positive_fact / denominator
    assert precision >= FACT_PRECISION_BAR, (
        f"fact precision {precision:.4f} ({true_positive_fact}/{denominator}) "
        f"below the pre-committed bar {FACT_PRECISION_BAR}"
    )

    true_fact_texts = {t for t, tl, _ in predictions if tl == "fact"}
    assert len(predicted_fact) >= MIN_PREDICTED_FACT_ITEMS, len(predicted_fact)

    recalled_true_fact = sum(
        1 for t, tl, pl in predictions if tl == "fact" and pl == "fact"
    )
    recall_on_fact = recalled_true_fact / len(true_fact_texts)
    assert recall_on_fact >= MIN_PREDICTED_FACT_RECALL_ON_TRUE_FACT, recall_on_fact
