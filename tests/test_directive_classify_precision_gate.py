"""Precision-over-recall bar for the directive classifier, run before the
classifier is wired live.

A false positive plants a phantom standing order that unconditionally
injects into every future session, while a conservative miss just leaves a
sentence as ordinary memory. The scoped always/never form is the noisiest of
the four owner-named signals, so it gets its own strict proof below in
addition to the overall precision bar.
"""

from __future__ import annotations

from iai_mcp.directive_classify import classify_is_directive

# Pre-committed BEFORE measurement. Do not adjust to make a run pass --
# narrow the classifier's patterns instead.
DIRECTIVE_PRECISION_BAR = 0.95

MIN_TOTAL_ITEMS = 40
MIN_TRUE_POSITIVE_ITEMS = 15
MIN_ADVERSARIAL_NEGATIVE_ITEMS = 10
MIN_PREDICTED_POSITIVE_ITEMS = 10
MIN_RECALL_ON_TRUE_POSITIVE = 0.6

# (text, is_directive, adversarial)
# adversarial=True marks bare-descriptive always/never sentences engineered
# to bait a naive `\balways\b`/`\bnever\b` classifier into a false positive:
# conjugated verb forms of the exact verbs the scoped pattern whitelists
# ("uses" vs "use", "checks" vs "check", "forgets" vs "forget", "runs" vs
# "run"), and non-adjacent policy words ("never got a reply"). Composed
# independently of directive_classify.py's own pattern strings.
LABELLED_SAMPLE: "list[tuple[str, bool, bool]]" = [
    # -- from now on (3) --
    ("from now on reply in English", True, False),
    ("from now on always confirm before deleting files", True, False),
    ("please, from now on use tabs not spaces", True, False),
    # -- across all sessions (2) --
    ("across all sessions, prefer concise answers", True, False),
    ("remember this across all sessions: I go by Alex", True, False),
    # -- save this for all sessions (1, literal) --
    ("save this for all sessions", True, False),
    # -- scoped always/never, policy-shaped (12) --
    ("always reply in English", True, False),
    ("always confirm before deleting a record", True, False),
    ("never use markdown in commit messages", True, False),
    ("never store passwords in plaintext", True, False),
    ("always call me Alex", True, False),
    ("never skip the release gate", True, False),
    ("always run the full test suite before merging", True, False),
    ("never merge without review", True, False),
    ("always keep answers under five sentences", True, False),
    ("never expose internal file paths in responses", True, False),
    ("always verify before claiming something is done", True, False),
    ("never assume the user wants a global change", True, False),
    # -- adversarial negatives: bare descriptive always/never (12) --
    ("the build always takes twenty minutes", False, True),
    ("she never uses the staging branch", False, True),
    ("that test always flakes on arm", False, True),
    ("we never got a reply from the vendor", False, True),
    ("the deploy always finishes around midnight", False, True),
    ("he never checks his email before noon", False, True),
    ("the cache always warms up after the first request", False, True),
    ("they never work on weekends", False, True),
    ("the server always restarts after a crash", False, True),
    ("I never got the confirmation email", False, True),
    ("the migration always takes longer than expected", False, True),
    ("she always forgets to update the changelog", False, True),
    # -- plain negatives, no signal (10) --
    ("the weather is nice today", False, False),
    ("the meeting is scheduled for next Monday", False, False),
    ("I like the new dashboard layout", False, False),
    ("maybe the regression is in the embedder", False, False),
    ("the report was shared with stakeholders", False, False),
    ("the team reviewed the proposal yesterday", False, False),
    ("the fix should land around Friday", False, False),
    ("she mentioned the roadmap during the call", False, False),
    ("the deploy will probably finish within the hour", False, False),
    ("the document lists several unrelated topics", False, False),
]


def test_labelled_sample_composition_is_non_vacuous():
    """Fixture must offer real false-positive opportunities before the
    precision number below means anything."""
    assert len(LABELLED_SAMPLE) >= MIN_TOTAL_ITEMS, len(LABELLED_SAMPLE)

    true_positive_count = sum(1 for _, label, _ in LABELLED_SAMPLE if label)
    assert true_positive_count >= MIN_TRUE_POSITIVE_ITEMS, true_positive_count

    adversarial_negative_count = sum(
        1 for _, label, adversarial in LABELLED_SAMPLE if adversarial and not label
    )
    assert adversarial_negative_count >= MIN_ADVERSARIAL_NEGATIVE_ITEMS, (
        adversarial_negative_count
    )


def test_classifier_precision_bar_holds():
    predictions = [
        (text, label, classify_is_directive(text))
        for text, label, _ in LABELLED_SAMPLE
    ]

    predicted_true = [(t, l, p) for t, l, p in predictions if p]
    true_positive = sum(1 for _, l, _ in predicted_true if l)
    false_positive = len(predicted_true) - true_positive

    denominator = true_positive + false_positive
    assert denominator > 0, "classifier predicted True on zero items -- undefined precision"

    precision = true_positive / denominator
    assert precision >= DIRECTIVE_PRECISION_BAR, (
        f"directive precision {precision:.4f} ({true_positive}/{denominator}) "
        f"below the pre-committed bar {DIRECTIVE_PRECISION_BAR}"
    )
    assert len(predicted_true) >= MIN_PREDICTED_POSITIVE_ITEMS, len(predicted_true)

    true_positive_total = sum(1 for _, label, _ in predictions if label)
    recall = true_positive / true_positive_total
    assert recall >= MIN_RECALL_ON_TRUE_POSITIVE, recall


def test_scoped_always_never_fires_on_policy_shape_not_on_descriptive():
    """The always/never form is proven separately and strictly: every
    policy-shaped positive in the sample must fire, and every adversarial
    descriptive negative that contains always/never must not."""
    always_never_positives = [
        text for text, label, _ in LABELLED_SAMPLE
        if label and ("always" in text.lower() or "never" in text.lower())
    ]
    always_never_adversarial_negatives = [
        text for text, label, adversarial in LABELLED_SAMPLE
        if not label and adversarial
        and ("always" in text.lower() or "never" in text.lower())
    ]
    assert len(always_never_positives) >= 10, len(always_never_positives)
    assert len(always_never_adversarial_negatives) >= 10, len(
        always_never_adversarial_negatives
    )

    misses = [t for t in always_never_positives if not classify_is_directive(t)]
    assert not misses, f"scoped always/never missed policy-shaped positives: {misses}"

    false_fires = [
        t for t in always_never_adversarial_negatives if classify_is_directive(t)
    ]
    assert not false_fires, (
        f"scoped always/never fired on bare descriptive text: {false_fires}"
    )
