"""Bounded contiguous-label plan; no label service is invoked here."""
from __future__ import annotations

from collections import defaultdict

from .common import digest
from .contracts import seal


def contiguous_blocks(units, width=40):
    """Frozen first/centred/last rule; overlap is deduplicated, gaps remain gaps."""
    n = len(units)
    starts = sorted({0, max(0, (n - width) // 2), max(0, n - width)})
    selected = set()
    for start in starts:
        selected.update(range(start, min(start + width, n)))
    return [units[i] for i in sorted(selected)]


def prepare(rows, units_by_attempt, *, model, input_binding, reusable=(), seed=42):
    """At most 16 B questions, one correct/wrong sibling each, 3840 unique units.

    Reuse identity includes exact sentence text/offset and trace hash. Prefix-only
    paired labels are a separate 200-unit calibration allocation, never silently
    substituted by labels created with the next sentence visible.
    """
    questions = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        if r["model"] == model and r["is_correct"] is not None and any(m["cohort"] == "B" for m in r["memberships"]):
            questions[r["question_id"]][r["is_correct"]].append(r)
    eligible = sorted((q for q, a in questions.items() if a[True] and a[False]), key=lambda q: digest([seed, q]))[:16]
    reuse = {r["unit_binding"]: r for r in reusable}
    chosen, attempts = [], []
    for q in eligible:
        for outcome in (True, False):
            attempt = min(questions[q][outcome], key=lambda r: digest([seed, r["attempt_id"]]))
            attempts.append(attempt["attempt_id"])
            for unit in contiguous_blocks(units_by_attempt[attempt["attempt_id"]]):
                binding = digest([attempt["trace_sha256"], unit])
                chosen.append(dict(question_id=q, attempt_id=attempt["attempt_id"],
                    trace_sha256=attempt["trace_sha256"], unit=unit, unit_binding=binding,
                    reuse=reuse.get(binding), observation="retrospective_lookahead"))
    if len(chosen) > 3840:
        raise AssertionError("Contiguous annotation budget exceeded")
    paired = sorted(chosen, key=lambda r: digest([seed, "paired", r["unit_binding"]]))[:200]
    return seal("analysis", dict(questions=eligible, attempts=attempts, units=chosen,
        prefix_only_pairs=[dict(unit_binding=r["unit_binding"], attempt_id=r["attempt_id"],
            unit=r["unit"], next_sentence_visible=False) for r in paired],
        new_lookahead_labels=sum(r["reuse"] is None for r in chosen),
        new_prefix_only_labels=len(paired), maximum_lookahead_labels=3840,
        label_agreement="pending_paired_annotation"), inputs={"inventory": input_binding},
        config=dict(seed=seed, model=model, questions=16, blocks=3, block_sentences=40,
                    block_rule="first_centered_last", paired_sentences=200),
        population="selected_B_sibling_pairs_conditional_on_eligibility")
