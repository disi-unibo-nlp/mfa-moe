"""Past-only routing/text rows for retrospectively adjudicated reasoning events."""
from __future__ import annotations

from .classes import class_dynamics
from .evaluation import latest_windows, text_features


def event_rows(trace, sentences, windows, identity):
    """Unknown annotations break histories; natural termination competes with exits.

    Labels are discovery strata and outcome targets, not deployable observations.
    Features stop before ANY token overlapping destination text, including a token
    assigned to its preceding sentence by the majority-overlap ownership rule.
    """
    class_dynamics(sentences)  # supplies gap-aware, observed-entry run ages
    replay = trace.metadata.get("token_replay") or {}
    offsets = replay.get("completion_offsets")
    if not offsets:
        return []
    output, history, previous = [], [], None
    for index, sentence in enumerate(sentences):
        label = sentence.get("label")
        contiguous = bool(previous and previous["sentence_index"] + 1 == sentence["sentence_index"]
                          and previous["segment"] == sentence["segment"] and previous["label"] and label)
        if not contiguous:
            history = []
        previous = sentence
        if label is None or sentence.get("token_end") is None:
            continue
        recurrence = bool(label in history and history[-1] != label)
        history.append(label)
        history = history[-6:]
        following = sentences[index + 1] if index + 1 < len(sentences) else None
        adjacent = bool(following and following["label"] and following["segment"] == sentence["segment"]
                        and following["sentence_index"] == sentence["sentence_index"] + 1)
        endpoint, destination_start = sentence["token_end"], None
        target, destination = None, None
        if adjacent:
            destination_start = next((i for i, (a, b) in enumerate(offsets)
                                      if b > following["start"] and a < following["end"]), None)
            if destination_start is None:
                continue
            endpoint = min(endpoint, destination_start)
            destination = following["label"]
            target = "stay" if destination == label else "exit"
        elif following is None and sentence["termination"] == "natural_termination":
            target = "natural_termination"
        if endpoint < 1:
            continue
        prefix_end = max(b for _, b in offsets[:endpoint])
        age = sentence.get("run_age_sentences")
        row = {**identity, "decision_token": endpoint, "feature_end_token": endpoint,
            "budget_population": "original", "base": {"token_position": endpoint,
                "dataset_" + trace.dataset: 1.}, "text": text_features(trace.cot_text[:prefix_end]),
            "state": {}, "routing_windows": latest_windows(windows, endpoint),
            "is_correct": identity["is_correct"], "remaining_tokens": len(offsets) - endpoint,
            "completion_tokens": len(offsets), "source_class": label, "next_class": destination,
            "next_event": target, "adjacent_labels": adjacent,
            "destination_start_token": destination_start,
            "run_age_bin": None if age is None else min(age, 10),
            "history_signature": tuple(history[-4:]), "is_recurrence": recurrence,
            "label_provenance": "retrospective_discovery_only",
            "transition_censored": target is None}
        output.append(row)
    return output
