"""Consecutive-sentence dynamics with explicit gap and endpoint censoring."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .contracts import require, seal


def analyze_transitions(rows, classes, *, total_sentences):
    classes = list(classes)
    nc = len(classes)
    attempts = defaultdict(list)
    for row in rows:
        require(set(("attempt_id", "family", "sentence_index", "segment", "label", "n_tokens")) <= set(row),
                "missing original sentence/segment identity")
        require(row["label"] in classes or row["label"] is None, "unknown reasoning label")
        attempts[row["attempt_id"]].append(row)
    per_family = defaultdict(lambda: np.zeros((nc, nc), dtype=np.int64))
    dwell, gaps, labelled = [], 0, 0
    for aid, sequence in attempts.items():
        sequence.sort(key=lambda r: r["sentence_index"])
        require(len({r["sentence_index"] for r in sequence}) == len(sequence), "duplicate sentence row")
        require(len({r["family"] for r in sequence}) == 1, "attempt changes family")
        run = None
        previous = None
        for row in sequence:
            adjacent = bool(previous is not None and row["sentence_index"] == previous["sentence_index"] + 1
                            and row["segment"] == previous["segment"] and
                            previous["label"] is not None and row["label"] is not None)
            if previous is not None and not adjacent:
                gaps += 1
            if adjacent:
                per_family[row["family"]][classes.index(previous["label"]), classes.index(row["label"])] += 1
            if run and (not adjacent or row["label"] != run["label"]):
                run["right_censored"] = not adjacent
                dwell.append(run)
                run = None
            if row["label"] is not None:
                labelled += 1
                if run is None:
                    run = {"attempt_id": aid, "family": row["family"], "label": row["label"],
                           "start_sentence": row["sentence_index"], "sentences": 0, "tokens": 0,
                           "left_censored": not adjacent}
                run["sentences"] += 1
                run["tokens"] += int(row["n_tokens"])
            previous = row
        if run:
            run["right_censored"] = True
            dwell.append(run)
    require(total_sentences >= labelled, "label coverage denominator too small")
    grouped = np.stack(list(per_family.values())) if per_family else np.zeros((0, nc, nc))
    counts = grouped.sum(0)
    source_mass = grouped.sum(-1)
    normalized = np.divide(grouped, source_mass[..., None], out=np.zeros_like(grouped, dtype=float),
                           where=source_mass[..., None] > 0)
    support = (source_mass > 0).sum(0)
    probability = np.divide(normalized.sum(0), support[:, None], out=np.zeros((nc, nc)), where=support[:, None] > 0)
    pair_support = (grouped > 0).sum(0)
    return seal({"schema": "expert-memory-transitions-v1", "classes": classes,
                 "counts": counts.astype(int).tolist(), "pair_family_support": pair_support.tolist(),
                 "source_family_support": support.tolist(), "family_balanced_probability": probability.tolist(),
                 "labelled_sentences": labelled, "total_sentences": total_sentences,
                 "label_coverage": labelled / total_sentences if total_sentences else None,
                 "broken_adjacencies": gaps, "dwell": dwell,
                 "dwell_scope": "observed lengths only; left/right censoring retained"})
