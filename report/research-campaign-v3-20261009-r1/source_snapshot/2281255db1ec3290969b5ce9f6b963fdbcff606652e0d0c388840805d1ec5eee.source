"""Sparse (model, layer, expert) associations, with question-level support."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .evaluation import bh_adjust

_EXPERT_KEYS = tuple(str(i) for i in range(65536))


def sentence_experts(sentences, selected, layers):
    if len(layers) != selected.shape[0] or len(set(layers)) != len(layers):
        raise ValueError("Expert layer identities mismatch")
    result = []
    for sentence in sentences:
        tokens = sentence["token_indices"]
        if not tokens:
            continue
        if max(tokens) >= selected.shape[1]:
            raise ValueError("Expert/token alignment mismatch")
        for i, layer in enumerate(layers):
            values = selected[i, tokens]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            counts = np.unique(np.asarray(values), return_counts=True)
            result.append({**{k: sentence[k] for k in ("model", "dataset", "trace_id", "question_id", "sentence_index")},
                "layer": layer, "counts": {_EXPERT_KEYS[int(e)] if int(e) < len(_EXPERT_KEYS) else str(int(e)): int(c) for e, c in zip(*counts)},
                "tokens": len(tokens), "label": sentence["label"], "next_class": sentence["next_class"]})
    return result


def _layer_associations(sentences, experts, min_problems=10, bootstrap=1000, seed=42):
    # Missing sparse entries mean zero selection; missing labels remain undefined.
    units = {(r["trace_id"], r["sentence_index"]): r for r in sentences if r["label"]}
    layer_units = defaultdict(set)
    selected = defaultdict(dict)
    for e in experts:
        unit = (e["trace_id"], e["sentence_index"])
        if unit in units:
            layer_units[(e["model"], e["dataset"], e["layer"])].add(unit)
            for expert, count in e["counts"].items():
                selected[(e["model"], e["dataset"], e["layer"], int(expert))][unit] = count / e["tokens"]
    rng, result = np.random.default_rng(seed), []
    for (model, dataset, layer, expert), values in sorted(selected.items()):
        universe = sorted(layer_units[(model, dataset, layer)])
        conditions = [(None, c) for c in sorted({units[u]["label"] for u in universe})]
        conditions += [(c, d) for c in sorted({units[u]["label"] for u in universe})
                       for d in sorted({units[u]["next_class"] for u in universe
                                        if units[u]["label"] == c and units[u]["next_class"]})]
        for source, target in conditions:
            eligible = [u for u in universe if source is None or
                        (units[u]["label"] == source and units[u]["next_class"] is not None)]
            positive = [u for u in eligible if units[u]["label" if source is None else "next_class"] == target]
            positive_set = set(positive)
            negative = [u for u in eligible if u not in positive_set]
            ps = {units[u]["question_id"] for u in positive}
            ns = {units[u]["question_id"] for u in negative}
            # Rank only contrasts supported in both arms and selected expert in >=10 problems.
            es = {units[u]["question_id"] for u in eligible if values.get(u, 0) > 0}
            supported = min(len(ps), len(ns), len(es)) >= min_problems
            def mean(group):
                return sum(values.get(u, 0) * units[u]["tokens"] for u in group) / sum(units[u]["tokens"] for u in group) if group else None
            pos, neg = mean(positive), mean(negative)
            # Token-level binary expert selection x class nPMI; association, not causality.
            total = sum(units[u]["tokens"] for u in eligible)
            joint = sum(values.get(u, 0) * units[u]["tokens"] for u in positive) / total if total else 0
            pe = sum(values.get(u, 0) * units[u]["tokens"] for u in eligible) / total if total else 0
            pc = sum(units[u]["tokens"] for u in positive) / total if total else 0
            npmi = math.log(joint / (pe * pc)) / -math.log(joint) if 0 < joint < 1 and pe * pc else None
            row = dict(model=model, dataset=dataset, layer=layer, expert=expert, source_class=source,
                target_class=target, positive_problems=len(ps), negative_problems=len(ns),
                expert_problems=len(es), positive_rate=pos, negative_rate=neg,
                rate_difference=pos - neg if pos is not None and neg is not None else None,
                npmi=npmi, estimand="token-weighted selection rate", status="descriptive" if supported else "insufficient_support",
                p_bootstrap=None, confidence_interval=None)
            if supported:
                # Question bootstrap with multiplicity; resample all questions jointly for both arms.
                problem_stats = defaultdict(lambda: [0., 0., 0., 0.])
                for u in eligible:
                    j = 0 if u in positive_set else 2
                    stats = problem_stats[units[u]["question_id"]]
                    stats[j] += values.get(u, 0) * units[u]["tokens"]
                    stats[j + 1] += units[u]["tokens"]
                data = np.array(list(problem_stats.values()))
                x = data[rng.integers(0, len(data), (bootstrap, len(data)))].sum(1)
                x = x[(x[:, 1] > 0) & (x[:, 3] > 0)]
                estimates = x[:, 0] / x[:, 1] - x[:, 2] / x[:, 3]
                row["confidence_interval"] = np.quantile(estimates, [.025, .975]).tolist()
                row["p_bootstrap"] = min(1., 2 * (min((estimates <= 0).sum(), (estimates >= 0).sum()) + 1) / (len(estimates) + 1))
            result.append(row)
    families = defaultdict(list)
    for i, row in enumerate(result):
        families[(row["model"], row["dataset"], "class" if row["source_class"] is None else "next_class")].append(i)
    for indices in families.values():
        for i, q in zip(indices, bh_adjust([result[i]["p_bootstrap"] for i in indices])):
            result[i]["q_bh"] = q
    return result


def associations(sentences, experts, min_problems=10, bootstrap=1000, seed=42):
    # Expand sparse rates for only one layer at a time; never pool numeric IDs.
    layers = defaultdict(list)
    for row in experts:
        layers[(row["model"], row["dataset"], row["layer"])].append(row)
    result = []
    for records in layers.values():
        result.extend(_layer_associations(sentences, records, min_problems, bootstrap, seed))
    families = defaultdict(list)
    for i, row in enumerate(result):
        families[(row["model"], row["dataset"], "class" if row["source_class"] is None else "next_class")].append(i)
    for indices in families.values():
        for i, q in zip(indices, bh_adjust([result[i]["p_bootstrap"] for i in indices])):
            result[i]["q_bh"] = q
    return result
