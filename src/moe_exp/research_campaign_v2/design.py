"""Outcome-blind cohorts, discovery statistics, nominations and byte-budget masks."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

from .contracts import CONFIGURATIONS, SEED, digest, require


def cohorts(rows, folds, limit):
    require(all(r["family"] in folds for r in rows), "family missing from original folds")
    by_benchmark = defaultdict(set)
    for r in rows:
        by_benchmark[r["benchmark"]].add(r["family"])
    selected = {b: set(sorted(fs, key=lambda f: (digest([SEED, b, f]), f))[:limit])
                for b, fs in sorted(by_benchmark.items())}
    return [dict(r, fold=folds[r["family"]]) for r in sorted(
        rows, key=lambda x: (x["benchmark"], x["family"], x["question"]))
        if r["family"] in selected[r["benchmark"]]]


def family_weights(rows):
    """One unit per family occurrence, equal benchmarks, arbitrary repeat counts."""
    cells = Counter((r["benchmark"], r["family"]) for r in rows)
    families = defaultdict(set)
    for b, f in cells:
        families[b].add(f)
    return np.array([1 / (len(families) * len(families[r["benchmark"]]) *
                          cells[r["benchmark"], r["family"]]) for r in rows])


def prefix_roster(sentences, selected, folds, model):
    """No replacement when the assigned progress third has no usable prefix."""
    wanted = {(r["benchmark"], r["family"]) for r in selected}
    by_cell = defaultdict(list)
    for s in sentences:
        if (s["benchmark"], s["family"]) in wanted:
            by_cell[s["benchmark"], s["family"]].append(s)
    cells = sorted(wanted, key=lambda c: (c[0], folds[c[1]], digest([SEED, "prefix", *c])))
    block = Counter()
    result = []
    for b, f in cells:
        key = (b, folds[f])
        third = block[key] % 3
        block[key] += 1
        compatible = [s for s in by_cell[b, f] if s.get("compatible",True) and min(2, int(s["progress"] * 3)) == third]
        chosen = min(compatible, key=lambda s: (digest([SEED, "boundary", model,
                     s["attempt_id"], s["sentence_index"]]), s["sentence_index"]), default=None)
        result.append({"model": model, "benchmark": b, "family": f, "fold": folds[f],
                       "progress_third": third, "prefix": chosen,
                       "status": "enrolled" if chosen else "upstream_unavailable"})
    # Balanced over benchmark x fold x assigned progress position, before any outcomes.
    blocks = defaultdict(list)
    for r in result:
        blocks[r["benchmark"], r["fold"], r["progress_third"]].append(r)
    for key, members in blocks.items():
        members.sort(key=lambda r: digest([SEED, "sweep", *key, r["family"]]))
        for i, r in enumerate(members):
            r["configuration_indices"] = [0, 1 + i % 8]
    return result


def weighted_quantile(values, weights, quantiles):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return [None] * len(quantiles)
    order = np.argsort(v[valid], kind="stable")
    v, w = v[valid][order], w[valid][order]
    c = np.cumsum(w) / w.sum()
    return [float(v[min(len(v) - 1, np.searchsorted(c, q))]) for q in quantiles]


def inclusion_statistics(counts, tokens, rows, train):
    """Per-family per-benchmark token inclusion; unsupported cells are NaN."""
    counts, tokens = np.asarray(counts), np.asarray(tokens)
    require(counts.shape[:2] == tokens.shape and len(rows) == len(counts), "count alignment")
    C, L, E = counts.shape[1:]
    by_cell = defaultdict(list)
    for i, r in enumerate(rows):
        if train[i]:
            by_cell[r["benchmark"], r["family"]].append(i)
    by_benchmark = defaultdict(list)
    for (b, _), ids in by_cell.items():
        n = tokens[ids].sum(0, dtype=np.float64)
        c = counts[ids].sum(0, dtype=np.float64)
        v = np.divide(c, n[:, None, None], out=np.full((C, L, E), np.nan),
                      where=n[:, None, None] > 0)
        by_benchmark[b].append(v)
    require(by_benchmark, "no discovery families")
    support = np.zeros((C, L, E), int)
    totals, weights = np.zeros((C, L, E)), np.zeros((C, L, E))
    for values in by_benchmark.values():
        a = np.stack(values)
        valid = np.isfinite(a)
        sup = valid.sum(0)
        mean = np.divide(np.nansum(a, axis=0), sup, out=np.zeros((C, L, E)), where=sup > 0)
        support += sup
        totals += mean * (sup > 0)
        weights += sup > 0
    rate = np.divide(totals, weights, out=np.full((C, L, E), np.nan), where=weights > 0)
    return rate, support


def enriched_set(excess, fraction):
    order = sorted((i for i, v in enumerate(excess) if np.isfinite(v) and v > 0),
                   key=lambda e: (-excess[e], e))
    if not order:
        return []
    threshold = sum(excess[e] for e in order) * fraction
    total, selected = 0., []
    for e in order:
        selected.append(e)
        total += excess[e]
        if total >= threshold:
            break
    return sorted(selected)


def nominate(excess, stability, layers, k):
    require(len(excess) == len(layers) == len(stability), "layer nomination alignment")
    result = []
    for indices in np.array_split(np.arange(len(layers)), 3):
        candidates = []
        for i in indices:
            ids = enriched_set(excess[i], .5)
            if ids:
                allocation = sum(excess[i, e] for e in ids) / k
                candidates.append((-allocation, -float(stability[i]), int(layers[i]), int(i)))
        best = min(candidates, default=None)
        result.append(None if best is None else best[3])
    return result


def random_control(ids, frequency, identity):
    """Match size and frequency-rank quintile counts, with recorded priorities."""
    E = len(frequency)
    order = sorted(range(E), key=lambda e: (float(frequency[e]), e))
    strata = {e: min(4, i * 5 // E) for i, e in enumerate(order)}
    demand = Counter(strata[e] for e in ids)
    chosen = []
    for s, count in sorted(demand.items()):
        candidates = [e for e in range(E) if strata[e] == s]
        candidates.sort(key=lambda e: (digest([SEED, "control", identity, e]), e))
        chosen += candidates[:count]
    return sorted(chosen)


def masks(score, expert_bytes, removal, k, *, unequal, layers):
    """Each method keeps its cross-layer scale; no implicit score normalization."""
    score, size = np.asarray(score, float), np.broadcast_to(expert_bytes, np.shape(score)).astype(float)
    require(score.ndim == 2 and np.isfinite(score).all() and (size > 0).all(), "invalid score/byte table")
    L, E = score.shape
    target = float(size.sum()) * (1 - removal)
    if not unequal:
        keep = int(round(E * (1 - removal)))
        require(keep >= k and math.isclose(keep, E * (1 - removal)), "nonintegral native keep count")
        chosen = [sorted(sorted(range(E), key=lambda e: (-score[i, e], e))[:keep]) for i in range(L)]
    else:
        chosen = [sorted(range(E), key=lambda e: (-score[i, e] / size[i, e], e))[:k] for i in range(L)]
        used = sum(size[i, e] for i in range(L) for e in chosen[i])
        require(used <= target, "budget below native minimum")
        pool = [(i, e) for i in range(L) for e in range(E) if e not in chosen[i]]
        pool.sort(key=lambda ie: (-score[ie] / size[ie], layers[ie[0]], ie[1]))
        for i, e in pool:
            if used + size[i, e] <= target + 1e-6:
                chosen[i].append(e)
                used += size[i, e]
        chosen = [sorted(ids) for ids in chosen]
    retained = sum(size[i, e] for i in range(L) for e in chosen[i])
    require(retained <= target + 1e-6, "retained bytes exceed comparison budget")
    return {str(l): ids for l, ids in zip(layers, chosen)}, retained


def config_values(index, sets, doses, durations):
    require(0 <= index < len(CONFIGURATIONS), "unknown threshold index")
    indices = ["LMH".index(x) for x in CONFIGURATIONS[index]]
    return {"experts": sets[indices[0]], "bias": doses[indices[1]],
            "pulse_tokens": durations[indices[2]], "configuration": CONFIGURATIONS[index]}
