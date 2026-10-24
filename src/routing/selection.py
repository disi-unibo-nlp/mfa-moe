"""Expert selection from saved routing counts: held-out coverage, inclusion statistics and nominations.

Counts are [attempt, class, layer, expert] token inclusions. `analyze` fits the outcome-free state-balanced and
labelled-frequency expert sets on four of five family folds and scores their coverage on the fifth; the other
functions rank experts per class and layer, nominate one layer per depth third and draw matched random controls.

    python -m routing.selection export OUT_DIR [--campaign DIR]

`export` writes OUT_DIR/selection_coverage.csv, the table behind the manuscript's selection figure: for each of the
seven captured models, fold and selector, the minimum class coverage on the held-out families. It reads the stored
SELECTION.json receipts of the seven-model expert selection (default report/expert-memory-study-v1).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import numpy as np

from models.roster import CAPTURED
from store import CLASSES, SEED, WORK, digest, file_sha, require, seal, verified

RECEIPTS = WORK / 'mfa-moe/repo/report/expert-memory-study-v1'  # reports are not tracked in git; they live in the main checkout


def family_folds(families, n=5):
    ordered = sorted(set(families), key=lambda f: (digest([SEED, "fold", f]), f))
    require(len(ordered) >= n, "insufficient independent families")
    return {f: i % n for i, f in enumerate(ordered)}


def validate_counts(counts, tokens, k):
    counts, tokens = np.asarray(counts), np.asarray(tokens)
    require(counts.ndim == 4 and tokens.shape == counts.shape[:2], "archive shape differs")
    require(np.issubdtype(counts.dtype, np.integer) and np.issubdtype(tokens.dtype, np.integer),
            "counts/tokens must be integer")
    require(bool((counts >= 0).all() and (tokens >= 0).all()), "negative count")
    require(bool((counts.sum(-1, dtype=np.uint64) == tokens[:, :, None] * k).all()),
            "selected slots differ from tokens times native k")


def scores(counts, families, train):
    """Counts are [attempt,class,layer,expert]; duplicates have total weight one."""
    counts = np.asarray(counts)
    families = np.asarray(families)
    train = np.asarray(train, dtype=bool)
    require(len(families) == len(counts) == len(train) and train.any(), "empty/misaligned training fold")
    grouped = np.stack([counts[train & (families == f)].mean(0, dtype=np.float64)
                        for f in sorted(set(families[train]))])
    mass = grouped.sum(-1)
    shares = np.divide(grouped, mass[..., None], out=np.zeros_like(grouped), where=mass[..., None] > 0)
    support = (mass > 0).sum(0)
    require(bool((support > 0).all()), "class/layer unsupported in training fold")
    balanced = (shares.sum(0) / support[..., None]).mean(0)
    pooled = grouped.sum((0, 1))
    frequency = pooled / pooled.sum(-1, keepdims=True)
    return balanced, frequency, support


def keep_set(score, keep, groups=1, *, model=None, layers=None, random=False):
    score = np.asarray(score)
    require(score.ndim == 2 and np.isfinite(score).all(), "invalid expert scores")
    L, E = score.shape
    require(E % groups == keep % groups == 0 and 0 < keep <= E, "nonintegral group quota")
    layers = list(range(L)) if layers is None else list(layers)
    require(len(layers) == L and len(set(layers)) == L, "layer map differs")
    result = []
    for i, layer in enumerate(layers):
        ids = []
        for group in range(groups):
            candidates = list(range(group * (E // groups), (group + 1) * (E // groups)))
            if random:
                require(model is not None, "random set needs model binding")
                candidates.sort(key=lambda e: (digest([SEED, model, layer, group, e]), e))
            else:
                candidates.sort(key=lambda e: (-score[i, e], e))
            ids += candidates[:keep // groups]
        result.append(sorted(ids))
    return np.asarray(result, dtype=np.int64)


def coverage(counts, families, mask):
    """Mean within-family class slot coverage, with explicit class support."""
    counts = np.asarray(counts, dtype=np.float64)
    grouped = np.stack([counts[np.asarray(families) == f].mean(0) for f in sorted(set(families))])
    total = grouped.sum(-1)
    kept = np.take_along_axis(grouped, np.asarray(mask)[None, None], axis=-1).sum(-1)
    support = (total > 0).sum(0)
    shares = np.divide(kept, total, out=np.zeros_like(kept), where=total > 0)
    result = np.divide(shares.sum(0), support, out=np.zeros_like(total[0]), where=support > 0)
    class_coverage = [float(row[sup > 0].mean()) if (sup > 0).any() else None
                      for row, sup in zip(result, support)]
    return {"coverage": [[float(x) if s else None for x, s in zip(row, sup)]
                         for row, sup in zip(result, support)],
            "family_support": support.tolist(),
            "class_mean_layer_coverage": class_coverage,
            "minimum_class_coverage": min((x for x in class_coverage if x is not None), default=None),
            "minimum_class_layer_coverage": float(result[support > 0].min()) if (support > 0).any() else None,
            "minimum_complete": bool((support > 0).all())}


def overlap(a, b):
    return [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]


def analyze(counts, families, *, keep, groups, layers, model, classes, folds=None):
    folds = folds or family_folds(families)
    require(set(families) <= set(folds), "family missing from common folds")
    random = keep_set(np.zeros(counts.shape[2:]), keep, groups, model=model, layers=layers, random=True)
    rows, fitted = [], []
    for fold in range(5):
        test = np.array([folds[f] == fold for f in families])
        require(test.any() and (~test).any(), "empty fold")
        b, f, support = scores(counts, families, ~test)
        masks = {"state_balanced": keep_set(b, keep, groups),
                 "labelled_frequency": keep_set(f, keep, groups), "random": random}
        fitted.append(masks)
        rows.append({"fold": fold, "train_families": len(set(np.asarray(families)[~test])),
                     "test_families": len(set(np.asarray(families)[test])), "training_support": support.tolist(),
                     "masks": {a: m.tolist() for a, m in masks.items()},
                     "held_out": {a: coverage(counts[test], np.asarray(families)[test], m) for a, m in masks.items()},
                     "balanced_frequency_jaccard": overlap(masks["state_balanced"], masks["labelled_frequency"])})
    b, f, support = scores(counts, families, np.ones(len(families), dtype=bool))
    final = {"state_balanced": keep_set(b, keep, groups), "labelled_frequency": keep_set(f, keep, groups), "random": random}
    stability = {a: [{"folds": [i, j], "layer_jaccard": overlap(fitted[i][a], fitted[j][a])}
                     for i in range(5) for j in range(i + 1, 5)] for a in final}
    return seal({"schema": "expert-selection-v1", "model": model, "classes": list(classes),
                 "layers": list(map(int, layers)), "groups": groups, "experts": counts.shape[-1],
                 "keep": keep, "folds_sha256": digest(folds), "folds": rows,
                 "stability": stability, "final_masks": {a: m.tolist() for a, m in final.items()},
                 "full_training_support": support.tolist(), "outcomes_used": False})


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


def counts_for(campaign, model, rows):
    import pandas as pd
    path = campaign["saved_inputs"][model]["counts"]
    with np.load(path, allow_pickle=False) as archive:
        ids, counts, tokens = (archive[k] for k in ("attempt_id", "counts", "tokens"))
        require(list(archive["classes"]) == list(CLASSES) + ["ALL_REASONING", "ALL_LABELLED"], "class order mismatch")
    index = pd.Index(ids).get_indexer([r["attempt_id"] for r in rows])
    require((index >= 0).all() and len(index) == len(ids), "archive row join failed")
    counts, tokens = counts[index], tokens[index]
    validate_counts(counts, tokens, campaign["models"][model]["native_k"])
    return counts, tokens


def export(campaign_dir, output):
    """Write selection_coverage.csv (model, fold, selector, minimum class coverage, test families, complete)."""
    campaign_dir, output = Path(campaign_dir), Path(output)
    campaign = verified(campaign_dir / 'CAMPAIGN.json')
    rows = []
    for model in CAPTURED:  # LFM2.5 has its own SELECTION_COVERAGE.json
        path = campaign_dir / model / 'SELECTION.json'
        if path.exists():
            receipt = verified(path)
            require(receipt['campaign_sha256'] == campaign['sha256'], 'foreign selection receipt')
            for fold in receipt['folds']:
                for arm, result in fold['held_out'].items():
                    rows.append({'model': model, 'fold': fold['fold'], 'arm': arm,
                                 'minimum_class_coverage': result['minimum_class_coverage'],
                                 'test_families': fold['test_families'], 'minimum_complete': result['minimum_complete']})
    require(rows, 'no SELECTION.json receipts found')
    output.mkdir(parents=True)
    with (output / 'selection_coverage.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return file_sha(output / 'selection_coverage.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m routing.selection', description='Selection coverage table for the paper.')
    parser.add_argument('command', choices=('export',))
    parser.add_argument('output', type=Path, help='new folder')
    parser.add_argument('--campaign', type=Path, default=RECEIPTS)
    args = parser.parse_args(argv)
    print(export(args.campaign, args.output), flush=True)


if __name__ == '__main__':
    main()
