"""Outcome-free family folds, ranking and held-out coverage."""
from __future__ import annotations

import numpy as np

from .contracts import digest, require, seal
from .protocol import SEED


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
