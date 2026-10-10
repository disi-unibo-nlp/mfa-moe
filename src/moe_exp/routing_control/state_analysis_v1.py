"""State transitions and strictly trailing native routing measurements."""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np

from .study_v1 import STATES, require


def prefix_status(text):
    """Prefix-only lexical proposals; semantic recognition needs a saved audit."""
    complete = bool(re.search(r"\\boxed\s*\{|\b(?:final\s+answer|answer\s+is|solution\s+is)\b", text, re.I))
    checked = bool(re.search(r"\b(?:check(?:ed|ing)?|substitut(?:e|ed|ing))\b", text, re.I) and
                   re.search(r"\b(?:satisf(?:y|ies)|consistent|correct|matches|agrees)\b", text, re.I))
    if complete:
        return "complete_checked" if checked else "complete_unchecked"
    if re.search(r"[=≈]\s*[-+]?\d|\b(?:result|value|we get|we obtain)\b", text, re.I):
        return "intermediate"
    return "unresolved"


def transition_summary(sentences, *, closed=False, hit_cap=False):
    """Missing labels and unobserved sentence gaps break adjacency and loops."""
    require(not (closed and hit_cap), "closure cannot be censored by a reasoning cap")
    rows = sorted(sentences, key=lambda r: r["sentence_index"])
    require(len({r["sentence_index"] for r in rows}) == len(rows), "duplicate sentence identity")
    counts, exits, reentries, loops = Counter(), Counter(), Counter(), Counter()
    dwell, current, visited, path = [], [], set(), []
    def flush(censor):
        nonlocal current
        if current:
            dwell.append({"state": current[0]["label"], "sentences": len(current),
                          "tokens": sum(r["token_end"] - r["token_start"] for r in current),
                          "censored": censor})
            current = []
    previous = None
    for row in rows:
        valid = row.get("label") in STATES and row.get("valid", True)
        adjacent = (previous is not None and row["sentence_index"] == previous["sentence_index"] + 1 and
                    row.get("segment", 0) == previous.get("segment", 0))
        if not valid or not adjacent:
            flush(True)
            previous = None
            visited, path = set(), []
        if not valid:
            continue
        if previous is not None:
            a, b = previous["label"], row["label"]
            counts[a, b] += 1
            if a != b:
                exits[a, b] += 1
                flush(False)
                if b in visited:
                    reentries[b] += 1
                    loops["→".join(path[path.index(b):] + [b])] += 1
                path.append(b)
        else:
            path = [row["label"]]
        visited.add(row["label"])
        current.append(row)
        previous = row
    if previous is not None and closed and rows and previous is rows[-1]:
        exits[previous["label"], "closure"] += 1
        flush(False)
    else:
        flush(True)
    matrix = [[counts[a, b] for b in STATES] for a in STATES]
    return {"states": list(STATES), "transition_counts": matrix,
        "destination_specific_exit_counts": [{"source": a, "destination": b, "count": n}
                                               for (a, b), n in sorted(exits.items())],
        "exit_probabilities": {a: {b: n / sum(v for (src, _), v in exits.items() if src == a)
                                    for (src, b), n in exits.items() if src == a}
                               for a in STATES if any(src == a for src, _ in exits)},
        "dwell": dwell, "reentry_counts": dict(reentries), "loops": dict(loops),
        "termination": "closure" if closed else "cap_censored" if hit_cap else "unobserved_end",
        "scope": "Observed contiguous labels only; closure is a competing terminal event."}


def validate_capture(tokens, ids, weights=None, *, n_layers=40, experts=256):
    """Every prediction row, including a closure-producing row, is retained."""
    ids = np.asarray(ids)
    require(ids.shape == (len(tokens), n_layers, 8) and np.issubdtype(ids.dtype, np.integer),
            "expert-ID rows do not align exactly with all emitted tokens")
    require((ids >= 0).all() and (ids < experts).all() and
            (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all(), "invalid native top-k IDs")
    if weights is not None:
        weights = np.asarray(weights)
        require(weights.shape == ids.shape and np.isfinite(weights).all() and (weights >= 0).all() and
                np.allclose(weights.sum(axis=-1), 1, atol=1e-5), "invalid normalized executed selected weights")
    return ids, weights


def trailing_features(ids, boundary, *, weights=None, window=64):
    """Source profiles end before the next prediction; future rows never enter."""
    ids = np.asarray(ids)
    require(type(boundary) is int and window in (64, 256) and 0 <= boundary <= len(ids),
            "invalid trailing boundary")
    require(ids.ndim == 3 and ids.shape[1:] == (40, 8), "routing shape differs")
    require((ids >= 0).all() and (ids < 256).all() and
            (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all(), "invalid routing IDs")
    require(weights is None or np.asarray(weights).shape == ids.shape, "executed weight rows differ")
    if boundary < window:
        return {"status": "UNAVAILABLE_SHORT_PREFIX", "window": window}
    def composition(left, right):
        out = np.stack([np.bincount(ids[left:right, layer].ravel(), minlength=256)
                        for layer in range(40)]).astype(float) / (right - left)
        return out
    composition_now = composition(boundary - window, boundary)
    selected = ids[boundary - window:boundary]
    overlap = np.any(selected[:-1, :, :, None] == selected[1:, :, None, :], axis=-1).sum(axis=-1)
    turnover = float(np.mean(1 - overlap / 8)) if len(selected) > 1 else None
    drift = velocity = acceleration = None
    if boundary >= 2 * window:
        previous = composition(boundary - 2 * window, boundary - window)
        drift = float(np.abs(composition_now - previous).sum(axis=-1).mean() / 16)
        velocity = (composition_now - previous) / window
        if boundary >= 3 * window:
            earlier = composition(boundary - 3 * window, boundary - 2 * window)
            acceleration = (composition_now - 2 * previous + earlier) / window ** 2
    executed = None
    if weights is not None:
        weights = np.asarray(weights)
        require(np.isfinite(weights[:boundary]).all() and (weights[:boundary] >= 0).all() and
                np.allclose(weights[:boundary].sum(axis=-1), 1, atol=1e-5), "executed weights are not normalized")
        executed = np.stack([np.bincount(selected[:, layer].ravel(),
                    weights=weights[boundary-window:boundary, layer].ravel(), minlength=256)
                    for layer in range(40)]) / window
    return {"status": "AVAILABLE", "window": window, "selection": composition_now,
            "executed_weight": executed, "turnover": turnover, "drift": drift,
            "velocity": velocity, "acceleration": acceleration,
            "router_probability_mass": None, "native_selection_boundary_gap": None,
            "confidence_status": "UNAVAILABLE_NATIVE_GATE_DISTRIBUTION_NOT_CAPTURED"}


def predictive_ablation(rows, baseline, static, temporal, shuffled, *, regularization=(.01, .1, 1.)):
    """Four held-out family folds with inner family-fold tuning and selection.

    Baseline must include prefix text features, position, state history and run
    age, computed without future data. All scale/feature/C fitting is local to
    training families. Order-shuffled routing preserves each window's IDs.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, brier_score_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .study_v1 import fold
    y = np.array([r["outcome"] for r in rows], dtype=int)
    families = np.array([r["family"] for r in rows])
    folds = np.array([fold(f) for f in families])
    arrays = [np.asarray(x, dtype=float) for x in (baseline, static, temporal, shuffled)]
    require(all(x.ndim == 2 and len(x) == len(rows) and np.isfinite(x).all() for x in arrays),
            "invalid predictive feature alignment")
    require(all(r.get("baseline_contract") == "prefix_text_position_state_history_run_age" for r in rows),
            "baseline contract is unqualified")
    variants = {"baseline": arrays[0], "composition": np.column_stack(arrays[:2]),
                "temporal": np.column_stack(arrays[:3]),
                "order_shuffled": np.column_stack([arrays[0], arrays[1], arrays[3]])}
    results = []
    for name, X in variants.items():
        probabilities = np.full(len(rows), np.nan)
        selections = []
        for held in range(4):
            train, test = folds != held, folds == held
            if not test.any() or len(set(y[train])) < 2:
                continue
            candidates = []
            def model(c):
                return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(64, X.shape[1])),
                                     LogisticRegression(C=c, max_iter=500, random_state=20261006))
            for c in regularization:
                losses = []
                for inner in sorted(set(folds[train])):
                    fit, valid = train & (folds != inner), train & (folds == inner)
                    if not valid.any() or len(set(y[fit])) < 2:
                        continue
                    fitted = model(c).fit(X[fit], y[fit])
                    losses.append(log_loss(y[valid], fitted.predict_proba(X[valid])[:, 1], labels=[0, 1]))
                if losses:
                    candidates.append((float(np.mean(losses)), c))
            if not candidates:
                continue
            c = min(candidates)[1]
            fitted = model(c).fit(X[train], y[train])
            probabilities[test] = fitted.predict_proba(X[test])[:, 1]
            selections.append({"held_fold": held, "C": c,
                "training_families": sorted(set(families[train])), "held_families": sorted(set(families[test])),
                "selected_features": fitted[1].get_support(indices=True).tolist()})
        valid = np.isfinite(probabilities)
        # Loss averages seeds/rows inside family before equal family weights.
        loss, brier = [], []
        for family in sorted(set(families[valid])):
            mask = valid & (families == family)
            loss.append(log_loss(y[mask], probabilities[mask], labels=[0, 1]))
            brier.append(brier_score_loss(y[mask], probabilities[mask]))
        results.append({"variant": name, "held_out_log_loss": float(np.mean(loss)) if loss else None,
            "brier": float(np.mean(brier)) if brier else None, "evaluated_families": len(loss),
            "folds": selections, "status": "AVAILABLE" if valid.all() else "UNAVAILABLE_INCOMPLETE_FOLDS"})
    return results
