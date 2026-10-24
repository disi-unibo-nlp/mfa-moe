"""Question-grouped nested evaluation; every data-dependent transform is fitted in-fold."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import math
import numpy as np
from .common import CLASSES


def grouped_folds(groups, n_splits, seed=42):
    unique = sorted(set(groups))
    if len(unique) < n_splits:
        raise ValueError("Insufficient questions for grouped folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {g: i % n_splits for i, g in enumerate(unique)}
    groups = np.asarray(groups)
    return [(np.flatnonzero([mapping[g] != f for g in groups]),
             np.flatnonzero([mapping[g] == f for g in groups])) for f in range(n_splits)]


def bh_adjust(pvalues):
    result = [None] * len(pvalues)
    order = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    q = 1.
    for rank in range(len(order), 0, -1):
        p, i = order[rank - 1]
        q = min(q, p * len(order) / rank)
        result[i] = q
    return result


def text_features(text, bins=32):
    """Fixed signed hash of prefix/current text: no corpus vocabulary or future labels."""
    import re
    features = Counter()
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    for word in words:
        h = hashlib.sha256(word.encode()).digest()
        features["text_" + str(int.from_bytes(h[:4], "big") % bins)] += 1 if h[4] % 2 else -1
    scale = max(1, len(words))
    return {**{k: v / scale for k, v in features.items()}, "text_chars": len(text),
            "text_words": len(words), "text_digit_fraction": sum(c.isdigit() for c in text) / max(1, len(text))}


def _thresholds(rows):
    samples = defaultdict(dict)
    for original in rows:
        row = _materialize(original, expert_columns=set())
        for layer, values in row.get("gaps", {}).items():
            for token, value in values:
                samples[layer.split("_W")[0]][(row["trace_id"], token)] = value
    return {layer: float(np.quantile(list(values.values()), .1))
            for layer, values in samples.items() if values}


def _features(row, level, thresholds, shuffled=False, expert_columns=None):
    row = _materialize(row, expert_columns=expert_columns)
    result = dict(row["base"])
    if level >= 1:
        result.update(row["static"])
    if level >= 2:
        result.update(row["shuffled"] if shuffled else row["temporal"])
        for layer, values in row.get("gaps", {}).items():
            threshold = thresholds.get(layer.split("_W")[0])
            if threshold is not None and values:
                result["weak_boundary_" + layer] = float(np.mean([v <= threshold for _, v in values]))
    return {k: float(v) for k, v in result.items() if v is not None and math.isfinite(float(v))}


def _design(train, test, level, min_problems, shuffled=False):
    thresholds = _thresholds(train) if level >= 2 else {}
    expert_support = defaultdict(set)
    if level >= 1:
        for row in train:
            for window in row.get("routing_windows", []):
                prefix = f"expert_L{window['layer']}_W{window['window']}_"
                for expert, rate in window["expert_rates"].items():
                    if rate > 0:
                        expert_support[prefix + expert].add(row["question_id"])
    screened = {k for k in sorted(expert_support, key=lambda k: (-len(expert_support[k]), k))
                if len(expert_support[k]) >= min_problems}
    screened = set(sorted(screened, key=lambda k: (-len(expert_support[k]), k))[:256])
    train_dict = [_features(r, level, thresholds, shuffled, screened) for r in train]
    test_dict = [_features(r, level, thresholds, shuffled, screened) for r in test]
    support = defaultdict(set)
    for row, features in zip(train, train_dict):
        for key, value in features.items():
            if value != 0:
                support[key].add(row["question_id"])
    columns = sorted(k for k in set().union(*(r.keys() for r in train_dict))
                     if not k.startswith("expert_") or len(support[k]) >= min_problems)
    # Bound model size using training-only support, with deterministic ties.
    experts = sorted((k for k in columns if k.startswith("expert_")),
                     key=lambda k: (-len(support[k]), k))[:256]
    columns = [k for k in columns if not k.startswith("expert_") or k in set(experts)]
    if not columns:
        columns = ["intercept_only"]
    a = np.array([[r.get(k, 0.) for k in columns] for r in train_dict])
    b = np.array([[r.get(k, 0.) for k in columns] for r in test_dict])
    mean, std = a.mean(0), a.std(0)
    keep = std > 1e-10
    a, b = (a[:, keep] - mean[keep]) / std[keep], (b[:, keep] - mean[keep]) / std[keep]
    return np.c_[np.ones(len(a)), a], np.c_[np.ones(len(b)), b], dict(
        columns=[k for k, use in zip(columns, keep) if use], thresholds=thresholds)


def _fit_predict(a, y, b, classes, penalty, groups):
    """Weighted multinomial ridge with unpenalized intercept, via SciPy."""
    from scipy.optimize import minimize
    group_counts = Counter(groups)
    weights = np.array([1 / group_counts[g] for g in groups], dtype=float)
    weights /= weights.sum()
    target = np.eye(classes)[y]
    def objective(flat):
        coef = flat.reshape(a.shape[1], classes)
        logits = a @ coef
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        loss = -(weights[:, None] * target * np.log(np.maximum(p, 1e-15))).sum()
        regularized = coef.copy()
        regularized[0] = 0
        grad = a.T @ (weights[:, None] * (p - target)) + penalty * regularized
        return loss + penalty / 2 * np.square(regularized).sum(), grad.ravel()
    fit = minimize(objective, np.zeros(a.shape[1] * classes), jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-9})
    if not fit.success:
        raise RuntimeError("Predictive optimizer did not converge: " + fit.message)
    logits = b @ fit.x.reshape(a.shape[1], classes)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def losses(y, predictions):
    return -np.log(np.maximum(predictions[np.arange(len(y)), y], 1e-15))


def question_mean(values, groups):
    bucket = defaultdict(list)
    for value, group in zip(values, groups):
        bucket[group].append(value)
    return np.array([np.mean(bucket[g]) for g in sorted(bucket)])


def calibration(y, p, groups, bins=10):
    group_counts = Counter(groups)
    weights = np.array([1 / group_counts[g] for g in groups])
    weights /= weights.sum()
    result = []
    for c in range(p.shape[1]):
        for i in range(bins):
            mask = (p[:, c] >= i / bins) & ((p[:, c] < (i + 1) / bins) if i + 1 < bins else (p[:, c] <= 1))
            if mask.any():
                w = weights[mask] / weights[mask].sum()
                result.append(dict(class_index=c, bin=i, rows=int(mask.sum()),
                    predicted=float(w @ p[mask, c]), observed=float(w @ (y[mask] == c)),
                    weight=float(weights[mask].sum())))
    return result


def evaluate(rows, *, seed=42, bootstrap=1000, min_problems=10):
    groups = [r["question_id"] for r in rows]
    labels = sorted(set(r["target"] for r in rows))
    supports = {str(label): len({r["question_id"] for r in rows if r["target"] == label}) for label in labels}
    if len(set(groups)) < max(12, min_problems) or len(labels) < 2:
        return dict(status="unsupported", reason="insufficient question/outcome support",
                    questions=len(set(groups)), class_support=supports)
    y = np.array([labels.index(r["target"]) for r in rows])
    outer = grouped_folds(groups, 4, seed)
    variants = [("history_text", 0, False), ("static_routing", 1, False),
                ("temporal_routing", 2, False), ("shuffled_temporal", 2, True)]
    outputs, all_predictions = {}, {}
    routing_available = any(r.get("routing_windows") or r.get("static") for r in rows)
    if not routing_available:
        variants = variants[:1]
        outputs.update({name: {"status": "unavailable", "reason": "no routing features"}
                        for name in ("static_routing", "temporal_routing", "shuffled_temporal")})
    splits = []
    for fold, (train, test) in enumerate(outer):
        splits.append(dict(fold=fold, training_questions=sorted({groups[i] for i in train}),
                           test_questions=sorted({groups[i] for i in test})))
    for name, level, shuffled in variants:
        predictions = np.zeros((len(rows), len(labels)))
        fold_audits = []
        for fold, (train, test) in enumerate(outer):
            training, testing = [rows[i] for i in train], [rows[i] for i in test]
            inner = grouped_folds([r["question_id"] for r in training], 3, seed + fold + 1)
            # Transform once per inner split; no held-out values enter screening/scaling/thresholds.
            designs = []
            for it, iv in inner:
                tr, va = [training[i] for i in it], [training[i] for i in iv]
                a, b, _ = _design(tr, va, level, min_problems, shuffled)
                designs.append((a, b, y[train[it]], y[train[iv]],
                                [r["question_id"] for r in tr], [r["question_id"] for r in va]))
            candidates = []
            for penalty in (.01, .1, 1., 10.):
                scores = []
                for a, b, ty, vy, tg, vg in designs:
                    p = _fit_predict(a, ty, b, len(labels), penalty, tg)
                    scores.append(question_mean(losses(vy, p), vg).mean())
                candidates.append((float(np.mean(scores)), penalty))
            penalty = min(candidates)[1]
            a, b, audit = _design(training, testing, level, min_problems, shuffled)
            predictions[test] = _fit_predict(a, y[train], b, len(labels), penalty,
                                             [groups[i] for i in train])
            fold_audits.append(dict(fold=fold, penalty=penalty, **audit))
        all_predictions[name] = predictions
        brier = np.square(predictions - np.eye(len(labels))[y]).sum(1)
        outputs[name] = dict(log_loss=float(question_mean(losses(y, predictions), groups).mean()),
            brier=float(question_mean(brier, groups).mean()), calibration=calibration(y, predictions, groups),
            folds=fold_audits)
    rng, comparisons = np.random.default_rng(seed), []
    for baseline, added in (("history_text", "static_routing"), ("static_routing", "temporal_routing"),
                            ("shuffled_temporal", "temporal_routing")):
        if baseline not in all_predictions or added not in all_predictions:
            continue
        delta = question_mean(losses(y, all_predictions[baseline]) - losses(y, all_predictions[added]), groups)
        samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(bootstrap)])
        p = min(1., 2 * (min(int((samples <= 0).sum()), int((samples >= 0).sum())) + 1) / (bootstrap + 1))
        comparisons.append(dict(baseline=baseline, added=added, log_loss_improvement=float(delta.mean()),
            confidence_interval=np.quantile(samples, [.025, .975]).tolist(), p_bootstrap=p,
            status="descriptive"))
    return dict(status="descriptive", classes=labels, class_support=supports, questions=len(set(groups)),
        outer_splits=splits, models=outputs, comparisons=comparisons,
        predictions=[dict(trace_id=r["trace_id"], question_id=r["question_id"], target=r["target"],
                          probabilities={name: p[i].tolist() for name, p in all_predictions.items()})
                     for i, r in enumerate(rows)])


def routing_features(windows, expert_columns=None):
    static, temporal, shuffled, gaps = {}, {}, {}, {}
    for w in windows:
        prefix = f"L{w['layer']}_W{w['window']}_"
        static.update({f"expert_{prefix}{e}": v for e, v in w["expert_rates"].items()
                       if expert_columns is None or f"expert_{prefix}{e}" in expert_columns})
        static[prefix + "router_local_entropy"] = w.get("full_local_entropy")
        for name in ("full_marginal_entropy", "full_entropy_difference", "full_jsd",
                     "mixture_local_entropy", "mixture_marginal_entropy", "mixture_entropy_difference",
                     "mixture_jsd", "mixture_effective_experts", "boundary_gap_mean", "set_turnover"):
            # Entropy decomposition and mean boundary gap are order invariant.
            # Putting these in the temporal increment confounds variability with order.
            if name not in ("full_jsd", "mixture_jsd", "set_turnover"):
                static[prefix + name] = w.get(name)
                continue
            temporal[prefix + name] = w.get(name)
            replacement = name.replace("_jsd", "_shuffled_jsd") if name.endswith("_jsd") else name
            shuffled[prefix + name] = w.get(replacement)
        if w.get("overlap_shuffled_lag1") is not None:
            shuffled[prefix + "set_turnover"] = 1 - w["overlap_shuffled_lag1"]
        for lag in (1, 4, 16, 64):
            temporal[prefix + f"overlap_lag{lag}"] = w.get(f"overlap_lag{lag}")
            shuffled[prefix + f"overlap_lag{lag}"] = w.get(f"overlap_shuffled_lag{lag}")
            static[prefix + f"overlap_expectation_lag{lag}"] = w.get(f"overlap_shuffled_lag{lag}")
        if w.get("boundary_gaps") is not None:
            gaps[prefix] = list(zip(range(w["start_token"], w["end_token"]), w["boundary_gaps"]))
    return dict(static=static, temporal=temporal, shuffled=shuffled, gaps=gaps)


def latest_windows(windows, endpoint):
    latest = {}
    for w in windows:
        if w["end_token"] <= endpoint:
            key = (w["layer"], w["window"])
            if key not in latest or w["end_token"] > latest[key]["end_token"]:
                latest[key] = w
    return list(latest.values())


def prediction_rows(trace, sentences, windows, landmarks=(256, 512, 1024, 2048)):
    from .common import keys
    rows, shared = [], keys(trace)
    for sentence in sentences:
        if sentence["label"] is None or sentence["next_class"] is None or sentence["token_end"] is None:
            continue
        endpoint = sentence["token_end"]
        base = {**text_features(sentence["text"]), "token_position": endpoint,
                "sentence_position": sentence["sentence_index"],
                "run_age_sentences": sentence.get("run_age_sentences"),
                "run_age_tokens": sentence.get("run_age_tokens"),
                "class_" + sentence["label"]: 1,
                "previous_" + str(sentence["previous_class"]): 1}
        rows.append({**shared, "task": "oracle_next_class", "target": sentence["next_class"], "base": base,
                     "feature_provenance": "retrospective_labels_not_deployable",
                     "routing_windows": latest_windows(windows, endpoint)})
    replay = trace.metadata.get("token_replay")
    if replay and trace.is_correct is not None:
        offsets = replay["completion_offsets"]
        for endpoint in landmarks:
            if len(offsets) < endpoint:
                continue
            # Prefix text and prefix routing only. No sentence labels, final length or finish flag.
            end_char = max(b for _, b in offsets[:endpoint])
            base = {**text_features(trace.cot_text[:end_char]), "landmark": endpoint}
            rows.append({**shared, "task": f"correctness_prefix_{endpoint}", "target": int(trace.is_correct),
                         "base": base, "routing_windows": latest_windows(windows, endpoint)})
    return rows


def _materialize(row, expert_columns=None):
    if "routing_windows" not in row:
        return row
    return {**row, **routing_features(row["routing_windows"], expert_columns)}
