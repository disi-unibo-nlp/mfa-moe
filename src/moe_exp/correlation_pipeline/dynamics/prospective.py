"""Nested question-held-out prediction with explicit decision-time feature blocks.

Retrospective labels are targets only. A deployable state readout requires
prefix-only training labels and is itself cross-fitted inside EVERY design fit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from .common import CLASSES
from .contracts import require_development, validate_feature
from .evaluation import _thresholds, bh_adjust, grouped_folds, routing_features
from .contrasts import joint_question_draws, cluster_estimate

LEVELS = ("position_dataset", "prefix_text", "prefix_state", "static_routing", "ordered_routing")


def weights(rows, cohort="A"):
    """Equal question weight; inverse B arm probability only within eligible questions.

    A is already one uniform draw per question; reweighting it by attempt count
    would overrepresent repeated datasets in a question-balanced population.
    """
    raw = []
    for row in rows:
        member = next((m for m in row["memberships"] if m["cohort"] == cohort), None)
        if member is None:
            raise ValueError("Evaluation population contains another cohort")
        p = member.get("arm_inclusion_probability") if cohort == "B" else 1.
        if p is None or not 0 < p <= 1:
            raise ValueError("Invalid inclusion probability")
        raw.append(1 / p)
    total = defaultdict(float)
    for r, w in zip(rows, raw):
        total[r["question_id"]] += w
    result = np.array([w / total[r["question_id"]] for r, w in zip(rows, raw)])
    return result / result.sum()


def _fit(a, y, b, *, penalty, sample_weights, classes=None):
    if classes is None:
        regularizer = np.eye(a.shape[1]) * penalty
        regularizer[0, 0] = 0
        coefficient = np.linalg.lstsq(a.T @ (sample_weights[:, None] * a) + regularizer,
                                     a.T @ (sample_weights * y), rcond=None)[0]
        return b @ coefficient
    from scipy.optimize import minimize
    target = np.eye(classes)[y.astype(int)]
    def objective(flat):
        beta = flat.reshape(a.shape[1], classes)
        logits = a @ beta
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        regularized = beta.copy()
        regularized[0] = 0
        loss = -(sample_weights[:, None] * target * np.log(np.maximum(p, 1e-15))).sum()
        gradient = a.T @ (sample_weights[:, None] * (p - target)) + penalty * regularized
        return loss + penalty / 2 * (regularized ** 2).sum(), gradient.ravel()
    result = minimize(objective, np.zeros(a.shape[1] * classes), jac=True, method="L-BFGS-B",
                      options={"maxiter": 400, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError("Predictive fit failed: " + str(result.message))
    logits = b @ result.x.reshape(a.shape[1], classes)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def _matrix(train, test):
    columns = sorted(set().union(*(r.keys() for r in train)))
    a = np.array([[r.get(c, 0.) for c in columns] for r in train], float)
    b = np.array([[r.get(c, 0.) for c in columns] for r in test], float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite feature")
    mean, scale = a.mean(0), a.std(0)
    keep = scale > 1e-10
    return (np.c_[np.ones(len(a)), (a[:, keep] - mean[keep]) / scale[keep]],
            np.c_[np.ones(len(b)), (b[:, keep] - mean[keep]) / scale[keep]],
            dict(columns=[c for c, k in zip(columns, keep) if k], mean=mean[keep].tolist(), scale=scale[keep].tolist()))


def _state_probabilities(train, test, cohort):
    """Train-only state model with out-of-question predictions on its own training rows."""
    if any(r.get("state_label_provenance") != "prefix_only" or r.get("state_target") not in CLASSES for r in train):
        raise ValueError("Prefix state readout requires prefix-only labels")
    groups = [r["question_id"] for r in train]
    if len(set(groups)) < 3:
        raise ValueError("Insufficient questions for cross-fitted state readout")
    state_y = np.array([CLASSES.index(r["state_target"]) for r in train])
    predictions = np.zeros((len(train), len(CLASSES)))
    for tr, va in grouped_folds(groups, 3, 42):
        training, validation = [train[i] for i in tr], [train[i] for i in va]
        a, b, _ = _matrix([r["text"] for r in training], [r["text"] for r in validation])
        predictions[va] = _fit(a, state_y[tr], b, penalty=1., sample_weights=weights(training, cohort), classes=len(CLASSES))
    a, b, audit = _matrix([r["text"] for r in train], [r["text"] for r in test])
    heldout = _fit(a, state_y, b, penalty=1., sample_weights=weights(train, cohort), classes=len(CLASSES))
    return predictions, heldout, audit


def estimated_histories(rows, probabilities):
    """History of earlier prefix readouts, never retrospective annotation history."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["attempt_id"]].append(i)
    result = [{} for _ in rows]
    for indices in groups.values():
        indices.sort(key=lambda i: rows[i]["decision_token"])
        past, last_class, run_start, switches, visited = [], None, None, 0, set()
        last_endpoint = -1
        for i in indices:
            endpoint = rows[i]["decision_token"]
            if endpoint <= last_endpoint:
                raise ValueError("Duplicate prefix readout endpoint")
            last_endpoint = endpoint
            current = int(probabilities[i].argmax())
            recurrence = current in visited and last_class != current
            if current != last_class:
                run_start = endpoint
                switches += int(last_class is not None)
            visited.add(current)
            past.append(probabilities[i])
            result[i] = {"estimated_state_" + c: float(p) for c, p in zip(CLASSES, probabilities[i])}
            result[i].update({"estimated_history_" + c: float(p) for c, p in zip(CLASSES, np.mean(past, 0))})
            result[i].update(estimated_run_tokens=endpoint - run_start,
                             estimated_switches=switches, estimated_recurrence=int(recurrence),
                             state_observations=len(past))
            last_class = current
    return result


def design(train, test, level, *, cohort="A", state_enabled=False, min_questions=10, shuffled=False):
    """Fit screening, scaling, gaps and state readouts on training questions only."""
    threshold = _thresholds(train) if level >= 3 else {}
    support = defaultdict(set)
    if level >= 3:
        for row in train:
            for w in row["routing_windows"]:
                for e, rate in w["expert_rates"].items():
                    if rate > 0:
                        support[f"expert_L{w['layer']}_W{w['window']}_{e}"].add(row["question_id"])
    expert_columns = set(sorted((k for k, q in support.items() if len(q) >= min_questions),
                               key=lambda k: (-len(support[k]), k))[:256])
    def features(row):
        validate_feature(row)
        values = dict(row["base"])
        if level >= 1:
            values.update(row["text"])
        if level >= 3:
            routing = routing_features(row["routing_windows"], expert_columns)
            values.update(routing["static"])
            for w in row["routing_windows"]:
                prefix = f"L{w['layer']}_W{w['window']}_"
                for expert, v in w.get("expert_weight_occupancy", {}).items():
                    if f"expert_{prefix}{expert}" in expert_columns:
                        values[f"occupancy_{prefix}{expert}"] = v
                gap_values = w.get("boundary_gaps")
                cutoff = threshold.get(f"L{w['layer']}")
                if gap_values and cutoff is not None:
                    weak = float(np.mean(np.array(gap_values) <= cutoff))
                    values[prefix + "weak_boundary_fraction"] = weak
                    if level >= 4:
                        overlap = w.get("overlap_shuffled_lag1") if shuffled else w.get("overlap_lag1")
                        if overlap is not None:
                            values[prefix + "weak_x_turnover"] = weak * (1 - overlap)
            if level >= 4:
                values.update(routing["shuffled"] if shuffled else routing["temporal"])
        return {k: float(v) for k, v in values.items() if v is not None}
    tr, te = [features(r) for r in train], [features(r) for r in test]
    state_audit = None
    if level >= 2 and state_enabled:
        p, q, state_audit = _state_probabilities(train, test, cohort)
        for dictionaries, originals, predictions in ((tr, train, p), (te, test, q)):
            for row, history in zip(dictionaries, estimated_histories(originals, predictions)):
                row.update(history)
    a, b, audit = _matrix(tr, te)
    audit.update(thresholds=threshold, expert_columns=sorted(expert_columns), state_readout=state_audit,
                 training_questions=sorted({r["question_id"] for r in train}))
    return a, b, audit


def _loss(y, prediction, regression):
    return (y - prediction) ** 2 if regression else -np.log(np.maximum(prediction[np.arange(len(y)), y.astype(int)], 1e-15))


def evaluate(rows, folds, *, target="is_correct", cohort="A", bootstrap=1000, min_questions=10):
    require_development(rows, folds)
    if cohort not in ("A", "B"):
        raise ValueError("Evaluate A and eligible B separately")
    all_rows = [r for r in rows if any(m["cohort"] == cohort for m in r["memberships"])]
    data = [r for r in all_rows if r.get(target) is not None]
    regression = target == "remaining_tokens"
    labels = None if regression else sorted(set(r[target] for r in data))
    questions = {r["question_id"] for r in data}
    if len(questions) < max(12, min_questions) or (not regression and len(labels) < 2):
        return dict(status="unsupported", questions=len(questions), target=target)
    if len({r["model"] for r in data}) != 1:
        raise ValueError("Fit model-specific readouts; share folds and resampling across models")
    for r in data:
        validate_feature(r)
    y = np.array([r[target] if regression else labels.index(r[target]) for r in data])
    state_enabled = all(r.get("state_label_provenance") == "prefix_only" and r.get("state_target") in CLASSES for r in data)
    variants = [(name, level, False) for level, name in enumerate(LEVELS) if level != 2 or state_enabled]
    variants.append(("shuffled_order", 4, True))
    results, predictions = {}, {}
    if not state_enabled:
        results["prefix_state"] = dict(status="unavailable", reason="prefix-only state labels/readout not qualified")
    outer = folds["payload"]["outer"]
    for name, level, shuffled in variants:
        output = np.zeros(len(data)) if regression else np.zeros((len(data), len(labels)))
        audits = []
        for f in range(4):
            train = [i for i, r in enumerate(data) if outer[r["question_id"]] != f]
            test = [i for i, r in enumerate(data) if outer[r["question_id"]] == f]
            if not train or not test:
                return dict(status="unsupported", reason="empty frozen outer fold")
            inner = folds["payload"]["inner"][str(f)]
            scores = defaultdict(list)
            for j in range(3):
                it = [i for i in train if inner[data[i]["question_id"]] != j]
                iv = [i for i in train if inner[data[i]["question_id"]] == j]
                if not it or not iv:
                    return dict(status="unsupported", reason="empty frozen inner fold")
                tr, va = [data[i] for i in it], [data[i] for i in iv]
                a, b, _ = design(tr, va, level, cohort=cohort, state_enabled=state_enabled,
                                 min_questions=min_questions, shuffled=shuffled)
                for penalty in (.01, .1, 1., 10.):
                    p = _fit(a, y[it], b, penalty=penalty, sample_weights=weights(tr, cohort),
                             classes=None if regression else len(labels))
                    scores[penalty].append(float(weights(va, cohort) @ _loss(y[iv], p, regression)))
            penalty = min(scores, key=lambda p: (np.mean(scores[p]), p))
            tr, te = [data[i] for i in train], [data[i] for i in test]
            a, b, audit = design(tr, te, level, cohort=cohort, state_enabled=state_enabled,
                                 min_questions=min_questions, shuffled=shuffled)
            output[test] = _fit(a, y[train], b, penalty=penalty, sample_weights=weights(tr, cohort),
                                classes=None if regression else len(labels))
            audits.append(dict(fold=f, penalty=penalty, **audit))
        predictions[name] = output
        w = weights(data, cohort)
        if regression:
            metrics = dict(mse=float(w @ ((y - output) ** 2)), mae=float(w @ abs(y - output)))
        else:
            calibration = []
            for c in range(len(labels)):
                for j in range(10):
                    mask = (output[:, c] >= j / 10) & ((output[:, c] < (j + 1) / 10) | ((j == 9) & (output[:, c] == 1)))
                    if mask.any():
                        ww = w[mask] / w[mask].sum()
                        calibration.append(dict(label=labels[c], bin=j, weight=float(w[mask].sum()),
                            predicted=float(ww @ output[mask, c]), observed=float(ww @ (y[mask] == c))))
            metrics = dict(log_loss=float(w @ _loss(y, output, False)),
                brier=float(w @ ((output - np.eye(len(labels))[y.astype(int)]) ** 2).sum(1)), calibration=calibration)
        results[name] = dict(status="predictive_evaluation", folds=audits, **metrics)
    comparisons = []
    sequence = [name for name, _, _ in variants if name != "shuffled_order"]
    draws = joint_question_draws(folds["payload"]["outer"], bootstrap=bootstrap)
    for baseline, added in [*zip(sequence, sequence[1:]), ("shuffled_order", "ordered_routing")]:
        delta = _loss(y, predictions[baseline], regression) - _loss(y, predictions[added], regression)
        bucket = defaultdict(list)
        for i, r in enumerate(data):
            bucket[r["question_id"]].append(i)
        effect = {q: float(weights([data[i] for i in idx], cohort) @ delta[idx]) for q, idx in bucket.items()}
        comparisons.append(dict(baseline=baseline, added=added, **cluster_estimate(effect, draws, min_questions=min_questions)))
    for row, q in zip(comparisons, bh_adjust([r["p"] for r in comparisons])):
        row["q_bh"] = q
    return dict(status="exploratory_predictive", target=target, labels=labels, cohort=cohort,
        population="representative_A" if cohort == "A" else "B_eligible_only", questions=len(questions),
        coverage=len(data) / max(1, len(all_rows)), models=results, comparisons=comparisons,
        dynamic_trigger_qualified=False,
        trigger_reason="Requires held-out trigger threshold, state-history baseline and useful-lead-time validation",
        predictions=[dict(question_id=r["question_id"], attempt_id=r["attempt_id"],
            decision_token=r["decision_token"], target=r[target], remaining_tokens=r.get("remaining_tokens"),
            predictions={name: p[i].tolist() for name, p in predictions.items()}) for i, r in enumerate(data)])
