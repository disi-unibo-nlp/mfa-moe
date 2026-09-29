"""Prespecified native-expert contrasts with question-balanced estimands.

Each question is first averaged over matched position/state strata. All experts,
including zeros, enter one model/contrast BH family for both primary estimands.
Retrospective contrasts nominate mechanisms, never online observations.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .contracts import require_development, validate_feature
from .evaluation import bh_adjust

CONTRASTS = ("accuracy", "efficient_success", "short_failure", "long_failure",
             "state_exit", "destination", "recovery")


def joint_question_draws(questions, *, bootstrap=1000, seed=42):
    """Use the SAME draws across models. Absent questions have zero contribution."""
    names = sorted(set(questions))
    if not names or bootstrap < 1:
        raise ValueError("Question bootstrap requires data and positive replicates")
    rng = np.random.default_rng(seed)
    return names, rng.integers(0, len(names), size=(bootstrap, len(names)))


def cluster_estimate(by_question, draws, *, min_questions=10):
    names, indices = draws
    values = np.array([by_question.get(q, np.nan) for q in names], dtype=float)
    supported = values[np.isfinite(values)]
    if not len(supported):
        return dict(effect=None, questions=0, ci=None, p=None, standardized_effect=None)
    n = len(supported)
    effect = float(supported.mean())
    sampled = values[indices]
    counts = np.isfinite(sampled).sum(1)
    boot = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
    # Null-centred bootstrap; unlike sign-tail counting this tests the null distribution.
    centered = np.abs(boot - effect)
    p = float((1 + (centered >= abs(effect)).sum()) / (len(boot) + 1)) if n >= min_questions else None
    std = float(supported.std(ddof=1)) if n > 1 else 0.
    return dict(effect=effect, questions=n, ci=np.quantile(boot, [.025, .975]).tolist(), p=p,
        standardized_effect=effect / std if std > 0 else None)


def _vector_estimates(values, draws, multiplicities, experts, min_questions):
    """One shared question bootstrap for every expert in a native layer.

    Multiplicity counts are exactly equivalent to indexing the same draws, but
    avoid allocating a replicates-by-global-questions array once per expert.
    """
    names, _ = draws
    columns = [i for i, q in enumerate(names) if q in values]
    if not columns:
        return [dict(effect=None, questions=0, ci=None, p=None,
                     standardized_effect=None) for _ in range(experts)]
    matrix = np.array([values[names[i]] for i in columns])
    if matrix.shape != (len(columns), experts) or not np.isfinite(matrix).all():
        raise ValueError("Invalid question expert contrasts")
    selected = multiplicities[:, columns]
    counts = selected.sum(1)
    boot = (selected @ matrix)[counts > 0] / counts[counts > 0, None]
    effect = matrix.mean(0)
    ci = np.quantile(boot, [.025, .975], axis=0)
    p = (1 + (np.abs(boot - effect) >= np.abs(effect)).sum(0)) / (len(boot) + 1)
    std = matrix.std(0, ddof=1) if len(columns) > 1 else np.zeros(experts)
    return [dict(effect=float(effect[e]), questions=len(columns), ci=ci[:, e].tolist(),
        p=float(p[e]) if len(columns) >= min_questions else None,
        standardized_effect=float(effect[e] / std[e]) if std[e] > 0 else None)
        for e in range(experts)]


def _matched_rows(rows, contrast, *, destination=None):
    if contrast not in CONTRASTS:
        raise ValueError("Unknown contrast")
    lengths = defaultdict(set)
    for r in rows:
        if r.get("is_correct") is True:
            lengths[(r["model"], r["question_id"])].add(r["completion_tokens"])
    for r in rows:
        validate_feature(r)
        match = [r["decision_token"]]
        arm = None
        if contrast == "accuracy":
            arm = r.get("is_correct")
        elif contrast == "efficient_success" and r.get("is_correct") is True:
            choices = sorted(lengths[(r["model"], r["question_id"])])
            if len(choices) > 1:
                midpoint = (choices[0] + choices[-1]) / 2
                if r["completion_tokens"] != midpoint:
                    arm = r["completion_tokens"] < midpoint
        elif contrast in ("short_failure", "long_failure"):
            if r.get("length_band") == contrast.split("_")[0] and r.get("is_correct") is not None:
                arm = not r["is_correct"]  # wrong minus correct within a frozen length band
        elif contrast in ("state_exit", "destination"):
            source, following = r.get("source_class"), r.get("next_class")
            if source and following and r.get("adjacent_labels") is True:
                if r.get("destination_start_token") is None:
                    raise ValueError("Transition needs destination timing")
                match.extend([source, r.get("run_age_bin")])
                if r.get("run_age_bin") is None:
                    continue
                if contrast == "state_exit":
                    arm = source != following
                elif following != source and destination is not None:
                    arm = following == destination
        elif contrast == "recovery" and r.get("history_signature") and r.get("is_recurrence"):
            arm = r.get("is_correct")
            match.extend([r.get("source_class"), tuple(r["history_signature"])])
        if arm is not None:
            yield r, tuple(match), bool(arm)


def estimate_contrast(rows, native_specs, contrast, *, destination=None, bootstrap=1000,
                      seed=42, min_questions=10, draws=None):
    """Binary selection probability and normalized executed weight, not slot share.

    Windows/positions are matching strata. They are averaged within question,
    never counted as independent observations. Unknown weights remain missing.
    """
    cells = defaultdict(lambda: defaultdict(lambda: {False: [], True: []}))
    for row, match, arm in _matched_rows(rows, contrast, destination=destination):
        spec = native_specs[row["model"]]
        for w in row["routing_windows"]:
            if w["layer"] not in spec["layers"]:
                raise ValueError("Non-native layer in contrast")
            for metric, field in (("selection_probability", "expert_rates"),
                                  ("weight_occupancy", "expert_weight_occupancy")):
                if field not in w:
                    continue
                vector = np.array([w[field].get(str(e), 0.) for e in range(spec["num_experts"])])
                expected = spec["top_k"] if metric == "selection_probability" else 1.
                if (vector < 0).any() or (vector > 1 + 1e-6).any() or not np.isclose(vector.sum(), expected, atol=1e-5):
                    raise ValueError("Expert rates have wrong estimand/normalization")
                key = (row["model"], w["layer"], w["window"], metric)
                cells[key][(row["question_id"], match)][arm].append(vector)
    if draws is None:
        draws = joint_question_draws([r["question_id"] for r in rows], bootstrap=bootstrap, seed=seed)
    multiplicities = np.stack([np.bincount(index, minlength=len(draws[0])) for index in draws[1]])
    result = []
    for (model, layer, window, metric), strata in sorted(cells.items()):
        per_question = defaultdict(list)
        arm_means = defaultdict(list)
        for (question, _), arms in strata.items():
            if arms[True] and arms[False]:
                positive, negative = np.mean(arms[True], 0), np.mean(arms[False], 0)
                per_question[question].append(positive - negative)
                arm_means[question].append((positive, negative))
        values = {q: np.mean(v, 0) for q, v in per_question.items()}
        estimates = _vector_estimates(values, draws, multiplicities,
                                      native_specs[model]["num_experts"], min_questions)
        for expert in range(native_specs[model]["num_experts"]):
            effects = {q: float(v[expert]) for q, v in values.items()}
            estimate = estimates[expert]
            secondary = {}
            if arm_means and metric == "selection_probability":
                positive, negative = np.mean([np.mean(v, axis=0)[:, expert] for v in arm_means.values()], axis=0)
                k = native_specs[model]["top_k"]
                joint, marginal = .5 * positive, .5 * (positive + negative)
                secondary = dict(positive_rate=float(positive), negative_rate=float(negative),
                    slot_share_difference=estimate["effect"] / k,
                    slot_share_enrichment=float(positive / negative) if negative > 0 else None,
                    binary_log_odds=float(np.log(positive / (1 - positive)) - np.log(negative / (1 - negative)))
                        if 0 < positive < 1 and 0 < negative < 1 else None,
                    binary_npmi=float(np.log(joint / (.5 * marginal)) / -np.log(joint))
                        if 0 < joint < 1 and marginal > 0 else None)
            result.append(dict(model=model, contrast=contrast, destination=destination, layer=layer,
                window=window, expert=expert, metric=metric, **estimate,
                question_effects=effects, secondary=secondary,
                status="descriptive" if estimate["questions"] < min_questions else "association",
                family=f"{model}/{contrast}/{destination or 'all'}/all_layers_experts_windows_primary_metrics"))
    families = defaultdict(list)
    for row in result:
        families[row["family"]].append(row)
    for family in families.values():
        for row, q in zip(family, bh_adjust([r["p"] for r in family])):
            row["q_bh"] = q
    return result


def retrospective_length_bands(rows, training_questions):
    """Training-only median lengths for premature-failure discovery controls.

    Final length is a retrospective stratifier, NEVER a deployable feature. Each
    attempt is counted once even if it supplies several overlapping landmarks.
    """
    lengths = defaultdict(dict)
    for r in rows:
        if r["question_id"] in training_questions:
            lengths[(r["model"], r["dataset"])][r["attempt_id"]] = r["completion_tokens"]
    cutoffs = {key: float(np.median(list(v.values()))) for key, v in lengths.items()}
    output = []
    for row in rows:
        cutoff = cutoffs.get((row["model"], row["dataset"]))
        band = None if cutoff is None or row["completion_tokens"] == cutoff else (
            "short" if row["completion_tokens"] < cutoff else "long")
        output.append({**row, "length_band": band, "length_band_provenance": "retrospective_training_median"})
    return output, cutoffs


def candidate_cards(rows, native_specs, folds, *, contrast="accuracy", min_questions=10,
                    max_groups=3, bootstrap=1000):
    """Compact singleton groups; train-only selection and held-out direction checks.

    Expanding to co-selection groups requires independent evidence. Rank cap and
    support are applied within each training fold, never to held-out outcomes.
    """
    require_development(rows, folds)
    selected = defaultdict(list)
    mapping = folds["payload"]["outer"]
    for fold in range(4):
        train = [r for r in rows if mapping[r["question_id"]] != fold]
        test = [r for r in rows if mapping[r["question_id"]] == fold]
        if not train or not test:
            continue
        estimates = estimate_contrast(train, native_specs, contrast, bootstrap=bootstrap,
                                      min_questions=min_questions)
        heldout = estimate_contrast(test, native_specs, contrast, bootstrap=bootstrap,
                                    min_questions=min_questions)
        key = lambda r: (r["model"], r["layer"], r["window"], r["expert"], r["metric"])
        checks = {key(r): r for r in heldout}
        for model in native_specs:
            eligible = [r for r in estimates if r["model"] == model
                        and r["metric"] == "selection_probability" and r["questions"] >= min_questions
                        and r["effect"] and r["q_bh"] is not None and r["q_bh"] <= .05]
            eligible.sort(key=lambda r: (-r["questions"], -abs(r["standardized_effect"] or 0.), r["layer"], r["expert"]))
            for r in eligible[:max_groups]:
                check = checks.get(key(r))
                if check and check["questions"] > 0:
                    selected[key(r)].append(dict(fold=fold, training_effect=r["effect"],
                        heldout_effect=check["effect"], training_questions=r["questions"],
                        heldout_questions=check["questions"],
                        standardized_effect=r["standardized_effect"],
                        agrees=bool(r["effect"] * check["effect"] > 0)))
    cards = []
    for (model, layer, window, expert, metric), checks in selected.items():
        # At least three folds must agree in the SAME signed direction.
        positive = sum(c["agrees"] and c["training_effect"] > 0 for c in checks)
        negative = sum(c["agrees"] and c["training_effect"] < 0 for c in checks)
        stable = max(positive, negative)
        cards.append(dict(model=model, layer=layer, experts=[expert], window=window,
            contrast=contrast, metric=metric, stable_folds=stable,
            direction=1 if positive >= negative else -1, folds=checks,
            minimum_training_questions=min(c["training_questions"] for c in checks),
            standardized_effect=float(np.mean([abs(c["standardized_effect"] or 0.) for c in checks])),
            eligible_for_mechanism_pilot=stable >= 3,
            claim="descriptive_association_not_action_benefit"))
    cards.sort(key=lambda r: (-r["stable_folds"], -r["minimum_training_questions"],
                              -r["standardized_effect"], r["layer"], r["experts"]))
    counts, final = defaultdict(int), []
    for row in cards:
        if counts[row["model"]] < max_groups:
            final.append(row)
            counts[row["model"]] += 1
    return final


def sibling_sequence_controls(rows, *, seed=42):
    """Exchange whole matched prefix routing sequences between sibling attempts.

    No donor may cross question/model/position coverage. A deterministic cyclic
    permutation has no fixed points; singletons are explicitly unsupported.
    """
    from .common import digest
    attempts = defaultdict(list)
    for row in rows:
        attempts[(row["model"], row["question_id"], row["attempt_id"])].append(row)
    groups = defaultdict(list)
    for key, sequence in attempts.items():
        sequence.sort(key=lambda r: r["decision_token"])
        signature = tuple((r["decision_token"], tuple(sorted((w["layer"], w["window"], w["end_token"])
                          for w in r["routing_windows"]))) for r in sequence)
        groups[(key[0], key[1], signature)].append(key)
    result = []
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda k: digest([seed, *k]))
        for i, target in enumerate(group):
            donor = group[(i + 1) % len(group)]
            for row, control in zip(attempts[target], attempts[donor]):
                result.append({**row, "routing_windows": control["routing_windows"],
                               "control_donor": donor[-1]})
    return result
