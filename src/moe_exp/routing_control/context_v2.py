"""Fold-local descriptive routing contexts; never fit on independent outcomes.

The context transform uses ID inclusion frequencies (sum eight per layer),
eight PCA components and K=1/2/4. Selected execution weights are deliberately
not mixed into this transform. All inner fits, supervised feature selection,
and regularization tuning consume training families only.
"""
from __future__ import annotations

import numpy as np

from .study_v1 import STATES, digest, fold, require

KS = (1, 2, 4)


def profiles(value):
    array = np.asarray(value, dtype=float)
    require(array.ndim == 3 and array.shape[1:] == (40, 256) and
            np.isfinite(array).all() and (array >= 0).all() and (array <= 1 + 1e-6).all() and
            np.allclose(array.sum(axis=2), 8, atol=1e-5), "invalid native ID composition")
    return array.reshape(len(array), -1)


def fit_context(value, families, k):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = profiles(value)
    require(k in KS and len(X) >= max(9, k) and len(families) == len(X),
            "eight-component PCA requires at least nine training rows")
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=8, svd_solver="randomized", random_state=20261006).fit(scaler.transform(X))
    Z = pca.transform(scaler.transform(X))
    cluster = KMeans(n_clusters=k, n_init=10, random_state=20261006).fit(Z)
    body = {"schema": "routing-context-transform-v2", "signal": "native_expert_ID_inclusion",
            "K": k, "components": 8, "training_families": sorted(set(families)),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
            "centroids": cluster.cluster_centers_.tolist(),
            "interpretation": "Descriptive routing contexts; no causal region or expert-function claim"}
    return {**body, "sha256": digest(body)}


def transform_context(model, value):
    require(model.get("sha256") == digest({k: v for k, v in model.items() if k != "sha256"}) and
            model.get("signal") == "native_expert_ID_inclusion" and model.get("components") == 8 and
            model.get("K") in KS, "changed routing-context transform")
    X = profiles(value)
    Z = ((X - np.asarray(model["mean"])) / np.asarray(model["scale"]) -
         np.asarray(model["pca_mean"])) @ np.asarray(model["pca_components"]).T
    distances = ((Z[:, None] - np.asarray(model["centroids"])[None]) ** 2).sum(axis=-1)
    labels = distances.argmin(axis=1)
    # Cluster-local PCA profiles let the predictive test detect changing local
    # associations, rather than only the additive effect of a cluster label.
    onehot = np.eye(model["K"])[labels]
    return labels, np.column_stack([Z, onehot, (Z[:, None] * onehot[:, :, None]).reshape(len(Z), -1)])


def context_associations(rows, value, contexts, *, direction_folds=3):
    """Compact support audit; associations and directions are family equal.

    Keep the strongest supported expert per source/status/context stratum.
    Multiple contexts require every occupied context to have a supported
    association, with ten positive AND ten comparison families.
    """
    from collections import defaultdict
    X = profiles(value)
    require(len(rows) == len(X) == len(contexts), "context association alignment differs")
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row["transition"], row["source_state"], row["prefix_status"], int(contexts[i])][row["family"]].append(i)
    result = []
    for (transition, state, status, context), families in sorted(groups.items()):
        positive, comparison, paired = set(), set(), {}
        for family, indices in families.items():
            p = [i for i in indices if rows[i]["outcome"]]
            n = [i for i in indices if not rows[i]["outcome"]]
            if p: positive.add(family)
            if n: comparison.add(family)
            if p and n: paired[family] = X[p].mean(axis=0) - X[n].mean(axis=0)
        if not paired:
            continue
        mean = np.mean(list(paired.values()), axis=0)
        held = [np.mean([v for f, v in paired.items() if fold(f) == k], axis=0)
                if any(fold(f) == k for f in paired) else None for k in range(4)]
        stable = np.sum([np.zeros(len(mean), bool) if h is None else h * mean > 0 for h in held], axis=0)
        # Inner training has only two of the four folds. Demand consistent
        # direction in every available inner fold; the outer/full fit still
        # requires three. Inner tuning must not make K>1 impossible by asking
        # for a held fold that was intentionally removed from its training.
        require(type(direction_folds) is int and direction_folds in (2, 3), "invalid fold direction rule")
        supported = (stable >= direction_folds) & (direction_folds >= 2) & (mean != 0) & (len(paired) >= 10)
        choices = np.flatnonzero(supported)
        expert_index = int(max(choices, key=lambda j: (abs(mean[j]), -j))) if len(choices) else int(np.argmax(abs(mean)))
        result.append({"transition": transition, "source_state": state, "prefix_status": status,
            "context": context, "layer": expert_index // 256, "expert": expert_index % 256,
            "association_difference": float(mean[expert_index]), "positive_families": len(positive),
            "comparison_families": len(comparison), "paired_families": len(paired),
            "consistent_folds": int(stable[expert_index]), "supported": bool(supported[expert_index])})
    return result


def _loss(y, p, families):
    from sklearn.metrics import log_loss
    return float(np.mean([log_loss(y[families == f], p[families == f], labels=[0, 1])
                          for f in sorted(set(families))]))


def nested_context_model(rows, baseline, value, *, family_boundary, regularization=(.01, .1, 1.)):
    """Nested family CV versus K=1; fallback unless three outer gains qualify.

    This assesses predictive context, not semantic prefix qualification. The
    baseline frame must already bind prefix text, position, state history and
    run age. A production gate additionally needs a discovery-frozen context
    acceptance rule and recognizer/worker qualification.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .families_v2 import require_discovery
    require_discovery([r["family"] for r in rows], family_boundary)
    require(rows and all(r.get("population") == "discovery" and type(r.get("outcome")) is bool and
            r.get("baseline_contract") == "prefix_text_position_state_history_run_age" and
            r.get("prediction_alignment_qualified") is True and r["source_state"] in STATES and
            r["prefix_status"] in ("intermediate", "complete_unchecked", "complete_checked", "unresolved")
            for r in rows), "unqualified prediction frame or family population")
    X = np.asarray(value, float)
    profiles(X)
    B = np.asarray(baseline, float)
    require(B.ndim == 2 and len(B) == len(rows) == len(X) and B.shape[1] > 0 and np.isfinite(B).all(),
            "invalid prefix baseline")
    require(regularization and all(c > 0 and np.isfinite(c) for c in regularization), "invalid regularization")
    F = np.array([r["family"] for r in rows])
    folds = np.array([fold(f) for f in F])
    y = np.array([r["outcome"] for r in rows], int)
    require(set(folds) == set(range(4)), "four outer family folds are required")

    def fit(train, test, k, c):
        model = fit_context(X[train], F[train].tolist(), k)
        contexts, train_features = transform_context(model, X[train])
        _, test_features = transform_context(model, X[test])
        associations = context_associations([r for r, keep in zip(rows, train) if keep], X[train], contexts,
                                           direction_folds=min(3, len(set(folds[train]))))
        supported = {r["context"] for r in associations if r["supported"]}
        if k > 1 and supported != set(range(k)):
            return None
        features = np.column_stack([B[train], train_features])
        classifier = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(64, features.shape[1])),
                                   LogisticRegression(C=c, max_iter=1000, random_state=20261006))
        classifier.fit(features, y[train])
        return classifier.predict_proba(np.column_stack([B[test], test_features]))[:, 1], model, associations

    def tune(train, ks):
        candidates = []
        for k in ks:
            for c in regularization:
                losses = []
                for inner in sorted(set(folds[train])):
                    fit_rows, valid = train & (folds != inner), train & (folds == inner)
                    if sum(fit_rows) < 9 or len(set(y[fit_rows])) < 2 or not valid.any():
                        continue
                    fitted = fit(fit_rows, valid, k, c)
                    if fitted is not None:
                        losses.append(_loss(y[valid], fitted[0], F[valid]))
                if len(losses) == len(set(folds[train])):
                    candidates.append((float(np.mean(losses)), k, c))
        return min(candidates) if candidates else None

    outer = []
    predictions = np.full((len(rows), 2), np.nan)
    for held in range(4):
        train, test = folds != held, folds == held
        choice, single = tune(train, KS), tune(train, (1,))
        if choice is None or single is None:
            outer.append({"held_fold": held, "status": "UNAVAILABLE_TRAINING_SUPPORT"})
            continue
        contextual, global_fit = fit(train, test, choice[1], choice[2]), fit(train, test, 1, single[2])
        if contextual is None:
            contextual = global_fit
            choice = single
        predictions[test] = np.column_stack([global_fit[0], contextual[0]])
        loss1, lossk = (_loss(y[test], predictions[test, j], F[test]) for j in (0, 1))
        outer.append({"held_fold": held, "status": "AVAILABLE", "selected_K": choice[1], "C": choice[2],
            "K1_log_loss": loss1, "context_log_loss": lossk, "predictive_gain": loss1 - lossk,
            "training_families": sorted(set(F[train])), "held_families": sorted(set(F[test])),
            "association_support": contextual[2], "transform_sha256": contextual[1]["sha256"]})
    gain_folds = sum(r.get("selected_K", 1) > 1 and r.get("predictive_gain", 0) > 0 for r in outer)
    all_rows = np.ones(len(rows), bool)
    choice = tune(all_rows, KS if gain_folds >= 3 else (1,))
    require(choice is not None, "no complete inner folds; context remains unavailable")
    fitted = fit(all_rows, all_rows, choice[1], choice[2])
    if fitted is None:
        choice = tune(all_rows, (1,))
        fitted = fit(all_rows, all_rows, 1, choice[2])
    body = {"schema": "routing-context-selection-v2", "population": "discovery",
        "family_boundary_sha256": family_boundary["sha256"], "discovery_families": family_boundary["discovery"],
        "selected_K": fitted[1]["K"], "positive_outer_folds": gain_folds, "outer_folds": outer,
        "model": fitted[1], "association_support": fitted[2],
        "held_out_log_loss": [_loss(y, predictions[:, j], F) for j in (0, 1)] if np.isfinite(predictions).all() else None,
        "status": "MULTIPLE_CONTEXTS_SUPPORTED" if fitted[1]["K"] > 1 else "FALLBACK_K1",
        "online_gate_status": "REQUIRES_FROZEN_ACCEPTANCE_RULE_AND_SEPARATE_QUALIFICATION"}
    return {**body, "sha256": digest(body)}
