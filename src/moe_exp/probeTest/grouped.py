"""CPU-only response-grouped evaluation of saved prospective episode activations.

Outer folds estimate performance on unseen responses. One inner grouped holdout
selects each label's layer; outer test labels never enter that selection. This
produces fold-specific estimators, not a replacement deployment probe suite.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from moe_exp.probeTest.data import EPISODE_LABELS
from moe_exp.probeTest.probe import load_activation_corpus, train_binary_probe, _save_pickle

logger = logging.getLogger(__name__)
METRICS = ("auc", "balanced_accuracy", "f1", "accuracy")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _unit_contract(units: list[dict]) -> list[list]:
    return [[u["response_id"], u["unit_index"], u["label"],
             hashlib.sha256(u["text"].encode()).hexdigest()] for u in units]


def make_split_plan(labels: np.ndarray, groups: np.ndarray, units: list[dict], *,
                    outer_folds: int = 5, validation_size: float = 0.2,
                    seed: int = 42) -> dict:
    if not 2 <= outer_folds <= len(set(groups)):
        raise ValueError("outer_folds must be between 2 and the number of responses")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between zero and one")
    splitter = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (development, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_size,
                                  random_state=seed + fold + 1)
        fit_local, val_local = next(inner.split(development, groups=groups[development]))
        fit, validation = development[fit_local], development[val_local]
        folds.append({"fold": fold,
                      "fit_responses": sorted(set(groups[fit].tolist())),
                      "validation_responses": sorted(set(groups[validation].tolist())),
                      "test_responses": sorted(set(groups[test].tolist()))})
    plan = {"schema_version": 1, "seed": seed, "outer_folds": outer_folds,
            "validation_size": validation_size,
            "outer_splitter": "StratifiedGroupKFold on seven-class labels",
            "inner_splitter": "one GroupShuffleSplit per outer fold",
            "unit_contract_sha256": _hash_json(_unit_contract(units)), "folds": folds}
    validate_split_plan(plan, labels, groups, units)
    return plan


def validate_split_plan(plan: dict, labels: np.ndarray, groups: np.ndarray,
                        units: list[dict]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported split-plan schema")
    if plan["unit_contract_sha256"] != _hash_json(_unit_contract(units)):
        raise ValueError("Split plan belongs to different units, labels, or response identities")
    all_groups = set(groups.tolist())
    held_out = []
    if len(plan["folds"]) != plan["outer_folds"]:
        raise ValueError("Incomplete split plan")
    for i, fold in enumerate(plan["folds"]):
        if fold["fold"] != i:
            raise ValueError("Invalid fold ordering")
        partitions = [set(fold[name + "_responses"]) for name in ("fit", "validation", "test")]
        if any(not p for p in partitions) or set.union(*partitions) != all_groups:
            raise ValueError("Partitions must cover every response")
        if any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Response leakage between partitions")
        for name, partition in zip(("fit", "validation", "test"), partitions):
            present = labels[np.isin(groups, sorted(partition))]
            for target in EPISODE_LABELS:
                if not np.any(present == target) or not np.any(present != target):
                    raise ValueError(f"Fold {i} {name} has insufficient binary support for {target}")
        held_out.extend(partitions[2])
    if len(held_out) != len(all_groups) or set(held_out) != all_groups:
        raise ValueError("Every response must appear in exactly one outer test fold")


def select_layer(activations: np.ndarray, y: np.ndarray, fit: np.ndarray,
                 validation: np.ndarray, *, seed: int, max_iter: int) -> tuple[int, list[dict]]:
    """Select exclusively on the provided inner validation partition."""
    rows = []
    for layer in range(activations.shape[1]):
        _, metrics = train_binary_probe(activations[fit, layer], y[fit],
                                        activations[validation, layer], y[validation],
                                        seed=seed, max_iter=max_iter)
        rows.append({"layer_idx": layer, "validation_auc": metrics["test_auc"],
                     "validation_f1": metrics["test_f1"],
                     "n_iter": metrics["n_iter"],
                     "convergence_warnings": metrics["convergence_warnings"]})
    best = max(rows, key=lambda r: (r["validation_auc"], -r["layer_idx"]))
    return best["layer_idx"], rows


def weighted_metrics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict:
    """Metrics for many response-bootstrap weight vectors, including tied-score AUROC."""
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    positive = weights @ y
    negative = weights @ (1 - y)
    prediction = probability > 0.5  # sklearn's binary decision threshold
    tp = weights @ (y * prediction)
    tn = weights @ ((1 - y) * ~prediction)
    fp = negative - tn
    fn = positive - tp
    order = np.argsort(probability, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(probability[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (pos * (np.cumsum(neg, axis=1) - 0.5 * neg)).sum(axis=1) / (positive * negative)
        balanced = 0.5 * (tp / positive + tn / negative)
        accuracy = (tp + tn) / (positive + negative)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                       out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return dict(auc=auc, balanced_accuracy=balanced, f1=f1, accuracy=accuracy)


def _fit_fold_target(activations: np.ndarray, position: np.ndarray, labels: np.ndarray,
                     groups: np.ndarray, fold: dict, target: str, output: Path,
                     contract: str, seed: int, max_iter: int) -> dict:
    checkpoint = output / "folds" / str(fold["fold"]) / (target.lower() + ".json")
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        if cached.get("run_contract_sha256") != contract:
            raise ValueError(f"Incompatible checkpoint: {checkpoint}")
        return cached
    y = (labels == target).astype(np.int8)
    fit = np.flatnonzero(np.isin(groups, fold["fit_responses"]))
    validation = np.flatnonzero(np.isin(groups, fold["validation_responses"]))
    test = np.flatnonzero(np.isin(groups, fold["test_responses"]))
    development = np.sort(np.r_[fit, validation])
    with threadpool_limits(limits=1):
        layer, candidates = select_layer(activations, y, fit, validation,
                                          seed=seed, max_iter=max_iter)
        probe, probe_metrics = train_binary_probe(
            activations[development, layer], y[development], activations[test, layer], y[test],
            seed=seed, max_iter=max_iter)
        baseline, baseline_metrics = train_binary_probe(
            position[development], y[development], position[test], y[test],
            seed=seed, max_iter=max_iter)
        probabilities = probe.predict_proba(activations[test, layer])[:, 1]
        position_probabilities = baseline.predict_proba(position[test])[:, 1]
    metrics = {method: {name: float(values[0]) for name, values in
                       weighted_metrics(y[test], scores, np.ones((1, len(test)))).items()}
               for method, scores in (("probe", probabilities), ("position", position_probabilities))}
    result = {"run_contract_sha256": contract, "fold": fold["fold"], "target": target,
              "selected_layer": layer, "selection_metric": "validation_auc",
              "validation_candidates": candidates, "test_indices": test.tolist(),
              "y_test": y[test].tolist(), "probabilities": probabilities.tolist(),
              "position_probabilities": position_probabilities.tolist(), "metrics": metrics,
              "fit_samples": len(fit), "validation_samples": len(validation),
              "refit_samples": len(development), "test_samples": len(test),
              "test_positive": int(y[test].sum()),
              "refit_n_iter": probe_metrics["n_iter"],
              "refit_convergence_warnings": probe_metrics["convergence_warnings"],
              "position_n_iter": baseline_metrics["n_iter"],
              "position_convergence_warnings": baseline_metrics["convergence_warnings"]}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _save_pickle({"classifier": probe, "position_classifier": baseline, "target": target,
                  "layer_idx": layer, "fold": fold["fold"], "protocol": "grouped",
                  "run_contract_sha256": contract}, checkpoint.with_suffix(".pkl"))
    _write_json(checkpoint, result)
    return result


def summarize_predictions(records: list[dict], groups: np.ndarray, *,
                          replicates: int, seed: int) -> dict:
    """Mean-fold metrics; bootstrap whole responses within each fixed outer test fold.

Intervals condition on the saved folds, layer selections, and fitted estimators;
refitting and alternative split seeds are not part of this bootstrap.
"""
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    fold_ids = sorted({r["fold"] for r in records})
    rng = np.random.default_rng(seed)
    estimates = {method: {target: {metric: [] for metric in METRICS}
                         for target in EPISODE_LABELS} for method in ("probe", "position")}
    draws = {method: {target: {metric: [] for metric in METRICS}
                     for target in EPISODE_LABELS} for method in estimates}
    for fold in fold_ids:
        fold_records = [r for r in records if r["fold"] == fold]
        indices = np.asarray(fold_records[0]["test_indices"])
        _, group_index = np.unique(groups[indices], return_inverse=True)
        n_groups = int(group_index.max()) + 1
        counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=replicates)
        weights = counts[:, group_index]
        for record in fold_records:
            if record["test_indices"] != indices.tolist():
                raise ValueError("Targets must share outer test folds")
            target = record["target"]
            for method, key in (("probe", "probabilities"), ("position", "position_probabilities")):
                values = weighted_metrics(record["y_test"], record[key], weights)
                for metric in METRICS:
                    estimates[method][target][metric].append(record["metrics"][method][metric])
                    draws[method][target][metric].append(values[metric])
    summary = {"targets": {}, "macro": {},
               "bootstrap": {"method": "outer_fold_stratified_response_percentile",
                             "replicates": replicates, "seed": seed,
                             "estimand": "unweighted mean of outer-fold metrics",
                             "conditional_on": "fixed folds, selected layers, fitted models",
                             "confidence_level": 0.95}}
    def package(point: float, samples: np.ndarray) -> dict:
        finite = samples[np.isfinite(samples)]
        return {"estimate": float(point),
                "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
                "valid_replicates": len(finite)}
    arrays: dict = {}
    points: dict = {}
    for target in EPISODE_LABELS:
        summary["targets"][target] = {}
        arrays[target], points[target] = {}, {}
        for method in estimates:
            arrays[target][method] = {m: np.mean(draws[method][target][m], axis=0) for m in METRICS}
            points[target][method] = {m: float(np.mean(estimates[method][target][m])) for m in METRICS}
            summary["targets"][target][method] = {
                m: package(points[target][method][m], arrays[target][method][m]) for m in METRICS}
        summary["targets"][target]["probe_minus_position"] = {
            m: package(points[target]["probe"][m] - points[target]["position"][m],
                       arrays[target]["probe"][m] - arrays[target]["position"][m]) for m in METRICS}
    for method in ("probe", "position", "probe_minus_position"):
        summary["macro"][method] = {}
        for metric in METRICS:
            if method == "probe_minus_position":
                point = np.mean([points[t]["probe"][metric] - points[t]["position"][metric]
                                 for t in EPISODE_LABELS])
                samples = np.mean([arrays[t]["probe"][metric] - arrays[t]["position"][metric]
                                   for t in EPISODE_LABELS], axis=0)
            else:
                point = np.mean([points[t][method][metric] for t in EPISODE_LABELS])
                samples = np.mean([arrays[t][method][metric] for t in EPISODE_LABELS], axis=0)
            summary["macro"][method][metric] = package(point, samples)
    return summary


def train_grouped_probes(*, manifest_path: Path, output_dir: Path,
                         split_plan_path: Path | None = None, outer_folds: int = 5,
                         validation_size: float = 0.2, seed: int = 42,
                         max_iter: int = 2000, workers: int = 4,
                         bootstrap_replicates: int = 5000, bootstrap_seed: int = 42) -> Path:
    if workers < 1 or bootstrap_replicates < 1 or max_iter < 1:
        raise ValueError("workers, max_iter, and bootstrap_replicates must be positive")
    started = time.perf_counter()
    torch.set_num_threads(1)
    activations, labels, groups, units, manifest = load_activation_corpus(manifest_path)
    order = np.asarray(sorted(range(len(units)), key=lambda i: (groups[i], units[i]["unit_index"])))
    activations = activations[order].numpy()
    labels, groups = labels[order], groups[order]
    units = [dict(units[i], source_sample_index=int(i), sample_index=j) for j, i in enumerate(order)]
    if len({(u["response_id"], u["unit_index"]) for u in units}) != len(units):
        raise ValueError("Duplicate response/unit identities")
    if not np.isfinite(activations).all():
        raise ValueError("Nonfinite saved activations")
    split_plan_path = split_plan_path or output_dir / "split_plan.json"
    if output_dir.exists() and not (output_dir / "run_config.json").exists():
        allowed = {split_plan_path.resolve()}
        if any(p.resolve() not in allowed for p in output_dir.iterdir()):
            raise ValueError("Output directory is not empty; use a separate grouped-results directory")
    if split_plan_path.exists():
        plan = json.loads(split_plan_path.read_text())
        validate_split_plan(plan, labels, groups, units)
        if (plan["seed"], plan["outer_folds"], plan["validation_size"]) != (
                seed, outer_folds, validation_size):
            raise ValueError("Requested settings differ from the existing split plan")
    else:
        plan = make_split_plan(labels, groups, units, outer_folds=outer_folds,
                               validation_size=validation_size, seed=seed)
        _write_json(split_plan_path, plan)
    prompt_tokens = {s["response_id"]: s["prompt_tokens"] for s in manifest["shards"]}
    position = np.asarray([[np.log1p(u["unit_index"]),
                            np.log1p(max(0, u["boundary_token_position"] + 1
                                         - prompt_tokens[u["response_id"]]))] for u in units])
    source_hashes = {str(manifest_path.resolve()): _hash_file(manifest_path)}
    for shard in manifest["shards"]:
        path = manifest_path.parent / "shards" / shard["activation_file"]
        source_hashes[str(path.resolve())] = _hash_file(path)
    config = {"protocol": "grouped", "model_id": manifest["model_id"],
              "model_revision": manifest["model_revision"], "quantization": manifest["quantization"],
              "boundary_definition": manifest["boundary_definition"],
              "activation_manifest": str(manifest_path.resolve()),
              "n_samples": len(labels), "n_responses": len(set(groups)),
              "n_hidden_states": activations.shape[1], "hidden_size": activations.shape[2],
              "targets": list(EPISODE_LABELS), "outer_folds": outer_folds,
              "validation_size": validation_size, "seed": seed,
              "selection_metric": "validation_auc", "tie_break": "lowest hidden-state index",
              "max_iter": max_iter, "C": 1.0, "solver": "lbfgs", "penalty": "l2",
              "class_weight": "balanced", "standardization": False, "pca": False,
              "position_features": ["log1p(prior annotated unit count)",
                                    "log1p(prior response token count)"],
              "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
              "split_plan_sha256": _hash_json(plan), "source_sha256": source_hashes,
              "code_sha256": {p.name: _hash_file(p) for p in (Path(__file__),
                               Path(__file__).with_name("probe.py"))},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "torch": torch.__version__,
                           "joblib": joblib.__version__}}
    contract = _hash_json(config)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains an incompatible run; choose a new directory")
    if output_dir.exists() and not config_path.exists() and any(output_dir.iterdir()):
        allowed = {split_plan_path.resolve()}
        if any(p.resolve() not in allowed for p in output_dir.iterdir()):
            raise ValueError("Output directory is not empty; use a separate grouped-results directory")
    _write_json(config_path, config)
    _write_json(output_dir / "split_plan.json", plan)
    (output_dir / "unit_index.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    _write_json(output_dir / "status.json", {"status": "running", "run_contract_sha256": contract})
    records = []
    with parallel_config(backend="loky", inner_max_num_threads=1), Parallel(n_jobs=workers) as parallel:
        for fold in plan["folds"]:
            logger.info("Grouped fold %d/%d: %d fit, %d validation, %d test responses",
                        fold["fold"] + 1, outer_folds, len(fold["fit_responses"]),
                        len(fold["validation_responses"]), len(fold["test_responses"]))
            results = parallel(delayed(_fit_fold_target)(activations, position, labels, groups,
                               fold, target, output_dir, contract, seed, max_iter)
                               for target in EPISODE_LABELS)
            records.extend(results)
            logger.info("Finished fold %d: mean label AUROC %.4f", fold["fold"] + 1,
                        np.mean([r["metrics"]["probe"]["auc"] for r in results]))
    counts = np.zeros((len(units), len(EPISODE_LABELS)), dtype=int)
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for record in records:
            target_index = EPISODE_LABELS.index(record["target"])
            for j, index in enumerate(record["test_indices"]):
                counts[index, target_index] += 1
                handle.write(json.dumps({"sample_index": index,
                    "response_id": units[index]["response_id"], "unit_index": units[index]["unit_index"],
                    "target": record["target"], "fold": record["fold"],
                    "selected_layer": record["selected_layer"], "y": record["y_test"][j],
                    "probability": record["probabilities"][j],
                    "position_probability": record["position_probabilities"][j]}) + "\n")
    if not np.all(counts == 1):
        raise ValueError("Incomplete or duplicate out-of-fold predictions")
    logger.info("Computing %d response bootstrap replicates", bootstrap_replicates)
    with threadpool_limits(limits=1):
        summary = summarize_predictions(records, groups, replicates=bootstrap_replicates,
                                        seed=bootstrap_seed)
    result = {"schema_version": 1, "status": "complete", "config": config,
              "run_contract_sha256": contract, "folds": records, "summary": summary,
              "execution": {"workers": workers, "blas_threads_per_worker": 1, "device": "cpu",
                            "elapsed_seconds_this_invocation": time.perf_counter() - started},
              "limitations": ["Intervals condition on fixed folds, layer selection and fitted models.",
                              "The 38-response corpus is reused; this is not an external-domain test.",
                              "No single global best layer is selected from outer-test results.",
                              "Comparison to the legacy result changes both splitting and selection."]}
    path = output_dir / "results.json"
    _write_json(path, result)
    _write_json(output_dir / "status.json", {"status": "complete", "run_contract_sha256": contract})
    return path
