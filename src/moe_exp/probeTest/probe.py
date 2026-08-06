"""Paper-matched layer-wise one-vs-rest probes for seven episode labels."""

from __future__ import annotations

import csv
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from moe_exp.probeTest.data import EPISODE_LABELS


logger = logging.getLogger(__name__)


def load_activation_corpus(
    manifest_path: Path,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Load validated shards as ``(N, L, H)`` plus labels and response groups."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError(f"Activation manifest is not complete: {manifest_path}")

    tensors: list[torch.Tensor] = []
    labels: list[str] = []
    groups: list[str] = []
    units: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    shard_dir = manifest_path.parent / "shards"
    for shard in manifest.get("shards", []):
        activation_path = shard_dir / shard["activation_file"]
        tensor = torch.load(activation_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"Expected a 3D tensor in {activation_path}")
        if tensor.shape[0] != shard["n_units"]:
            raise ValueError(
                f"Unit count mismatch in {activation_path}: {tensor.shape[0]} vs {shard['n_units']}"
            )
        layer_shape = (int(tensor.shape[1]), int(tensor.shape[2]))
        if expected_shape is None:
            expected_shape = layer_shape
        elif layer_shape != expected_shape:
            raise ValueError(
                f"Layer/hidden shape mismatch in {activation_path}: {layer_shape} vs {expected_shape}"
            )
        tensors.append(tensor.to(torch.float32))
        for unit_index, (label, text, span, boundary) in enumerate(
            zip(
                shard["labels"],
                shard["texts"],
                shard["char_spans"],
                shard["boundary_token_positions"],
                strict=True,
            )
        ):
            if label not in EPISODE_LABELS:
                raise ValueError(f"Unknown episode label {label!r} in {activation_path}")
            labels.append(label)
            groups.append(shard["response_id"])
            units.append(
                {
                    "sample_index": len(units),
                    "response_id": shard["response_id"],
                    "unit_index": unit_index,
                    "label": label,
                    "text": text,
                    "char_span": span,
                    "boundary_token_position": boundary,
                }
            )
    if not tensors:
        raise ValueError(f"No activation shards listed in {manifest_path}")

    activations = torch.cat(tensors, dim=0)
    if len(labels) != activations.shape[0]:
        raise ValueError("Activation and metadata sample counts differ")
    return (
        activations,
        np.asarray(labels),
        np.asarray(groups),
        units,
        manifest,
    )


def train_binary_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 2000,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Fit the exact logistic-regression configuration from the ACL code."""
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=max_iter,
        penalty="l2",
        solver="lbfgs",
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x_train, y_train)
    convergence_warnings = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]

    train_predictions = classifier.predict(x_train)
    test_predictions = classifier.predict(x_test)
    train_probabilities = classifier.predict_proba(x_train)[:, 1]
    test_probabilities = classifier.predict_proba(x_test)[:, 1]
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "train_f1": float(f1_score(y_train, train_predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "train_auc": float(roc_auc_score(y_train, train_probabilities)),
        "test_auc": float(roc_auc_score(y_test, test_probabilities)),
        "train_precision": float(
            precision_score(y_train, train_predictions, zero_division=0)
        ),
        "test_precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "train_recall": float(recall_score(y_train, train_predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "n_negative_train": int((y_train == 0).sum()),
        "n_negative_test": int((y_test == 0).sum()),
        "confusion_matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
        "n_iter": int(classifier.n_iter_[0]),
        "convergence_warnings": convergence_warnings,
    }
    return classifier, metrics


def _save_pickle(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_unit_index(units: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")


def _write_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target",
        "layer_idx",
        "train_accuracy",
        "test_accuracy",
        "train_f1",
        "test_f1",
        "train_auc",
        "test_auc",
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
        "n_train_samples",
        "n_test_samples",
        "n_positive_train",
        "n_positive_test",
        "n_negative_train",
        "n_negative_test",
        "n_iter",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_layerwise_accuracy(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_target = {
        target: sorted(
            (row for row in rows if row["target"] == target),
            key=lambda row: row["layer_idx"],
        )
        for target in EPISODE_LABELS
    }
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for axis, target in zip(flat_axes, EPISODE_LABELS, strict=False):
        target_rows = by_target[target]
        axis.plot(
            [row["layer_idx"] for row in target_rows],
            [row["test_accuracy"] for row in target_rows],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
        )
        axis.set_title(f"{target} vs all")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.45, 1.01)
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Test accuracy")

    macro_axis = flat_axes[-1]
    layers = sorted({row["layer_idx"] for row in rows})
    macro_accuracy = [
        float(np.mean([row["test_accuracy"] for row in rows if row["layer_idx"] == layer]))
        for layer in layers
    ]
    macro_axis.plot(layers, macro_accuracy, marker="o", markersize=2.5, linewidth=1.4)
    macro_axis.set_title("Macro mean")
    macro_axis.grid(alpha=0.25)
    macro_axis.set_ylim(0.45, 1.01)
    macro_axis.set_xlabel("Hidden-state index")
    macro_axis.set_ylabel("Test accuracy")
    figure.suptitle("Schoenfeld episode boundary probes")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_layerwise_probes(
    *,
    manifest_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 2000,
    make_plot: bool = True,
) -> Path:
    """Train seven binary probes at every hidden-state index."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    activations, labels, groups, units, manifest = load_activation_corpus(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifiers_dir = output_dir / "classifiers"
    splits_dir = output_dir / "splits"
    classifiers_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_unit_index(units, output_dir / "unit_index.jsonl")

    n_samples, n_hidden_states, hidden_size = map(int, activations.shape)
    indices = np.arange(n_samples)
    rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for target in EPISODE_LABELS:
        y = (labels == target).astype(np.int8)
        class_counts = np.bincount(y, minlength=2)
        if class_counts.min() < 2:
            raise ValueError(
                f"Not enough samples to probe {target}: negative={class_counts[0]}, "
                f"positive={class_counts[1]}"
            )
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        np.savez_compressed(
            splits_dir / f"{target.lower()}.npz",
            train_indices=train_indices,
            test_indices=test_indices,
            train_response_ids=groups[train_indices],
            test_response_ids=groups[test_indices],
        )
        shared_responses = sorted(
            set(groups[train_indices].tolist()) & set(groups[test_indices].tolist())
        )
        split_summaries[target] = {
            "positive": int(class_counts[1]),
            "negative": int(class_counts[0]),
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            "responses_shared_between_train_and_test": shared_responses,
        }

        for layer_index in range(n_hidden_states):
            logger.info("Training %s vs all at hidden-state index %d", target, layer_index)
            layer = activations[:, layer_index, :].numpy()
            classifier, metrics = train_binary_probe(
                layer[train_indices],
                y[train_indices],
                layer[test_indices],
                y[test_indices],
                seed=seed,
                max_iter=max_iter,
            )
            row = {"target": target, "layer_idx": layer_index, **metrics}
            rows.append(row)
            _save_pickle(
                {
                    "classifier": classifier,
                    "target": target,
                    "layer_idx": layer_index,
                    "model_id": manifest["model_id"],
                    "boundary_definition": manifest["boundary_definition"],
                    "feature_size": hidden_size,
                    "protocol": {
                        "classifier": "LogisticRegression",
                        "one_vs_rest": True,
                        "C": 1.0,
                        "class_weight": "balanced",
                        "max_iter": max_iter,
                        "penalty": "l2",
                        "solver": "lbfgs",
                        "test_size": test_size,
                        "seed": seed,
                        "standardization": False,
                        "pca": False,
                    },
                },
                classifiers_dir / f"{target.lower()}_layer_{layer_index:02d}.pkl",
            )

    best_by_target: dict[str, Any] = {}
    for target in EPISODE_LABELS:
        target_rows = [row for row in rows if row["target"] == target]
        best = max(target_rows, key=lambda row: (row["test_accuracy"], -row["layer_idx"]))
        best_by_target[target] = {
            "layer_idx": best["layer_idx"],
            "test_accuracy": best["test_accuracy"],
            "test_f1": best["test_f1"],
            "test_auc": best["test_auc"],
        }

    results = {
        "config": {
            "activation_manifest": str(manifest_path.resolve()),
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "quantization": manifest["quantization"],
            "boundary_definition": manifest["boundary_definition"],
            "n_samples": n_samples,
            "n_hidden_states": n_hidden_states,
            "hidden_size": hidden_size,
            "targets": list(EPISODE_LABELS),
            "classifier": "seven binary one-vs-rest logistic regressions per hidden-state index",
            "test_size": test_size,
            "split": "sentence-level stratified random split (ACL paper protocol)",
            "seed": seed,
            "max_iter": max_iter,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "standardization": False,
            "pca": False,
            "known_limitation": (
                "The paper-matched sentence split can place units from one response in both sets; "
                "response IDs are retained in each split for auditing."
            ),
        },
        "splits": split_summaries,
        "best_by_target": best_by_target,
        "results": rows,
    }
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_results_csv(rows, output_dir / "layerwise_metrics.csv")
    if make_plot:
        _plot_layerwise_accuracy(rows, output_dir / "layerwise_accuracy.png")
    return results_path
