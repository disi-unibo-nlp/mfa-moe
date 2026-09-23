from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from moe_exp.probeTest import grouped
from moe_exp.probeTest.data import EPISODE_LABELS
from moe_exp.probeTest.run import build_parser


def corpus():
    units = [dict(response_id=f"r{r:02d}", unit_index=i, label=label, text=f"{r}:{i}")
             for r in range(12) for i, label in enumerate(list(EPISODE_LABELS) * 2)]
    return np.array([u["label"] for u in units]), np.array([u["response_id"] for u in units]), units


def test_grouped_partitions_and_shared_contract():
    labels, groups, units = corpus()
    plan = grouped.make_split_plan(labels, groups, units, outer_folds=3)
    assert plan == grouped.make_split_plan(labels, groups, units, outer_folds=3)
    tests = []
    for fold in plan["folds"]:
        fit, validation, test = [set(fold[k + "_responses"]) for k in ("fit", "validation", "test")]
        assert not (fit & validation or fit & test or validation & test)
        assert fit | validation | test == set(groups)
        tests.extend(test)
    assert Counter(tests) == Counter(set(groups))
    changed = copy.deepcopy(units)
    changed[0]["text"] += " changed"
    with pytest.raises(ValueError, match="different units"):
        grouped.validate_split_plan(plan, labels, groups, changed)
    leaking = copy.deepcopy(plan)
    leaking["folds"][0]["fit_responses"].append(leaking["folds"][0]["test_responses"][0])
    with pytest.raises(ValueError, match="leakage"):
        grouped.validate_split_plan(leaking, labels, groups, units)


def test_layer_selection_only_observes_inner_partitions(monkeypatch):
    x = np.arange(8 * 3 * 2).reshape(8, 3, 2)
    y = np.array([0, 1] * 4)
    fit, validation = np.array([0, 1, 2, 3]), np.array([4, 5])
    seen = []

    def fake_train(x_fit, y_fit, x_val, y_val, **kwargs):
        layer = len(seen)
        np.testing.assert_equal(x_fit, x[fit, layer])
        np.testing.assert_equal(x_val, x[validation, layer])
        np.testing.assert_equal(y_fit, y[fit])
        np.testing.assert_equal(y_val, y[validation])
        seen.append(layer)
        return None, dict(test_auc=[.6, .9, .9][layer], test_f1=.5,
                          n_iter=1, convergence_warnings=[])

    monkeypatch.setattr(grouped, "train_binary_probe", fake_train)
    best, rows = grouped.select_layer(x, y, fit, validation, seed=42, max_iter=10)
    assert best == 1  # tied validation AUROC uses the lower index
    assert len(rows) == 3


def test_weighted_bootstrap_metrics_match_sklearn_with_ties():
    rng = np.random.default_rng(12)
    y = np.array([0, 1] * 12)
    scores = rng.choice([.1, .3, .5, .7, .9], size=len(y))
    weights = rng.integers(1, 5, size=(10, len(y)))
    result = grouped.weighted_metrics(y, scores, weights)
    for i, w in enumerate(weights):
        assert result["auc"][i] == pytest.approx(roc_auc_score(y, scores, sample_weight=w))
        assert result["balanced_accuracy"][i] == pytest.approx(balanced_accuracy_score(y, scores > .5, sample_weight=w))
        assert result["f1"][i] == pytest.approx(f1_score(y, scores > .5, sample_weight=w))
        assert result["accuracy"][i] == pytest.approx(accuracy_score(y, scores > .5, sample_weight=w))
    unsupported = grouped.weighted_metrics(y, scores, y)
    assert np.isnan(unsupported["auc"][0])


def write_corpus(path: Path):
    labels, groups, units = corpus()
    shard_dir = path / "shards"
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(5)
    shards = []
    for response in sorted(set(groups)):
        current = [u for u in units if u["response_id"] == response]
        values = rng.normal(size=(len(current), 2, 8)).astype(np.float32)
        for i, u in enumerate(current):
            values[i, :, EPISODE_LABELS.index(u["label"])] += 4
        torch.save(torch.from_numpy(values), shard_dir / f"{response}.pt")
        shards.append(dict(activation_file=f"{response}.pt", response_id=response,
                           n_units=len(current), labels=[u["label"] for u in current],
                           texts=[u["text"] for u in current],
                           char_spans=[[i, i+1] for i in range(len(current))],
                           boundary_token_positions=list(range(2, 2+len(current))), prompt_tokens=3))
    manifest = dict(status="complete", model_id="synthetic", model_revision="main",
                    quantization="none", boundary_definition="pre-unit", shards=shards)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path / "manifest.json"


def test_grouped_run_oof_resume_and_output_protection(tmp_path, monkeypatch):
    manifest = write_corpus(tmp_path / "activations")
    output = tmp_path / "grouped"
    args = dict(manifest_path=manifest, output_dir=output, outer_folds=3,
                workers=2, bootstrap_replicates=50)
    result = json.loads(grouped.train_grouped_probes(**args).read_text())
    assert result["status"] == "complete"
    assert "best_by_target" not in result
    assert len(result["folds"]) == 3 * 7
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    counts = Counter((r["response_id"], r["unit_index"], r["target"]) for r in predictions)
    assert len(counts) == 12 * 14 * 7
    assert set(counts.values()) == {1}
    plan = json.loads((output / "split_plan.json").read_text())
    for row in predictions:
        fold = plan["folds"][row["fold"]]
        assert row["response_id"] in fold["test_responses"]
        assert row["response_id"] not in fold["fit_responses"] + fold["validation_responses"]
    for row in result["folds"]:
        best = max(row["validation_candidates"], key=lambda r: (r["validation_auc"], -r["layer_idx"]))
        assert row["selected_layer"] == best["layer_idx"]
        assert row["refit_samples"] == row["fit_samples"] + row["validation_samples"]
    assert result["summary"]["macro"]["probe"]["auc"]["estimate"] > .9

    def no_refit(*args, **kwargs):
        raise AssertionError("Completed checkpoints should not be refitted")
    monkeypatch.setattr(grouped, "select_layer", no_refit)
    resumed = json.loads(grouped.train_grouped_probes(**dict(args, workers=1)).read_text())
    assert resumed["summary"] == result["summary"]
    with pytest.raises(ValueError, match="incompatible run"):
        grouped.train_grouped_probes(**dict(args, max_iter=99))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "results.json").write_text("legacy")
    with pytest.raises(ValueError, match="not empty"):
        grouped.train_grouped_probes(**dict(args, output_dir=legacy))
    assert [p.name for p in legacy.iterdir()] == ["results.json"]


def test_cli_keeps_sentence_protocol_default():
    parser = build_parser()
    args = parser.parse_args(["probe", "--manifest", "x", "--output-dir", "y"])
    assert args.protocol == "sentence"
    args = parser.parse_args(["probe", "--manifest", "x", "--output-dir", "y", "--protocol", "grouped"])
    assert args.protocol == "grouped"


def test_bootstrap_resamples_complete_responses_and_pairs_baseline():
    # Response A is ranked perfectly; response B is ranked in reverse.
    # Resampling two whole responses can therefore yield AUROC 0 or 1.
    groups = np.array(["a"] * 3 + ["b"] * 5)
    y = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    probability = np.array([0., .1, .9, 1., 0., 1., 0., 1.])
    point = {k: float(v[0]) for k, v in grouped.weighted_metrics(y, probability, np.ones(len(y))).items()}
    records = [dict(fold=0, target=t, test_indices=list(range(len(y))), y_test=y.tolist(),
                    probabilities=probability.tolist(), position_probabilities=probability.tolist(),
                    metrics={"probe": point, "position": point}) for t in EPISODE_LABELS]
    summary = grouped.summarize_predictions(records, groups, replicates=500, seed=42)
    assert summary["macro"]["probe"]["auc"]["ci95"] == [0., 1.]
    for values in summary["macro"]["probe_minus_position"].values():
        assert values["estimate"] == 0.
        assert values["ci95"] == [0., 0.]
