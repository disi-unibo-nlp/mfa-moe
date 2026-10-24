"""Allocated analysis of existing count/sentence archives, without route recapture."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import time
import zipfile

import numpy as np
import pandas as pd

from .contracts import allocated, check_files, digest, file_sha, load, require, save
from .protocol import KEEP, MODELS
from .selection import analyze, coverage, family_folds, keep_set, overlap, scores, validate_counts
from .transitions import analyze_transitions

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
ANALYSIS = BASE / "v3_analysis/results-r2"
DYNAMICS = BASE / "dynamics-routing/results"
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")


def inventory():
    result = {}
    for model in MODELS:
        root = ANALYSIS / model / "A"
        p = root / "class_counts.npz"
        with zipfile.ZipFile(p) as z:
            size = sum(info.file_size for info in z.infolist())
        prov = json.loads((root / "provenance.json").read_text())
        result[model] = {"count_archive": str(p), "bytes": p.stat().st_size,
                         "uncompressed_bytes": size, "attempts": str(root / "attempts.parquet"),
                         "provenance": str(root / "provenance.json"), "spec": prov["spec"],
                         "sentences": str(DYNAMICS / model / "A/sentences.parquet"),
                         "histogram": str(DYNAMICS / model / "A/hist.npz")}
    return result


def common_training_folds(parent, inputs):
    qf = {q: f for f, qs in parent["new_parent_pools"]["families"].items() for q in qs}
    sets = []
    for model in MODELS:
        frame = pd.read_parquet(inputs[model]["attempts"], columns=["attempt_id", "question"])
        require(frame.attempt_id.is_unique and frame.question.is_unique, "not one historical attempt per question")
        sets.append(set(frame.question))
    require(all(qs == sets[0] for qs in sets) and len(sets[0]) == 1547 and sets[0] <= set(qf),
            "historical seven-model question population differs")
    families = {qf[q] for q in sets[0]}
    require(len(families) == 1486, "historical duplicate-family count differs")
    return qf, family_folds(families)


def read_archive(item, qf):
    frame = pd.read_parquet(item["attempts"], columns=["attempt_id", "question"])
    with np.load(item["count_archive"], allow_pickle=False) as z:
        ids, counts, tokens, classes = (z[k] for k in ("attempt_id", "counts", "tokens", "classes"))
    require(len(set(ids)) == len(ids), "duplicate count identity")
    index = pd.Index(ids).get_indexer(frame.attempt_id)
    require(bool((index >= 0).all()) and len(index) == len(ids), "archive/table identity differs")
    counts, tokens = counts[index], tokens[index]
    spec = item["spec"]
    require(counts.shape[2:] == (len(spec["layers"]), spec["num_experts"]), "native geometry differs")
    validate_counts(counts, tokens, spec["top_k"])
    require(list(classes) == list(CLASSES) + ["ALL_REASONING", "ALL_LABELLED"], "class order differs")
    require(bool((counts[:, :7].sum(1, dtype=np.uint64) == counts[:, 8]).all()) and
            bool((tokens[:, :7].sum(1) == tokens[:, 8]).all()), "labelled pooled archive is inconsistent")
    families = np.array([qf[q] for q in frame.question])
    return counts, tokens, families


def measure(campaign, output):
    allocated()
    start = time.monotonic()
    check_files(campaign["code_files"])
    check_files(campaign["saved_source_files"])
    inputs = campaign["saved_inputs"]
    largest = max(MODELS, key=lambda m: (inputs[m]["uncompressed_bytes"], inputs[m]["bytes"]))
    qf, folds = common_training_folds(load(campaign["parent_family_path"]), inputs)
    load_start = time.monotonic()
    counts, tokens, families = read_archive(inputs[largest], qf)
    load_seconds = time.monotonic() - load_start
    fit_start = time.monotonic()
    train = np.array([folds[f] != 0 for f in families])
    b, f, support = scores(counts[:, :7], families, train)
    groups = campaign["models"][largest]["groups"]
    masks = [keep_set(b, KEEP[largest], groups), keep_set(f, KEEP[largest], groups),
             keep_set(b, KEEP[largest], groups, model=largest,
                      layers=inputs[largest]["spec"]["layers"], random=True)]
    for mask in masks:
        coverage(counts[~train, :7], families[~train], mask)
    _, all_frequency, _ = scores(counts[:, 7:8], families, train)
    coverage(counts[~train, :7], families[~train], keep_set(all_frequency, KEEP[largest], groups))
    fit_seconds = time.monotonic() - fit_start
    transition_start = time.monotonic()
    common = set.intersection(*(set(pd.read_parquet(inputs[m]["sentences"], columns=["question"]).question)
                                for m in MODELS))
    require(len(common) == 522 and len({qf[q] for q in common}) == 509, "transition population differs")
    sentences = pd.read_parquet(inputs[largest]["sentences"])
    sentences = sentences[sentences.question.isin(common)].copy()
    sentences["family"] = sentences.question.map(qf)
    sentences["label"] = sentences.cls.map(dict(enumerate(CLASSES)))
    attempts = pd.read_parquet(inputs[largest]["attempts"], columns=["question", "sentence_units"])
    analyze_transitions(sentences.to_dict("records"), CLASSES,
                        total_sentences=int(attempts[attempts.question.isin(common)].sentence_units.sum()))
    transition_seconds = time.monotonic() - transition_start
    elapsed = time.monotonic() - start
    return save(output, {"schema": "expert-memory-cpu-price-v1", "model": largest,
                         "load_seconds": load_seconds, "one_fold_seconds": fit_seconds,
                         "one_model_transition_seconds": transition_seconds,
                         "total_seconds": elapsed, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "families": len(set(families)), "folds_sha256": digest(folds),
                         "training_support": support.tolist(), "campaign_sha256": campaign["sha256"],
                         "archive_sha256": file_sha(inputs[largest]["count_archive"]),
                         "remaining_work_estimate_seconds": 1.25 * (7 * load_seconds + 42 * fit_seconds + 7 * transition_seconds),
                         "price_scope": "archive loading, complete selection/coverage fold including all-reasoning sensitivity, one model transitions; conservative extrapolation"})


def selection(campaign, model, output, price):
    allocated()
    started = time.monotonic()
    require(price["campaign_sha256"] == campaign["sha256"], "CPU price belongs to another campaign")
    check_files(campaign["code_files"])
    check_files(campaign["saved_source_files"])
    qf, folds = common_training_folds(load(campaign["parent_family_path"]), campaign["saved_inputs"])
    counts, _, families = read_archive(campaign["saved_inputs"][model], qf)
    result = analyze(counts[:, :7], families, keep=KEEP[model], groups=campaign["models"][model]["groups"],
                     layers=campaign["saved_inputs"][model]["spec"]["layers"], model=model,
                     classes=CLASSES, folds=folds)
    sensitivity = []
    for fold in range(5):
        train = np.array([folds[f] != fold for f in families])
        _, frequency, _ = scores(counts[:, 7:8], families, train)
        mask = keep_set(frequency, KEEP[model], campaign["models"][model]["groups"])
        sensitivity.append({"fold": fold, "mask": mask.tolist(),
                            "labelled_coverage": coverage(counts[~train, :7], families[~train], mask),
                            "balanced_overlap": overlap(mask, result["folds"][fold]["masks"]["state_balanced"])})
    return save(output, {**result, "all_reasoning_sensitivity": sensitivity,
                         "campaign_sha256": campaign["sha256"], "source_files": campaign["saved_source_files"],
                         "elapsed_seconds": time.monotonic() - started,
                         "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def transitions(campaign, output):
    allocated()
    started = time.monotonic()
    check_files(campaign["code_files"])
    check_files(campaign["saved_source_files"])
    frames = {m: pd.read_parquet(campaign["saved_inputs"][m]["sentences"]) for m in MODELS}
    common = set.intersection(*(set(f.question) for f in frames.values()))
    parent = load(campaign["parent_family_path"])
    qf = {q: f for f, qs in parent["new_parent_pools"]["families"].items() for q in qs}
    require(len(common) == 522 and len({qf[q] for q in common}) == 509, "common transition population differs")
    results = {}
    files = {}
    for model, frame in frames.items():
        frame = frame[frame.question.isin(common)].copy()
        frame["family"] = frame.question.map(qf)
        frame["label"] = frame.cls.map(dict(enumerate(CLASSES)))
        require(frame.label.notna().all() and frame.segment.notna().all(), "sentence labels/segments unresolved")
        attempts = pd.read_parquet(campaign["saved_inputs"][model]["attempts"],
                                   columns=["attempt_id", "question", "sentence_units"])
        total = int(attempts[attempts.question.isin(common)].sentence_units.sum())
        results[model] = analyze_transitions(frame.to_dict("records"), CLASSES, total_sentences=total)
        for field in ("sentences", "histogram"):
            p = campaign["saved_inputs"][model][field]
            files[p] = file_sha(p)
    return save(output, {"schema": "expert-memory-seven-transitions-v1", "questions": sorted(common),
                         "families": 509, "models": results, "source_files": files,
                         "campaign_sha256": campaign["sha256"], "new_inference": False,
                         "elapsed_seconds": time.monotonic() - started,
                         "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
