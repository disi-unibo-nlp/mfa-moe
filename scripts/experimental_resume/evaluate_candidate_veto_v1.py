"""Family-held-out discovery evaluation of a prefix-only candidate veto.

The model sees the original problem, previous completed sentence and current
completed sentence. `next_sentence`, correctness and future text are never
read. This is an exploratory Qwen-label model, not a truth or causal judge.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
BASE = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery"
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FULL_RATINGS = BASE / "ratings-v22-fullprefix-v2-6c10499b-1ef8863f"
OLD_RATINGS = BASE / "ratings-f5b2e28c-74c16a5d"
VERSION = "candidate-prefix-veto-charlogit-v1-discovery"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def prefix_features(unit):
    # `inputs.next_sentence` can be present in the cache. Never inspect it.
    fields = unit["inputs"]
    problem, previous, current = (fields[key] for key in
                                  ("problem_statement", "previous_sentence", "sentence"))
    if not all(isinstance(value, str) for value in (problem, previous, current)):
        raise ValueError("unit prefix fields must be text")
    return ("PROBLEM " + problem[-1000:] + "\nPREVIOUS " + previous[-600:] +
            "\nCURRENT " + current)


def all_ratings(directory):
    binding = sealed(directory / "BINDING.json")
    summary = sealed(directory / "SUMMARY.json")
    if summary["binding_sha256"] != binding["sha256"]:
        raise ValueError("rating summary binding differs")
    parts = [sealed(path) for path in sorted((directory / "batches").glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))]
    if any(part["binding_sha256"] != binding["sha256"] for part in parts):
        raise ValueError("rating batch binding differs")
    rows = [row for part in parts for row in part["records"]]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("duplicated rating UID")
    return binding, summary, parts, rows


def fit_model(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    if len(set(labels)) != 2:
        raise ValueError("both start classes required for model fit")
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                         min_df=2, max_features=20000,
                                         sublinear_tf=True),
                         LogisticRegression(C=.5, class_weight="balanced",
                                            solver="liblinear", max_iter=1000)).fit(texts, labels)


def inner_threshold(texts, labels, families):
    import numpy as np
    from sklearn.model_selection import GroupKFold
    groups = sorted(set(families))
    if len(groups) < 3:
        return 1.01, {"reason": "fewer_than_three_training_families"}
    predictions = np.full(len(texts), np.nan)
    for train, hold in GroupKFold(n_splits=min(3, len(groups))).split(texts, labels, families):
        if len(set(labels[i] for i in train)) != 2:
            continue
        model = fit_model([texts[i] for i in train], [labels[i] for i in train])
        predictions[hold] = model.predict_proba([texts[i] for i in hold])[:, 1]
    valid = np.isfinite(predictions)
    if valid.sum() < 10:
        return 1.01, {"reason": "insufficient_inner_predictions"}
    best = None
    for threshold in np.arange(.40, .951, .05):
        selected = valid & (predictions >= threshold)
        n = int(selected.sum())
        if n < 4:
            continue
        precision = sum(labels[i] for i in np.flatnonzero(selected)) / n
        recall = sum(labels[i] for i in np.flatnonzero(selected)) / max(1, sum(labels[i] for i in np.flatnonzero(valid)))
        if precision >= .70:
            key = (recall, precision, -float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold), n, float(precision), float(recall))
    if best is None:
        return 1.01, {"reason": "no_inner_threshold_at_70pct_precision_and_four_fires"}
    return best[1], {"inner_selected": best[2], "inner_precision": best[3],
                     "inner_recall": best[4]}


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("veto family CV requires CPU Slurm")
    import numpy as np
    from sklearn.model_selection import GroupKFold
    source_path = (BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json" if args.source == "full"
                   else BASE / "TRANSITION_AUDIT_FIXTURES.json")
    ratings_dir = FULL_RATINGS if args.source == "full" else OLD_RATINGS
    output = REPO / "report/experimental-resume-v1" / (
        "CANDIDATE_VETO_FULLPREFIX_DISCOVERY_v1.json" if args.source == "full"
        else "CANDIDATE_VETO_ADAPTED_EXPLORATORY_v1.json")
    source = sealed(source_path)
    units = sealed(BASE / "UNITS.json")
    sensitivity = sealed(REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json")
    binding, summary, parts, ratings = all_ratings(ratings_dir)
    binding_source = binding.get("frame_sha256", binding.get("fixtures_sha256"))
    if args.source == "old":
        original = sealed(BASE / "TRANSITION_AUDIT_CANDIDATES.json")
        source_binding_ok = (binding_source == original["sha256"] and
                             {r["uid"] for r in original["records"]} ==
                             {r["uid"] for r in source["records"]})
    else:
        source_binding_ok = binding_source == source["sha256"]
    if (not source_binding_ok or len(ratings) != len(source["records"])
            or {r["uid"] for r in ratings} != {r["uid"] for r in source["records"]}):
        raise ValueError("rating assignments differ from frozen frame")
    if args.source == "full" and (len(ratings) != 372 or summary["counts"].get("ratings") != 744):
        raise ValueError("full 744-rating audit is incomplete")
    v22_fires = {(e["attempt_id"], e["sentence_index"]) for e in
                 sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|candidate_to_verify"]}
    by_unit = {(u["attempt_id"], u["sentence_index"]): u for u in units["records"]}
    rated_by_uid = {r["uid"]: r for r in ratings}
    examples, unresolved = [], Counter()
    for row in source["records"]:
        if row["transition"] != "candidate_to_verify":
            continue
        if args.source == "full":
            key = row["attempt_id"], row["sentence_index"]
            fired = row["analysis_meta"]["v22_fired"]
            family = row["family"]
        else:
            meta = row["analysis_meta"]
            key = meta["attempt_id"], meta["source_sentence_index"]
            fired = key in v22_fires
            family = meta["family"]
        if not fired:
            continue
        unit = by_unit[key]
        if unit["family"] != family or key not in v22_fires:
            raise ValueError("rated fire differs from frozen unit inventory")
        if unit["inputs"]["sentence"] != row["reader_input"]["triggering_sentence"]:
            raise ValueError("rated trigger differs from native sentence")
        readers = rated_by_uid[row["uid"]]["readers"]
        if any(reader["rating"] is None or reader["finish_reason"] != "stop" for reader in readers):
            unresolved["parse_or_length"] += 1
            continue
        labels = [reader["rating"]["start"] for reader in readers]
        if labels[0] != labels[1]:
            unresolved["reader_disagreement"] += 1
            continue
        examples.append({"uid": row["uid"], "family": family,
                         "text": prefix_features(unit), "label": bool(labels[0])})
    texts = [r["text"] for r in examples]
    labels = [int(r["label"]) for r in examples]
    families = [r["family"] for r in examples]
    if len(set(labels)) != 2 or len(set(families)) < 5:
        raise ValueError("insufficient resolved family-disjoint positive and negative starts")
    outer = GroupKFold(n_splits=5)
    out_pred = np.zeros(len(examples), dtype=bool)
    fold_rows = []
    for fold, (train, hold) in enumerate(outer.split(texts, labels, families)):
        train_texts = [texts[i] for i in train]
        train_labels = [labels[i] for i in train]
        train_families = [families[i] for i in train]
        threshold, inner = inner_threshold(train_texts, train_labels, train_families)
        if threshold <= 1.0 and len(set(train_labels)) == 2:
            model = fit_model(train_texts, train_labels)
            scores = model.predict_proba([texts[i] for i in hold])[:, 1]
            out_pred[hold] = scores >= threshold
        fold_rows.append({"fold": fold, "train_families": len(set(train_families)),
                          "heldout_families": len(set(families[i] for i in hold)),
                          "heldout_rows": len(hold), "threshold": threshold,
                          "inner": inner, "heldout_selected": int(out_pred[hold].sum()),
                          "heldout_true_selected": sum(labels[i] for i in hold if out_pred[i])})
    selected = np.flatnonzero(out_pred)
    total_positive = sum(labels)
    observed = {"resolved_rated_fires": len(examples), "positive_both_readers": total_positive,
                "unresolved": dict(unresolved), "selected": len(selected),
                "selected_true": sum(labels[i] for i in selected),
                "sample_precision": (sum(labels[i] for i in selected) / len(selected)) if len(selected) else None,
                "sample_recall": (sum(labels[i] for i in selected) / total_positive) if total_positive else None,
                "selected_families": len({families[i] for i in selected})}
    final_threshold, final_inner = inner_threshold(texts, labels, families)
    event_rows = sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|candidate_to_verify"]
    burden = {"all_v22_fires": len(event_rows), "all_v22_families": len({e["family"] for e in event_rows}),
              "exploratory_model_threshold": final_threshold, "inner": final_inner}
    if final_threshold <= 1.0:
        model = fit_model(texts, labels)
        event_texts = [prefix_features(by_unit[e["attempt_id"], e["sentence_index"]]) for e in event_rows]
        event_pred = model.predict_proba(event_texts)[:, 1] >= final_threshold
        burden["predicted_kept_fires"] = int(event_pred.sum())
        burden["predicted_kept_families"] = len({e["family"] for e, keep in zip(event_rows, event_pred) if keep})
    body = {"schema": "candidate-veto-family-cv-v1", "job_id": os.environ["SLURM_JOB_ID"],
            "version": VERSION, "source": args.source,
            "source_frame_sha256": source["sha256"], "rating_binding_sha256": binding["sha256"],
            "rating_summary_sha256": summary["sha256"],
            "rating_batch_sha256s": [p["sha256"] for p in parts],
            "units_sha256": units["sha256"], "v22_sensitivity_sha256": sensitivity["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "features_allowlist": ["problem_statement", "previous_sentence", "sentence"],
            "model": "char 3-5 TF-IDF (20k max, min_df=2) + balanced logistic C=0.5",
            "threshold_policy": "inner family CV: maximal recall at >=0.70 apparent precision and >=4 selected; otherwise abstain",
            "outer_family_cv": fold_rows, "outer_oof": observed,
            "full_native_fire_burden_exploratory": burden,
            "interpretation": ("Old start ratings were adapted and readers saw later text; all estimates exploratory. "
                               if args.source == "old" else
                               "Qwen full-prefix starts are same-model LLM audits, not independent human truth. ") +
                              "Nested family CV is within discovery families; the final native-fire burden is fitted on discovery and cannot be treated as validation."}
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(output), "outer_oof": observed,
                      "burden": burden}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["old", "full"], required=True)
    main(parser.parse_args())
