"""Class dynamics and expert nominations from saved labelled sentences.

`derive(campaign, model)` is the saved-data analysis of one campaign-v3 model. From its labelled sentences it writes
  DYNAMICS.json                 class runs, transitions and re-entries (gaps in the labelling stay censored)
  native_dynamics_estimands.parquet, NATIVE_DYNAMICS_SUMMARY.json
                                per attempt: class occupancy, dwell tokens, re-entry and next-class rates, then a
                                10,000-draw family bootstrap
  SCORES.npz, association_atlas.parquet, NOMINATIONS.json
                                per fold: which experts each class uses more than the other classes, and the nominated
                                layer of each depth third
  NATIVE_ACCURACY.json, DERIVED.json
`dynamics_summary(binding)` builds the same per-attempt panel for LFM2.5 from its published trajectories.

    python -m routing.dynamics CAMPAIGN_DIR MODEL [MODEL ...] [--root NEW_ROOT]

CAMPAIGN_DIR holds CAMPAIGN.json (report/research-campaign-v3-20261014-r5-r2). With --root everything is recomputed
into NEW_ROOT, which must hold the campaign's inputs (links to the real files) but none of the outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from routing.selection import (counts_for, enriched_set, family_weights, inclusion_statistics, nominate,
                               random_control, weighted_quantile)
from stats import bootstrap_panel, clean_rows, unknown_bounds
from store import CLASSES, SEED, WORK, canonical, digest, file_sha, private, publish, require, verified

DEPTH_PATHS = WORK/'tools/tmp/claude-analysis-2026-09-24/depth-paths/results/part1'


def trajectories(rows):
    """Original sentence index and segment must both establish adjacency."""
    ordered = sorted(rows, key=lambda r: (r["attempt_id"], r["sentence_index"]))
    transitions, runs, reentries = [], [], []
    previous, run, seen = None, None, set()
    for r in ordered:
        adjacent = previous is not None and r["attempt_id"] == previous["attempt_id"] and r["segment"] == previous["segment"] and r["sentence_index"] == previous["sentence_index"] + 1
        if not adjacent:
            if run:
                runs.append({**run, "right_censored": True})
            seen = set()
            run = {"attempt_id": r["attempt_id"], "class": r["class"], "sentences": 1,
                   "tokens": r["n_tokens"], "left_censored": True}
        else:
            transitions.append({"source": previous["class"], "destination": r["class"],
                                "attempt_id": r["attempt_id"], "boundary": r["sentence_index"]})
            if r["class"] == previous["class"]:
                run["sentences"] += 1
                run["tokens"] += r["n_tokens"]
            else:
                runs.append({**run, "right_censored": False})
                if r["class"] in seen:
                    reentries.append({"attempt_id": r["attempt_id"], "class": r["class"], "boundary": r["sentence_index"]})
                seen.add(previous["class"])
                run = {"attempt_id": r["attempt_id"], "class": r["class"], "sentences": 1,
                       "tokens": r["n_tokens"], "left_censored": False}
        previous = r
    if run:
        runs.append({**run, "right_censored": True})
    return {"transitions": transitions, "dwell": runs, "reentry": reentries,
            "gaps_are_censored": True}


def by_attempt(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["attempt_id"]].append(item)
    return grouped


def attempt_panel(sentences, dynamics, sources):
    """One row per attempt: label coverage, and per class its occupancy, dwell tokens, re-entry and next-class rates.

    Args:
        sentences: Labelled sentences (attempt_id, class, n_tokens).
        dynamics: The attempt's runs, transitions and re-entries, as `trajectories` returns them.
        sources: attempt_id -> source record (benchmark, family, completion_tokens).
    """
    runs, steps, reentries = by_attempt(dynamics["dwell"]), by_attempt(dynamics["transitions"]), by_attempt(dynamics["reentry"])
    rows = []
    for aid, sent in by_attempt(sentences).items():
        source = sources[aid]
        n = sum(s["n_tokens"] for s in sent)
        row = {"benchmark": source["benchmark"], "family": source["family"], "attempt_id": aid,
               "observed_labelled_tokens": n,
               "label_coverage": n/source["completion_tokens"] if source["completion_tokens"] else None}
        for c in CLASSES:
            row[c+":occupancy"] = sum(s["n_tokens"] for s in sent if s["class"] == c)/n if n else None
            dwell = [d for d in runs[aid] if d["class"] == c]
            uncensored = [d["tokens"] for d in dwell if not d["left_censored"] and not d["right_censored"]]
            row[c+":observed_dwell_tokens"] = float(np.mean([d["tokens"] for d in dwell])) if dwell else None
            row[c+":uncensored_dwell_tokens"] = float(np.mean(uncensored)) if uncensored else None
            trans = [d for d in steps[aid] if d["source"] == c]
            row[c+":reentry_per_observed_boundary"] = (sum(d["class"] == c for d in reentries[aid])/len(trans)
                                                       if trans else None)
            for next_class in CLASSES:
                row[c+":next:"+next_class] = sum(d["destination"] == next_class for d in trans)/len(trans) if trans else None
        rows.append(row)
    return rows


def attempt_key(row):
    return (row["dataset"], row["problem_id"], int(row["sample_id"]), row["trace_sha256"])


def joined_accuracy(outcomes, metadata, qf):
    rows = []
    require(len(metadata) == len(outcomes), "exact native metadata population differs")
    for outcome in outcomes:
        key = attempt_key(outcome)
        require(key in metadata, "exact original attempt metadata missing")
        source = metadata[key]
        require(source["completion_tokens"] == int(outcome["completion_tokens"]) and
                source["finish_reason"] == outcome["termination"], "exact native attempt metadata mismatch")
        question = outcome["dataset"] + "|" + outcome["source_problem_id"]
        require(question in qf, "native outcome absent from family graph")
        strict = outcome["strict_correct"]
        strict = None if strict is None or (isinstance(strict, float) and strict != strict) else bool(strict)
        lower, upper = unknown_bounds(strict, source["finish_reason"], source["answer_present"])
        adjudicated = outcome["adjudicated_correct"]
        adjudicated = None if adjudicated is None or (isinstance(adjudicated, float) and adjudicated != adjudicated) else float(adjudicated)
        rows.append({"benchmark": outcome["dataset"], "family": qf[question],
                     "attempt_identity": list(key), "attempt_id": digest(key),
                     "strict": None if strict is None else float(strict),
                     "historical_adjudicated": adjudicated,
                     "unknown_lower": lower, "unknown_upper": upper,
                     "tokens": int(outcome["completion_tokens"]),
                     "termination": source["finish_reason"], "answer_present": source["answer_present"],
                     "original_source": source["location"]})
    return rows


def native_accuracy(campaign, model):
    import pandas as pd
    from schemas import TraceRecord
    from labeling.spans import trace_digest
    from generation.scoring import extract_answer
    root = Path(campaign["root"])
    if (root/model/"NATIVE_ESTIMANDS.json").exists() and (root/model/"NATIVE_ACCURACY.json").exists():
        estimands, summary = verified(root/model/"NATIVE_ESTIMANDS.json"), verified(root/model/"NATIVE_ACCURACY.json")
        require(estimands.get("exact_attempt_metadata"), "inherited native accuracy lacks exact attempt binding")
        return estimands, summary
    outcomes = pd.read_parquet(campaign["outcomes"])
    outcomes = outcomes[(outcomes.corpus == "card-v3") & (outcomes.model == model)].to_dict("records")
    require(len({attempt_key(r) for r in outcomes}) == len(outcomes), "duplicate native outcome identity")
    expected = {attempt_key(r) for r in outcomes}
    sources = verified(root/model/"SOURCES.json")["records"]
    paths = sorted({Path(json.loads(r["source_location"])["path"]) for r in sources})
    metadata = {}
    for path in paths:
        with path.open("rb") as stream:
            offset = 0
            for raw in stream:
                trace = json.loads(raw)
                key = (trace["dataset"], trace["problem_id"], int(trace["sample_id"]), trace_digest(TraceRecord(**trace)))
                require(key in expected and key not in metadata, "native source contains an unexpected or duplicate attempt")
                replay = trace["metadata"]["token_replay"]
                final = trace["metadata"].get("assistant_content") or trace["metadata"].get("reasoning_content") or ""
                metadata[key] = {"completion_tokens": len(replay["completion_token_ids"]),
                                 "finish_reason": trace["metadata"]["finish_reason"],
                                 "answer_present": bool(extract_answer(final, "math")),
                                 "location": {"path": str(path), "byte_offset": offset, "line_bytes": len(raw),
                                              "line_sha256": hashlib.sha256(raw).hexdigest()}}
                offset += len(raw)
    rows = joined_accuracy(outcomes, metadata, verified(root/"FOLDS.json")["question_families"])
    estimands = publish(root/model/"NATIVE_ESTIMANDS.json", {"records": rows,
        "historical_adjudicated_not_used": True, "exact_attempt_metadata": True})
    summary = publish(root/model/"NATIVE_ACCURACY.json", bootstrap_panel(rows,
        ("strict", "unknown_lower", "unknown_upper", "tokens"), panel="native:"+model))
    return estimands, summary


def write_once(path, write):
    """Write a data file atomically; a file already there must have the same bytes. Returns its receipt."""
    private(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_name(f".{path.name}.pending")
    try:
        with pending.open("wb") as stream:
            write(stream)
        if path.exists():
            require(file_sha(pending) == file_sha(path), "conflicting immutable data artifact")
        else:
            pending.rename(path)
    finally:
        pending.unlink(missing_ok=True)
    return {"path": str(path), "sha256": file_sha(path)}


def fold_nominations(model, fold, counts, tokens, sources, family_fold, layers, k, experts, sentence_rows, folds):
    """Nominated sites, atlas rows and expert scores of one discovery fold (families of other folds)."""
    train = family_fold != fold
    rates, support = inclusion_statistics(counts, tokens, sources, train)
    # Class versus the other observed classes; counts are token inclusions.
    other_counts = counts[:, :7].sum(1)[:, None] - counts[:, :7]
    other_tokens = tokens[:, :7].sum(1)[:, None] - tokens[:, :7]
    other_rate, other_support = inclusion_statistics(other_counts, other_tokens, sources, train)
    excess = rates[:7] - other_rate
    excess[(support[:7] < 2) | (other_support < 2)] = np.nan
    # State-informed saliency is the mean of supported class inclusion rates,
    # the same class balancing as the earlier state-balanced score. Global
    # unequal allocation uses this unchanged score per byte.
    supported = np.isfinite(rates[:7])
    state = np.divide(np.nansum(rates[:7], axis=0), supported.sum(0),
                      out=np.full_like(rates[7], np.nan), where=supported.sum(0) > 0)
    random = np.array([[int(digest([SEED, "mask", model, l, e])[:13], 16) / 16**13
                        for e in range(experts)] for l in layers])
    scores = np.stack([state, rates[7], rates[8], random])
    stab = np.zeros((7, len(layers)))
    for excluded in range(5):
        if excluded == fold:
            continue
        sub = train & (family_fold != excluded)
        sr, _ = inclusion_statistics(counts, tokens, sources, sub)
        so, _ = inclusion_statistics(other_counts, other_tokens, sources, sub)
        sx = sr[:7] - so
        for c in range(7):
            for i in range(len(layers)):
                a, b = set(enriched_set(excess[c, i], .5)), set(enriched_set(sx[c, i], .5))
                stab[c, i] += len(a & b) / len(a | b) / 4 if a | b else 0.
    discovered_sentences = [s for s in sentence_rows if folds[s["family"]] != fold]
    durations = weighted_quantile([s["n_tokens"] for s in discovered_sentences],
                                 family_weights(discovered_sentences), (.25, .5, .75))
    durations = [max(1, int(round(v))) if v is not None else None for v in durations]
    upper = durations[2]
    nominations, atlas = [], []
    for c, name in enumerate(CLASSES):
        sites = nominate(excess[c], stab[c], layers, k)
        for depth, i in enumerate(sites):
            record = {"model": model, "fold": fold, "class": name, "depth_third": depth,
                      "layer": None if i is None else layers[i], "pulse_tokens": durations,
                      "windows": [v * upper for v in (4, 8, 16)] if upper else None,
                      "status": "nominated" if i is not None and upper else "insufficient_support"}
            if i is not None:
                sets = [enriched_set(excess[c, i], f) for f in (.25, .5, .75)]
                controls = [random_control(ids, rates[7, i], [model, fold, name, depth, len(ids)]) for ids in sets]
                record.update(expert_sets=sets, control_sets=controls,
                    central_excess_slot_allocation=float(np.nansum(excess[c, i, sets[1]]) / k),
                    stability=float(stab[c, i]), family_support=int(support[c, i].min()))
            nominations.append(record)
        for i, l in enumerate(layers):
            for e in range(experts):
                value = excess[c, i, e]
                atlas.append({"model": model, "fold": fold, "class": name, "layer": l, "expert": e,
                    "inclusion_rate": float(rates[c, i, e]) if np.isfinite(rates[c, i, e]) else None,
                    "other_class_inclusion": float(other_rate[c, i, e]) if np.isfinite(other_rate[c, i, e]) else None,
                    "inclusion_difference": float(value) if np.isfinite(value) else None,
                    "enrichment": float(rates[c, i, e] / other_rate[c, i, e]) if other_rate[c, i, e] > 0 else None,
                    "family_support": int(support[c, i, e]), "selection_stability": float(stab[c, i])})
    return nominations, atlas, scores, support


def derive(campaign, model):
    import pandas as pd
    root = Path(campaign["root"])
    if campaign.get("resume_from") and (root/model/"NOMINATIONS.json").exists():
        _, accuracy = native_accuracy(campaign, model)
        data = clean_rows(pd.read_parquet(root/model/"native_dynamics_estimands.parquet").to_dict("records"))
        columns = sorted({k for r in data for k in r if k not in ("benchmark", "family", "attempt_id", "unavailable_reasons")})
        summary = publish(root/model/"NATIVE_DYNAMICS_SUMMARY.json", bootstrap_panel(data, columns, panel="native-dynamics:"+model))
        return publish(root/model/"DERIVED.json", {"status": "complete", "original_nominations_preserved": True,
            "nominations_sha256": verified(root/model/"NOMINATIONS.json")["sha256"],
            "native_accuracy_sha256": accuracy["sha256"], "dynamics_summary_sha256": summary["sha256"],
            "analysis_binding": campaign["analysis_binding"], "compatible_artifacts_referenced": True})
    sources = verified(root/model/"SOURCES.json")["records"]
    folds = verified(root/"FOLDS.json")["folds"]
    counts, tokens = counts_for(campaign, model, sources)
    contract = campaign["models"][model]
    layers, k = contract["layers"], contract["native_k"]
    saved = pd.read_parquet(campaign["saved_inputs"][model]["sentences"])
    qf = verified(root/"FOLDS.json")["question_families"]
    sentence_rows = [{"attempt_id": r.attempt_id, "question": r.question,
        "benchmark": r.dataset, "family": qf[r.question], "class": CLASSES[int(r.cls)],
        "sentence_index": int(r.sentence_index), "segment": int(r.segment),
        "n_tokens": int(r.n_tokens), "token_start": int(r.tok_start), "token_end": int(r.tok_end)}
        for r in saved.itertuples()]
    dynamics = publish(root/model/"DYNAMICS.json", trajectories(sentence_rows))
    panel = attempt_panel(sentence_rows, dynamics, {s["attempt_id"]: s for s in sources})
    write_once(root/model/"native_dynamics_estimands.parquet",
               lambda stream: pd.DataFrame(panel).to_parquet(stream, index=False))
    columns = sorted({k for r in panel for k in r if k not in ("benchmark", "family", "attempt_id")})
    publish(root/model/"NATIVE_DYNAMICS_SUMMARY.json", bootstrap_panel(panel, columns, panel="native-dynamics:"+model))
    family_fold = np.array([folds[r["family"]] for r in sources])
    nominations, atlas, scores, support = [], [], [], []
    for fold in range(5):
        fold_records, fold_atlas, fold_scores, fold_support = fold_nominations(
            model, fold, counts, tokens, sources, family_fold, layers, k, contract["experts"], sentence_rows, folds)
        nominations += fold_records
        atlas += fold_atlas
        scores.append(fold_scores)
        support.append(fold_support)
    score_file = write_once(root/model/"SCORES.npz", lambda stream: np.savez_compressed(
        stream, scores=np.stack(scores), support=np.stack(support)))
    score_file["bytes"] = Path(score_file["path"]).stat().st_size
    atlas_file = write_once(root/model/"association_atlas.parquet",
                            lambda stream: pd.DataFrame(atlas).to_parquet(stream, index=False))
    atlas_file["rows"] = len(atlas)
    nomination_file = publish(root/model/"NOMINATIONS.json", {"records": nominations,
        "outcomes_used": False, "layers": layers, "folds_sha256": digest(folds)})
    # Strict native fields are used uniformly beside historically adjudicated
    # descriptive estimates. No candidate-verification/J1 judge is restarted.
    _, accuracy = native_accuracy(campaign, model)
    return publish(root/model/"DERIVED.json", {"status": "complete", "scores": score_file,
        "atlas": atlas_file, "nominations_sha256": nomination_file["sha256"],
        "native_accuracy_sha256": accuracy["sha256"], "dynamics_sha256": dynamics["sha256"],
        "source_conditioned_association_input": campaign["saved_inputs"][model]["histogram"],
        "hidden_behavior_reused": campaign["saved_inputs"][model]["features"],
        "existing_depth_paths_reused": str(DEPTH_PATHS/(model+".json"))})


def dynamics_summary(binding):
    """Native-dynamics panel for LFM2.5: `attempt_panel` of its published trajectories, then the family bootstrap."""
    import pandas as pd
    root = Path(binding['root'])
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    require(file_sha(Path(inputs['sentences'])) == inputs['sentences_sha256'], 'LFM derived sentences changed')
    sentences = pd.read_parquet(inputs['sentences']).to_dict('records')
    dynamics = verified(root/'lfm25/DYNAMICS.json')
    recomputed = trajectories(sentences)
    require(all(canonical(dynamics[k]) == canonical(recomputed[k]) for k in ('dwell', 'transitions', 'reentry')),
            'published LFM trajectories differ from the current sentences')
    rows = attempt_panel(sentences, dynamics, {s['attempt_id']: s for s in inputs['sources']})
    pd.DataFrame(rows).to_parquet(Path(binding['config']['bulk_root'])/'lfm25/derived/native_dynamics_estimands.parquet',
                                  index=False)
    columns = sorted({k for r in rows for k in r if k not in ('benchmark', 'family', 'attempt_id')})
    return publish(root/'lfm25/NATIVE_DYNAMICS_SUMMARY.json',
                   bootstrap_panel(rows, columns, panel='native-dynamics:lfm25'))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m routing.dynamics',
                                     description='Recompute the native dynamics and nominations of campaign-v3 models.')
    parser.add_argument('campaign', type=Path, help='directory holding CAMPAIGN.json')
    parser.add_argument('models', nargs='+')
    parser.add_argument('--root', type=Path, help='write all outputs here and recompute them from the inputs')
    args = parser.parse_args(argv)
    campaign = json.loads((args.campaign/'CAMPAIGN.json').read_text())
    if args.root:
        campaign['root'], campaign['resume_from'] = str(args.root), None
    for model in args.models:
        print(model, derive(campaign, model)['sha256'], flush=True)


if __name__ == '__main__':
    main()
