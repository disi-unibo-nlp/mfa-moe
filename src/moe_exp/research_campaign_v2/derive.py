"""Saved-data estimands and discovery-only nominations; no trained models."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np

from .artifacts import frame, npz
from .contracts import CLASSES, METHODS, SEED, digest, load, publish, require
from .design import enriched_set, family_weights, inclusion_statistics, nominate, random_control, weighted_quantile
from .inputs import counts_for
from .statistics import bootstrap_panel, trajectories


def derive(campaign, model):
    import pandas as pd
    root = Path(campaign["root"])
    if campaign.get("resume_from") and (root/model/"NOMINATIONS.json").exists():
        from .accuracy import native_accuracy
        _, accuracy=native_accuracy(campaign,model)
        data=pd.read_parquet(root/model/"native_dynamics_estimands.parquet").to_dict("records")
        columns=sorted({k for r in data for k in r if k not in ("benchmark","family","attempt_id")})
        summary=publish(root/model/"NATIVE_DYNAMICS_SUMMARY.json",bootstrap_panel(data,columns,panel="native-dynamics:"+model))
        return publish(root/model/"DERIVED.json",{"status":"complete","original_nominations_preserved":True,
            "nominations_sha256":load(root/model/"NOMINATIONS.json")["sha256"],
            "native_accuracy_sha256":accuracy["sha256"],"dynamics_summary_sha256":summary["sha256"],
            "analysis_binding":campaign["analysis_binding"],"compatible_artifacts_referenced":True})
    sources = load(root / model / "SOURCES.json")["records"]
    folds = load(root / "FOLDS.json")["folds"]
    counts, tokens = counts_for(campaign, model, sources)
    contract = campaign["models"][model]
    layers, k = contract["layers"], contract["native_k"]
    saved = pd.read_parquet(campaign["saved_inputs"][model]["sentences"])
    qf = load(root / "FOLDS.json")["question_families"]
    sentence_rows = [{"attempt_id": r.attempt_id, "question": r.question,
        "benchmark": r.dataset, "family": qf[r.question], "class": CLASSES[int(r.cls)],
        "sentence_index": int(r.sentence_index), "segment": int(r.segment),
        "n_tokens": int(r.n_tokens), "token_start": int(r.tok_start), "token_end": int(r.tok_end)}
        for r in saved.itertuples()]
    dynamics = publish(root / model / "DYNAMICS.json", trajectories(sentence_rows))
    source_lookup={s["attempt_id"]:s for s in sources}
    by_attempt=defaultdict(list)
    for s in sentence_rows:by_attempt[s["attempt_id"]].append(s)
    native_dynamics_rows=[]
    dwell_by_attempt=defaultdict(list)
    transitions_by_attempt=defaultdict(list)
    reentry_by_attempt=defaultdict(list)
    for d in dynamics["dwell"]:dwell_by_attempt[d["attempt_id"]].append(d)
    for d in dynamics["transitions"]:transitions_by_attempt[d["attempt_id"]].append(d)
    for d in dynamics["reentry"]:reentry_by_attempt[d["attempt_id"]].append(d)
    for aid,sent in by_attempt.items():
        source=source_lookup[aid];n=sum(s["n_tokens"] for s in sent)
        row={"benchmark":source["benchmark"],"family":source["family"],"attempt_id":aid,
            "observed_labelled_tokens":n,"label_coverage":n/source["completion_tokens"] if source["completion_tokens"] else None}
        for c in CLASSES:
            row[c+":occupancy"]=sum(s["n_tokens"] for s in sent if s["class"]==c)/n if n else None
            dwell=[d for d in dwell_by_attempt[aid] if d["class"]==c]
            row[c+":observed_dwell_tokens"]=float(np.mean([d["tokens"] for d in dwell])) if dwell else None
            row[c+":uncensored_dwell_tokens"]=float(np.mean([d["tokens"] for d in dwell if not d["left_censored"] and not d["right_censored"]])) if any(not d["left_censored"] and not d["right_censored"] for d in dwell) else None
            trans=[d for d in transitions_by_attempt[aid] if d["source"]==c]
            row[c+":reentry_per_observed_boundary"]=sum(d["class"]==c for d in reentry_by_attempt[aid])/len(trans) if trans else None
            for next_class in CLASSES:
                row[c+":next:"+next_class]=sum(d["destination"]==next_class for d in trans)/len(trans) if trans else None
        native_dynamics_rows.append(row)
    frame(root/model/"native_dynamics_estimands.parquet",native_dynamics_rows)
    columns=sorted({k for r in native_dynamics_rows for k in r if k not in ("benchmark","family","attempt_id")})
    publish(root/model/"NATIVE_DYNAMICS_SUMMARY.json",bootstrap_panel(native_dynamics_rows,columns,panel="native-dynamics:"+model))
    nominations, atlas, scores, support_tables = [], [], [], []
    family_fold = np.array([folds[r["family"]] for r in sources])
    for fold in range(5):
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
                            for e in range(contract["experts"])] for l in layers])
        scores.append(np.stack([state, rates[7], rates[8], random]))
        support_tables.append(support)
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
                for e in range(contract["experts"]):
                    value = excess[c, i, e]
                    atlas.append({"model": model, "fold": fold, "class": name, "layer": l, "expert": e,
                        "inclusion_rate": float(rates[c, i, e]) if np.isfinite(rates[c, i, e]) else None,
                        "other_class_inclusion": float(other_rate[c, i, e]) if np.isfinite(other_rate[c, i, e]) else None,
                        "inclusion_difference": float(value) if np.isfinite(value) else None,
                        "enrichment": float(rates[c, i, e] / other_rate[c, i, e]) if other_rate[c, i, e] > 0 else None,
                        "family_support": int(support[c, i, e]), "selection_stability": float(stab[c, i])})
    score_file = npz(root / model / "SCORES.npz", scores=np.stack(scores), support=np.stack(support_tables))
    atlas_file = frame(root / model / "association_atlas.parquet", atlas)
    nomination_file = publish(root / model / "NOMINATIONS.json", {"records": nominations,
        "outcomes_used": False, "layers": layers, "folds_sha256": digest(folds)})
    # Strict native fields are used uniformly beside historically adjudicated
    # descriptive estimates. No candidate-verification/J1 judge is restarted.
    from .accuracy import native_accuracy
    _, accuracy = native_accuracy(campaign, model)
    return publish(root / model / "DERIVED.json", {"status": "complete", "scores": score_file,
        "atlas": atlas_file, "nominations_sha256": nomination_file["sha256"],
        "native_accuracy_sha256": accuracy["sha256"], "dynamics_sha256": dynamics["sha256"],
        "source_conditioned_association_input": campaign["saved_inputs"][model]["histogram"],
        "hidden_behavior_reused": campaign["saved_inputs"][model]["features"],
        "existing_depth_paths_reused": str(Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/results/part1") / (model + ".json"))})
