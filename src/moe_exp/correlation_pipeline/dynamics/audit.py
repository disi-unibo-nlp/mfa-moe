"""Audit legacy scoring and association sensitivity without changing existing outputs."""
from collections import Counter, defaultdict
import numpy as np
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord
from moe_exp.correlation_pipeline.provenance import file_sha256
from .common import keys, termination, write_json


def correlation(x, y, controls=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if controls is not None:
        z = np.c_[np.ones(len(x)), np.asarray(controls, float)]
        x = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def audit(paths, output):
    rows, seen = [], set()
    for path in paths:
        for record in iter_jsonl(path):
            t = TraceRecord(**record)
            k = keys(t)
            if k["trace_id"] in seen:
                raise ValueError("Duplicate composite attempt")
            seen.add(k["trace_id"])
            compact = t.metadata.get("correlation_features", {})
            rows.append({**k, "termination": termination(t), "tokens": compact.get("token_count"),
                "features": compact.get("values", {}), "layers": t.model_logs.layer_indices,
                "scoring_method": t.scoring_method, "forward_provenance": t.metadata.get("forward_provenance"),
                "label_units": len(t.metadata.get("reasoning_annotation", {}).get("units", [])),
                "exact_token_replay": "token_replay" in t.metadata})
    return summarize(rows, paths, output)


def summarize(rows, paths, output):
    datasets = {}
    features = ("router_confidence_mean_layers", "router_margin_mean_layers",
                "router_boundary_margin_mean_layers", "hidden_norm_mean_layers",
                "hidden_step_distance_mean_layers", "hidden_router_geometry_mean_layers")
    for model, dataset in sorted({(r["model"], r["dataset"]) for r in rows}):
        group = [r for r in rows if (r["model"], r["dataset"]) == (model, dataset)]
        scored = [r for r in group if r["is_correct"] is not None]
        attempts = Counter(r["question_id"] for r in group)
        result = dict(traces=len(group), questions=len(attempts), scored=len(scored),
            correct=sum(r["is_correct"] for r in scored), caps=sum(r["termination"] == "generation_cap" for r in group),
            attempts_per_question=dict(Counter(attempts.values())), scoring_methods=dict(Counter(r["scoring_method"] for r in group)),
            exact_replay=sum(r["exact_token_replay"] for r in group),
            labeled_units=sum(r["label_units"] for r in group),
            layer_sets=[list(x) for x in sorted({tuple(r["layers"] or []) for r in group})],
            forward_provenance_available=sum(r["forward_provenance"] is not None for r in group), associations={})
        for feature in features:
            valid = [r for r in scored if r["features"].get(feature) is not None and r["tokens"] is not None]
            x = [r["features"][feature] for r in valid]
            y = [int(r["is_correct"]) for r in valid]
            controls = [[np.log1p(r["tokens"]), int(r["termination"] == "generation_cap")] for r in valid]
            noncap = [r for r in valid if r["termination"] != "generation_cap"]
            by_question = defaultdict(list)
            for r in valid:
                by_question[r["question_id"]].append(r)
            within_x, within_y, contrasts = [], [], []
            for question_rows in by_question.values():
                if len({r["is_correct"] for r in question_rows}) < 2:
                    continue
                xx = np.array([r["features"][feature] for r in question_rows])
                yy = np.array([int(r["is_correct"]) for r in question_rows])
                within_x.extend(xx - xx.mean())
                within_y.extend(yy - yy.mean())
                contrasts.append(float(xx[yy == 1].mean() - xx[yy == 0].mean()))
            result["associations"][feature] = dict(n=len(valid), marginal=correlation(x, y),
                length_termination_partial=correlation(x, y, controls) if valid else None,
                noncapped=correlation([r["features"][feature] for r in noncap],
                                     [int(r["is_correct"]) for r in noncap]),
                within_question=correlation(within_x, within_y),
                mixed_outcome_questions=len(contrasts),
                mean_within_question_difference=float(np.mean(contrasts)) if contrasts else None)
        datasets[model + "/" + dataset] = result
    result = dict(status="descriptive", sources=[dict(sha256=file_sha256(p), file=str(p)) for p in paths],
                  datasets=datasets, intervention_reversal="unverified: corresponding intervention artifacts not supplied",
                  position_sensitivity="unavailable in whole-trace input; inspect position-view artifacts separately",
                  expert_identity_warning="Legacy expert IDs pooled across layers cannot support layer-specific claims",
                  precision_warning="Generation and replay checkpoints/quantization may differ; see source records",
                  primary_policy="Retain capped attempts; noncapped estimates are sensitivity analyses only")
    write_json(output, result)
    return result


def audit_tables(path, output):
    """Bounded audit of existing compact CSVs, including all saved position views."""
    import csv
    import math
    from pathlib import Path
    from .common import digest
    def number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    def read_table(table):
        records = []
        with Path(table).open() as handle:
            for r in csv.DictReader(handle):
                records.append(dict(model=r["model_id"], dataset=r["dataset"],
                    trace_id=digest([r["model_id"],r["dataset"],r["problem_id"],r.get("sample_id")]),
                    question_id=digest([r["dataset"],r.get("source_problem_id") or r["problem_id"]]),
                    is_correct=bool(int(r["is_correct"])) if r["is_correct"] in ("0","1") else None,
                    termination="generation_cap" if r.get("generation_hit_token_limit")=="1" else "natural_termination" if r.get("generation_finish_reason")=="stop" else "unknown_termination",
                    tokens=number(r.get("token_count")),
                    features={k:number(v) for k,v in r.items() if k.endswith("_mean_layers")},
                    layers=sorted(int(k.rsplit("_l",1)[1]) for k in r if k.startswith("router_confidence_l")),
                    scoring_method=r.get("scoring_method"), forward_provenance=None,
                    label_units=0, exact_token_replay=False))
        if len({r["trace_id"] for r in records}) != len(records):
            raise ValueError("Duplicate CSV identity")
        return records
    path=Path(path)
    result=summarize(read_table(path),[path],output)
    position=[]
    for table in sorted((path.parent/"views-v1/position").glob("*/trace_features.csv")):
        rows=read_table(table)
        for dataset in sorted({r["dataset"] for r in rows}):
            valid=[r for r in rows if r["dataset"]==dataset and r["is_correct"] is not None
                   and r["features"].get("router_confidence_mean_layers") is not None]
            position.append(dict(position=table.parent.name,dataset=dataset,n=len(valid),
                router_confidence_r=correlation([r["features"]["router_confidence_mean_layers"] for r in valid],
                                                [int(r["is_correct"]) for r in valid])))
    result["position_sensitivity"]=position or "unavailable: no saved position tables"
    result["coverage_note"]="CSV audit; label coverage, exact replay and missing checkpoint provenance cannot be recovered from CSV. Those zero counts are unavailable, not evidence of absent source metadata."
    result["primary_policy"]="All CSV attempts retained. Compare scoring denominator to source manifests before claiming complete benchmark coverage."
    write_json(output,result)
    return result
