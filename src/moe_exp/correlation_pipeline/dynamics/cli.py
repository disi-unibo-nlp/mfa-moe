"""correlation-dynamics: independent class-only, routing-only and combined analysis."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from . import SCHEMA_VERSION
from .common import DEFAULT_CONFIG, artifact_path, digest, identity, keys, termination, write_json, write_rows
from .classes import class_dynamics, saved_layout, sentence_table, summaries
from moe_exp.correlation_pipeline.provenance import code_provenance, file_sha256
from moe_exp.correlation_pipeline.spans import trace_digest
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord


def analyze(paths, output, *, annotations=(), mode="combined", raw_routing=False,
            evaluate_models=False, plots=False, config=None, bootstrap=1000):
    config = {**DEFAULT_CONFIG, **(config or {})}
    output = Path(output)
    annotation_map = {}
    for path in annotations:
        for a in iter_jsonl(path):
            key = (a.get("source_model"), a["dataset"], a["problem_id"], a.get("sample_id", 0))
            if key in annotation_map:
                raise ValueError("Duplicate annotation identity")
            annotation_map[key] = a
    inputs = [{"sha256": file_sha256(p), "role": "traces"} for p in paths]
    inputs += [{"sha256": file_sha256(p), "role": "annotations"} for p in annotations]
    # Artifact contents, not absolute paths, bind resume and remain valid on relocation.
    tensors = {}
    if mode != "class-only":
        for path in paths:
            for record in iter_jsonl(path):
                for name in ("selected_experts", "router_logits", "expert_weights"):
                    if name != "selected_experts" and not raw_routing:
                        continue
                    found = artifact_path(path, record.get("model_logs", {}).get(name))
                    if found:
                        tensors[(digest([record["model_id"], record["dataset"], record["problem_id"],
                                         record.get("sample_id", 0)]), name)] = file_sha256(found)
    binding = dict(schema_version=SCHEMA_VERSION, inputs=inputs,
        tensors=[dict(trace_id=k[0], signal=k[1], sha256=v) for k, v in sorted(tensors.items())],
        mode=mode, raw_routing=raw_routing, evaluate_models=evaluate_models, plots=plots,
        bootstrap=bootstrap, config=config, code=code_provenance())
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("binding") != binding:
            raise ValueError("Dynamics output has a different source/schema/config/code contract; choose a new output directory")
        if saved.get("status") == "complete" and all((output / p).is_file() and file_sha256(output / p) == h
                                                    for p, h in saved.get("outputs", {}).items()):
            return saved
        raise ValueError("Incomplete/corrupt dynamics bundle; choose a new output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty unbound output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {"status": "running", "binding": binding})
    sentences, windows, expert_rows, coverage, predictions, seen = [], [], [], [], [], set()
    for path in paths:
        for record in iter_jsonl(path):
            trace = TraceRecord(**record)
            key = tuple(identity(trace))
            if key in seen:
                raise ValueError("Duplicate trace identity")
            seen.add(key)
            annotation = annotation_map.get(key)
            # Legacy annotation files may omit source_model, but never join across two sources.
            if annotation is None:
                annotation = annotation_map.get((None, trace.dataset, trace.problem_id, trace.sample_id))
            layout = saved_layout(trace)
            current = sentence_table(trace, annotation, layout) if mode != "routing-only" else []
            sentences.extend(current)
            current_windows, availability = [], {}
            compact = trace.metadata.get("routing_dynamics")
            selected_path = artifact_path(path, trace.model_logs.selected_experts)
            if mode != "class-only":
                from .routing import reduce_arrays, probabilities
                if compact is not None:
                    if compact.get("schema_version") != SCHEMA_VERSION or compact.get("config") != config:
                        raise ValueError("Incompatible compact dynamics schema/config")
                    if compact.get("trace_sha256") != trace_digest(trace):
                        raise ValueError("Stale compact dynamics")
                    if compact.get("forward_provenance") != trace.metadata.get("forward_provenance"):
                        raise ValueError("Compact dynamics checkpoint provenance mismatch")
                    actual_annotation = annotation or trace.metadata.get("reasoning_annotation")
                    if compact.get("annotation_sha256") != digest(actual_annotation):
                        raise ValueError("Compact dynamics annotation contract mismatch")
                    current_windows = compact["windows"]
                    availability = compact.get("semantics", {})
                if selected_path is not None:
                    import torch
                    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
                    layers = trace.model_logs.layer_indices
                    if layers is None:
                        # Legacy tensors do not prove original layer identity.
                        raise ValueError("Explicit router layer identities required")
                    if layout["token_count"] is None or selected.shape[1] != layout["token_count"]:
                        raise ValueError("Native expert/token alignment unavailable or mismatched")
                    from .experts import sentence_experts
                    expert_rows.extend(sentence_experts(current, selected, layers))
                    if compact is None:
                        router_path = artifact_path(path, trace.model_logs.router_logits) if raw_routing else None
                        weights_path = artifact_path(path, trace.model_logs.expert_weights) if raw_routing else None
                        router = torch.load(router_path, map_location="cpu", weights_only=True) if router_path else None
                        weights = torch.load(weights_path, map_location="cpu", weights_only=True) if weights_path else None
                        # Old scalars cannot recover full probabilities or executed mixture weights.
                        n_experts = trace.metadata.get("correlation_features", {}).get("num_experts")
                        if n_experts is None and router is not None:
                            n_experts = router.shape[-1]
                        if n_experts is None:
                            raise ValueError("Model expert count unavailable; refusing to infer from observed IDs")
                        class ProbabilitySlices:
                            shape = router.shape if router is not None else None
                            def __getitem__(self, ix):
                                return probabilities(router[ix].float().numpy())
                        availability = dict(num_experts=n_experts, deterministic_topk=False,
                            native_selection=bool(trace.metadata.get("router_capture_version")),
                            selection_scores="unavailable", executed_weights="available" if weights is not None else "unavailable")
                        current_windows = reduce_arrays(selected, layers=layers, model=trace.model_id,
                            probabilities_array=ProbabilitySlices() if router is not None else None,
                            weights=weights, semantics=availability, config=config,
                            token_indices=layout["reasoning_tokens"])
                        del router, weights
                    del selected
                elif compact is None:
                    availability = {"status": "unavailable", "reason": "no native routing artifacts"}
                windows.extend([{**keys(trace), **w} for w in current_windows])
            # Assign run ages before building retrospective rows.
            class_dynamics(current)
            if evaluate_models:
                from .evaluation import prediction_rows
                predictions.extend(prediction_rows(trace, current, current_windows))
            coverage.append({**keys(trace), "sentences": len(current),
                "labeled_sentences": sum(r["label"] is not None for r in current),
                "adjacent_labeled_pairs": sum(r["label"] is not None and r["next_class"] is not None for r in current),
                "token_alignment": layout["token_count"] is not None,
                "reasoning_tokens": len(layout["reasoning_tokens"]) if layout["reasoning_tokens"] is not None else None,
                "termination": termination(trace), "scoring_method": trace.scoring_method,
                "generation_config": trace.metadata.get("generation_config"),
                "forward_provenance": trace.metadata.get("forward_provenance"),
                "router_layers": trace.model_logs.layer_indices, "routing_windows": len(current_windows),
                "routing_availability": availability})
    tables = class_dynamics(sentences)
    tables.update(summaries(tables, config["min_problems"]))
    tables.update(sentences=sentences, windows=windows, expert_sentences=expert_rows, coverage=coverage)
    from .experts import associations
    tables["expert_associations"] = associations(sentences, expert_rows,
        min_problems=config["min_problems"], bootstrap=bootstrap, seed=config["seed"])
    evaluation = {}
    if evaluate_models:
        from .evaluation import evaluate, bh_adjust
        by_task = defaultdict(list)
        for row in predictions:
            by_task[(row["model"], row["dataset"], row["task"])].append(row)
        for key, rows in sorted(by_task.items()):
            evaluation["/".join(key)] = evaluate(rows, seed=config["seed"], bootstrap=bootstrap,
                                                min_problems=config["min_problems"])
        # All searched landmarks/comparisons in one declared model/dataset/target family.
        families = defaultdict(list)
        for key, result in evaluation.items():
            target = key.rsplit("/", 1)[1]
            family = key.rsplit("/", 1)[0] + ("/correctness" if target.startswith("correctness") else "/next_class")
            families[family].extend(result.get("comparisons", []))
        for comparisons in families.values():
            for row, q in zip(comparisons, bh_adjust([r["p_bootstrap"] for r in comparisons])):
                row["q_bh"] = q
                row["status"] = "incrementally_predictive" if q is not None and q < .05 and row["confidence_interval"][0] > 0 else "unsupported"
    else:
        evaluation = {"status": "unavailable", "reason": "evaluation was not requested"}
    for name, rows in tables.items():
        write_rows(output / (name + ".jsonl"), rows)
    write_json(output / "evaluation.json", evaluation)
    summary = dict(status="descriptive", traces=len(coverage), questions=len({r["question_id"] for r in coverage}),
        sentences=len(sentences), labeled_sentences=sum(r["label"] is not None for r in sentences),
        adjacent_labeled_pairs=len(tables["transitions"]), routing_windows=len(windows),
        capped_attempts=sum(r["termination"] == "generation_cap" for r in coverage),
        scored_attempts=sum(r["is_correct"] is not None for r in coverage),
        correct_attempts=sum(r["is_correct"] is True for r in coverage),
        limitations=["Associations do not establish beneficial forced expert activation.",
                     "Missing metrics are unavailable; no distributions are reconstructed from scalar summaries.",
                     "Corrected overlap is a transform of raw overlap, not independent evidence."])
    write_json(output / "summary.json", summary)
    from .report import render_report
    render_report(output, summary, tables, evaluation, plots=plots)
    outputs = {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*"))
               if p.is_file() and p != manifest_path}
    result = dict(status="complete", binding=binding, outputs=outputs, summary=summary)
    write_json(manifest_path, result)
    return result


def main(argv=None):
    # Keep all historical subcommands; the versioned campaign stages are opt-in.
    import sys
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied and supplied[0] == "steering":
        from .steering_cli import main as steering_main
        return steering_main(supplied[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("steering", help="Versioned campaign inventory, folds, extraction and pilot stages")
    p = sub.add_parser("analyze")
    p.add_argument("--traces", nargs="+", type=Path, required=True)
    p.add_argument("--annotations", nargs="*", type=Path, default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("class-only", "routing-only", "combined"), default="combined")
    p.add_argument("--raw-routing", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    p = sub.add_parser("prepare-pilot")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("audit")
    p.add_argument("--traces", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        result = analyze(args.traces, args.output, annotations=args.annotations, mode=args.mode,
                         raw_routing=args.raw_routing, evaluate_models=args.evaluate, plots=args.plots,
                         bootstrap=args.bootstrap)
    elif args.command == "prepare-pilot":
        from .pilot import prepare
        result = prepare(args.traces, args.fixtures, args.program, args.output, seed=args.seed)
    else:
        from .audit import audit, audit_tables
        result = audit_tables(args.traces[0], args.output) if len(args.traces) == 1 and args.traces[0].suffix == ".csv" else audit(args.traces, args.output)
    print(json.dumps(result.get("summary", {"status": result.get("status")}), indent=2))
