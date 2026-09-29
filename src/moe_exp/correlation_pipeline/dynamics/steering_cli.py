"""Campaign steering stages. Commands never submit jobs or start an inference server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import freeze_folds, load, save, seal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--manifests", type=Path, required=True)
    inv.add_argument("--acceptance", type=Path, required=True)
    inv.add_argument("--outcomes", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    fold = sub.add_parser("folds")
    fold.add_argument("--inventory", type=Path, required=True)
    fold.add_argument("--output", type=Path, required=True)
    base = sub.add_parser("baselines")
    base.add_argument("--inventory", type=Path, required=True)
    base.add_argument("--output", type=Path, required=True)
    base.add_argument("--bootstrap", type=int, default=1000)
    ex = sub.add_parser("extract")
    ex.add_argument("--inventory", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--limit-per-cohort", type=int)
    ex.add_argument("--include-weights", action="store_true")
    ex.add_argument("--landmarks", type=int, nargs="+", default=[1024, 2048, 4096])
    ex.add_argument("--shard-index", type=int, default=0)
    ex.add_argument("--shard-count", type=int, default=1)
    ex.add_argument("--max-output-bytes", type=int)
    merge = sub.add_parser("merge-extractions")
    merge.add_argument("--inventory", type=Path, required=True)
    merge.add_argument("--manifests", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    sizing = sub.add_parser("resources")
    sizing.add_argument("--inventory", type=Path, required=True)
    sizing.add_argument("--sizing", type=Path, required=True)
    sizing.add_argument("--output", type=Path, required=True)
    sizing.add_argument("--startup-attempt-id")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--features", type=Path, required=True)
    analysis.add_argument("--folds", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--bootstrap", type=int, default=1000)
    analysis.add_argument("--evaluate", action="store_true")
    pilot = sub.add_parser("pilot-plan")
    for name in ("prefixes", "folds", "candidate", "calibration", "qualification", "output"):
        pilot.add_argument("--" + name, type=Path, required=True)
    pilot.add_argument("--trigger", type=Path)
    paired = sub.add_parser("pilot-analysis")
    paired.add_argument("--branches", type=Path, required=True)
    paired.add_argument("--results", type=Path, nargs="+", required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    if args.stage == "inventory":
        from .campaign import freeze_inventory, inventory_report
        result = freeze_inventory(args.manifests, args.acceptance, args.outcomes)
        save(args.output, result)
        inventory_report(result, args.output.with_suffix(".md"))
        summary = dict(attempts=len(result["payload"]["attempts"]), cohorts=len(result["payload"]["cohorts"]))
    elif args.stage == "folds":
        inv = load(args.inventory, "inventory")
        result = freeze_folds(inv["payload"]["attempts"], inv["binding"])
        save(args.output, result)
        summary = dict(reserved=len(result["payload"]["reserved_questions"]),
                       development=len(result["payload"]["outer"]))
    elif args.stage == "baselines":
        from .baselines import summarize, render
        result = summarize(load(args.inventory, "inventory"), bootstrap=args.bootstrap)
        save(args.output, result)
        render(result, args.output.with_suffix(".md"))
        summary = dict(tables=len(result["payload"]["tables"]))
    elif args.stage == "extract":
        from .campaign import extract
        result = extract(args.inventory, args.output, limit_per_cohort=args.limit_per_cohort,
                         include_weights=args.include_weights, landmarks=args.landmarks,
                         shard_index=args.shard_index, shard_count=args.shard_count,
                         max_output_bytes=args.max_output_bytes)
        summary = dict(attempts=len(result["payload"]["features"]))
    elif args.stage == "merge-extractions":
        from .campaign import merge_extractions
        result = merge_extractions(args.inventory, args.manifests)
        save(args.output, result)
        summary = dict(attempts=len(result["payload"]["features"]))
    elif args.stage == "resources":
        from .campaign import measured_resource_estimate
        result = measured_resource_estimate(load(args.inventory, "inventory"), load(args.sizing, "analysis"),
                                           startup_attempt_id=args.startup_attempt_id)
        save(args.output, result)
        summary = dict(models=len(result["payload"]["models"]))
    elif args.stage == "analyze":
        result = analyze(args.features, args.folds, bootstrap=args.bootstrap, evaluate=args.evaluate)
        save(args.output, result)
        summary = dict(status=result["payload"]["status"])
    elif args.stage == "pilot-plan":
        from .experiment import freeze_branches
        read = lambda path: json.loads(path.read_text())
        result = freeze_branches(read(args.prefixes), load(args.folds, "folds"), read(args.candidate),
            read(args.calibration), read(args.qualification), trigger=read(args.trigger) if args.trigger else None)
        save(args.output, result)
        summary = dict(branches=len(result["payload"]["branches"]))
    else:
        from .experiment import analyze_pilot
        from .contracts import file_binding
        branches = load(args.branches, "branches")
        results = [load(path, "analysis")["payload"] for path in args.results]
        result = seal("analysis", analyze_pilot(branches, results, bootstrap=args.bootstrap),
            inputs={"branches": branches["binding"], "results": [file_binding(p) for p in args.results]},
            config=dict(bootstrap=args.bootstrap), population=branches["population"])
        save(args.output, result)
        summary = dict(status=result["payload"]["status"])
    print(json.dumps(dict(binding=result["binding"], **summary)))


def analyze(features_path, folds_path, *, bootstrap=1000, evaluate=False):
    from .contrasts import (CONTRASTS, candidate_cards, estimate_contrast,
                            joint_question_draws, retrospective_length_bands)
    from .contracts import file_binding
    manifest, folds = load(features_path, "analysis"), load(folds_path, "folds")
    if manifest["population"] == "sizing_only":
        raise ValueError("Sizing samples cannot be used for scientific analysis")
    if manifest["inputs"]["inventory"] != folds["inputs"]["inventory"]:
        raise ValueError("Features and folds have different inventory bindings")
    rows, specs = [], {}
    for ref in manifest["payload"]["features"]:
        if file_binding(ref["path"]) != ref:
            raise ValueError("Feature file changed")
        artifact = load(ref["path"], "features")
        if artifact["inputs"] != manifest["inputs"]:
            raise ValueError("Feature artifact is from another inventory")
        feature_config = {k: v for k, v in artifact["config"].items() if k not in ("shard_index", "shard_count")}
        manifest_config = {k: v for k, v in manifest["config"].items() if k not in ("shard_index", "shard_count")}
        if (artifact["code"] != manifest["code"] or feature_config != manifest_config
                or artifact["population"] != manifest["population"]):
            raise ValueError("Feature configuration/code/population differs from its manifest")
        attempt = artifact["payload"]["attempt"]
        spec = {k: attempt[k] for k in ("layers", "num_experts", "top_k")}
        if attempt["model"] in specs and specs[attempt["model"]] != spec:
            raise ValueError("Model has incompatible native identities")
        specs[attempt["model"]] = spec
        rows.extend(r for r in artifact["payload"]["rows"] if r["question_id"] in folds["payload"]["outer"])
    if not rows:
        raise ValueError("No eligible development features")
    draws = joint_question_draws(folds["payload"]["outer"], bootstrap=bootstrap)
    contrasts, candidates = {}, []
    retrospective, length_cutoffs = retrospective_length_bands(rows, set(folds["payload"]["outer"]))
    for name in CONTRASTS:
        if name == "destination":
            from .common import CLASSES
            contrasts[name] = {d: estimate_contrast(rows, specs, name, destination=d,
                                                  draws=draws) for d in CLASSES}
        else:
            contrasts[name] = estimate_contrast(retrospective, specs, name, draws=draws)
    for name in ("accuracy", "efficient_success"):
        candidates.extend(candidate_cards(rows, specs, folds, contrast=name, bootstrap=bootstrap))
    evaluation = {}
    if evaluate:
        from .prospective import evaluate as fit
        for model in specs:
            for cohort in ("A", "B"):
                for target in ("is_correct", "remaining_tokens", "next_event", "next_class"):
                    evaluation[f"{model}/{cohort}/{target}"] = fit(
                        [r for r in rows if r["model"] == model], folds,
                        target=target, cohort=cohort, bootstrap=bootstrap)
        from collections import defaultdict
        from .evaluation import bh_adjust
        families = defaultdict(list)
        for key, value in evaluation.items():
            families[key.split("/")[0]].extend(value.get("comparisons", []))
        for model, comparisons in families.items():
            for comparison, q in zip(comparisons, bh_adjust([c["p"] for c in comparisons])):
                comparison["q_bh"] = q
                comparison["multiplicity_family"] = model + "/all_prospective_targets_and_cohorts"
    # One overall maximum of three compact groups per model across discovery contrasts.
    candidates.sort(key=lambda c: (-c["stable_folds"], -c["minimum_training_questions"], -c["standardized_effect"]))
    capped_candidates = []
    for model in specs:
        unique = set()
        for c in (c for c in candidates if c["model"] == model):
            key = (c["layer"], tuple(c["experts"]))
            if key not in unique and len(unique) < 3:
                capped_candidates.append(c)
                unique.add(key)
    natural = [r for r in retrospective if not r["capped"]]
    sensitivity = {name: estimate_contrast(natural, specs, name, draws=draws)
                   for name in ("accuracy", "efficient_success", "short_failure", "long_failure")}
    payload = dict(status="exploratory_analysis", contrasts=contrasts, candidate_cards=capped_candidates,
        natural_completion_sensitivity=sensitivity,
        retrospective_length_cutoffs=[dict(model=m, dataset=d, cutoff=v) for (m, d), v in length_cutoffs.items()],
        prospective=evaluation, coverage=manifest["payload"]["coverage"],
        unavailable=["State contrasts require contiguous, timed labels",
                     "Length/difficulty-adjusted regression is not a deployable predictor",
                     "No causal pilot has run"])
    return seal("analysis", payload, inputs={"features": manifest["binding"], "folds": folds["binding"]},
                config=dict(bootstrap=bootstrap, evaluate=evaluate), population=manifest["population"])


if __name__ == "__main__":
    main()
