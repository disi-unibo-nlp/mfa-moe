"""Private CLI; real archive fits and model execution require approved Slurm cards."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .contracts import REPO, allocated, check_files, digest, file_sha, load, require, save
from .protocol import MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run-card")
    run.add_argument("--campaign", required=True)
    run.add_argument("--card", required=True)
    run.add_argument("--authorization", required=True)
    run.add_argument("--deadline-epoch", type=float, required=True)
    cards = sub.add_parser("make-card")
    cards.add_argument("--campaign", required=True)
    cards.add_argument("--stage", required=True)
    cards.add_argument("--model", choices=MODELS)
    cards.add_argument("--request", required=True)
    cards.add_argument("--price")
    cards.add_argument("--seconds", type=int)
    cards.add_argument("--output", required=True)
    submit = sub.add_parser("submit")
    for name in ("campaign", "card", "authorization", "root"):
        submit.add_argument("--" + name, required=True)
    reconcile = sub.add_parser("reconcile-submission")
    for name in ("card", "root"):
        reconcile.add_argument("--" + name, required=True)
    masks = sub.add_parser("freeze-masks")
    masks.add_argument("--campaign", required=True)
    masks.add_argument("--selection-root", required=True)
    masks.add_argument("--output", required=True)
    population = sub.add_parser("freeze-population")
    population.add_argument("--campaign", required=True)
    population.add_argument("--exposure-ledger", required=True)
    population.add_argument("--output", required=True)
    roster = sub.add_parser("freeze-assignments")
    for name in ("population", "frozen-masks", "output"):
        roster.add_argument("--" + name, required=True)
    roster.add_argument("--stage", choices=("pilot", "final"), required=True)
    roster.add_argument("--sample-size")
    for command in ("size-final", "analyze"):
        p = sub.add_parser(command)
        for name in ("campaign", "grades", "assignments", "output"):
            p.add_argument("--" + name, required=True)
    price = sub.add_parser("price")
    for name in ("campaign", "result", "closeout", "output"):
        price.add_argument("--" + name, required=True)
    price.add_argument("--source-result")
    plan = sub.add_parser("plan-final")
    for name in ("campaign", "assignments", "frozen-masks", "prices", "availability", "root", "output"):
        plan.add_argument("--" + name, required=True)
    approve = sub.add_parser("approve-final-plan")
    for name in ("plan", "review", "output"):
        approve.add_argument("--" + name, required=True)
    export = sub.add_parser("export")
    for name in ("campaign", "root", "output"):
        export.add_argument("--" + name, required=True)
    status = sub.add_parser("status")
    status.add_argument("--campaign", required=True)
    status.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    result = None
    if args.command == "prepare":
        from .campaign import prepare as do_prepare
        result = do_prepare(args.output)
    elif args.command == "run-card":
        from .stages import run
        require(args.deadline_epoch > time.time(), "allocation already expired")
        result = run(load(args.campaign), load(args.card), load(args.authorization), deadline=args.deadline_epoch)
    elif args.command == "make-card":
        from .cards import card
        result = save(args.output, card(load(args.campaign), args.stage, model=args.model, seconds=args.seconds,
                                       price=load(args.price) if args.price else None,
                                       request=json.loads(Path(args.request).read_text())))
    elif args.command == "submit":
        from .cards import submit as do_submit
        result = do_submit(args.card, args.authorization, args.campaign, args.root)
    elif args.command == "reconcile-submission":
        from .cards import reconcile_submission
        result = reconcile_submission(args.card, args.root)
    elif args.command == "freeze-masks":
        from .campaign import freeze_selections
        result = freeze_selections(load(args.campaign), {m: Path(args.selection_root) / m / "SELECTION.json" for m in MODELS}, args.output)
    elif args.command == "freeze-population":
        from .population import freeze_population
        campaign = load(args.campaign)
        exposure = load(args.exposure_ledger)
        check_files(exposure["source_files"])
        sources = {name: load(REPO / "report/routing-study-v1" / name) for name in campaign["population_source_seals"]}
        require(all(sources[name]["sha256"] == expected for name, expected in campaign["population_source_seals"].items()),
                "population source is outside the campaign closure")
        result = save(args.output, {**freeze_population(sources["FULL_MATH_QUESTION_TABLE.json"],
                      sources["FULL_MATH_DUPLICATE_GRAPH.json"], sources["INDEPENDENT_FAMILY_MANIFEST.json"], exposure),
                      "campaign_sha256": campaign["sha256"]})
    elif args.command == "freeze-assignments":
        from .population import assignments
        require(args.stage == "pilot" or args.sample_size, "final sample size not frozen")
        size = load(args.sample_size) if args.sample_size else None
        population, freeze = load(args.population), load(args.frozen_masks)
        if size:
            require(size["campaign_sha256"] == freeze["campaign_sha256"] and
                    size["population_sha256"] == population["sha256"] and
                    size["frozen_masks_sha256"] == freeze["sha256"], "final size belongs to foreign pilot sources")
        result = save(args.output, assignments(population, freeze["models"], stage=args.stage,
                                              final_size=size["N"] if size else None,
                                              frozen_masks_sha256=freeze["sha256"],
                                              sample_size_sha256=size["sha256"] if size else None))
    elif args.command in ("size-final", "analyze"):
        from .stages import arm_summaries, score_grid
        from .statistics import blinded_discordance, paired_analysis, sample_size
        campaign, roster, grades = load(args.campaign), load(args.assignments), load(args.grades)
        require(roster["campaign_sha256"] == campaign["sha256"] and
                grades["campaign_sha256"] == campaign["sha256"], "foreign sizing/analysis campaign")
        if args.command == "size-final":
            require(roster["stage"] == "pilot", "sizing from nonpilot outcomes forbidden")
            # Unknown semantic grades are treated as discordant for conservative sizing.
            value = sample_size(blinded_discordance(score_grid(grades, roster, semantic=True)))
            result = save(args.output, {**value, "grades_sha256": grades["sha256"],
                          "assignments_sha256": roster["sha256"], "campaign_sha256": campaign["sha256"],
                          "population_sha256": roster["population_sha256"],
                          "frozen_masks_sha256": roster["frozen_masks_sha256"]})
        else:
            allocated()
            require(roster["stage"] == "final", "prospective analysis from nonfinal population")
            result = save(args.output, {"schema": "expert-memory-seven-accuracy-v1",
                          "strict": paired_analysis(score_grid(grades, roster)),
                          "semantic_bounds": paired_analysis(score_grid(grades, roster, semantic=True)),
                          "arms": arm_summaries(grades, roster),
                          "grades_sha256": grades["sha256"], "assignments_sha256": roster["sha256"],
                          "campaign_sha256": campaign["sha256"]})
    elif args.command == "price":
        from .pricing import price_generation
        retained, closeout, campaign = load(args.result), load(args.closeout), load(args.campaign)
        receipt = load(args.source_result) if args.source_result else retained
        if args.source_result:
            require(retained.get("alias_receipt_sha256") == receipt["sha256"] and
                    retained["assignment"]["selection_sha256"] == receipt["assignment"]["selection_sha256"] and
                    retained["assignment"]["alias_of_arm"] == receipt["assignment"]["arm"], "foreign price policy alias")
        require(closeout["result_sha256"] == receipt["sha256"] and receipt["campaign_sha256"] == campaign["sha256"],
                "retained price receipt binding differs")
        a = retained["assignment"]
        require(a["stage"] == "pilot", "price measurement outside pilot")
        value = price_generation({**receipt, "closeout_seconds": closeout["seconds"],
                                   "artifact_bytes": Path(args.result).stat().st_size + Path(args.closeout).stat().st_size},
                                 model=a["model"], arm=a["arm"], context=40960)
        result = save(args.output, {**value, "closeout_sha256": closeout["sha256"], "campaign_sha256": campaign["sha256"],
                      "retained_result_sha256": retained["sha256"], "selection_sha256": a["selection_sha256"],
                      "policy_alias": bool(args.source_result)})
    elif args.command == "plan-final":
        from .pricing import final_plan
        from .stages import resolve_result
        from .protocol import ARMS
        campaign, roster, freeze = load(args.campaign), load(args.assignments), load(args.frozen_masks)
        paths, availability = load(args.prices), load(args.availability)
        check_files(availability["source_files"])
        require(availability["reviewed_epoch"] > time.time() - 3600 and
                availability["remaining_seconds"] <= datetime(2026, 10, 13, 23, 59, 59, tzinfo=timezone.utc).timestamp() - time.time(),
                "availability is stale or extends past the October 13 computation cutoff")
        prices = {(m, a): load(paths["models"][m][a]) for m in MODELS for a in ARMS}
        completed = [r["uid"] for r in roster["records"]
                     if (Path(args.root) / "results" / (r["uid"] + ".result.json")).exists()]
        for row in roster["records"]:
            if row["uid"] in completed:
                resolve_result(args.root, row, roster)
        value = final_plan(campaign, roster, freeze, prices, availability, completed_uids=completed)
        result = save(args.output, {**value, "price_source_files": {
            paths["models"][m][a]: file_sha(paths["models"][m][a]) for m in MODELS for a in ARMS}})
    elif args.command == "approve-final-plan":
        from .pricing import approve_final_plan
        plan = load(args.plan)
        check_files(plan["source_files"])
        check_files(plan["price_source_files"])
        result = save(args.output, approve_final_plan(plan, load(args.review)))
    elif args.command == "export":
        from .paper import export as do_export
        result = do_export(load(args.campaign), args.root, args.output)
    elif args.command == "status":
        campaign = load(args.campaign)
        check_files(campaign["scientific_files_preserved"])
        check_files(campaign["readiness_files_preserved"])
        result = {"models": {m: {"selection": (Path(args.root) / m / "SELECTION.json").exists(),
                      "qualification": load(Path(args.root) / m / "qualification/QUALIFICATION.json")["status"]
                      if (Path(args.root) / m / "qualification/QUALIFICATION.json").exists() else "UNMEASURED"} for m in MODELS},
                  "unresolved_assignments": [str(p) for p in (Path(args.root) / "results").glob("*.start-*.json")
                      if not p.with_name(p.name.split(".start-")[0] + ".result.json").exists()],
                  "performs_dispatch": False}
    print(json.dumps({"command": args.command, "sha256": result.get("sha256"),
                      "status": result.get("status", "COMPLETE"),
                      **({"observation": result} if args.command == "status" else {})}, sort_keys=True))
    return 0
