"""Bounded stage dispatch. No stage submits a successor or touches v3 ownership."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .contracts import allocated, check_files, digest, file_sha, load, require, save
from .journal import Journal
from .protocol import ARMS, MODELS


def validate_request(campaign, card, authorization):
    require(card["campaign_sha256"] == campaign["sha256"] and
            authorization["approved_card_sha256"] == card["sha256"] and
            authorization["resources"] == card["resources"] and authorization.get("user_quote"),
            "stage lacks exact workload/resource authorization")
    check_files(campaign["code_files"])
    check_files(campaign["scientific_files_preserved"])
    check_files(campaign["readiness_files_preserved"])
    require(card["request"] is not None, "missing finite request")


def generate(campaign, card, *, deadline):
    from .execution import Backend, loader_binding
    allocated(gpu=True)
    request = card["request"]
    cutoff = datetime(2026, 10, 13, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    require(time.time() < cutoff, "computation cutoff reached")
    deadline = min(deadline, cutoff)
    model, arm = card["model"], request["arm"]
    require(model in MODELS and arm in ARMS, "unrecognized model/arm")
    freeze = load(request["frozen_masks"])
    selection = freeze["models"][model]
    qualification = load(request["qualification"])
    require(qualification["status"] == "PASS" and qualification["selection_sha256"] == selection["sha256"] and
            qualification["campaign_sha256"] == campaign["sha256"], "exact frozen loader unqualified")
    roster, population = load(request["assignments"]), load(request["population"])
    require(roster["population_sha256"] == population["sha256"] and
            population["campaign_sha256"] == campaign["sha256"] and
            freeze["campaign_sha256"] == campaign["sha256"] and
            roster["campaign_sha256"] == campaign["sha256"] and
            roster["frozen_masks_sha256"] == freeze["sha256"], "foreign family/mask freeze")
    from .population import assignments
    expected = assignments(population, freeze["models"], stage=roster["stage"],
                           final_size=roster["families"] if roster["stage"] == "final" else None,
                           frozen_masks_sha256=freeze["sha256"], sample_size_sha256=roster["sample_size_sha256"])
    require(expected == roster, "assignment roster does not reproduce its frozen sources")
    records = [r for r in roster["records"] if r["model"] == model and r["arm"] == arm]
    chosen = request["assignment_uids"]
    require(chosen and len(chosen) == len(set(chosen)) and set(chosen) <= {r["uid"] for r in records},
            "foreign or duplicated finite assignments")
    records = [r for r in records if r["uid"] in chosen]
    require(all(r["selection_sha256"] == selection["sha256"] for r in records), "assignment selection differs")
    if card["stage"] == "pricing":
        require(roster["stage"] == "pilot" and len(records) == 1, "pricing must retain one original pilot assignment")
    else:
        require(request.get("price_path"), "unpriced bulk inference forbidden")
        price = load(request["price_path"])
        require(price["model"] == model and price["arm"] == arm and price["context"] == 40960 and
                price["campaign_sha256"] == campaign["sha256"] and
                price["selection_sha256"] == selection["sha256"] and
                card["price_basis"]["sha256"] == price["sha256"],
                "role/context price differs")
        from .pricing import shard_price
        require(shard_price(price, len(records))["reservation_seconds"] <= card["resources"]["seconds"],
                "finite card underprices its assignments")
    if card["stage"] == "final":
        size = load(request["sample_size"])
        feasibility = load(request["feasibility"])
        require(roster["stage"] == "final" and roster["families"] == size["N"] and
                roster["sample_size_sha256"] == size["sha256"] and
                size["campaign_sha256"] == campaign["sha256"] and
                size["population_sha256"] == population["sha256"] and
                size["frozen_masks_sha256"] == freeze["sha256"] and
                feasibility["status"] == "FEASIBLE_APPROVED" and
                feasibility["dispatch_authorized"] is True and
                feasibility["campaign_sha256"] == campaign["sha256"] and
                feasibility["frozen_masks_sha256"] == freeze["sha256"] and
                feasibility["assignments_sha256"] == roster["sha256"], "complete final allocation not approved")
        require(any(s["model"] == model and s["arm"] == arm and s["assignment_uids"] == chosen
                    for s in feasibility["shards"]), "card is outside the complete approved allocation")
    root = Path(request["root"])
    journal = Journal(root / "results")
    catalog = {r["question_id"]: r["problem"] for r in population["pilot"] + population["final_order"]}
    pending = []
    for assignment in records:
        if assignment["alias_of_arm"] != arm:
            source = next(r for r in roster["records"] if r["model"] == model and r["family"] == assignment["family"]
                          and r["arm"] == assignment["alias_of_arm"])
            receipt = load(root / "results" / (source["uid"] + ".result.json"))
            start = journal.begin(assignment)
            if start["status"] != "ALREADY_COMMITTED":
                journal.commit(assignment, {"alias_of_uid": source["uid"], "alias_receipt_sha256": receipt["sha256"],
                                           "execution_complete": receipt["execution_complete"],
                                           "no_duplicate_inference": True})
            continue
        # Begin is written just before actual invocation, never for an entire unrun shard.
        if not (root / "results" / (assignment["uid"] + ".result.json")).exists():
            pending.append(assignment)
        else:
            resolve_result(root, assignment, roster)
            materialize_aliases(journal, root, assignment, roster)
    if not pending:
        return {"status": "ALREADY_COMMITTED", "assignments": len(records)}
    binding_path = root / "bindings" / (card["sha256"] + ".json")
    if binding_path.exists():
        binding = load(binding_path)
        require(binding["selection_sha256"] == selection["sha256"] and binding["arm"] == arm,
                "existing loader binding differs")
    else:
        binding = loader_binding(campaign, model, selection, arm, binding_path)
    backend = Backend(campaign, model, binding_path, deadline=deadline)
    try:
        if card["stage"] == "pricing":
            lengths = {r["uid"]: len(backend.prompt_ids(catalog[r["question_id"]]))
                       for r in roster["records"] if r["model"] == model and r["arm"] == arm}
            largest = min(lengths, key=lambda uid: (-lengths[uid], uid))
            require(pending[0]["uid"] == largest, "retained pricing must use the original largest-input pilot assignment")
        # Bound equal-cache and backend invariants to observed qualification ranks.
        case = next(c for c in qualification["cases"] if c["arm"] == arm)
        if "alias_of" in case:
            case = next(c for c in qualification["cases"] if c["arm"] == case["alias_of"])
        current = backend.llm.collective_rpc("memory_status")
        require([r["cache_sha256"] for r in current] == [r["cache_sha256"] for r in case["loader"]] and
                [r["backend"] for r in current] == [r["backend"] for r in case["loader"]],
                "resolved cache/backend differs from qualification")
        for assignment in pending:
            if card["stage"] != "pricing":
                require(time.time() + 1.25 * price["worst_cap_assignment_seconds"] < deadline,
                        "insufficient measured time for next assignment; leave it unstarted")
            begin = journal.begin(assignment)
            if begin["status"] == "ALREADY_COMMITTED":
                continue
            try:
                result = backend.generate(assignment, catalog[assignment["question_id"]])
            except Exception as exc:
                result = {"execution_complete": False, "text": "", "generated_tokens": 0,
                          "finish_reason": "execution_failure", "type": type(exc).__name__, "message": str(exc)[:2000]}
            closeout_start = time.monotonic()
            # Save compact invariants and completions, without route archives.
            result.update(campaign_sha256=campaign["sha256"], binding_sha256=binding["sha256"],
                          card_sha256=card["sha256"], closeout_seconds=0.)
            receipt = journal.commit(assignment, result)
            save(root / "closeout" / (assignment["uid"] + ".json"),
                 {"result_sha256": receipt["sha256"], "seconds": time.monotonic() - closeout_start})
            materialize_aliases(journal, root, assignment, roster)
            require(result["execution_complete"], "assignment failure committed; infrastructure reconciliation required")
    finally:
        backend.close()
    return {"status": "COMPLETE", "assignments": len(records)}


def materialize_aliases(journal, root, source, roster):
    receipt = load(Path(root) / "results" / (source["uid"] + ".result.json"))
    for alias in roster["records"]:
        if alias["model"] == source["model"] and alias["family"] == source["family"] and \
                alias["alias_of_arm"] == source["arm"] and alias["arm"] != source["arm"]:
            started = journal.begin(alias)
            if started["status"] != "ALREADY_COMMITTED":
                journal.commit(alias, {"alias_of_uid": source["uid"], "alias_receipt_sha256": receipt["sha256"],
                                       "execution_complete": receipt["execution_complete"], "no_duplicate_inference": True})


def resolve_result(root, assignment, roster):
    root = Path(root)
    result = load(root / "results" / (assignment["uid"] + ".result.json"))
    require(result["assignment"] == assignment, "result assignment differs")
    if "alias_of_uid" in result:
        source = next(r for r in roster["records"] if r["uid"] == result["alias_of_uid"])
        require(source["selection_sha256"] == assignment["selection_sha256"] and
                source["family"] == assignment["family"] and source["seed"] == assignment["seed"] and
                source["model"] == assignment["model"] and source["alias_of_arm"] == source["arm"],
                "invalid policy alias")
        observed = load(root / "results" / (source["uid"] + ".result.json"))
        require(observed["sha256"] == result["alias_receipt_sha256"], "alias source changed")
        return observed
    return result


def grade(roster, root, questions, output, *, terminal=None):
    allocated()
    from .grading import score
    source = {r["id"]: r for r in questions["records"]}
    rows = []
    missing_uids = set()
    if terminal is not None:
        require(terminal.get("complete") is True and terminal.get("assignments_sha256") == roster["sha256"] and
                terminal.get("accounting_files"), "missing-assignment terminal reconciliation incomplete")
        check_files(terminal["accounting_files"])
        missing_uids = set(terminal["missing_assignment_uids"])
        require(missing_uids <= {a["uid"] for a in roster["records"]}, "foreign missing-assignment identity")
    for assignment in roster["records"]:
        try:
            result = resolve_result(root, assignment, roster)
        except FileNotFoundError:
            require(assignment["uid"] in missing_uids, "missing result is not terminally reconciled")
            result = {"execution_complete": False, "type": "AssignmentMissingAtCloseout", "text": "",
                      "sha256": None}
        rows.append({"assignment": assignment, "result_sha256": result["sha256"],
                     **score(result, source[assignment["question_id"]]["solution"]),
                     "memory": result.get("memory_after"), "prefill_memory": result.get("prefill_memory"),
                     "generated_tokens": result.get("generated_tokens"), "finish_reason": result.get("finish_reason")})
    return save(output, {"schema": "expert-memory-grades-v1", "records": rows,
                         "assignments_sha256": roster["sha256"], "question_table_sha256": questions["sha256"],
                         "campaign_sha256": roster["campaign_sha256"],
                         "terminal_reconciliation_sha256": terminal["sha256"] if terminal else None,
                         "strict_scorer_files": {str(p): file_sha(p) for p in (
                             Path(__file__).parents[1] / "correlation_pipeline/scoring.py",
                             Path(__file__).parents[1] / "utils.py")}})


def score_grid(grades, roster, *, semantic=False):
    require(grades["assignments_sha256"] == roster["sha256"], "grades belong to another roster")
    assigned = {r["uid"]: r for r in roster["records"]}
    require(len(assigned) == len(roster["records"]), "duplicated roster assignments")
    questions = sorted({r["family"] for r in assigned.values()})
    grid = np.full((len(questions), 7, 4), np.nan)
    seen = set()
    for row in grades["records"]:
        r = row["assignment"]
        require(r["uid"] not in seen and assigned.get(r["uid"]) == r, "duplicated/foreign grade")
        seen.add(r["uid"])
        value = row["strict_score"] if not semantic else \
            row["semantic_lower"] if row["semantic_lower"] == row["semantic_upper"] else np.nan
        grid[questions.index(r["family"]), MODELS.index(r["model"]), ARMS.index(r["arm"])] = value
    require(seen == set(assigned) and len(assigned) == 28 * len(questions), "all-assignment grade grid incomplete")
    return grid


def arm_summaries(grades, roster):
    score_grid(grades, roster)  # Complete all-assignment identity before summarizing.
    rows = []
    for model in MODELS:
        for arm in ARMS:
            items = [r for r in grades["records"] if r["assignment"]["model"] == model and r["assignment"]["arm"] == arm]
            measured = [r for r in items if r.get("memory")]
            metrics = {}
            for field, getter in {
                "expert_bytes": lambda r: sum(x["total_expert_bytes"] for x in r["memory"]),
                "model_tensor_bytes": lambda r: sum(x["total_model_tensor_bytes"] for x in r["memory"]),
                "allocated_bytes": lambda r: sum(x["memory"]["allocated"] for x in r["memory"]),
                "reserved_bytes": lambda r: sum(x["memory"]["reserved"] for x in r["memory"]),
                "nvml_process_bytes": lambda r: sum(p["bytes"] for x in r["memory"] for p in x["memory"]["nvml_process_residence"]),
                "decode_sum_rank_peak_bytes": lambda r: sum(x["memory"]["peak_allocated"] for x in r["memory"]),
                "prefill_sum_rank_peak_bytes": lambda r: sum(x["peak_allocated"] for x in r["prefill_memory"]),
            }.items():
                values = [getter(r) for r in measured if field != "prefill_sum_rank_peak_bytes" or r.get("prefill_memory")]
                metrics[field] = {"mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)} if values else None
            rows.append({"model": model, "arm": arm, "assigned_families": len(items),
                         "strict_accuracy": sum(r["strict_score"] for r in items) / len(items),
                         "semantic_bounds": [sum(r[k] for r in items) / len(items) for k in ("semantic_lower", "semantic_upper")],
                         "memory_observed_families": len(measured), "memory": metrics,
                         "caps": sum(r.get("finish_reason") == "length" for r in items),
                         "assigned_failures": sum(r["status"] in ("ASSIGNED_MISSING", "EXECUTION_FAILURE") for r in items)})
    return rows


def run(campaign, card, authorization, *, deadline):
    from .cards import validate_allocation
    from .execution import qualify
    from .saved_data import measure, selection, transitions
    validate_request(campaign, card, authorization)
    validate_allocation(card)
    stage, request = card["stage"], card["request"]
    if stage == "measure-cpu":
        return measure(campaign, request["output"])
    if stage == "selection":
        return selection(campaign, card["model"], request["output"], load(request["price_path"]))
    if stage == "transitions":
        return transitions(campaign, request["output"])
    if stage == "qualification":
        selection_value = load(request["selection_path"])
        return qualify(campaign, card["model"], selection_value, request["output"], deadline=deadline)
    if stage in ("pricing", "pilot", "final"):
        return generate(campaign, card, deadline=deadline)
    if stage == "grade":
        return grade(load(request["assignments"]), request["root"], load(request["questions"]), request["output"],
                     terminal=load(request["terminal_reconciliation"]) if request.get("terminal_reconciliation") else None)
    if stage == "analyze":
        allocated()
        from .statistics import paired_analysis
        roster, grades = load(request["assignments"]), load(request["grades"])
        require(roster["stage"] == "final", "analysis requires final assignment population")
        return save(request["output"], {"schema": "expert-memory-seven-accuracy-v1",
                    "strict": paired_analysis(score_grid(grades, roster)),
                    "semantic_bounds": paired_analysis(score_grid(grades, roster, semantic=True)),
                    "arms": arm_summaries(grades, roster),
                    "grades_sha256": grades["sha256"], "assignments_sha256": roster["sha256"],
                    "campaign_sha256": campaign["sha256"]})
    raise ValueError("unimplemented finite stage")
