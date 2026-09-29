"""Frozen mechanism-pilot specifications and paired, prefix-conditional analysis."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .common import digest
from .contracts import seal, validate
from .contrasts import cluster_estimate, joint_question_draws

ARMS = ("baseline", "positive", "negative", "sham", "reweight")


def freeze_branches(prefixes, folds, candidate, calibration, qualification, *,
                    seeds=(42, 43), questions=16, trigger=None):
    """All eligibility must be prefix-only. This function never reads final outcomes."""
    validate(folds, "folds")
    if questions != 16 or len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("The first pilot is fixed at 16 questions and two seeds")
    if not candidate.get("eligible_for_mechanism_pilot") or candidate.get("stable_folds", 0) < 3:
        raise ValueError("Candidate has not passed held-out sign stability")
    if not candidate.get("experts") or qualification.get("status") != "passed":
        raise ValueError("Native backend qualification required")
    if not all(qualification.get(k) is True for k in
               ("selected_ids_equal", "dispatched_weights_equal", "output_equal", "attribution_equal")):
        raise ValueError("Incomplete native no-op parity")
    if qualification.get("historical_backend_revalidated") is not True:
        raise ValueError("Historical/intervention backend candidate comparison required")
    if trigger is not None and not trigger.get("qualified"):
        raise ValueError("Unvalidated dynamic trigger")
    for arm in ARMS[1:]:
        if arm not in calibration or not calibration[arm].get("development_only"):
            raise ValueError("Each direction requires development-prefix dose calibration")
        if calibration[arm].get("layer", candidate["layer"]) != candidate["layer"]:
            raise ValueError("Calibration refers to a different native layer")
        if not calibration[arm].get("prefix_binding"):
            raise ValueError("Calibration must bind exact development prefixes")
        for metric in ("membership_change", "weight_l1"):
            if not math.isfinite(calibration[arm].get(metric, float("nan"))):
                raise ValueError("Missing measured routing displacement")
        if arm != "reweight" and not .03 <= calibration[arm]["membership_change"] <= .07:
            raise ValueError("Selection-bias calibration is outside the prespecified 5% +/-2% dose")
    if calibration["positive"]["dose"] <= 0 or calibration["negative"]["dose"] >= 0:
        raise ValueError("Positive and negative doses must have opposite signed directions")
    if set(calibration["sham"]["experts"]) & set(candidate["experts"]):
        raise ValueError("Sham and target expert groups overlap")
    if len(calibration["sham"]["experts"]) != len(candidate["experts"]):
        raise ValueError("Sham group size mismatch")
    if not calibration["sham"].get("prevalence_matched"):
        raise ValueError("Sham prevalence matching is unverified")
    reserved = folds["payload"]["reserved_questions"]
    eligible = defaultdict(list)
    for prefix in prefixes:
        q = prefix["question_id"]
        if q not in reserved or prefix["model"] != candidate["model"]:
            continue
        if prefix.get("reasoning_at_branch") is not True or prefix.get("budget_population") != "original":
            continue
        endpoint = prefix["prefix_tokens"]
        if trigger is None and endpoint != 2048:
            continue
        if trigger is not None and (not 1024 <= endpoint <= 4096 or prefix.get("first_eligible_trigger") is not True):
            continue
        if len(prefix["prefix_token_ids"]) != endpoint or not prefix["prompt_token_ids"]:
            raise ValueError("Exact prefix length mismatch")
        expected_hash = digest([prefix["prompt_token_ids"], prefix["prefix_token_ids"]])
        if prefix.get("prefix_sha256") != expected_hash:
            raise ValueError("Prefix hash mismatch")
        config = prefix["generation_config"]
        if config["max_tokens"] <= endpoint:
            continue
        if any(key not in config for key in ("temperature", "top_p", "top_k")):
            raise ValueError("Missing original decoding settings")
        eligible[q].append(prefix)
    selected = []
    for q in reserved:
        if q in eligible:
            # No final correctness/length ranking. Earlier eligible trigger first.
            selected.append(min(eligible[q], key=lambda p: (p["prefix_tokens"], p["attempt_id"])))
        if len(selected) == questions:
            break
    if len(selected) != questions:
        raise ValueError("Insufficient eligible reserved questions; do not relax outcome-blind selection")
    branches = []
    for prefix in selected:
        for seed in seeds:
            for arm in ARMS:
                dose = 0. if arm == "baseline" else calibration[arm]["dose"]
                action = "noop" if arm == "baseline" else "selected_reweight" if arm == "reweight" else "selection_bias"
                experts = calibration["sham"]["experts"] if arm == "sham" else candidate["experts"]
                policy = dict(layer=candidate["layer"], experts=experts, action=action, dose=dose,
                              start_token=prefix["prefix_tokens"], pulse_tokens=256)
                from moe_exp.moe_guiding.native import NativePolicy
                NativePolicy(**policy)
                decoding = {k: v for k, v in prefix["generation_config"].items() if k not in ("seed", "max_tokens")}
                row = dict(question_id=prefix["question_id"], attempt_id=prefix["attempt_id"],
                    model=prefix["model"], prefix_sha256=prefix["prefix_sha256"],
                    prompt_token_ids=prefix["prompt_token_ids"], prefix_token_ids=prefix["prefix_token_ids"],
                    original_max_tokens=prefix["generation_config"]["max_tokens"],
                    max_new_tokens=prefix["generation_config"]["max_tokens"] - prefix["prefix_tokens"],
                    seed=seed, arm=arm, decoding=decoding, policy=policy,
                    closing_sequences=prefix["closing_sequences"],
                    source_binding=prefix["source_binding"])
                row["branch_id"] = digest(row)
                branches.append(row)
    # Fixed randomized execution order, paired question/seed identity remains intact.
    branches.sort(key=lambda b: digest([42, b["branch_id"]]))
    return seal("branches", dict(branches=branches, candidate=candidate, calibration=calibration,
        qualification=qualification, trigger=trigger, maximum_continuations=160,
        abort_reasons=["integrity_failure", "native_parity_failure", "attribution_failure", "resource_limit"],
        estimand="prefix_conditional_accuracy_cost", timing_arms="gated_on_measurable_manipulation"),
        inputs={"folds": folds["binding"], "prefixes": digest(prefixes),
                "qualification": digest(qualification), "candidate": digest(candidate)},
        config=dict(questions=questions, seeds=list(seeds), original_budget=True, pulse_tokens=256,
                    target_changed_sets=.05, dose_tolerance=.02),
        population="reserved_A_only_exploratory_exact_prefixes")


def add_timing_arms(manifest, manipulation_receipt):
    validate(manifest, "branches")
    if manipulation_receipt.get("manifest_binding") != manifest["binding"] or not manipulation_receipt.get("measurable_routing_change"):
        raise ValueError("Timing arms require observed manipulation in the frozen first pilot")
    base = [b for b in manifest["payload"]["branches"] if b["arm"] == "positive"]
    if len(base) != 32:
        raise ValueError("Expected the same 16 question-prefix blocks and two seeds")
    branches = []
    for b in base:
        for arm in ("delayed", "continuous"):
            p = dict(b["policy"])
            if arm == "delayed":
                # Randomization is fixed before observing outcomes; never before branch.
                delay = 1 + int(digest([b["question_id"], b["seed"], "timing"])[:8], 16) % 256
                p["start_token"] += delay
            else:
                p["pulse_tokens"] = None
            row = {k: v for k, v in b.items() if k != "branch_id"}
            row.update(arm=arm, policy=p)
            row["branch_id"] = digest(row)
            branches.append(row)
    return seal("branches", dict(branches=branches, maximum_continuations=64,
        reference_branches=[b for b in manifest["payload"]["branches"] if b["arm"] == "baseline"],
        reference_manifest_binding=manifest["binding"],
        qualification=manifest["payload"]["qualification"], estimand="prefix_conditional_timing_effect"),
        inputs={"first_pilot": manifest["binding"], "manipulation": digest(manipulation_receipt)},
        config=manifest["config"], population=manifest["population"])


def analyze_pilot(manifest, results, *, bootstrap=2000, seed=42):
    """Average seeds inside questions, pair arms, then bootstrap QUESTIONS."""
    validate(manifest, "branches")
    expected = {b["branch_id"]: b for b in manifest["payload"]["branches"]}
    references = {b["branch_id"]: b for b in manifest["payload"].get("reference_branches", [])}
    expected.update(references)
    seen, blocks = set(), defaultdict(dict)
    metrics = ("is_correct", "total_tokens", "reasoning_tokens", "capped", "seconds",
               "controller_seconds", "expert_executions", "membership_change", "weight_l1",
               "state_exits", "state_dwell", "state_recurrence")
    for r in results:
        required_binding = (manifest["payload"]["reference_manifest_binding"]
                            if r["branch_id"] in references else manifest["binding"])
        if r["branch_id"] not in expected or r["branch_id"] in seen or r["manifest_binding"] != required_binding:
            raise ValueError("Duplicate, unknown or incompatible pilot result")
        b = expected[r["branch_id"]]
        if r.get("status") != "complete":
            raise ValueError("Incomplete pilot result")
        if r["total_tokens"] > b["original_max_tokens"]:
            raise ValueError("Branch exceeded original budget")
        seen.add(r["branch_id"])
        blocks[(b["question_id"], b["seed"])][b["arm"]] = r
    if seen != set(expected):
        return dict(status="incomplete", completed=len(seen), expected=len(expected), analysis=None)
    unresolved = sum(r.get("is_correct") is None for r in results)
    arms = sorted({b["arm"] for b in expected.values()} - {"baseline"})
    draws = joint_question_draws([q for q, _ in blocks], bootstrap=bootstrap, seed=seed)
    contrasts = []
    for arm in arms:
        for metric in metrics:
            differences = defaultdict(list)
            for (q, _), block in blocks.items():
                if "baseline" not in block or arm not in block:
                    raise ValueError("Unpaired question-seed block")
                a, b = block[arm].get(metric), block["baseline"].get(metric)
                if a is not None and b is not None:
                    differences[q].append(float(a) - float(b))
            values = {q: float(np.mean(d)) for q, d in differences.items()}
            contrasts.append(dict(arm=arm, metric=metric, **cluster_estimate(values, draws)))
    means = {}
    for arm in ["baseline", *arms]:
        per_question = defaultdict(list)
        for (q, _), block in blocks.items():
            if block[arm].get("is_correct") is not None:
                per_question[q].append([float(block[arm]["is_correct"]), block[arm]["total_tokens"]])
        if len(per_question) == len(draws[0]) and all(len(v) == 2 for v in per_question.values()):
            values = np.array([np.mean(v, 0) for v in per_question.values()])
            means[arm] = dict(accuracy=float(values[:, 0].mean()), tokens=float(values[:, 1].mean()))
    frontier = [arm for arm, point in means.items() if not any(
        other["accuracy"] >= point["accuracy"] and other["tokens"] <= point["tokens"]
        and (other["accuracy"] > point["accuracy"] or other["tokens"] < point["tokens"])
        for name, other in means.items() if name != arm)]
    tail = {}
    for arm in ["baseline", *arms]:
        values = {q: [block[arm]["total_tokens"] for (qq, _), block in blocks.items() if qq == q]
                  for q in draws[0]}
        point = float(np.quantile([v for group in values.values() for v in group], .9))
        samples = [np.quantile([v for i in indices for v in values[draws[0][i]]], .9) for indices in draws[1]]
        tail[arm] = dict(p90_total_tokens=point, ci=np.quantile(samples, [.025, .975]).tolist())
    return dict(status="awaiting_adjudication" if unresolved else "complete", unresolved_outcomes=unresolved,
        claim="causal_prefix_conditional_exploratory", contrasts=contrasts, tail_cost=tail,
        arm_means=means, point_estimate_frontier=frontier,
        limitations=["No whole-prompt value or noninferiority claim", "Only 16 distinct questions",
                     "Unresolved accuracy remains missing, never a scored failure", "Frontier is a point estimate with paired uncertainty reported separately"])
