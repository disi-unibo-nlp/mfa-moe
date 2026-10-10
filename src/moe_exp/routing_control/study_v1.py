"""Prospective routing study contracts, separate from frozen historical estimators.

These utilities operate on explicit execution and family identities. Missing
measurements stay missing; short mechanism outcomes never become answer grades.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "routing-study-v1"
STATES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
TRANSITIONS = ("candidate_to_verify", "approach_to_commit", "checked_candidate_to_close")
REFERENCE_TARGETS = {(28, 189), (28, 9), (24, 159)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            result.update(block)
    return result.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    require(value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
            "changed artifact seal: " + str(path))
    return value


def save(path, body):
    """Immutable, exclusive artifact creation; no historical overwrite."""
    path = Path(path)
    value = {**body, "sha256": digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(sealed(path) == value, "existing artifact differs: " + str(path))
    else:
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    require(sealed(path) == value, "artifact readback failed")
    return value


def fold(family):
    return int(hashlib.sha256(("dense-verify-fold-v1|" + family).encode()).hexdigest()[:8], 16) % 4


def association_table(rows, *, signal="selection"):
    """Matched within-family/source/status risk differences, equal family weights.

    Each row contains a source prefix profile and a subsequent outcome proxy.
    A family contributes only if it has positive and comparison rows within the
    same source state and prefix status. Held-out directions are evaluated for
    each identity, with no held-out target selection.
    """
    require(signal in ("selection", "executed_weight"), "unknown routing signal")
    grouped = defaultdict(list)
    for row in rows:
        require(row["source_state"] in STATES and row["prefix_status"] in
                ("intermediate", "complete_unchecked", "complete_checked", "unresolved"),
                "invalid source state or prefix status")
        if row.get("outcome") is None:
            continue
        require(type(row["outcome"]) is bool, "outcome must be boolean or missing")
        if row.get(signal) is None:
            continue
        profile = np.asarray(row[signal], dtype=float)
        require(profile.shape == (40, 256) and np.isfinite(profile).all() and
                (profile >= 0).all() and (profile <= 1 + 1e-6).all(), "invalid expert profile")
        grouped[row["transition"], row["source_state"], row["prefix_status"], row["family"]].append(row)
    conditions = defaultdict(dict)
    for (transition, state, status, family), group in grouped.items():
        positive = [np.asarray(r[signal]) for r in group if r["outcome"]]
        comparison = [np.asarray(r[signal]) for r in group if not r["outcome"]]
        if positive and comparison:
            conditions[transition, state, status][family] = np.mean(positive, axis=0) - np.mean(comparison, axis=0)
    table = []
    for (transition, state, status), values in sorted(conditions.items()):
        families = sorted(values)
        array = np.stack([values[f] for f in families])
        mean = array.mean(axis=0)
        held = [array[[fold(f) == k for f in families]].mean(axis=0)
                if any(fold(f) == k for f in families) else None for k in range(4)]
        for layer in range(40):
            for expert in range(256):
                effect = float(mean[layer, expert])
                directions = [None if h is None else float(h[layer, expert]) for h in held]
                stable = sum(d is not None and d * effect > 0 for d in directions)
                supported = len(families) >= 10 and stable >= 3 and effect != 0
                table.append({"transition": transition, "source_state": state, "prefix_status": status,
                    "layer": layer, "expert": expert, "signal": signal, "risk_difference": effect,
                    "positive_families": len(families), "comparison_families": len(families),
                    "held_out_effects": directions, "consistent_folds": stable,
                    "status": "SUPPORTED_ASSOCIATION" if supported else "UNAVAILABLE_INSUFFICIENT_SUPPORT",
                    "semantic_qualification": "class_transition_proxy_only"})
    return table


def select_singletons(table):
    """At most two new identities; semantic prefix recognition must qualify first."""
    eligible = [r for r in table if r["status"] == "SUPPORTED_ASSOCIATION" and
                r.get("prefix_recognition_qualified") is True and r["transition"] in TRANSITIONS and
                (r["layer"], r["expert"]) not in REFERENCE_TARGETS]
    eligible.sort(key=lambda r: (-abs(r["risk_difference"]), -r["consistent_folds"],
                                  r["transition"], r["layer"], r["expert"]))
    chosen, identities = [], set()
    for row in eligible:
        identity = row["layer"], row["expert"]
        if identity not in identities:
            chosen.append(row)
            identities.add(identity)
        if len(chosen) == 2:
            break
    return chosen


def screen_assignments(targets, prefixes, random_experts):
    """Freeze same-prefix ±1 singleton screen with operator-matched controls."""
    require(len(targets) <= 2 and len({(t["layer"], t["expert"]) for t in targets}) == len(targets),
            "screen permits at most two new singletons")
    assignments = []
    for target in targets:
        require(target["transition"] in TRANSITIONS and target.get("prefix_recognition_qualified") is True,
                "target lacks qualified prefix recognition")
        selected = sorted((p for p in prefixes if p["transition"] == target["transition"] and
                           p.get("eligible") is True),
                          key=lambda p: (digest([SCHEMA, p["family"]]), p["uid"]))[:24]
        for prefix in selected:
            for seed in (0, 1):
                identity = f'L{target["layer"]}E{target["expert"]}'
                random = random_experts.get((identity, prefix["family"], seed))
                require(type(random) is int and 0 <= random < 256 and random != target["expert"],
                        "missing exposure-matched random expert")
                for arm, bias, expert in (("native", 0., None), ("positive", 1., target["expert"]),
                                         ("negative", -1., target["expert"]), ("random_positive", 1., random),
                                         ("random_negative", -1., random)):
                    row = {"target_id": identity, "prefix_uid": prefix["uid"], "family": prefix["family"],
                        "transition": target["transition"], "seed": seed, "arm": arm,
                        "layer": target["layer"], "expert": expert, "bias": bias,
                        "pulse_tokens": 256, "continuation_cap": 1024, "top_k": 8,
                        "shared_expert": "unchanged", "prefix_ids_sha256": prefix["prefix_ids_sha256"]}
                    assignments.append({**row, "uid": digest([SCHEMA, row])})
    return assignments


def delayed_fire(*, opportunity_position, position, desired_act_occurred, closed, delay=64):
    require(type(delay) is int and delay >= 0 and position >= opportunity_position,
            "invalid delayed opportunity")
    return position >= opportunity_position + delay and not desired_act_occurred and not closed


def freeze_independent_families(questions, edges, previously_used, *, revision):
    """Stratified 32/96 reservation over the full, duplicate-connected source.

    Edges must include previous-source aliases, so transitive exclusions remain
    valid. Selection consumes no outcomes, gold answers or native traces.
    """
    require(bool(revision), "source revision is required")
    require(len({q["id"] for q in questions}) == len(questions), "duplicate source question IDs")
    parent = {q["id"]: q["id"] for q in questions}
    for q in previously_used:
        parent.setdefault(q, q)
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    for a, b in edges:
        require(a in parent and b in parent, "duplicate edge outside source universe")
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    blocked = {find(q) for q in previously_used}
    components = defaultdict(list)
    for q in questions:
        if find(q["id"]) not in blocked:
            components[find(q["id"])].append(q)
    strata = defaultdict(list)
    for members in components.values():
        representative = min(members, key=lambda q: digest([SCHEMA, revision, q["id"]]))
        require(representative.get("subject") and representative.get("level") is not None,
                "subject and level are required for stratification")
        family = digest(sorted(q["id"] for q in members))
        strata[str(representative["subject"]), str(representative["level"])].append(
            {"family": family, "question_id": representative["id"],
             "subject": representative["subject"], "level": representative["level"]})
    eligible = sum(map(len, strata.values()))
    if eligible < 128:
        return {"status": "UNAVAILABLE_FAMILY_SHORTAGE", "eligible": eligible, "required": 128,
                "source_revision": revision, "validation": [], "final": []}
    for pool in strata.values():
        pool.sort(key=lambda q: digest([SCHEMA, revision, q["family"]]))
    # Proportional deficit sampling; deterministic ties, retaining subject/level
    # balance both for the first 32 and for all 128 families.
    sizes, taken, order = {k: len(v) for k, v in strata.items()}, defaultdict(int), []
    for i in range(128):
        key = max((k for k in strata if taken[k] < sizes[k]),
                  key=lambda k: ((i + 1) * sizes[k] / eligible - taken[k], k))
        order.append(strata[key][taken[key]])
        taken[key] += 1
    return {"status": "FROZEN", "source_revision": revision, "eligible": eligible,
        "validation": order[:32], "final": order[32:], "seeds": [0, 1], "cap": 16384,
        "maximum_final_assignments": 768, "outcomes_used_in_selection": False,
        "source_questions_sha256": digest(questions), "duplicate_edges_sha256": digest(edges),
        "previously_used_sha256": digest(sorted(previously_used))}


def utility_frontier(rows, families, policy_ids, *, draws=50000, seed=20261006):
    """Paired family max-standardized-deviation intervals across all contrasts.

    Seeds are averaged inside family. Complete-family inference is accompanied
    by worst-case accuracy bounds; missing cost has no finite upper bound.
    """
    require(len(set(families)) == len(families) and len(set(policy_ids)) == len(policy_ids),
            "duplicate family or policy")
    require("native" not in policy_ids and draws >= 100, "invalid comparison design")
    expected = {(f, s, a) for f in families for s in (0, 1) for a in ("native", *policy_ids)}
    by_id = {}
    for row in rows:
        key = row["family"], row["seed"], row["arm"]
        require(key in expected and key not in by_id, "foreign or duplicate utility assignment")
        correct = row.get("correct")
        require(correct is None or type(correct) is bool, "invalid answer grade")
        for name in ("generated_tokens", "controller_tokens"):
            require(row.get(name) is None or type(row[name]) is int and row[name] >= 0,
                    "invalid token accounting")
        by_id[key] = row
    def complete(key):
        row = by_id.get(key, {})
        return all(row.get(k) is not None for k in ("correct", "generated_tokens", "controller_tokens"))
    complete_families = [f for f in families if all(complete((f, s, a))
                        for s in (0, 1) for a in ("native", *policy_ids))]
    contrasts, bounds, arm_means = [], [], {}
    for policy in policy_ids:
        for endpoint in ("accuracy_pp", "total_tokens"):
            contrasts.append((policy, endpoint))
    data = np.zeros((len(complete_families), len(contrasts)))
    def value(row, endpoint):
        return 100 * row["correct"] if endpoint == "accuracy_pp" else row["generated_tokens"] + row["controller_tokens"]
    for j, (policy, endpoint) in enumerate(contrasts):
        for i, family in enumerate(complete_families):
            data[i, j] = np.mean([value(by_id[family, s, policy], endpoint) -
                                 value(by_id[family, s, "native"], endpoint) for s in (0, 1)])
        if endpoint == "accuracy_pp":
            limits = []
            for family in families:
                seed_bounds = []
                for s in (0, 1):
                    a = by_id.get((family, s, policy), {}).get("correct")
                    n = by_id.get((family, s, "native"), {}).get("correct")
                    seed_bounds.append(((0 if a is None else a) - (1 if n is None else n),
                                        (1 if a is None else a) - (0 if n is None else n)))
                limits.append(np.mean(seed_bounds, axis=0) * 100)
            bounds.append(None if not limits else np.mean(limits, axis=0).tolist())
        else:
            bounds.append(None)
    estimates = data.mean(axis=0) if len(data) else np.full(len(contrasts), np.nan)
    intervals, q = [None] * len(contrasts), None
    if len(data) >= 2:
        rng = np.random.default_rng(seed)
        boot = np.concatenate([data[rng.integers(0, len(data), size=(min(1000, draws - i), len(data)))].mean(axis=1)
                               for i in range(0, draws, 1000)])
        scale = boot.std(axis=0, ddof=1)
        active = scale > 0
        q = float(np.quantile(np.max(np.abs((boot[:, active] - estimates[active]) / scale[active]), axis=1), .95)) if active.any() else 0.
        # Degenerate components cannot establish retention; expose their status.
        intervals = [[float(e - q * s), float(e + q * s)] for e, s in zip(estimates, scale)]
    for arm in ("native", *policy_ids):
        arm_means[arm] = {endpoint: float(np.mean([value(by_id[f, s, arm], endpoint)
                         for f in complete_families for s in (0, 1)])) if complete_families else None
                         for endpoint in ("accuracy_pp", "total_tokens")}
        exposures = [by_id[f, s, arm].get("intervened_positions") for f in complete_families for s in (0, 1)]
        arm_means[arm]["intervened_positions"] = float(np.mean(exposures)) if exposures and all(x is not None for x in exposures) else None
    frontier = []
    for arm, point in arm_means.items():
        if point["accuracy_pp"] is None:
            continue
        dominated = any(other["accuracy_pp"] >= point["accuracy_pp"] and
                        other["total_tokens"] <= point["total_tokens"] and
                        (other["accuracy_pp"] > point["accuracy_pp"] or other["total_tokens"] < point["total_tokens"])
                        for other in arm_means.values() if other["accuracy_pp"] is not None)
        if not dominated:
            frontier.append(arm)
    return {"schema": "routing-study-utility-frontier-v1", "assigned": len(expected),
        "missing_execution_assignments": len(expected - by_id.keys()), "complete_families": complete_families,
        "assigned_families": len(families), "arm_means": arm_means, "frontier": frontier,
        "contrasts": [{"policy": p, "endpoint": e, "estimate": None if not len(data) else float(estimates[j]),
            "simultaneous_ci95": intervals[j], "missing_accuracy_bounds": bounds[j],
            "missing_cost_upper_bound": "unbounded" if e == "total_tokens" and len(data) < len(families) else None,
            "degenerate": bool(len(data) and np.ptp(data[:, j]) == 0)} for j, (p, e) in enumerate(contrasts)],
        "inference": "paired family bootstrap; simultaneous max standardized deviation; complete families",
        "draws": draws, "seed": seed, "accuracy_loss_margin": None,
        "claim_limit": "Nonsignificance and degenerate intervals do not establish accuracy retention."}


def choose_policy(result):
    require(result["assigned_families"] == 32 and len(result["complete_families"]) == 32,
            "policy selection requires all 32 validation families")
    points = result["arm_means"]
    eligible = [p for p in points if p != "native" and
                points[p]["total_tokens"] <= points["native"]["total_tokens"]]
    require(len(points) <= 3, "validation allows at most two shortlisted policies")
    if not eligible:
        return {"status": "NO_COST_IMPROVING_WINNER", "selected_policy": None}
    require(all(points[p]["intervened_positions"] is not None for p in eligible),
            "selection tie-break needs intervention exposure")
    winner = min(eligible, key=lambda p: (-points[p]["accuracy_pp"], points[p]["total_tokens"],
                                        points[p]["intervened_positions"], p))
    return {"status": "SELECTED_ON_VALIDATION", "selected_policy": winner}


def stage_price(*, prompt_tokens, output_tokens, controller_prompt_tokens, controller_tokens,
                label_prompt_tokens, label_tokens, grade_prompt_tokens, grade_tokens,
                prefill_tps, decode_tps, load_seconds, loads, shutdown_seconds,
                overhead_seconds, retry_factor, gpus):
    values = locals()
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values.values()),
            "invalid measured resource input")
    require(prefill_tps > 0 and decode_tps > 0 and retry_factor >= 1 and loads >= 1 and gpus >= 1,
            "missing measured throughput or recovery load")
    prefill = prompt_tokens + controller_prompt_tokens + label_prompt_tokens + grade_prompt_tokens
    decode = output_tokens + controller_tokens + label_tokens + grade_tokens
    seconds = retry_factor * (prefill / prefill_tps + decode / decode_tps) + loads * (load_seconds + shutdown_seconds) + overhead_seconds
    return {"schema": "routing-study-stage-price-v1", "inputs": values,
            "complete_seconds": seconds, "GPU_hours": gpus * seconds / 3600,
            "status": "PRICED_REQUIRES_STAGE_AUTHORIZATION"}
