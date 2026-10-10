"""Fresh family reservation with explicit exposure reconciliation."""
from __future__ import annotations

from collections import defaultdict

from .contracts import digest, require, seal
from .protocol import ARMS, MODELS, SEED


def components(question_ids, edges):
    parent = {q: q for q in question_ids}

    def root(q):
        parent.setdefault(q, q)
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for a, b in edges:
        x, y = root(a), root(b)
        if x != y:
            parent[max(x, y)] = min(x, y)
    grouped = defaultdict(list)
    for q in sorted(parent):
        grouped[root(q)].append(q)
    return {digest(qs): qs for qs in grouped.values()}


def proportional_order(records):
    """Deterministic weighted fair interleaving, proportional at each prefix."""
    strata = defaultdict(list)
    for row in records:
        strata[(row["subject"], row["level"])].append(row)
    for key in strata:
        strata[key].sort(key=lambda r: (digest([SEED, "population", r["family"]]), r["family"]))
    offsets = {s: 0 for s in strata}
    ordered = []
    for _ in range(len(records)):
        candidates = [s for s in strata if offsets[s] < len(strata[s])]
        chosen = min(candidates, key=lambda s: ((offsets[s] + .5) / len(strata[s]), s))
        ordered.append(strata[chosen][offsets[chosen]])
        offsets[chosen] += 1
    return ordered


def freeze_population(questions, graph, verification, exposure, *, expected_pre_exposure=4351):
    require(exposure.get("complete") is True and exposure.get("reviewed_through_utc") and
            exposure.get("source_files"), "complete later-exposure ledger required before reservation")
    source = {r["id"]: r for r in questions["records"]}
    families = components(source, graph["edges"])
    reserved = verification["validation"] + verification["final"]
    require(len(reserved) == 128, "verification reservation differs")
    excluded = set(graph["previously_used"]) | {r["question_id"] for r in reserved}
    later = set(exposure.get("question_ids", []))
    later_families = set(exposure.get("family_ids", []))
    eligible, pre_exposure = [], 0
    for family, members in families.items():
        available = sorted(set(members) & set(source))
        if not available or set(members) & excluded:
            continue
        pre_exposure += 1
        if set(members) & later or family in later_families:
            continue
        q = min(available, key=lambda q: (digest([SEED, "representative", q]), q))
        row = source[q]
        eligible.append({"family": family, "question_id": q, "subject": row["subject"],
                         "level": row["level"], "problem": row["problem"]})
    require(pre_exposure == expected_pre_exposure, "fresh population count differs; reconcile source graph")
    ordered = proportional_order(eligible)
    require(len(ordered) >= 32 + 640, "insufficient fresh families for minimum final sample")
    return seal({"schema": "expert-memory-population-v1", "pre_later_exposure_families": pre_exposure,
                 "later_exposure_excluded_families": pre_exposure - len(ordered),
                 "exposure_sha256": exposure["sha256"], "reviewed_through_utc": exposure["reviewed_through_utc"],
                 "pilot": ordered[:32], "final_order": ordered[32:], "outcomes_used": False,
                 "source_sha256": questions["sha256"], "duplicate_graph_sha256": graph["sha256"],
                 "verification_reservation_sha256": verification["sha256"]})


def assignments(population, masks, *, stage, final_size=None, frozen_masks_sha256=None, sample_size_sha256=None):
    require(set(masks) == set(MODELS), "all seven final masks must be frozen")
    campaigns = {m.get("campaign_sha256") for m in masks.values()}
    require(len(campaigns) == 1 and None not in campaigns, "selection campaign differs")
    require(stage in ("pilot", "final"), "unknown prospective population")
    rows = population["pilot"] if stage == "pilot" else population["final_order"][:final_size]
    require(len({r["family"] for r in rows}) == len(rows) and
            not ({r["family"] for r in population["pilot"]} &
                 {r["family"] for r in population["final_order"]}), "pilot/final family isolation differs")
    if stage == "pilot":
        require(len(rows) == 32, "pilot needs exactly 32 independent families")
    if stage == "final":
        require(type(final_size) is int and final_size % 32 == 0 and 640 <= final_size <= 3712 and
                len(rows) == final_size, "final size is absent or infeasible")
    result = []
    for row in rows:
        seed = int(digest([SEED, row["family"]])[:8], 16) % (2**31)
        for model in MODELS:
            policies = {"native": "native"}
            aliases = {}
            for arm in ARMS[1:]:
                signature = digest(masks[model]["final_masks"][arm])
                aliases[arm] = next((a for a, s in policies.items() if s == signature), arm)
                policies.setdefault(arm, signature)
            for arm in ARMS:
                value = {"model": model, "family": row["family"], "question_id": row["question_id"],
                         "seed": seed, "arm": arm, "stage": stage,
                         "population_sha256": population["sha256"],
                         "selection_sha256": masks[model]["sha256"], "alias_of_arm": aliases.get(arm, arm)}
                result.append({**value, "uid": digest(value)})
    return seal({"schema": "expert-memory-assignments-v1", "stage": stage, "families": len(rows),
                 "records": result, "population_sha256": population["sha256"],
                 "campaign_sha256": next(iter(campaigns)), "frozen_masks_sha256": frozen_masks_sha256,
                 "sample_size_sha256": sample_size_sha256})
