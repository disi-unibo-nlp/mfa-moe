"""X2 dose-finding builder for the continuation-cap proposal (steering-v1).

A thin layer over the SNAPSHOT's own constructors (``manifests.make_request`` /
``build_manifest`` / ``validate_manifest`` / ``assign_shards``, ``policies.*``); it edits no
snapshot code. Production builds require a verified frozen cap snapshot and explicit tree::

    steer_submit.py cpu --snapshot $SNAP --name st-x2prep-<what> \\
        '$CLIENT_PY $S/addenda/x2/x2_build.py --name x2-v1 --n-shards 1 \\
         --expect-tree $TREE --max-new-tokens 1024 --out $S/runs/x2'

Design (transcribed from the task; PREREG v0.2 5.3-5.6, 6, 7, 11 G3, 12): the 48 split-v1 dev-cal
questions; eligible = the first lexical Explore onset (lexicon-v1, the streaming TriggerFSM of the
snapshot) with decision-token index >= 4,096 in the card-v3 sample_00 trace, decided before any
continuation, no replacement; one branch per eligible question at prefix_len = decision index + 1
(the row that produces the first new token is the first steered row, exactly as the onset
schedule would start its pulse), landmark = prefix_len, one 256-token pulse; seeds k = 0, 1;
85 conditions per (question, seed): N, 42 target cells (bias +/-{.5,1,2,4}, force +/-,
reweight +/-{1,2} x L1/BAND/ALL on targets-v1) and the same 42 cells on random-sets-v2 with set k
paired with seed k.

The builder refuses the old s1 API, which cannot express a continuation cap. The cap proposal
passes ``max_new_tokens`` to make_request and checks every request before writing it. The CLI
defaults N to the sham landmark policy. Dry builds against an unfrozen proposal require
``--dry-unfrozen`` and a name beginning ``x2-dryrun``; they are never production manifests.
"""
from __future__ import annotations

import argparse
import inspect
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moe_steer import engine
from moe_steer import manifests as M
from moe_steer import policies
from moe_steer.spec import Policy, PolicyTable, digest, seal, verify
from moe_steer.trigger import Lexicon, TriggerFSM, VocabBytes, offline_sentence_matches

ELIGIBILITY_SCHEMA = "x2prep-eligibility-v1"
ENUMERATION_SCHEMA = "x2prep-enumeration-v1"
SEEDS = (0, 1)
SCOPES = policies.SCOPES
PULSE_LEN = 256
ONSET_FLOOR = 4096
THINK_END_ID = 248069
ONSET_SCHEDULE = {"kind": "onset", "pulse_len": 256, "cooldown": 0, "min_reasoning_index": 0}
BIAS_MAGNITUDES = (0.5, 1.0, 2.0, 4.0)
REWEIGHT_MAGNITUDES = (1.0, 2.0)
SIGNS = (1, -1)
N_CONDITIONS = 85
N_TARGET_CELLS = 42
ROUTED_WINDOW = 64
SUBSPLIT = "dev-cal"
HORIZONS = (1024, 768)
NLL_CELLS = N_TARGET_CELLS + 1
ARM_LABELS = ("N", "E+", "E-", "M+", "M-")
# expected_len option (iii) of the AMBIGUITIES table: decode-token equivalents of the prefill.
PREFILL_PER_DECODE_TOKEN = 19656.0 / 1474.0


class X2BuildError(AssertionError):
    """An X2 design assertion failed (confirm leak, pairing, condition count, ...)."""


@dataclass(frozen=True)
class Cell:
    """One of the 42 target cells; its random counterpart is ``policy@random{k}``."""

    policy: str
    scope: str
    kind: str
    sign: int
    magnitude: float


# ------------------------------------------------------------------ policy table

def grid() -> list[tuple[str, int, float]]:
    """(operator kind, sign, magnitude) of the 14 cells of one scope (force has magnitude 0)."""
    cells = [("bias", s, m) for s in SIGNS for m in BIAS_MAGNITUDES]
    cells += [("force", s, 0.0) for s in SIGNS]
    cells += [("reweight", s, m) for s in SIGNS for m in REWEIGHT_MAGNITUDES]
    return cells


def e_policy(
    inputs: policies.Inputs, scope: str, cell: tuple[str, int, float], schedule: Any
) -> Policy:
    """Target-set policy of one grid cell (the snapshot's bias/force/reweight constructors)."""
    kind, sign, magnitude = cell
    targets = inputs.scope(scope)
    if kind == "bias":
        return policies.bias(targets, sign, magnitude, schedule)
    if kind == "force":
        return policies.force(targets, sign, schedule)
    return policies.reweight(targets, sign, magnitude, schedule)


def x2_table(inputs: policies.Inputs, n_policy: str = "zero") -> tuple[PolicyTable, list[Cell]]:
    """Zero policy, the 42 E policies, their random twins for sets 0 and 1 (and a sham N)."""
    if n_policy not in ("zero", "sham"):
        raise X2BuildError(f"n_policy must be zero or sham, got {n_policy!r}")
    schedule = policies.landmark_schedule(PULSE_LEN)
    cells: list[Cell] = []
    e_policies: list[Policy] = []
    for scope in SCOPES:
        for cell in grid():
            policy = e_policy(inputs, scope, cell, schedule)
            e_policies.append(policy)
            cells.append(Cell(policy.name, scope, *cell))
    m_policies = [policies.matched_random(p, inputs, k) for k in SEEDS for p in e_policies]
    extra = [policies.sham(inputs.scope("ALL"), schedule)] if n_policy == "sham" else []
    return policies.build_table([*e_policies, *m_policies, *extra]), cells


def n_policy_name(n_policy: str) -> str:
    return "zero" if n_policy == "zero" else "sham_ALL_landmark"


def condition_policies(
    cells: Sequence[Cell], seed_k: int, n_policy: str
) -> list[tuple[str, str]]:
    """(arm, policy name) of the 85 conditions of one (question, seed_k)."""
    out = [("N", n_policy_name(n_policy))]
    out += [("E+" if c.sign > 0 else "E-", c.policy) for c in cells]
    out += [("M+" if c.sign > 0 else "M-", f"{c.policy}@random{seed_k}") for c in cells]
    return out


# ------------------------------------------------------------------ eligibility

def first_onset(
    ids: Sequence[int], lexicon: Lexicon, vocab: VocabBytes, uid: str, floor: int = ONSET_FLOOR
) -> dict[str, Any]:
    """Stream the trace through the onset FSM; the first decision with index >= ``floor``.

    Returns {"streaming_first", "marker", "sentence_first_o", "n_onsets_before_floor",
    "think_end"}; decisions after ``</think>`` cannot occur (the FSM closes reasoning).
    """
    fsm = TriggerFSM(ONSET_SCHEDULE, lexicon, vocab, uid)
    found: dict[str, Any] | None = None
    before = 0
    for o, token in enumerate(ids):
        for event in fsm.feed(o, int(token)):
            if event["type"] != "onset":
                continue
            if event["o"] < floor:
                before += 1
            elif found is None:
                found = {"o": event["o"], "marker": event["marker"],
                         "sentence_first_o": event["sentence_first_o"]}
        if found is not None or fsm.think_end is not None:
            break
    return {
        "streaming_first": None if found is None else found["o"],
        "marker": None if found is None else found["marker"],
        "sentence_first_o": None if found is None else found["sentence_first_o"],
        "n_onsets_before_floor": before,
        "think_end": fsm.think_end,
    }


def offline_first(
    ids: Sequence[int], offsets: Sequence[Sequence[int]], cot_text: str, lexicon: Lexicon,
    floor: int = ONSET_FLOOR,
) -> int | None:
    """Offline (spans.py) cross-check: first onset decision >= ``floor`` (a4_trigger_stats way)."""
    think = list(ids).index(THINK_END_ID) if THINK_END_ID in ids else None
    boundary = offsets[think][0] if think is not None else len(cot_text)
    rows = offline_sentence_matches(cot_text[:boundary], offsets, lexicon, full_text=cot_text)
    decisions = [r["decision"] for r in rows if r["onset"] and r["decision"] is not None]
    return next((d for d in decisions if d >= floor), None)


def eligibility_row(
    info: Mapping[str, Any], trace: Mapping[str, Any], lexicon: Lexicon, vocab: VocabBytes,
    *, cross_check: bool = True,
) -> dict[str, Any]:
    """Eligibility record of one dev-cal question from its card-v3 sample_00 trace."""
    replay = trace["metadata"]["token_replay"]
    ids = [int(t) for t in replay["completion_token_ids"]]
    if int(trace["sample_id"]) != 0 or str(trace["problem_id"]) != info["prompt_ref"]["problem_id"]:
        raise X2BuildError(f"{info['question']}: the trace is not the sample_00 prompt_ref trace")
    if [int(t) for t in replay["prompt_token_ids"]] != info["prompt_token_ids"]:
        raise X2BuildError(f"{info['question']}: trace prompt ids differ from the prompt table")
    found = first_onset(ids, lexicon, vocab, f"x2-elig|{info['question']}")
    row: dict[str, Any] = {
        "question": info["question"], "dataset": info["dataset"],
        "source_problem_id": info["source_problem_id"], "bucket": info["bucket"],
        "trace_ref": dict(info["prompt_ref"]), "sample_id": int(trace["sample_id"]),
        "prompt_tokens": int(info["prompt_tokens"]), "trace_completion_tokens": len(ids),
        "card_median_completion_tokens": float(info["median_completion_tokens"]),
        "think_end": found["think_end"],
        "n_onsets_before_floor": found["n_onsets_before_floor"],
        "streaming_first_onset": found["streaming_first"], "marker": found["marker"],
        "sentence_first_o": found["sentence_first_o"],
    }
    if cross_check:
        row["offline_first_onset"] = offline_first(
            ids, replay["completion_offsets"], trace["cot_text"], lexicon)
    onset = found["streaming_first"]
    if onset is None:
        reason = ("reasoning_ended_before_onset" if found["think_end"] is not None
                  else "no_onset_ge_4096_in_trace")
        row.update(eligible=False, reason=reason, onset_o=None, prefix_len=None,
                   prefix_sha256=None, branch_prompt_tokens=None, context=None)
        return row
    prefix_len = onset + 1
    if prefix_len >= len(ids) or THINK_END_ID in ids[:prefix_len]:
        row.update(eligible=False, reason="no_room_or_think_end_in_prefix", onset_o=onset,
                   prefix_len=None, prefix_sha256=None, branch_prompt_tokens=None, context=None)
        return row
    first_o = found["sentence_first_o"]
    row.update(
        eligible=True, reason="first onset >= 4096 (streaming FSM)", onset_o=onset,
        prefix_len=prefix_len, prefix_sha256=digest(ids[:prefix_len]),
        branch_prompt_tokens=int(info["prompt_tokens"]) + prefix_len,
        context=vocab.decode(ids[first_o:prefix_len]),
    )
    return row


def compute_eligibility(
    infos: Sequence[Mapping[str, Any]], store: Any, lexicon: Lexicon, vocab: VocabBytes,
    *, cross_check: bool = True,
) -> list[dict[str, Any]]:
    """Eligibility rows of ``infos`` (sorted by question key); one trace read per question."""
    rows = []
    for info in sorted(infos, key=lambda i: i["question"]):
        ref = info["prompt_ref"]
        trace = store.trace(ref["dataset"], ref["problem_id"])
        rows.append(eligibility_row(info, trace, lexicon, vocab, cross_check=cross_check))
    return rows


# ------------------------------------------------------------------ requests and manifest

def build_requests(
    name: str, table: PolicyTable, cells: Sequence[Cell], infos: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]], *, max_new_tokens: int, n_policy: str,
) -> list[dict[str, Any]]:
    """The 85 x eligible x 2 branch requests (snapshot ``make_request``; expected_len = horizon)."""
    if "max_new_tokens" not in inspect.signature(M.make_request).parameters:
        raise X2BuildError("X2 requires the s2 continuation-cap adapter; s1 cannot enforce the horizon")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row["eligible"]:
            continue
        info = infos[row["question"]]
        parent = {"trace_ref": dict(row["trace_ref"]), "prefix_len": int(row["prefix_len"])}
        for k in SEEDS:
            for arm, policy_name in condition_policies(cells, k, n_policy):
                out.append(M.make_request(
                    name, table, info, arm=arm, policy_name=policy_name, seed_k=k,
                    parent=parent, landmark=int(row["prefix_len"]),
                    expected_len=int(max_new_tokens), max_new_tokens=max_new_tokens,
                ))
    return out


def routed_block(cells: Sequence[Cell], horizon: int) -> dict[str, Any]:
    """Routed-window options: ALL-scope target membership only, 64-row windows, 256 pulse rows."""
    all_policy = next(c.policy for c in cells if c.scope == "ALL")
    return {"target_policies": [all_policy], "window": ROUTED_WINDOW, "after_pulse": horizon}


def confirm_sets(split: Mapping[str, Any], infos: Mapping[str, Mapping[str, Any]]) -> dict:
    """Confirm question keys, their duplicate-group ids and prompt hashes (alias check)."""
    by_q = {M.question_key(q["dataset"], q["source_problem_id"]): q for q in split["questions"]}
    confirm = {k for k, q in by_q.items() if q["split"] == "confirm"}
    return {
        "keys": confirm,
        "groups": {by_q[k]["group_id"] for k in confirm},
        "prompts": {infos[k]["prompt_sha256"] for k in confirm if k in infos},
        "by_q": by_q,
    }


def assert_no_confirm(
    questions: Sequence[str], split: Mapping[str, Any], infos: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """0 confirm questions, aliases or duplicate groups; only dev-cal questions (asserted)."""
    conf = confirm_sets(split, infos)
    problems = []
    for q in questions:
        entry = conf["by_q"].get(q)
        if entry is None:
            problems.append(f"{q}: not in split-v1")
            continue
        if entry["split"] != "dev" or entry.get("subsplit") != SUBSPLIT:
            problems.append(f"{q}: split {entry['split']}/{entry.get('subsplit')} is not dev-cal")
        if q in conf["keys"]:
            problems.append(f"{q}: is a confirm question")
        if entry["group_id"] in conf["groups"]:
            problems.append(f"{q}: shares a duplicate group with a confirm question")
        if infos[q]["prompt_sha256"] in conf["prompts"]:
            problems.append(f"{q}: prompt hash equals a confirm prompt (alias)")
    if problems:
        raise X2BuildError("confirm exclusion failed: " + "; ".join(problems[:10]))
    return {"n_questions": len(questions), "n_confirm_questions_in_split": len(conf["keys"]),
            "intersection_with_confirm": 0, "duplicate_groups_in_split":
            len(split["duplicate_groups"]), "split_sha256": split["sha256"]}


def check_manifest(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], cells: Sequence[Cell], *,
    split: Mapping[str, Any], infos: Mapping[str, Mapping[str, Any]], n_policy: str,
    max_new_tokens: int, expect_tree: str | None,
) -> dict[str, Any]:
    """Every design assertion of the task; raises X2BuildError, returns the evidence."""
    verify(manifest)
    M.validate_manifest(manifest)
    if expect_tree is not None and manifest["code_tree"] != expect_tree:
        raise X2BuildError(f"code_tree {manifest['code_tree']} is not the frozen tree")
    table = M.manifest_table(manifest)
    eligible = {r["question"]: r for r in rows if r["eligible"]}
    requests = manifest["requests"]
    confirm = assert_no_confirm(sorted(manifest["questions"]), split, infos)
    if set(manifest["questions"]) != set(eligible):
        raise X2BuildError("manifest questions differ from the eligible questions")
    if len(requests) != N_CONDITIONS * len(eligible) * len(SEEDS):
        raise X2BuildError(f"{len(requests)} requests != 85 x {len(eligible)} x 2")
    expected_names = {k: {p for _, p in condition_policies(cells, k, n_policy)} for k in SEEDS}
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for r in requests:
        row = eligible[r["question"]]
        prefix = int(row["prefix_len"])
        parent = r["parent"]
        if parent is None or parent["prefix_len"] != prefix or "trace_ref" not in parent:
            raise X2BuildError(f"{r['uid']}: missing or wrong parent")
        if r["landmark"] != prefix:
            raise X2BuildError(f"{r['uid']}: landmark {r['landmark']} is not the branch point")
        if r["schedule_overrides"] or r["switch_penalty"] is not None or r["seed_k"] not in SEEDS:
            raise X2BuildError(f"{r['uid']}: unexpected override, switch penalty or seed")
        if r["expected_len"] != max_new_tokens:
            raise X2BuildError(f"{r['uid']}: expected_len {r['expected_len']} != horizon")
        cap = min(M.request_cap(r["prompt_tokens"], prefix), max_new_tokens)
        if r["max_tokens"] != cap or cap != max_new_tokens:
            raise X2BuildError(f"{r['uid']}: max_tokens {r['max_tokens']} != X2 horizon {max_new_tokens}")
        policy = table.policies[r["policy_index"]]
        k = policies.random_set_index(policy)
        if r["arm"].startswith("M"):
            if k != r["seed_k"] or not policy.name.endswith(f"@random{r['seed_k']}"):
                raise X2BuildError(f"{r['uid']}: random set {k} is not paired with seed")
        elif k is not None:
            raise X2BuildError(f"{r['uid']}: a non-M arm uses a random set")
        if r["policy_name"] != "zero" and (policy.schedule.kind != "landmark"
                                           or policy.schedule.pulse_len != PULSE_LEN):
            raise X2BuildError(f"{r['uid']}: policy {policy.name} is not a 256-token landmark")
        if r["arm"] == "N":
            if policy.operator.kind != "none":
                raise X2BuildError(f"{r['uid']}: the N arm edits routing")
        elif policy.operator.sign != (1 if r["arm"].endswith("+") else -1):
            raise X2BuildError(f"{r['uid']}: arm {r['arm']} disagrees with its policy sign")
        groups[(r["question"], r["seed_k"])].append(r)
    for key, members in groups.items():
        names = [m["policy_name"] for m in members]
        if len(members) != N_CONDITIONS or len(set(names)) != N_CONDITIONS:
            raise X2BuildError(f"{key}: {len(members)} requests, {len(set(names))} conditions")
        if set(names) != expected_names[key[1]]:
            raise X2BuildError(f"{key}: the condition set differs from the design")
        if len({m["shard"] for m in members}) != 1:
            raise X2BuildError(f"{key}: arms are spread over several shards")
    if len(groups) != len(eligible) * len(SEEDS):
        raise X2BuildError("a (question, seed) group is missing")
    arms = Counter(r["arm"] for r in requests)
    if set(arms) - set(ARM_LABELS):
        raise X2BuildError(f"unexpected arms {sorted(arms)}")
    return {
        "confirm_exclusion": confirm, "n_groups": len(groups), "arms": dict(sorted(arms.items())),
        "conditions_per_group": N_CONDITIONS, "validate_manifest": "passed",
        "set_k_paired_with_seed_k": True, "seal_verified": True,
        "all_requests_have_parent_and_landmark_at_branch_point": True,
        "horizon_enforced_by_manifest": True,
    }


def build_x2_manifest(
    world: M.World, rows: Sequence[Mapping[str, Any]], *, name: str, n_shards: int,
    max_new_tokens: int, n_policy: str = "zero", code_tree: str | None = None,
) -> tuple[dict[str, Any], list[Cell]]:
    """Policy table, requests and the sealed manifest (snapshot ``build_manifest``)."""
    table, cells = x2_table(world.inputs, n_policy)
    requests = build_requests(
        name, table, cells, world.infos, rows, max_new_tokens=max_new_tokens, n_policy=n_policy)
    manifest = M.build_manifest(
        name, "x2", table, requests, world.infos, n_shards=n_shards, seals=world.seals,
        code_tree=code_tree, lexicon_path=world.lexicon_path, vocab_path=world.vocab_path,
        routed=routed_block(cells, max_new_tokens),
    )
    return manifest, cells


# ------------------------------------------------------------------ enumeration

def percentiles(values: Sequence[float]) -> dict[str, float]:
    """min / median / p90 / max (p90 and median by linear interpolation) of ``values``."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"n": 0}

    def at(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {"n": len(ordered), "min": ordered[0], "median": at(0.5), "p90": at(0.9),
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def alt_balance(
    name: str, manifest: Mapping[str, Any], n_shards: int, max_new_tokens: int
) -> dict[str, Any]:
    """Prefill-token balance of the shards under expected_len options (ii) built and (iii)."""
    out: dict[str, Any] = {}
    for label, fn in (
        ("ii_horizon_constant", lambda r: max_new_tokens),
        ("iii_decode_equivalents", lambda r: int(round(
            max_new_tokens + (r["prompt_tokens"] + r["parent"]["prefix_len"])
            / PREFILL_PER_DECODE_TOKEN))),
    ):
        copies = [dict(r, expected_len=fn(r), shard=None, order=None)
                  for r in manifest["requests"]]
        M.assign_shards(name, copies, n_shards)
        prefill = [0] * n_shards
        count = [0] * n_shards
        for r in copies:
            prefill[r["shard"]] += r["prompt_tokens"] + r["parent"]["prefix_len"]
            count[r["shard"]] += 1
        out[label] = {"requests": count, "prefill_tokens": prefill,
                      "prefill_max_over_mean": max(prefill) / (sum(prefill) / n_shards)}
    return out


def enumerate_manifest(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], cells: Sequence[Cell], *,
    max_new_tokens: int, n_policy: str,
) -> dict[str, Any]:
    """Counts of one manifest: arms, cells, shards, prefill and decode tokens, NLL positions."""
    requests = manifest["requests"]
    cell_of = {c.policy: c for c in cells}
    by_arm = Counter(r["arm"] for r in requests)
    by_cell: Counter[str] = Counter()
    for r in requests:
        base = r["policy_name"].split("@")[0]
        role = "M" if "@random" in r["policy_name"] else ("N" if r["arm"] == "N" else "E")
        by_cell[f"{role}|{base}" + (f"|k{r['seed_k']}" if role == "M" else "")] += 1

    def n_prefill(rs: Sequence[Mapping[str, Any]]) -> int:
        return sum(r["prompt_tokens"] + r["parent"]["prefix_len"] for r in rs)

    shards = []
    for shard in range(manifest["n_shards"]):
        members = [r for r in requests if r["shard"] == shard]
        arms = Counter(r["arm"] for r in members)
        shards.append({
            "shard": shard, "requests": len(members), "groups": len({
                (r["question"], r["seed_k"]) for r in members}),
            "expected_tokens": manifest["shards"][shard]["expected_tokens"],
            "prefill_tokens": n_prefill(members),
            "decode_tokens_cap": len(members) * max_new_tokens,
            "arms": dict(sorted(arms.items())),
        })
    nll = [r for r in requests if r["arm"] in ("N", "E+", "E-")]
    nll_positions = sum(r["prompt_tokens"] + r["parent"]["prefix_len"] + PULSE_LEN for r in nll)
    eligible = [r for r in rows if r["eligible"]]
    return {
        "manifest": {"name": manifest["name"], "sha256": manifest["sha256"],
                     "code_tree": manifest["code_tree"], "n_shards": manifest["n_shards"],
                     "policy_table_digest": manifest["inputs"]["policy_table"],
                     "n_policies": len(manifest["policy_table"]["policies"]),
                     "routed": manifest["routed"], "n_policy": n_policy_name(n_policy)},
        "requests": len(requests), "by_arm": dict(sorted(by_arm.items())),
        "by_cell": dict(sorted(by_cell.items())), "cells": len(cell_of),
        "by_shard": shards,
        "prefill_tokens_total": n_prefill(requests),
        "decode_tokens_cap_total": len(requests) * max_new_tokens,
        "nll_sequences": len(nll), "nll_positions_total": nll_positions,
        "uncapped_decode_tokens_counterfactual": sum(
            max(r["trace_completion_tokens"] - r["prefix_len"], 0) * N_CONDITIONS * len(SEEDS)
            for r in eligible),
        "expected_len_alternatives": alt_balance(
            manifest["name"], manifest, manifest["n_shards"], max_new_tokens),
    }


def eligibility_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and prefix-length distributions of the eligibility rows."""
    eligible = [r for r in rows if r["eligible"]]
    return {
        "n_dev_cal": len(rows), "n_eligible": len(eligible),
        "n_ineligible": len(rows) - len(eligible),
        "by_bucket": {b: {"dev_cal": sum(r["bucket"] == b for r in rows),
                          "eligible": sum(r["bucket"] == b for r in eligible)}
                      for b in sorted({r["bucket"] for r in rows})},
        "ineligible": [{k: r[k] for k in ("question", "bucket", "reason", "trace_completion_tokens",
                                          "think_end", "n_onsets_before_floor")}
                       for r in rows if not r["eligible"]],
        "onset_o": percentiles([r["onset_o"] for r in eligible]),
        "prefix_len": percentiles([r["prefix_len"] for r in eligible]),
        "branch_prompt_tokens_incl_prompt": percentiles(
            [r["branch_prompt_tokens"] for r in eligible]),
        "streaming_offline_disagreements": [
            {"question": r["question"], "streaming": r["streaming_first_onset"],
             "offline": r.get("offline_first_onset")}
            for r in rows if "offline_first_onset" in r
            and r["offline_first_onset"] != r["streaming_first_onset"]],
    }


# ------------------------------------------------------------------ files

def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    """Pretty canonical JSON; refuses to replace a file whose content differs."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    if final.exists():
        if final.read_text() != text:
            raise X2BuildError(f"refusing to overwrite differing file {final}")
        return
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(final)


def load_or_compute_eligibility(
    path: Path, world: M.World, store: Any, lexicon: Lexicon, vocab: VocabBytes
) -> dict[str, Any]:
    """Reuse a verified eligibility.json whose inputs match, else compute and write it."""
    inputs = {
        "split_sha256": world.seals["split"], "prompt_table_sha256": world.seals["prompt_table"],
        "lexicon_sha256": world.seals["lexicon"], "vocab_file_sha256": file_sha256(
            world.vocab_path), "onset_floor": ONSET_FLOOR, "onset_schedule": ONSET_SCHEDULE,
        "branch_index_rule": "prefix_len = decision token index + 1",
    }
    if path.is_file():
        record = json.loads(path.read_text())
        verify(record)
        if record["inputs"] != inputs:
            raise X2BuildError(f"{path} was computed from different inputs")
        return record
    infos = M.select(world.infos, split="dev", subsplit=SUBSPLIT)
    rows = compute_eligibility(infos, store, lexicon, vocab)
    record = seal({"schema": ELIGIBILITY_SCHEMA, "inputs": inputs,
                   "summary": eligibility_summary(rows), "rows": rows})
    write_json_once(path, record)
    return record


# ------------------------------------------------------------------ main

def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_unfrozen and not args.name.startswith("x2-dryrun"):
        raise X2BuildError("--dry-unfrozen is restricted to names starting with x2-dryrun")
    if "max_new_tokens" not in inspect.signature(M.make_request).parameters:
        raise X2BuildError("X2 requires s2; the imported make_request cannot enforce continuation caps")
    out = Path(args.out)
    world = M.load_world()
    split = M.load_split()
    lexicon = Lexicon.load(world.lexicon_path)
    vocab = VocabBytes.load(world.vocab_path)
    store = M.TraceStore(cache_dir=out / "trace-offsets")
    elig = load_or_compute_eligibility(out / "eligibility.json", world, store, lexicon, vocab)
    rows = elig["rows"]
    multi = len(args.n_shards) > 1
    results: dict[str, Any] = {}
    for n_shards in args.n_shards:
        manifest, cells = build_x2_manifest(
            world, rows, name=args.name, n_shards=n_shards, max_new_tokens=args.max_new_tokens,
            n_policy=args.n_policy)
        evidence = check_manifest(
            manifest, rows, cells, split=split, infos=world.infos, n_policy=args.n_policy,
            max_new_tokens=args.max_new_tokens,
            expect_tree=None if args.dry_unfrozen else args.expect_tree)
        path = out / (f"{args.name}.n{n_shards}.json" if multi else f"{args.name}.json")
        M.write_manifest(path, manifest)
        info = enumerate_manifest(
            manifest, rows, cells, max_new_tokens=args.max_new_tokens, n_policy=args.n_policy)
        info.update(path=str(path), file_sha256=file_sha256(path), assertions=evidence)
        results[str(n_shards)] = info
    enumeration = {
        "schema": ENUMERATION_SCHEMA, "name": args.name,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
        "snapshot": {"code_tree": engine.code_tree_sha256(), "expected": args.expect_tree,
                     "dry_unfrozen": args.dry_unfrozen,
                     "moe_steer": str(Path(sys.modules["moe_steer"].__file__).resolve().parent)},
        "design": {"questions": f"split-v1 {SUBSPLIT}", "seeds": list(SEEDS),
                   "conditions_per_group": N_CONDITIONS, "pulse_len": PULSE_LEN,
                   "max_new_tokens_intended": args.max_new_tokens,
                   "horizon_enforced_by_manifest": True, "n_arm_policy": n_policy_name(
                       args.n_policy), "onset_floor": ONSET_FLOOR},
        "inputs": {"split": world.seals["split"], "targets": world.seals["targets"],
                   "random_sets": world.seals["random_sets"], "lexicon": world.seals["lexicon"],
                   "prompt_table": world.seals["prompt_table"],
                   "eligibility_json": file_sha256(out / "eligibility.json")},
        "eligibility": elig["summary"], "manifests": results,
    }
    target = Path(args.enumeration) if args.enumeration else out / f"{args.name}.enumeration.json"
    write_json_once(target, enumeration)
    return enumeration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="manifest name (the uid salt)")
    ap.add_argument("--n-shards", type=int, nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, choices=HORIZONS, required=True)
    ap.add_argument("--out", required=True, help="absolute output directory")
    ap.add_argument("--n-policy", choices=("zero", "sham"), default="sham",
                    help="N arm: the zero policy (PREREG literal) or sham_ALL_landmark")
    ap.add_argument("--enumeration", help="path of the enumeration JSON (default under --out)")
    ap.add_argument("--expect-tree", required=True, help="expected 64-hex run snapshot tree")
    ap.add_argument("--dry-unfrozen", action="store_true", help="proposal-tree dry build only")
    args = ap.parse_args(argv)
    if not os.path.isabs(args.out):
        ap.error("--out must be an absolute path")
    if len(args.expect_tree) != 64 or any(c not in "0123456789abcdef" for c in args.expect_tree):
        ap.error("--expect-tree must be a lowercase 64-hex digest")
    if args.dry_unfrozen and not args.name.startswith("x2-dryrun"):
        ap.error("--dry-unfrozen requires an x2-dryrun name")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enumeration = run(args)
    brief = {n: {"requests": i["requests"], "sha256": i["manifest"]["sha256"],
                 "path": i["path"]} for n, i in enumeration["manifests"].items()}
    print(json.dumps({"eligible": enumeration["eligibility"]["n_eligible"],
                      "ineligible": enumeration["eligibility"]["n_ineligible"],
                      "manifests": brief}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
