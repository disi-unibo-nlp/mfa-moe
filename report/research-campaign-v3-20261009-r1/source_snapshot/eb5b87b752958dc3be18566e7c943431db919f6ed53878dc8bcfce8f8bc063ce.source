"""Candidate-specific proposals, blinded readers, and prefix enrollment.

Cheap proposals are deliberately not called semantic truth. GPU audit receipts
must qualify a recognizer before it controls scientific generation.
"""
from __future__ import annotations

from collections import defaultdict
import re

from moe_exp.routing_control.transitions_v24 import _completed_sentences, _closed_math
from .contracts import digest, identity, require, seal, verify

CHECK = re.compile(r"\b(check(?:ing|ed)?|verif(?:y|ying|ied)|substitut(?:e|ing|ed)|satisf(?:y|ies)|test(?:ing|ed)?)\b", re.I)
INTENT = re.compile(r"\b(?:will|should|need to|let(?:'s| us)|want to|must)\s+(?:now\s+)?(?:check|verify|test|substitute)\b", re.I)
RELATION = re.compile(r"(?<![<>=])=(?![=])|\\(?:equiv|leq|geq)|[≠≤≥]")


def candidate_proposals(problem, text):
    require(isinstance(problem, str) and isinstance(text, str), "prefix text must be strings")
    units = list(_completed_sentences(text))
    result, checked = [], set()
    for index, (start, end, sentence) in enumerate(units):
        math = _closed_math(sentence)
        if math is None:
            continue
        expressions = list(math)
        if not expressions and re.search(r"\\boxed\s*\{[^{}]+\}", sentence):
            expressions = [sentence]
        plain = re.search(r"\b(?:final answer|the answer|answer)\s+(?:is|:)\s*([^.;\n]+)", sentence, re.I)
        if not expressions and plain:
            expressions = ["answer=" + plain.group(1)]
        if not expressions and RELATION.search(sentence):
            expressions = [sentence]
        is_check = bool(CHECK.search(sentence) and RELATION.search(sentence) and not INTENT.search(sentence))
        if is_check:
            # Only a visible reference to this candidate can mark it checked.
            for candidate in result:
                compact = re.sub(r"\s+", "", sentence)
                # Equal numbers in different candidates do not establish identity.
                expression = re.sub(r"\s+", "", candidate["expression"])
                reference = re.search(r"(?<![A-Za-z0-9_])" + re.escape(candidate["lhs"]) +
                                      r"(?![A-Za-z0-9_])", sentence)
                value = re.search(r"(?<![A-Za-z0-9_.])" + re.escape(candidate["value"]) +
                                  r"(?![A-Za-z0-9_.])", sentence)
                if expression in compact or (reference and value):
                    checked.add(candidate["candidate_id"])
            continue
        for expression in expressions:
            match = re.search(r"([A-Za-z][A-Za-z_0-9{}\\]*|\\boxed\s*\{[^{}]+\})\s*=\s*([^=;\n]+)", expression)
            boxed = re.search(r"\\boxed\s*\{([^{}]+)\}", expression)
            if match:
                lhs, value = match.group(1).strip(), match.group(2).strip().rstrip("., $")
                canonical = lhs + "=" + value
            elif boxed:
                lhs, value, canonical = "answer", boxed.group(1).strip(), "answer=" + boxed.group(1).strip()
            else:
                continue
            if not value or len(canonical) > 512:
                continue
            cid = identity("candidate", re.sub(r"\s+", "", canonical))
            result.append({"candidate_id": cid, "expression": canonical, "lhs": lhs, "value": value,
                "kind": "complete" if boxed or re.search(r"\b(?:final answer|answer is|the answer)\b", sentence, re.I) else "intermediate",
                "symbolic": bool(re.search(r"[A-Za-z\\]", value)), "sentence_index": index,
                "char_start": start, "char_end": end, "originating_sentence": sentence,
                "problem_sha256": digest(problem), "constraint_text": problem,
                "constraint_scope": "original problem; specific relation requires semantic audit"})
    # Duplicate assertions of one value do not create a later favorable opportunity.
    unique = {}
    for row in result:
        unique.setdefault(row["candidate_id"], {**row, "checked": row["candidate_id"] in checked})
    return list(unique.values())


def eligible_proposal(problem, text):
    units = list(_completed_sentences(text))
    if not units or text[units[-1][1]:].strip():
        return None
    candidates = candidate_proposals(problem, text)
    return next((r for r in candidates if not r["checked"] and r["char_end"] == units[-1][1]), None)


def reader_messages(record, *, role):
    require(role in ("eligibility", "outcome"), "unknown reader role")
    visible = {k: record[k] for k in ("problem", "prefix_text")}
    # Keep the reader blind to the cheap detector's asserted answer.
    visible["candidate"] = {k: record["candidate"][k] for k in
                            ("expression", "originating_sentence") if k in record["candidate"]}
    if role == "outcome":
        visible["window_text"] = record["window_text"]
    rubric = ("Use only the supplied original problem, emitted prefix, and candidate. Numeric and symbolic, "
        "intermediate and complete candidates are eligible. A substantive verification performs a calculation "
        "or constraint comparison checking THIS candidate against a condition of the original problem. "
        "Intending to check or checking a different candidate is false. Gold answers and future text are absent. ")
    if role == "eligibility":
        keys = "candidate_valid, constraint_bound, already_checked, eligible"
        request = "Judge whether this specific candidate is complete enough to check and remains unchecked at the end of this prefix."
    else:
        keys = "substantive_verification, evidence"
        request = "Judge whether the supplied continuation window performs that check, excluding the triggering sentence. Evidence must be a verbatim substring of window_text, or null."
    import json
    return [{"role": "system", "content": rubric + request + " Return exactly one JSON object with keys " + keys +
             ". Boolean judgments are true, false, or null if ambiguous. No explanatory prose outside JSON."},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)}]


def parse_rating(text, role):
    import json
    try:
        value = json.loads(text.rsplit("</think>", 1)[-1].strip())
    except (ValueError, AttributeError):
        return None
    boolean = ("candidate_valid", "constraint_bound", "already_checked", "eligible") if role == "eligibility" else ("substantive_verification",)
    expected = set(boolean) | ({"evidence"} if role == "outcome" else set())
    if not isinstance(value, dict) or set(value) != expected or any(
            value[k] is not None and type(value[k]) is not bool for k in boolean):
        return None
    if role == "outcome" and value["evidence"] is not None and not isinstance(value["evidence"], str):
        return None
    if role == "eligibility" and value["eligible"] is True and not (
            value["candidate_valid"] is True and value["constraint_bound"] is True and value["already_checked"] is False):
        return None
    return value


def outcome(record, ratings):
    require(record.get("horizon") == 1024 and record.get("arm_blinded") is True, "outcome horizon/blinding differs")
    indexed = {r["reader"]: r for r in ratings}
    require(len(indexed) == len(ratings) and set(indexed) <= {0, 1}, "duplicate reader")
    votes = []
    for reader in (0, 1):
        r = indexed.get(reader)
        if not r or r.get("finish_reason") != "stop" or r.get("input_sha256") != digest(record):
            return {"success": None, "reason": "MISSING_OR_UNBOUND_READER"}
        parsed = parse_rating(r["text"], "outcome")
        if parsed is None or parsed["substantive_verification"] is None:
            return {"success": None, "reason": "AMBIGUOUS_READER"}
        evidence = parsed["evidence"]
        if parsed["substantive_verification"] and (not evidence or evidence not in record["window_text"]):
            return {"success": None, "reason": "INVALID_EVIDENCE_SPAN"}
        votes.append(parsed["substantive_verification"])
    if votes[0] != votes[1]:
        return {"success": None, "reason": "READER_DISAGREEMENT"}
    if votes[0] is False and record.get("execution_complete") is not True:
        return {"success": None, "reason": "INCOMPLETE_EXECUTION"}
    return {"success": votes[0], "reason": "AGREEING_READER_DEFINED_OUTCOME"}


def audit_frame(records, boundary, model):
    verify(boundary); verify(model)
    ordered = sorted(boundary["discovery"], key=lambda f: digest(["verification-audit-split", f]))
    splits = {f: "apparent" if i < 36 else "held_out" for i, f in enumerate(ordered)}
    indexed = defaultdict(list)
    for row in records:
        require(row["family"] in splits and row["prefix_only"] is True and row["population"] == "discovery",
                "reader frame sees independent/future data")
        require(row["stratum"] in ("proposal", "checked", "intent", "unfiltered"), "unknown audit stratum")
        indexed[splits[row["family"]], row["stratum"]].append(row)
    selected = []
    # Enriched precision frame; unfiltered rows retain a separate coverage role.
    quotas = {"apparent": {"proposal": 48, "checked": 12, "intent": 12, "unfiltered": 24},
              "held_out": {"proposal": 16, "checked": 4, "intent": 4, "unfiltered": 8}}
    shortages = []
    seen = set()
    for split, strata in quotas.items():
        for stratum, count in strata.items():
            per_family = defaultdict(list)
            for r in indexed[split, stratum]:
                per_family[r["family"]].append(r)
            ordered_families = sorted(per_family, key=lambda f: digest(["audit-family", split, stratum, f]))
            for f in per_family:
                per_family[f].sort(key=lambda r: digest(["audit-row", r["uid"]]))
            candidates = [per_family[f][i] for i in range(max(map(len, per_family.values()), default=0))
                          for f in ordered_families if i < len(per_family[f])]
            rows = [r for r in candidates if r["uid"] not in seen][:count]
            shortages.append({"split": split, "stratum": stratum, "requested": count, "available": len(rows)})
            for row in rows:
                selected.append({**row, "split": split})
                seen.add(row["uid"])
    return seal({"schema": "verification-semantic-frame-v3", "model_sha256": model["sha256"],
        "boundary_sha256": boundary["sha256"], "records": selected, "shortages": shortages,
        "family_split": splits, "splits_frozen_before_ratings": True,
        "uncertainty_unit": "family", "truth_scope": "two draws from the frozen reader; not independent humans"})


def qualify_recognizer(frame, ratings):
    verify(frame, "verification-semantic-frame-v3")
    by_record = {r["uid"]: r for r in frame["records"]}
    by_uid = defaultdict(dict)
    for row in ratings:
        verify(row, "verification-reader-receipt-v3")
        require(row["reader"] in (0, 1) and row["reader"] not in by_uid[row["uid"]] and
                row["frame_sha256"] == frame["sha256"] and row["uid"] in by_record and
                row["role"] == "eligibility" and row["input_sha256"] == digest(by_record[row["uid"]]) and
                row["checkpoint"], "duplicate/unbound semantic rating")
        by_uid[row["uid"]][row["reader"]] = row
    summaries, reasons = {}, []
    for decision in ("eligible", "already_checked"):
        family_votes = defaultdict(list)
        held = defaultdict(list)
        for row in frame["records"]:
            pair = by_uid[row["uid"]]
            parsed = [parse_rating(pair[r]["text"], "eligibility") if r in pair and pair[r].get("finish_reason") == "stop" else None for r in (0, 1)]
            proposed = row["proposed"][decision]
            yes = all(p is not None and p[decision] is True for p in parsed)
            if proposed:
                (family_votes if row["split"] == "apparent" else held)[row["family"]].append(int(yes))
        precision = sum(sum(v)/len(v) for v in family_votes.values()) / len(family_votes) if family_votes else 0.
        summaries[decision] = {"apparent_families": len(family_votes), "apparent_family_precision": precision,
            "held_out_family_values": {f: sum(v)/len(v) for f, v in held.items()},
            "held_out_support": len(held), "missing_and_disagreement_count_as_unconfirmed": True}
        values = [sum(v)/len(v) for v in held.values()]
        if values:
            import random
            rng = random.Random(7301)
            draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(5000))
            summaries[decision]["held_out_family_interval95"] = [draws[124], draws[4874]]
        if len(family_votes) < 10 or precision < .70:
            reasons.append(decision + ":SUPPORT_OR_PRECISION")
        if not held:
            reasons.append(decision + ":HELD_OUT_UNCERTAINTY_UNAVAILABLE")
    coverage = []
    symbolic = []
    for record in frame["records"]:
        pair = by_uid[record["uid"]]
        parsed = [parse_rating(pair[r]["text"], "eligibility") if r in pair and pair[r].get("finish_reason") == "stop" else None for r in (0, 1)]
        vote = parsed[0]["eligible"] if all(p is not None for p in parsed) and parsed[0]["eligible"] == parsed[1]["eligible"] else None
        entry = {"uid": record["uid"], "family": record["family"], "split": record["split"],
                 "proposed": record["proposed"]["eligible"], "reader_eligible": vote}
        if record["stratum"] == "unfiltered":
            coverage.append(entry)
        if record.get("candidate", {}).get("symbolic"):
            symbolic.append(entry)
    return seal({"schema": "verification-recognizer-qualification-v3", "status": "FAIL" if reasons else "PASS",
        "frame_sha256": frame["sha256"], "model_sha256": frame["model_sha256"], "prefix_only": True,
        "method": "cheap", "decisions": summaries, "failure_reasons": reasons,
        "unfiltered_coverage_rows": coverage, "symbolic_audit_rows": symbolic,
        "reader_checkpoint": next(iter({r["checkpoint"] for r in ratings}), None),
        "held_out_gate": "coverage and uncertainty; no invented precision threshold"})


def enroll_prefixes(model, source_rows, recognition):
    verify(model); verify(recognition)
    require(recognition["status"] == "PASS" and recognition["model_sha256"] == model["sha256"], "recognizer unqualified")
    families, prefixes, unavailable = set(), [], []
    for row in source_rows:
        require(row["seed"] == 0 and row["arm"] == "native" and row["population"] == "final" and
                row["family"] not in families, "wrong or duplicate source-native path")
        families.add(row["family"])
        if row.get("execution_complete") is not True:
            unavailable.append({"family": row["family"], "status": "UNKNOWN_SOURCE_EXECUTION"})
            continue
        candidates = row.get("qualified_opportunities", [])
        if candidates:
            first = min(candidates, key=lambda p: (p["position"], p["candidate"]["candidate_id"]))
            require(first["prefix_only"] is True and first["recognizer_sha256"] == recognition["sha256"] and
                    digest(first["prefix_token_ids"]) == first["prefix_ids_sha256"], "unqualified opportunity")
            prefixes.append({**first, "family": row["family"], "source_uid": row["uid"]})
            prefixes[-1]["original_prompt_ids_sha256"] = row["original_prompt_ids_sha256"]
        else:
            unavailable.append({"family": row["family"], "status": "NO_ELIGIBLE_PREFIX"})
    require(len(families) == 96, "final source enrollment incomplete")
    return seal({"schema": "verification-prefix-enrollment-v3", "model_sha256": model["sha256"],
        "recognizer_sha256": recognition["sha256"], "source_seed": 0, "source_families": sorted(families),
        "prefixes": prefixes, "unavailable": unavailable, "frozen_before_treatment": True,
        "population": "native-seed-0 eligible prefixes; not all original prompts"})
