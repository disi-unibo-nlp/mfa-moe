"""Explicit scientific gates and enrollment; no effect-based policy search."""
from __future__ import annotations

from pathlib import Path
import time

from .contracts import check_files, digest, identity, load, require, require_generation_binding, save, seal
from .backend import Backend
from .execution import prerequisites
from .protocol import assignments
from .semantics import enroll_prefixes, qualify_recognizer, outcome
from .views import outcome_view, reasoning_limit


def recognition(campaign_path, key, frame_path, ratings_dir, output):
    campaign = load(campaign_path); check_files(campaign["code_files"])
    frame = load(frame_path)
    reader = load(campaign_path.parent / "READER.json")
    ratings = [load(p) for p in ratings_dir.glob("*.reader-*.json")]
    require(all(r["reader_contract_sha256"] == reader["sha256"] and r["checkpoint"] == reader["checkpoint"] and
                r["runtime_sha256"] == campaign["runtime_sha256"] for r in ratings),
            "reader identity changed")
    return save(output, qualify_recognizer(frame,ratings))


def calibrate(campaign_path,key,qualification_path,nomination_path,frame_path,source,output,deadline):
    campaign, model = prerequisites(campaign_path,key,qualification_path)
    nomination = load(nomination_path); frame = load(frame_path)
    require(nomination["status"] == "NOMINATED" and nomination["model_sha256"] == model["sha256"] and
            nomination["frame_sha256"] == frame["sha256"], "nomination failed or changed")
    selected = nomination["selected"]
    discovery = load(campaign_path.parent/key/"DISCOVERY_ASSIGNMENTS.json")
    allowed = {r["family"]:r for r in discovery["assignments"]}
    first = {}
    for record in sorted(frame["records"],key=lambda r:(r["position"],r["uid"])):
        if record["split"] == "apparent" and record["proposed"]["eligible"]:
            first.setdefault(record["family"],record)
    chosen = sorted(first, key=lambda f:digest(["calibration-families",f]))[:16]
    require(len(chosen)>=10,"insufficient discovery candidate families for calibration")
    enrollment = save(output/"ENROLLMENT.json",{"schema":"verification-calibration-enrollment-v3",
        "model_sha256":model["sha256"],"frame_sha256":frame["sha256"],"families":chosen,
        "selection":"first apparent qualified lexical prefix; family hash order; no causal outcomes"})
    public = load(campaign_path.parent/"PUBLIC_QUESTIONS.json")
    backend = Backend(model,deadline=deadline)
    try:
        for family in chosen:
            record = first[family]; original = load(source/(record["source_uid"]+".json"))
            require_generation_binding(original,campaign,model)
            base = allowed[family]
            body = {**base,"cap":64,"prefix":{**record,"candidate":record["candidate"],
                    "original_prompt_ids_sha256":original["original_prompt_ids_sha256"]}}
            body.pop("uid"); body["uid"] = identity(body)
            problem = public["questions"][body["question_id"]]["problem"]
            receipt = backend.generate(body,problem,action={"layer":selected["layer"],"expert":selected["expert"],"bias":0.},bank=True)
            from .storage import commit_generation
            commit_generation(output/"native-predictions",receipt)
        policies = backend.llm.collective_rpc("verification_policy",kwargs={"contract":model,
            "nomination_sha256":nomination["sha256"],"target":selected["expert"],"layer":selected["layer"]})
        require(len(policies)==2 and policies[0]["sha256"] == policies[1]["sha256"],"TP calibrated policy differs")
        return save(output/"POLICY.json",{**policies[0],"enrollment_sha256":enrollment["sha256"],
            "campaign_sha256":campaign["sha256"],"code_files":campaign["code_files"]})
    finally:
        backend.close()


def validation(campaign_path,key,policy_path,recognition_path,source,output):
    campaign = load(campaign_path); check_files(campaign["code_files"])
    model = load(campaign["models"][key]); policy = load(policy_path); detector = load(recognition_path)
    manifest = load(campaign_path.parent/key/"VALIDATION_ASSIGNMENTS.json")
    reasons = []
    require(policy["model_sha256"]==model["sha256"] and policy["campaign_sha256"]==campaign["sha256"] and
            detector["model_sha256"]==model["sha256"],"foreign validation policy/detector")
    if policy["status"] != "PASS" or detector["status"] != "PASS":
        reasons.append("POLICY_OR_RECOGNIZER_UNQUALIFIED")
    for assignment in manifest["assignments"]:
        path = source/(assignment["uid"]+".json")
        if not path.exists():
            reasons.append("MISSING_NATIVE_VALIDATION:"+assignment["uid"])
            continue
        receipt = load(path)
        require_generation_binding(receipt,campaign,model)
        if receipt["assignment"] != assignment or receipt["engineering_only"] or not receipt["execution_complete"]:
            reasons.append("INVALID_VALIDATION_RECEIPT:"+assignment["uid"])
        if receipt["recognizer_sha256"]!=detector["sha256"]:
            reasons.append("CHANGED_OR_MISSING_DETECTOR")
        if any(p["recognizer_sha256"] != detector["sha256"] for p in receipt["qualified_opportunities"]):
            reasons.append("CHANGED_DETECTOR")
    return save(output,{"schema":"verification-validation-v3","status":"FAIL" if reasons else "PASS",
        "campaign_sha256":campaign["sha256"],"model_sha256":model["sha256"],"policy_sha256":policy["sha256"],
        "recognizer_sha256":detector["sha256"],"native_assignments":len(manifest["assignments"]),
        "failure_reasons":reasons,"effect_estimates_used":False,
        "role":"all32 families/two seeds native pipeline coverage; online action semantics qualified by engineering fixtures"})


def freeze_final(campaign_path,key,source,policy_path,recognition_path,output):
    campaign = load(campaign_path); check_files(campaign["code_files"])
    model = load(campaign["models"][key]); policy = load(policy_path); detector = load(recognition_path)
    manifest = load(campaign_path.parent/key/"FINAL_NATIVE_ASSIGNMENTS.json")
    rows = []
    for assignment in manifest["assignments"]:
        if assignment["seed"] != 0:
            continue
        path = source/(assignment["uid"]+".json")
        receipt = load(path) if path.exists() else None
        if receipt:
            require_generation_binding(receipt,campaign,model)
            require(receipt["assignment"]==assignment and receipt["recognizer_sha256"]==detector["sha256"] and
                    receipt["engineering_only"] is False,"changed or unrecognized final source")
        else:
            terminal=load(source/"TERMINAL.json")
            require(terminal["manifest_sha256"]==manifest["sha256"] and assignment["uid"] in terminal["unknown_assignments"],
                    "source path still pending; cannot freeze as no-eligible")
        rows.append({"uid":assignment["uid"],"family":assignment["family"],"seed":0,"arm":"native","population":"final",
            "execution_complete":bool(receipt and receipt["execution_complete"]),
            "original_prompt_ids_sha256":receipt["original_prompt_ids_sha256"] if receipt else None,
            "qualified_opportunities":receipt["qualified_opportunities"] if receipt else []})
    # Freeze every seed0 path, including failed/no-eligible paths, before either
    # treated experiment is dispatched. This file is immutable.
    prefixes = save(output/"PREFIXES.json",enroll_prefixes(model,rows,detector))
    public = load(campaign_path.parent/"PUBLIC_QUESTIONS.json")
    final = [{"family":row["family"],"question_id":qid} for qid,row in public["questions"].items() if row["population"]=="final"]
    save(output/"CONDITIONAL_ASSIGNMENTS.json",assignments(model,"conditional",final,policy=policy,prefixes=prefixes))
    return save(output/"UTILITY_ASSIGNMENTS.json",assignments(model,"utility",final,policy=policy))


def outcome_frame(campaign_path,key,manifest_path,source,output):
    from transformers import AutoTokenizer
    campaign = load(campaign_path); check_files(campaign["code_files"])
    model = load(campaign["models"][key]); manifest = load(manifest_path)
    public = load(campaign_path.parent/"PUBLIC_QUESTIONS.json")
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"],local_files_only=True)
    rows, binding, missing = [], {}, []
    for assignment in manifest["assignments"]:
        path = source/(assignment["uid"]+".json")
        if not path.exists():
            missing.append(assignment["uid"]); continue
        receipt = load(path)
        require_generation_binding(receipt,campaign,model)
        require(receipt["assignment"] == assignment, "foreign conditional receipt")
        row = outcome_view(receipt,public["questions"][assignment["question_id"]]["problem"],
            tokenizer.decode(assignment["prefix"]["prefix_token_ids"],skip_special_tokens=False))
        row["window_text"] = tokenizer.decode(row["window_token_ids"],skip_special_tokens=False)
        rows.append(row); binding[row["uid"]] = assignment["uid"]
    frame = save(output,{"schema":"verification-outcome-frame-v3","records":rows,
        "missing_assignments":missing,"assignment_manifest_sha256":manifest["sha256"],
        "reader_contract_sha256":load(campaign_path.parent/"READER.json")["sha256"]})
    save(output.with_name(output.stem+"-PRIVATE-BINDING.json"),{"schema":"verification-outcome-arm-binding-v3",
         "frame_sha256":frame["sha256"],"assignment_uids":binding,"reader_access":"FORBIDDEN"})
    return frame
