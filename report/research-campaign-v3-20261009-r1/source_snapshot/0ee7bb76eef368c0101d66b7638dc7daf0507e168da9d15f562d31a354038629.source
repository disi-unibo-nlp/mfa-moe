"""Resume finite scientific shards only from source-bound qualified contracts."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .accounting import begin_attempt, end_attempt, assignment_ledger
from .contracts import check_files, digest, load, require, require_generation_binding, save, verify
from .backend import Backend
from .policy import action_for
from .storage import commit_generation, capture_arrays


def prerequisites(campaign_path, key, qualification_path):
    campaign = load(campaign_path)
    check_files(campaign["code_files"])
    model = load(campaign["models"][key])
    check_files(model["config_files"])
    qualification = load(qualification_path, "verification-backend-qualification-v3")
    from .prepare import runtime_binding
    require(qualification["status"] == "PASS" and qualification["model_sha256"] == model["sha256"] and
            qualification["campaign_sha256"] == campaign["sha256"] and
            qualification["code_files"] == campaign["code_files"], "backend scientific prerequisite failed or changed")
    require(runtime_binding()["sha256"]==campaign["runtime_sha256"] and
            qualification["runtime"]["sha256"]==campaign["runtime_sha256"],"execution runtime changed")
    return campaign, model


def run_shard(campaign_path, key, qualification_path, manifest_path, output, deadline,
              *, policy_path=None, recognition_path=None, begin=0, count=None, validation_path=None, price_path=None):
    campaign, model = prerequisites(campaign_path, key, qualification_path)
    manifest = load(manifest_path, "verification-assignments-v3")
    require(manifest["model_sha256"] == model["sha256"], "foreign assignment roster")
    from .protocol import protocol, validate_assignment
    require(manifest["protocol_sha256"]==protocol()["sha256"],"assignment protocol differs")
    for row in manifest["assignments"]:validate_assignment(row)
    stage = manifest["stage"]
    if stage in ("utility", "conditional", "final_native"):
        validation = load(validation_path, "verification-validation-v3")
        require(validation["status"] == "PASS" and validation["model_sha256"] == model["sha256"] and
                validation["campaign_sha256"] == campaign["sha256"], "final stage validation missing or failed")
    needed = ("generator_load","prefill","generation","capture","closeout","recovery") + (("controller",) if stage in ("validation","final_native","utility") else ())
    measured = False
    if price_path is not None:
        prices = load(price_path, "verification-prices-v3")
        require(prices["model_sha256"] == model["sha256"] and prices["code_files"] == campaign["code_files"], "price binding differs")
        measured = all(prices["components"][r]["status"] == "MEASURED" for r in needed)
    if not measured:
        require(stage in ("discovery","validation") and count == 1, "unpriced work must be one retained native measurement assignment")
        existing_uids = {p.stem for p in output.glob("*.json") if len(p.stem) == 64}
        selected_uids = {a["uid"] for a in manifest["assignments"][begin:begin+1]}
        require(existing_uids <= selected_uids, "first measurement already exists; collect its prices before another assignment")
    policy = load(policy_path) if policy_path else None
    recognition = load(recognition_path) if recognition_path else None
    if stage in ("validation", "final_native"):
        require(recognition is not None and recognition["status"] == "PASS" and
                recognition["model_sha256"] == model["sha256"], "native source requires qualified online recognition")
    if stage in ("utility","conditional","final_native"):
        require(policy is not None and recognition is not None and
                validation["policy_sha256"]==policy["sha256"] and
                validation["recognizer_sha256"]==recognition["sha256"],"validation belongs to another policy/detector")
    if stage in ("utility", "conditional"):
        require(policy and policy["status"] == "PASS" and recognition and recognition["status"] == "PASS",
                "unqualified policy/recognizer")
    public = load(campaign_path.parent / "PUBLIC_QUESTIONS.json")
    expected_population="discovery" if stage=="discovery" else "validation" if stage=="validation" else "final"
    require(all(public["questions"][r["question_id"]]["family"]==r["family"] and
                public["questions"][r["question_id"]]["population"]==expected_population for r in manifest["assignments"]),
            "question/family/population differs from campaign")
    assignments = manifest["assignments"][begin:begin+count if count is not None else None]
    require(assignments, "empty shard")
    output.mkdir(parents=True, exist_ok=True)
    invocation = str(os.environ["SLURM_JOB_ID"]) + ":" + str(time.time_ns())
    backend, error = None, None
    try:
        backend = Backend(model, deadline=deadline)
        for assignment in assignments:
            uid = assignment["uid"]
            if (output / (uid+".json")).exists():
                existing = load(output / (uid+".json"))
                require_generation_binding(existing,campaign,model)
                require(existing["assignment"] == assignment and not existing["engineering_only"], "changed resume receipt")
                require(existing["recognizer_sha256"]==(recognition["sha256"] if recognition else None),"changed resume detector")
                continue
            if time.time() >= deadline-120:
                break
            path = begin_attempt(output,assignment,invocation)
            stamp = time.monotonic()
            try:
                problem = public["questions"][assignment["question_id"]]["problem"]
                if measured:
                    inputs = len(backend.prompt_ids(problem))+len((assignment.get("prefix") or {}).get("prefix_token_ids",[]))
                    require(inputs <= prices["components"]["prefill"]["max_context_input_tokens"] and
                            inputs+assignment["cap"] <= prices["components"]["generation"]["max_total_context_tokens"],
                            "planned context exceeds measured prefill/generation coverage")
                action = action_for(policy,assignment) if stage in ("conditional","utility") else None
                if stage in ("validation", "final_native"):
                    require(policy is not None and policy["status"] == "PASS", "native scientific capture requires nominated layer")
                    action = {"layer":policy["layer"],"expert":policy["target"],"bias":0.}
                receipt = backend.generate(assignment,problem,action=action,recognition=recognition,
                                           capture_all=stage == "discovery",policy=policy)
                receipt = commit_generation(output,receipt)
                end_attempt(path,receipt=receipt,elapsed_seconds=time.monotonic()-stamp)
            except Exception as exc:
                end_attempt(path,error={"type":type(exc).__name__,"message":str(exc)[:1000]},elapsed_seconds=time.monotonic()-stamp)
                raise  # process may be compromised; supervisor may resume once
    except Exception as exc:
        error = {"type":type(exc).__name__,"message":str(exc)[:1000]}
    finally:
        if backend is not None:
            backend.close()
    receipts = [load(p) for p in output.glob("*.json") if ".attempt-" not in p.name and len(p.stem) == 64]
    for receipt in receipts:require_generation_binding(receipt,campaign,model)
    attempts = [load(p) for p in output.glob("*.attempt-*.json")]
    # The complete roster is retained even when this was one finite shard.
    ledger = assignment_ledger(manifest,receipts,attempts)
    save(output / "LEDGER.json",ledger,replace=True)
    return save(output / ("SHARD-"+invocation.replace(":","-")+".json"),
        {"schema":"verification-shard-v3","manifest_sha256":manifest["sha256"],"job_id":os.environ["SLURM_JOB_ID"],
         "slice":[begin,len(assignments)],"error":error,"ledger_sha256":ledger["sha256"],
         "status":"FAIL" if error else "COMPLETE" if all((output/(a["uid"]+".json")).exists() for a in assignments) else "INCOMPLETE",
         "scientific_success": "NOT_INFERRED_FROM_SCHEDULER"})


def make_audit(campaign_path, key, source, output):
    from transformers import AutoTokenizer
    from .semantics import audit_frame
    from .views import audit_records
    campaign = load(campaign_path)
    check_files(campaign["code_files"])
    model = load(campaign["models"][key])
    public = load(campaign_path.parent / "PUBLIC_QUESTIONS.json")
    manifest = load(campaign_path.parent / key / "DISCOVERY_ASSIGNMENTS.json")
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"],local_files_only=True)
    rows = []
    available = []
    missing = []
    for assignment in manifest["assignments"]:
        path = source / (assignment["uid"]+".json")
        if path.exists():
            receipt = load(path)
            require_generation_binding(receipt,campaign,model)
            require(receipt["assignment"] == assignment, "foreign discovery capture")
            rows.extend(audit_records(receipt,public["questions"][assignment["question_id"]]["problem"],tokenizer))
            available.append(receipt["sha256"])
        else:
            missing.append(assignment["uid"])
    frame = audit_frame(rows,load(campaign_path.parent/"FAMILIES.json"),model)
    return save(output,{**frame,"source_receipts":available,"missing_discovery_assignments":missing,
                       "reader_contract_sha256":load(campaign_path.parent/"READER.json")["sha256"]})


def rate_frame(campaign_path, frame_path, output, deadline, role="eligibility"):
    from .readers import Reader
    campaign = load(campaign_path)
    check_files(campaign["code_files"])
    frame = load(frame_path)
    contract = load(campaign_path.parent/"READER.json")
    from .prepare import runtime_binding
    require(runtime_binding()["sha256"] == campaign["runtime_sha256"], "reader runtime changed")
    check_files(contract["config_files"])
    require(frame["reader_contract_sha256"] == contract["sha256"], "foreign reader frame")
    output.mkdir(parents=True,exist_ok=True)
    reader = Reader(contract["checkpoint"])
    try:
        save(output/"RUNTIME.json", {"schema":"verification-reader-runtime-v3","reader_sha256":contract["sha256"],
            "frame_sha256":frame["sha256"],"load_seconds":reader.load_seconds,"job_id":os.environ["SLURM_JOB_ID"],
            "runtime_sha256":campaign["runtime_sha256"]})
        for row in frame["records"]:
            for draw in (0,1):
                path = output / (row["uid"]+f".reader-{draw}.json")
                if path.exists():
                    existing = load(path)
                    require(existing["frame_sha256"] == frame["sha256"] and existing["input_sha256"] == digest(row)
                            and existing["checkpoint"] == contract["checkpoint"] and existing["reader_contract_sha256"] == contract["sha256"]
                            and existing["runtime_sha256"] == campaign["runtime_sha256"] and existing["reader"] == draw and existing["role"] == role,
                            "changed reader resume input or deployment")
                    continue
                require(time.time() < deadline-120,"reader allocation deadline")
                receipt = reader.rate(row,role,draw)
                save(path,{**receipt,"frame_sha256":frame["sha256"],"reader_contract_sha256":contract["sha256"],
                           "runtime_sha256":campaign["runtime_sha256"]})
    finally:
        reader.close()
    return {"status":"COMPLETE","ratings":len(list(output.glob("*.reader-*.json")))}


def fit_nomination(campaign_path,key,frame_path,ratings_dir,source,output):
    from .nomination import nominate
    from .views import past_profile_rows
    campaign = load(campaign_path); check_files(campaign["code_files"])
    model = load(campaign["models"][key]); frame = load(frame_path)
    ratings = [load(p) for p in ratings_dir.glob("*.reader-*.json")]
    reader = load(campaign_path.parent/"READER.json")
    require(all(r["frame_sha256"] == frame["sha256"] and r["role"] == "eligibility" and
                r["reader_contract_sha256"] == reader["sha256"] and r["checkpoint"] == reader["checkpoint"] and
                r["runtime_sha256"] == campaign["runtime_sha256"] for r in ratings), "nomination reader deployment differs")
    source_ids = {r["source_uid"] for r in frame["records"]}
    receipts = {uid:load(source/(uid+".json")) for uid in source_ids}
    for receipt in receipts.values():require_generation_binding(receipt,campaign,model)
    captures = {uid:capture_arrays(receipt) for uid,receipt in receipts.items()}
    rows = past_profile_rows(frame,ratings,captures,model)
    nominated = nominate(model,rows,load(campaign_path.parent/"FAMILIES.json"))
    return save(output,{**nominated,"frame_sha256":frame["sha256"],"reader_labels_sha256":__import__("moe_exp.verification_study.contracts",fromlist=["digest"]).digest(ratings)})
