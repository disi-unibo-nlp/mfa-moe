"""Join sealed assignments, blinded readers and gold-only grades without deletion."""
from __future__ import annotations

import os
from pathlib import Path
import time

from .contracts import check_files, digest, identity, load, require, require_generation_binding, save, seal, verify
from .registry import MODEL_KEYS
from .semantics import outcome


def engagement(receipt):
    """Observed same-state actuator dose; zero active rows is not a zero effect."""
    import numpy as np
    from .storage import capture_arrays
    action = receipt.get("action")
    if not action or not receipt["pulse_fired"]:
        return {"status":"NO_ACTIVE_PULSE", "active_rows":0}
    arrays = capture_arrays(receipt)
    stem = f"L{action['layer']}_"
    active = arrays[stem+"active"]
    n = int(active.sum())
    if not n:
        return {"status":"NO_ACTIVE_PULSE", "active_rows":0}
    native = arrays[stem+"native_ids"][active]
    executed = arrays[stem+"executed_ids"][active]
    nw = arrays[stem+"native_weights"][active]
    ew = arrays[stem+"executed_weights"][active]
    width = max(int(native.max()),int(executed.max()))+1
    ni = np.zeros((n,width));ei = np.zeros((n,width));rows=np.arange(n)[:,None]
    ni[rows,native]=nw;ei[rows,executed]=ew
    before = float((native==action["expert"]).any(axis=1).mean())
    after = float((executed==action["expert"]).any(axis=1).mean())
    return {"status":"MEASURED", "active_rows":n, "native_exposure":before,
            "executed_inclusion":after,"inclusion_increment":after-before,
            "relative_weight_displacement":float((np.abs(ei-ni).sum(axis=1)/np.abs(ni).sum(axis=1)).mean()),
            "scope":"same-state active rows on the realized branch; descriptive, not a new treatment-selection gate"}


def require_cpu():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            os.environ.get("SLURM_JOB_ACCOUNT")=="iscrc_miosr" and
            os.environ.get("SLURM_JOB_PARTITION")=="lrd_all_serial", "grading/analysis requires allocated serial CPU step")


def grade_strict(campaign_path,key,manifest_path,source,output,*,begin=0,count=None):
    require_cpu()
    from moe_exp.utils import last_boxed
    from moe_exp.correlation_pipeline.continuation import final_answer
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    from moe_exp.correlation_pipeline.answer_equivalence import candidate_text
    campaign=load(campaign_path);check_files(campaign["code_files"])
    model=load(campaign["models"][key]);manifest=load(manifest_path)
    require(manifest["stage"] in ("final_native","utility") and manifest["model_sha256"]==model["sha256"],"foreign utility grading roster")
    gold=load(campaign_path.parent/"GRADING_ONLY.json")
    public=load(campaign_path.parent/"PUBLIC_QUESTIONS.json")
    require(gold["public_catalog_sha256"]==public["sha256"],"grading/public catalog differs")
    output.mkdir(parents=True,exist_ok=True)
    residuals=[]
    for assignment in manifest["assignments"][begin:begin+count if count is not None else None]:
        uid=assignment["uid"];path=source/(uid+".json")
        if not path.exists():continue
        receipt=load(path);require(receipt["assignment"]==assignment and not receipt["engineering_only"],"foreign grading receipt")
        require_generation_binding(receipt,campaign,model)
        qid=assignment["question_id"]
        reference=gold["questions"][qid]
        reference_text=reference.get("gold") or last_boxed(reference.get("solution",""))
        require(reference_text is not None,"gold extraction failed; do not fabricate answer")
        text=final_answer(receipt["emitted_text"],model["reasoning_parser"])
        blind={"uid":identity("blind-grade",uid),"problem":public["questions"][qid]["problem"],
               "gold":reference_text,"content_text":text or "","finish_reason":receipt["finish_reason"]}
        grade_path=output/(uid+".json")
        if grade_path.exists():
            old=load(grade_path);require(old["input_sha256"]==digest(blind),"grade resume input changed")
            if old["accuracy"] is None:
                residuals.append({"uid":blind["uid"],"problem":blind["problem"],"gold":reference_text,
                                  "candidate":candidate_text(text),"grade_sha256":old["sha256"]})
            continue
        started=time.monotonic()
        strict=score_completion_detailed({"gold_answer":reference_text},answer_type="math",model_text=text or "") if text else {
            "model_answer":"","is_correct":False,"method":"no_native_final_answer","math_verify_status":"not_run"}
        # Preserve the existing corrected scorer convention: uncapped strict
        # rejects require equivalence adjudication; unresolved cases stay unknown.
        primary=True if strict["is_correct"] is True else False if receipt["finish_reason"]=="length" or not text else None
        row=save(grade_path,{"schema":"verification-grade-v3","assignment_uid":uid,"blind_uid":blind["uid"],
            "manifest_sha256":manifest["sha256"],"generation_sha256":receipt["sha256"],"input_sha256":digest(blind),
            "strict":strict,"accuracy":primary,"scorer_contract":3,"seconds":time.monotonic()-started,
            "status":"PENDING_J1" if primary is None else "COMPLETE","gold_visible_only_offline":True})
        if primary is None:
            residuals.append({"uid":blind["uid"],"problem":blind["problem"],"gold":reference_text,
                              "candidate":candidate_text(text),"grade_sha256":row["sha256"]})
    return save(output/"J1_FRAME.json",{"schema":"verification-j1-frame-v3","records":residuals,
        "manifest_sha256":manifest["sha256"],"reader_contract_sha256":load(campaign_path.parent/"READER.json")["sha256"],
        "prompt_version":1,"parser_version":2,"votes":3,"output_cap":8192,"blinding":"question/gold/candidate; no arm/model/seed"},replace=True)


def assemble(campaign_path,root,output):
    require_cpu()
    campaign=load(campaign_path);check_files(campaign["code_files"])
    reader_contract=load(campaign_path.parent/"READER.json")
    records,enrollment,sources=[],{},{}
    for key in MODEL_KEYS:
        model=load(campaign["models"][key]);base=root/key
        prefix_path=base/"final/PREFIXES.json"
        prefix=load(prefix_path) if prefix_path.exists() else None
        if prefix:
            require(prefix["model_sha256"]==model["sha256"] and prefix["frozen_before_treatment"],"foreign or unfrozen enrollment")
        enrollment[key]=[r["family"] for r in prefix["prefixes"]] if prefix else None
        for stage,manifest_path,source in (
            ("final_native",campaign_path.parent/key/"FINAL_NATIVE_ASSIGNMENTS.json",base/"final_native"),
            ("utility",base/"final/UTILITY_ASSIGNMENTS.json",base/"utility"),
            ("conditional",base/"final/CONDITIONAL_ASSIGNMENTS.json",base/"conditional")):
            if not manifest_path.exists():continue
            manifest=load(manifest_path)
            # Do not pretend a merely planned final-native roster was executed.
            if not source.exists():continue
            require(manifest["model_sha256"]==model["sha256"],"foreign closeout roster")
            sources[str(manifest_path)]=manifest["sha256"]
            frame_path=base/"OUTCOME_FRAME.json"
            frame=load(frame_path) if stage=="conditional" and frame_path.exists() else None
            binding=load(frame_path.with_name(frame_path.stem+"-PRIVATE-BINDING.json")) if frame else None
            if frame:
                require(frame["assignment_manifest_sha256"]==manifest["sha256"] and binding["frame_sha256"]==frame["sha256"] and
                        frame["reader_contract_sha256"]==reader_contract["sha256"],"outcome/arm binding differs")
                views={binding["assignment_uids"][r["uid"]]:r for r in frame["records"]}
                ratings_by_uid={}
                for path in (base/"outcome-ratings").glob("*.reader-*.json"):
                    rating=load(path)
                    require(rating["frame_sha256"]==frame["sha256"] and rating["role"]=="outcome" and
                            rating["reader_contract_sha256"]==reader_contract["sha256"] and rating["checkpoint"]==reader_contract["checkpoint"] and
                            rating["runtime_sha256"]==campaign["runtime_sha256"], "foreign outcome vote or deployment")
                    ratings_by_uid.setdefault(rating["uid"],[]).append(rating)
            else:
                views,ratings_by_uid={},{}
            for assignment in manifest["assignments"]:
                uid=assignment["uid"];path=source/(uid+".json")
                receipt=load(path) if path.exists() else None
                if receipt:
                    require_generation_binding(receipt,campaign,model)
                    require(receipt["assignment"]==assignment and not receipt["engineering_only"],"foreign science receipt")
                row={"uid":uid,"model":key,"family":assignment["family"],"seed":assignment["seed"],"arm":assignment["arm"],
                    "experiment":"conditional" if stage=="conditional" else "utility",
                    "assignment_sha256":digest(assignment),"execution_status":"COMPLETE" if receipt else "UNKNOWN",
                    "total_output_tokens":receipt["generated_tokens"]+receipt["controller_tokens"] if receipt else None,
                    "controller_method":"cheap","no_fire":not receipt["pulse_fired"] if receipt else None,
                    "closure":receipt["closed"] if receipt else None,"generated_tokens":receipt["generated_tokens"] if receipt else None,
                    "accuracy":None,"semantic":None,"source_generation_sha256":receipt["sha256"] if receipt else None,
                    "engagement":engagement(receipt) if receipt else {"status":"UNKNOWN_EXECUTION"}}
                if stage=="conditional":
                    view=views.get(uid)
                    measured=outcome(view,ratings_by_uid.get(view["uid"],[])) if view else {"success":None,"reason":"MISSING_EXECUTION_OR_READER_FRAME"}
                    row.update(semantic=measured["success"],semantic_status=measured["reason"])
                    row["candidate_kind"]=assignment["prefix"]["candidate"].get("kind","unspecified")
                else:
                    grade_path=base/(stage+"-grades")/(uid+".json")
                    grade=load(grade_path) if grade_path.exists() else None
                    if grade:
                        require(receipt and grade["generation_sha256"]==receipt["sha256"] and grade["manifest_sha256"]==manifest["sha256"],"foreign gold grade")
                        row.update(accuracy=grade["accuracy"],strict_accuracy=grade["strict"]["is_correct"],grade_status=grade["status"])
                        j1_path=base/(stage+"-grades")/"j1"/(grade["blind_uid"]+".json")
                        if grade["accuracy"] is None and j1_path.exists():
                            j1=load(j1_path)
                            require(j1["grade_sha256"]==grade["sha256"] and j1["qualified"] is True,"unqualified or foreign J1 adjudication")
                            qualified=load(Path(j1["qualification_path"]),"verification-j1-qualification-v3")
                            require(qualified["sha256"]==j1["qualification_sha256"] and qualified["status"]=="PASS" and
                                    qualified["reader_contract_sha256"]==reader_contract["sha256"] and qualified["runtime_sha256"]==campaign["runtime_sha256"] and
                                    j1["checkpoint"]==reader_contract["checkpoint"] and j1["prompt_version"]==1 and j1["parser_version"]==2,
                                    "J1 qualification/deployment differs")
                            row["accuracy"]={"EQUIVALENT":True,"NOT_EQUIVALENT":False}.get(j1["verdict"])
                            row["grade_status"]="J1_"+j1["verdict"]
                attempts=[load(p) for p in source.glob(uid+".attempt-*.json")]
                require(len(attempts)<=2,"retry budget exceeded")
                row["research_attempt_seconds"]=sum(a.get("elapsed_seconds",0) for a in attempts)
                row["attempts"]=len(attempts)
                records.append(row)
    from collections import Counter
    reader_diagnostics=Counter((r["model"],r["arm"],r.get("candidate_kind","unspecified"),r.get("semantic_status","unknown"))
                               for r in records if r["experiment"]=="conditional")
    return save(output,{"schema":"verification-analysis-input-v3","campaign_sha256":campaign["sha256"],
        "records":records,"conditional_enrollment":enrollment,"source_manifests":sources,
        "reader_diagnostics":[{"model":k[0],"arm":k[1],"candidate_kind":k[2],"semantic_status":k[3],"assigned":n}
                              for k,n in sorted(reader_diagnostics.items())],
        "all_assignment_accounting":True,"unqualified_models_are_not_null_effects":True})


def grade_j1(campaign_path,frame_path,qualification_path,output,deadline):
    from .readers import Reader
    campaign=load(campaign_path);check_files(campaign["code_files"])
    frame=load(frame_path);contract=load(campaign_path.parent/"READER.json")
    qualified=load(qualification_path,"verification-j1-qualification-v3")
    from .prepare import runtime_binding
    require(runtime_binding()["sha256"]==campaign["runtime_sha256"],"J1 execution runtime changed")
    require(qualified["status"]=="PASS" and qualified["reader_contract_sha256"]==contract["sha256"] and
            qualified["runtime_sha256"]==campaign["runtime_sha256"] and frame["reader_contract_sha256"]==contract["sha256"],
            "current J1 deployment is not qualified")
    reader=Reader(contract["checkpoint"]);output.mkdir(parents=True,exist_ok=True)
    try:
        for item in frame["records"]:
            path=output/(item["uid"]+".json")
            if path.exists():
                old=load(path);require(old["input_sha256"]==digest(item) and old["qualification_sha256"]==qualified["sha256"],"changed J1 input or qualification")
                continue
            require(time.time()<deadline-120,"J1 allocation deadline")
            verdict=reader.grade(item)
            save(path,{**verdict,"qualified":True,"qualification_sha256":qualified["sha256"],
                       "qualification_path":str(qualification_path.resolve())})
    finally:
        reader.close()
    return {"status":"COMPLETE"}


def j1_audit(campaign_path,output,deadline):
    """Requalify exact reader deployment on the existing sealed agent consensus."""
    import json
    from collections import Counter
    from .readers import Reader
    from .prepare import STAGING,runtime_binding
    campaign=load(campaign_path);check_files(campaign["code_files"])
    contract=load(campaign_path.parent/"READER.json")
    root=STAGING/"claude-analysis-2026-09-24/campaign-v3/j1"
    audit_path=root/"audit2_items.json";consensus_path=root/"audit2_consensus.json"
    from .contracts import file_sha
    require(file_sha(consensus_path)=="8605916f162e606fef8deef549dd143dc45653098fc5d33ee7e3016faf90e56c", "frozen J1 consensus changed")
    audit=json.loads(audit_path.read_text());consensus=json.loads(consensus_path.read_text())
    require(consensus["audit_sha256"]==audit["audit_sha256"] and consensus["frozen_before_j1_verdicts"] and len(consensus["labels"])==100,
            "J1 historical audit/consensus binding differs")
    items={item["item_id"]:item for item in audit["items"]}
    output.mkdir(parents=True,exist_ok=True)
    reader=Reader(contract["checkpoint"]);errors=[];strata=Counter();false_equivalents=[]
    try:
        for label in consensus["labels"]:
            item=items[label["item_id"]]
            visible={"uid":identity("J1-qualification",item["item_id"]),**{k:item[k] for k in ("problem","gold","candidate")}}
            path=output/(visible["uid"]+".json")
            if path.exists():
                verdict=load(path);require(verdict["input_sha256"]==digest(visible),"J1 audit input changed")
            else:
                require(time.time()<deadline-120,"J1 audit deadline")
                verdict=save(path,reader.grade(visible))
            if verdict["verdict"]!=label["verdict"]:
                errors.append(item["item_id"]);strata[item["stratum"]]+=1
                if item["stratum"]=="clear_wrong_control" and verdict["verdict"]=="EQUIVALENT":
                    false_equivalents.append(item["item_id"])
    finally:
        reader.close()
    matches=100-len(errors)
    return save(output/"J1_QUALIFICATION.json",{"schema":"verification-j1-qualification-v3",
        "status":"PASS" if matches>=95 and not false_equivalents and all(n<=2 for n in strata.values()) else "FAIL",
        "reader_contract_sha256":contract["sha256"],"runtime_sha256":runtime_binding()["sha256"],
        "matches":matches,"of":100,"per_stratum_errors":dict(strata),"false_equivalents":false_equivalents,
        "source_audit_sha256":file_sha(audit_path),"source_consensus_sha256":file_sha(consensus_path),
        "gate":"preserved95/100; zero false EQUIVALENT clear-wrong; <=2 errors per stratum",
        "truth_scope":"frozen Claude/Codex-DeepSeek agent consensus, not independent humans"})


def terminal_source(manifest_path,source,output):
    """Only seal failed source paths after their observed Slurm jobs are terminal."""
    from .campaign import command
    manifest=load(manifest_path);unknown=[];states={}
    terminal={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED"}
    for assignment in manifest["assignments"]:
        uid=assignment["uid"]
        if (source/(uid+".json")).exists():continue
        attempts=[load(p) for p in source.glob(uid+".attempt-*.json")]
        require(attempts,"unattempted source path remains pending")
        for attempt in attempts:
            job_id=attempt["job_id"]
            state=command(["sacct","-nP","-j",str(job_id),"--format=JobIDRaw,State,ExitCode"])
            top=[line.split("|")[1].split()[0].rstrip("+") for line in state.splitlines() if line.split("|")[0]==str(job_id)]
            require(len(top)==1 and top[0] in terminal,"source attempt job is not terminal")
            states[str(job_id)]=state
        unknown.append(uid)
    return save(output,{"schema":"verification-terminal-source-v3","manifest_sha256":manifest["sha256"],
        "unknown_assignments":unknown,"observed_scheduler_states":states,
        "meaning":"execution unknown; never reclassified as no eligible prefix"})
