"""Finite offline J1 execution and full-roster residual freezing.

Reader.grade, its three fixed votes and the original audit gates are unchanged.
This extension does not edit the qualified native scientific executor.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from moe_exp.verification_study.contracts import (check_files,digest,file_sha,identity,
    load,private_path,require,require_generation_binding,save,seal)
from moe_exp.verification_study.registry import REPO
from verification_study_v3_reader_r1 import selected_records

AUDIT_ROOT=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/j1")
AUDIT_SHA="06deeb4d7eeb5923a99c812aa8796f3dec97cccf6e5d93827813b2db90c60971"
CONSENSUS_SHA="8605916f162e606fef8deef549dd143dc45653098fc5d33ee7e3016faf90e56c"


def binding(campaign):
    names=("verification_study_v3_j1_r1.py","verification_study_v3_extension_r1.py",
           "verification_study_v3_extension_r1_gpu.sbatch","verification_study_v3_extension_r1_cpu.sbatch")
    paths=[REPO/"scripts/experimental_resume"/name for name in names]
    return seal({"schema":"verification-j1-executor-v3","campaign_sha256":campaign["sha256"],
        "scientific_executor_sha256":digest(campaign["code_files"]),"runtime_sha256":campaign["runtime_sha256"],
        "reader_contract_sha256":campaign["reader_sha256"],"code_files":{str(p):file_sha(p) for p in paths},
        "behavior":"unchanged r4 Reader.grade, original UID seeds, three votes, prompt1/parser2, cap8192"})


def deployment(campaign_path,frame_path):
    campaign=load(campaign_path);check_files(campaign["code_files"])
    contract=load(campaign_path.parent/"READER.json","verification-reader-contract-v3");check_files(contract["config_files"])
    frame=load(frame_path)
    require(contract["sha256"]==campaign["reader_sha256"] and frame["reader_contract_sha256"]==contract["sha256"] and
            frame.get("prompt_version")==1 and frame.get("parser_version")==2 and frame.get("votes")==3 and
            frame.get("output_cap")==8192,"J1 frame/deployment differs")
    require(len({r["uid"] for r in frame["records"]})==len(frame["records"]) and
            all(re.fullmatch(r"[a-f0-9]{64}",r["uid"]) for r in frame["records"]),"unsafe or duplicate J1 identities")
    if frame["schema"]=="verification-j1-audit-frame-v3":
        expected=audit_body(campaign)
        require(all(frame.get(k)==v for k,v in expected.items()),"J1 audit frame changed original items, labels or gate inputs")
    else:
        require(frame["schema"]=="verification-j1-frame-v3" and frame.get("full_roster_frozen") is True,
                "production J1 requires a full-roster frozen residual frame")
    return campaign,contract,frame,binding(campaign)


def audit_frame(campaign_path,output):
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu();campaign=load(campaign_path);check_files(campaign["code_files"])
    return save(output,audit_body(campaign))


def audit_body(campaign):
    audit_path=AUDIT_ROOT/"audit2_items.json";truth_path=AUDIT_ROOT/"audit2_consensus.json"
    require(file_sha(audit_path)==AUDIT_SHA and file_sha(truth_path)==CONSENSUS_SHA,"frozen J1 audit/consensus changed")
    audit=json.loads(audit_path.read_text());truth=json.loads(truth_path.read_text())
    require(truth["audit_sha256"]==audit["audit_sha256"] and truth["frozen_before_j1_verdicts"] is True and
            len(truth["labels"])==100 and len(audit["items"])==100,"J1 audit freeze/size differs")
    items={item["item_id"]:item for item in audit["items"]}
    require(len(items)==100 and len({r["item_id"] for r in truth["labels"]})==100,"duplicate J1 audit identity")
    records=[];labels={}
    for label in truth["labels"]:
        item=items[label["item_id"]];uid=identity("J1-qualification",item["item_id"])
        records.append({"uid":uid,**{k:item[k] for k in ("problem","gold","candidate")}})
        labels[uid]={"item_id":item["item_id"],"verdict":label["verdict"],"stratum":item["stratum"]}
    return {"schema":"verification-j1-audit-frame-v3","records":records,"labels":labels,
        "reader_contract_sha256":campaign["reader_sha256"],"prompt_version":1,"parser_version":2,"votes":3,"output_cap":8192,
        "source_audit_sha256":AUDIT_SHA,"source_consensus_sha256":CONSENSUS_SHA,
        "population":"frozen_J1_audit","scope":"one shared pinned grader deployment for all seven generators",
        "truth_scope":"frozen agent consensus, not independent human truth"}


def freeze_residuals(campaign_path,key,manifest_path,source,grades,output):
    from moe_exp.verification_study.closeout import require_cpu,grade_strict,terminal_source
    require_cpu();campaign=load(campaign_path);check_files(campaign["code_files"])
    model=load(campaign["models"][key]);manifest=load(manifest_path)
    require(manifest["stage"] in {"final_native","utility"} and manifest["model_sha256"]==model["sha256"],"foreign J1 residual roster")
    missing=[r["uid"] for r in manifest["assignments"] if not (source/(r["uid"]+".json")).exists()]
    proof_path=source/("TERMINAL-GRADING-"+os.environ["SLURM_JOB_ID"]+".json") if missing else None
    proof=terminal_source(manifest_path,source,proof_path) if missing else None
    generations={};grade_receipts={}
    for assignment in manifest["assignments"]:
        generation_path=source/(assignment["uid"]+".json");grade_path=grades/(assignment["uid"]+".json")
        if not generation_path.exists():
            require(assignment["uid"] in proof["unknown_assignments"],"pending generation cannot become frozen grading missingness")
            continue
        generation=load(generation_path);require_generation_binding(generation,campaign,model)
        require(generation["assignment"]==assignment and grade_path.exists(),"strict grading is still incomplete")
        grade=load(grade_path,"verification-grade-v3")
        require(grade["generation_sha256"]==generation["sha256"] and grade["manifest_sha256"]==manifest["sha256"] and
                grade["assignment_uid"]==assignment["uid"] and grade["scorer_contract"]==3,"foreign strict grade before J1 freeze")
        generations[assignment["uid"]]=generation["sha256"];grade_receipts[assignment["uid"]]=grade["sha256"]
    # Rebuild from the whole roster, never the last shard's residual view.
    frame=grade_strict(campaign_path,key,manifest_path,source,grades,begin=0,count=None)
    return save(output,{**frame,"full_roster_frozen":True,"full_roster_assignments":len(manifest["assignments"]),
        "source_generations":generations,"source_grades":grade_receipts,"unknown_generations":missing,
        "terminal_source_proof_path":str(proof_path) if proof_path else None,
        "terminal_source_proof_sha256":proof["sha256"] if proof else None})


def validate_runtime(value,campaign,contract,frame,executor):
    require(value.get("schema")=="verification-j1-runtime-v3" and value.get("campaign_sha256")==campaign["sha256"] and
            value.get("reader_contract_sha256")==contract["sha256"] and value.get("runtime_sha256")==campaign["runtime_sha256"] and
            value.get("checkpoint")==contract["checkpoint"] and value.get("frame_sha256")==frame["sha256"] and
            value.get("j1_executor")==executor and re.fullmatch(r"\d+",str(value.get("job_id"))) and
            type(value.get("load_seconds")) in (float,int) and math.isfinite(value["load_seconds"]) and value["load_seconds"]>=0,
            "foreign J1 invocation deployment or measured load")
    rows=selected_records(frame,*value["slice"])
    require(value["selected_uids"]==[r["uid"] for r in rows],"J1 runtime slice changed")


def validate_verdict(path,item,campaign,contract,frame,executor,qualified=None):
    value=load(path,"verification-j1-verdict-v3")
    require(path.name==item["uid"]+".json" and value["uid"]==item["uid"] and value["input_sha256"]==digest(item) and
            value["grade_sha256"]==item.get("grade_sha256") and value["frame_sha256"]==frame["sha256"] and
            value["checkpoint"]==contract["checkpoint"] and value["prompt_version"]==1 and value["parser_version"]==2 and
            value.get("j1_executor_sha256")==executor["sha256"] and value.get("runtime_sha256")==campaign["runtime_sha256"],
            "foreign J1 UID/input/frame/grade/deployment/executor")
    require(len(value["draws"])==3 and len(value["votes"])==3,"J1 must retain three draws")
    seed=int(digest(["verification-J1",item["uid"]])[:8],16)%2000000000
    from moe_exp.correlation_pipeline.answer_equivalence import parse_verdict
    decisions=[]
    for index,draw in enumerate(value["draws"]):
        require(type(draw["vote"]) is int and draw["vote"]==index and type(draw["seed"]) is int and draw["seed"]==seed+index and
                type(draw["seconds"]) in (int,float) and math.isfinite(draw["seconds"]) and draw["seconds"]>=0 and
                type(draw["input_tokens"]) is int and 0<draw["input_tokens"]<=49152-8192 and
                type(draw["output_tokens"]) is int and 0<=draw["output_tokens"]<=8192,"foreign J1 fixed vote or measured cost")
        decisions.append(parse_verdict(draw["text"].rsplit("</think>",1)[-1]) if draw["finish_reason"]=="stop" else "TRUNCATED")
    top,n=Counter(decisions).most_common(1)[0]
    expected=top if n>=2 and top in {"EQUIVALENT","NOT_EQUIVALENT"} else "UNCERTAIN"
    require(value["votes"]==decisions and value["verdict"]==expected,"J1 three-vote rule differs")
    runtime_path=private_path(value["invocation_runtime_path"])
    require(runtime_path.is_relative_to(path.parent/"invocations") and runtime_path.name=="RUNTIME.json","foreign J1 runtime path")
    runtime=load(runtime_path);validate_runtime(runtime,campaign,contract,frame,executor)
    require(runtime["sha256"]==value["invocation_runtime_sha256"] and runtime["job_id"]==value["job_id"] and
            item["uid"] in runtime["selected_uids"],"J1 invocation identity changed")
    if qualified is not None:
        require(value.get("qualified") is True and value.get("qualification_sha256")==qualified["sha256"] and
                load(value["qualification_path"])["sha256"]==qualified["sha256"],"changed J1 deployment qualification")
    else:require(value.get("qualified") is False,"an audit verdict cannot self-qualify")
    return value


def qualification(campaign_path,frame_path,source,output):
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu();campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    require(frame["schema"]=="verification-j1-audit-frame-v3" and len(frame["records"])==100 and
            frame["source_audit_sha256"]==AUDIT_SHA and frame["source_consensus_sha256"]==CONSENSUS_SHA,
            "J1 qualification needs the original full100 audit")
    require(set(frame["labels"])=={r["uid"] for r in frame["records"]},"J1 audit labels differ")
    missing=[];errors=[];strata=Counter();false_equivalents=[];receipts={}
    for item in frame["records"]:
        path=source/(item["uid"]+".json")
        if not path.exists():missing.append(item["uid"]);continue
        verdict=validate_verdict(path,item,campaign,contract,frame,executor)
        receipts[item["uid"]]=verdict["sha256"];label=frame["labels"][item["uid"]]
        if verdict["verdict"]!=label["verdict"]:
            errors.append(label["item_id"]);strata[label["stratum"]]+=1
            if label["stratum"]=="clear_wrong_control" and verdict["verdict"]=="EQUIVALENT":false_equivalents.append(label["item_id"])
    matches=100-len(errors)-len(missing)
    status="INCOMPLETE" if missing else "PASS" if matches>=95 and not false_equivalents and all(n<=2 for n in strata.values()) else "FAIL"
    require(not missing,"J1 audit incomplete; no deployment qualification can be sealed")
    return save(output,{"schema":"verification-j1-qualification-v3","status":status,
        "reader_contract_sha256":contract["sha256"],"runtime_sha256":campaign["runtime_sha256"],"matches":matches,"of":100,
        "per_stratum_errors":dict(strata),"false_equivalents":false_equivalents,"errors":errors,
        "source_audit_sha256":AUDIT_SHA,"source_consensus_sha256":CONSENSUS_SHA,"frame_sha256":frame["sha256"],
        "j1_executor":executor,"audit_receipts":receipts,"gate":"preserved95/100; zero false EQUIVALENT clear-wrong; <=2 errors per stratum",
        "truth_scope":"frozen Claude/Codex-DeepSeek agent consensus, not independent humans"})


def run_slice(campaign_path,frame_path,output,deadline,begin,count,qualification_path=None,*,reader_factory=None,check_runtime=True):
    campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    if check_runtime:
        from moe_exp.verification_study.backend import require_compute
        from moe_exp.verification_study.prepare import runtime_binding
        require_compute();require(runtime_binding()["sha256"]==campaign["runtime_sha256"],"J1 installed runtime changed")
    qualified=load(qualification_path,"verification-j1-qualification-v3") if qualification_path else None
    if qualified is not None:
        require(qualified["status"]=="PASS" and qualified["reader_contract_sha256"]==contract["sha256"] and
                qualified["runtime_sha256"]==campaign["runtime_sha256"] and qualified.get("j1_executor")==executor,
                "current J1 deployment is not qualified")
        require(frame["schema"]=="verification-j1-frame-v3","production J1 must read the frozen residual frame")
    else:require(frame["schema"]=="verification-j1-audit-frame-v3","unqualified J1 may read only its frozen100 audit")
    rows=selected_records(frame,begin,count);output=private_path(output);output.mkdir(parents=True,exist_ok=True)
    by_uid={r["uid"]:r for r in frame["records"]}
    for path in output.glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{64}",path.stem):continue
        require(path.stem in by_uid,"foreign J1 output identity")
        validate_verdict(path,by_uid[path.stem],campaign,contract,frame,executor,qualified)
    job=os.environ["SLURM_JOB_ID"];invocation=job+"-"+str(time.time_ns());slot=output/"invocations"/invocation
    identity_data={"campaign_sha256":campaign["sha256"],"frame_sha256":frame["sha256"],"job_id":job,
        "slice":[begin,count],"selected_uids":[r["uid"] for r in rows],"j1_executor":executor,
        "qualification_sha256":qualified["sha256"] if qualified else None}
    reader=None;error=None;completed=0
    with ExitStack() as stack:
        for item in sorted(rows,key=lambda r:r["uid"]):
            path=output/"locks"/(item["uid"]+".lock");path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open("a"));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            existing=output/(item["uid"]+".json")
            if existing.exists():validate_verdict(existing,item,campaign,contract,frame,executor,qualified)
        missing=[r for r in rows if not (output/(r["uid"]+".json")).exists()]
        for item in missing:require(len(list(output.glob(item["uid"]+".attempt-*.json")))<2,"J1 item retry budget exhausted")
        save(slot/"INTENT.json",{"schema":"verification-j1-invocation-intent-v3",**identity_data,"started_epoch":time.time()})
        try:
            if missing:
                require(time.time()<deadline-120,"J1 deadline before load")
                if reader_factory is None:
                    from moe_exp.verification_study.readers import Reader
                    reader_factory=Reader
                reader=reader_factory(contract["checkpoint"])
                runtime=save(slot/"RUNTIME.json",{"schema":"verification-j1-runtime-v3",**identity_data,
                    "reader_contract_sha256":contract["sha256"],"runtime_sha256":campaign["runtime_sha256"],
                    "checkpoint":contract["checkpoint"],"load_seconds":reader.load_seconds})
                for item in missing:
                    require(time.time()<deadline-120,"J1 allocation deadline")
                    attempt_path=output/(item["uid"]+".attempt-"+invocation+".json")
                    attempt={"schema":"verification-j1-attempt-v3","uid":item["uid"],"input_sha256":digest(item),
                        "frame_sha256":frame["sha256"],"job_id":job,"j1_executor_sha256":executor["sha256"],"started_epoch":time.time()}
                    save(attempt_path,{**attempt,"status":"STARTED"});started=time.monotonic()
                    try:
                        verdict=reader.grade(item)
                        receipt=save(output/(item["uid"]+".json"),{**verdict,"frame_sha256":frame["sha256"],
                            "runtime_sha256":campaign["runtime_sha256"],"j1_executor_sha256":executor["sha256"],"job_id":job,
                            "invocation_runtime_path":str(slot/"RUNTIME.json"),"invocation_runtime_sha256":runtime["sha256"],
                            "qualified":qualified is not None,"qualification_sha256":qualified["sha256"] if qualified else None,
                            "qualification_path":str(qualification_path.resolve()) if qualified else None})
                        validate_verdict(output/(item["uid"]+".json"),item,campaign,contract,frame,executor,qualified)
                        save(attempt_path,{**attempt,"status":"COMPLETE","receipt_sha256":receipt["sha256"],
                            "elapsed_seconds":time.monotonic()-started},replace=True);completed+=1
                    except Exception as exc:
                        save(attempt_path,{**attempt,"status":"FAIL","elapsed_seconds":time.monotonic()-started,
                            "error":{"type":type(exc).__name__,"message":str(exc)[:2000]}},replace=True)
                        raise
        except Exception as exc:error={"type":type(exc).__name__,"message":str(exc)[:2000]}
        finally:
            if reader is not None:
                try:reader.close()
                except Exception as exc:error=error or {"type":type(exc).__name__,"message":str(exc)[:2000]}
        complete=all((output/(r["uid"]+".json")).exists() for r in rows)
        result=save(slot/"RESULT.json",{"schema":"verification-j1-slice-v3",**identity_data,
            "status":"FAIL" if error else "COMPLETE" if complete else "INCOMPLETE","new_verdicts":completed,
            "loaded":reader is not None,"error":error,"ended_epoch":time.time()})
    require(result["status"]=="COMPLETE","J1 slice failed: "+str(error))
    return result


def probe(campaign_path,frame_path,begin=0,count=None):
    from transformers import AutoTokenizer
    from moe_exp.verification_study.readers import normalize_prompt_ids
    from moe_exp.correlation_pipeline.answer_equivalence import SYSTEM
    campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    rows=selected_records(frame,begin,len(frame["records"])-begin if count is None else count)
    tokenizer=AutoTokenizer.from_pretrained(contract["checkpoint"],local_files_only=True);sizes=[]
    for item in rows:
        content=f"Problem:\n{item['problem']}\n\nReference answer:\n{item['gold']}\n\nCandidate:\n{item['candidate']}"
        sizes.append(len(normalize_prompt_ids(tokenizer.apply_chat_template(
            [{"role":"system","content":SYSTEM},{"role":"user","content":content}],tokenize=True,
            add_generation_prompt=True,enable_thinking=True,return_dict=False))))
    require(all(n+8192<=49152 for n in sizes),"J1 context exceeds unchanged deployment; truncation forbidden")
    return seal({"schema":"verification-j1-input-shape-v3","frame_sha256":frame["sha256"],"j1_executor_sha256":executor["sha256"],
        "begin":begin,"count":len(rows),"uids":[r["uid"] for r in rows],"input_tokens":sizes,
        "max_input_tokens":max(sizes),"output_cap_each_vote":8192,"votes":3,"largest_record_index":begin+sizes.index(max(sizes))})


def collect_prices(campaign_path,key,root,output):
    """J1 has separate load and token prices; strict grades cannot price it."""
    from verification_study_v3_reader_r1 import collect_reader_prices
    base=collect_reader_prices(campaign_path,key,root,output.with_name(output.stem+"-reader.json"))
    campaign=load(campaign_path);model=load(campaign["models"][key]);executor=binding(campaign)
    components={name:[] for name in ("j1_load","j1_generation","j1_item")}
    pairs=[(root/"J1/AUDIT_FRAME.json",root/"J1/audit",None)]
    for stage in ("final_native","utility"):
        grades=root/key/(stage+"-grades")
        pairs.append((grades/"J1_FROZEN_FRAME.json",grades/"j1",root/"J1/J1_QUALIFICATION.json"))
    for frame_path,directory,qualification_path in pairs:
        if not directory.exists():continue
        files=[p for p in directory.glob("*.json") if re.fullmatch(r"[a-f0-9]{64}",p.stem)]
        runtimes=list(directory.glob("invocations/*/RUNTIME.json"))
        if not files and not runtimes:continue
        parent,contract,frame,bound=deployment(campaign_path,frame_path)
        qualified=load(qualification_path) if qualification_path else None
        if qualified is not None:
            require(qualified["status"]=="PASS" and qualified["j1_executor"]==bound and
                    qualified["runtime_sha256"]==campaign["runtime_sha256"] and qualified["reader_contract_sha256"]==contract["sha256"],
                    "J1 price belongs to an unqualified deployment")
        rows={r["uid"]:r for r in frame["records"]}
        def sample(role,path,value,seconds,units,input_tokens=0,total_tokens=0):
            if units:
                components[role].append({"role":role,"seconds":seconds,"units":units,"model_sha256":model["sha256"],
                    "execution_sha256":campaign["runtime_sha256"],"receipt_path":str(path),"receipt_sha256":value["sha256"],
                    "j1_executor_sha256":executor["sha256"],"context_input_tokens":input_tokens,"total_context_tokens":total_tokens,
                    "price_scope":"shared pinned J1 grader; identical hardware/runtime; model key identifies the generator paper cell"})
        for path in runtimes:
            runtime=load(path);validate_runtime(runtime,parent,contract,frame,bound)
            sample("j1_load",path,runtime,runtime["load_seconds"],1)
        for path in files:
            require(path.stem in rows,"foreign J1 price identity")
            value=validate_verdict(path,rows[path.stem],parent,contract,frame,bound,qualified)
            seconds=sum(d["seconds"] for d in value["draws"]);tokens=sum(d["output_tokens"] for d in value["draws"])
            inputs=max(d["input_tokens"] for d in value["draws"])
            total=max(d["input_tokens"]+d["output_tokens"] for d in value["draws"])
            sample("j1_generation",path,value,seconds,tokens,inputs,total)
            sample("j1_item",path,value,seconds,1,inputs,total)
    extended={name:{"status":"MEASURED" if rows else "UNMEASURED",
        "seconds_per_unit_max":max((r["seconds"]/r["units"] for r in rows),default=None),
        "max_context_input_tokens":max((r["context_input_tokens"] for r in rows),default=0),
        "max_total_context_tokens":max((r["total_context_tokens"] for r in rows),default=0),"samples":rows}
        for name,rows in components.items()}
    return save(output,{**base,"components":{**base["components"],**extended},"j1_executor":executor,
        "missing":base["missing"]+[name for name,rows in components.items() if not rows],
        "j1_price_contract":"one cold load; prefill-inclusive seconds per emitted token; plan3*8192 tokens/item; preserve item costs separately"})
