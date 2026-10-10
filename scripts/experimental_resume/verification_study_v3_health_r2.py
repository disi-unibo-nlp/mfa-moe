"""Finite afterany health signals; the persistent Goal is the sole dispatcher.

This separately sealed observer never edits the scientific executor, relaxes a
gate, submits scientific work, retries an assignment, or runs a daemon.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study.campaign import command, preflight
from moe_exp.verification_study.contracts import (check_files, digest, file_sha, load,
    private_path, require, require_generation_binding, save, seal)
from moe_exp.verification_study.registry import REPO

SCRIPT=Path(__file__).resolve()
SHELL=SCRIPT.with_name("verification_study_v3_health_r2.sbatch")
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


def arguments(card):
    raw=card.get("payload_arguments",[])
    result={}
    for index,item in enumerate(raw):
        if item.startswith("--"):
            require(index+1<len(raw) and not raw[index+1].startswith("--"),"invalid observed task argument")
            require(item not in result,"duplicate observed argument")
            result[item]=raw[index+1]
    return result


def expected_work(card,root):
    args=arguments(card);stage=card["stage"];key=card["model"]
    output=Path(args["--output"]) if "--output" in args else root/key/"qualification" if stage=="qualify" else None
    required=[];inputs={};uids=[]
    for flag in ("--manifest","--frame","--qualification","--recognizer","--policy","--validation","--prices"):
        if flag in args:
            path=Path(args[flag]);value=load(path)
            inputs[str(path)]=value["sha256"]
    if stage=="run":
        manifest=load(args["--manifest"],"verification-assignments-v3")
        begin=int(args.get("--begin",0));count=int(args["--count"]) if "--count" in args else None
        rows=manifest["assignments"][begin:begin+count if count is not None else None]
        require(rows,"empty health slice")
        uids=[row["uid"] for row in rows]
        required=[str(output/(uid+".json")) for uid in uids]
    elif stage=="rate":
        frame=load(args["--frame"])
        uids=[row["uid"] for row in frame["records"]]
        required=[str(output/(uid+f".reader-{draw}.json")) for uid in uids for draw in (0,1)]
        required.append(str(output/"RUNTIME.json"))
    elif stage=="calibrate":required=[str(output/"POLICY.json"),str(output/"ENROLLMENT.json")]
    elif stage=="freeze-final":required=[str(output/name) for name in ("PREFIXES.json","CONDITIONAL_ASSIGNMENTS.json","UTILITY_ASSIGNMENTS.json")]
    elif stage=="qualify":required=[str(output/"QUALIFICATION.json")]
    elif stage=="evidence":required=[str(REPO/"paper/6a510c6db0d9b08aa438c142/generated/PROVENANCE.json")]
    elif stage=="grade-strict":required=[str(output/"J1_FRAME.json")]
    elif stage=="j1-audit":required=[str(output/"J1_QUALIFICATION.json")]
    elif stage=="grade-j1":
        frame=load(args["--frame"]);uids=[row["uid"] for row in frame["records"]]
        required=[str(output/(uid+".json")) for uid in uids]
    else:
        require(output is not None,"health stage lacks canonical output")
        required=[str(output)]
    if stage=="grade-strict":
        manifest=load(args["--manifest"])
        begin=int(args.get("--begin",0));count=int(args["--count"]) if "--count" in args else None
        selected=manifest["assignments"][begin:begin+count if count is not None else None]
        uids=[r["uid"] for r in selected if (Path(args["--source"])/(r["uid"]+".json")).exists()]
        required.extend(str(output/(uid+".json")) for uid in uids)
    return {"stage":stage,"model":key,"output":str(output) if output else None,
            "inputs":inputs,"expected_uids":uids,"expected_uids_sha256":digest(uids),"required_artifacts":required}


def register(campaign_path,submission_path,root,test_only=False):
    campaign=load(campaign_path);check_files(campaign["code_files"])
    predecessor=load(submission_path,"verification-submission-v3")
    card_path=submission_path.parent/"JOB_CARD.json";card=load(card_path,"verification-job-card-v3")
    require(predecessor["job_card_sha256"]==card["sha256"] and card["campaign_sha256"]==campaign["sha256"],"foreign health predecessor")
    work=expected_work(card,root)
    health=seal({"schema":"verification-health-card-v3","campaign_path":str(campaign_path),
        "campaign_sha256":campaign["sha256"],"predecessor_job_id":predecessor["job_id"],
        "predecessor_submission_path":str(submission_path),"predecessor_submission_sha256":predecessor["sha256"],
        "job_card_path":str(card_path),"job_card_sha256":card["sha256"],"result_root":str(root),
        "observer_files":{str(p):file_sha(p) for p in (SCRIPT,SHELL)},"expected_work":work,
        "claim":card["claim"],"artifact":"sealed completion/failure event and actual component prices for the predecessor's paper cell",
        "failure_interpretation":"execution failure, scientific gate failure and observer/accounting failure remain distinct; no causal-null inference",
        "resources":{"partition":"lrd_all_serial","qos":"normal","gpus":0,"cpus":1,"memory_gb":4,"walltime_seconds":600,"billing_core_hours_max":0},
        "authorization":"user: add health signals on completion/failure and continue autonomously until the approved campaign ends",
        "scientific_dispatch":"persistent Goal only; callback emits facts and prices, with zero recovery allocations"})
    slot=private_path(root/"health/submissions"/health["sha256"]);slot.mkdir(parents=True,exist_ok=True)
    path=slot/"HEALTH_CARD.json";save(path,health)
    receipt_path=slot/"SUBMISSION.json"
    if receipt_path.exists() and not test_only:return load(receipt_path)
    if (slot/"INTENT.json").exists() and not test_only:
        query=command(["squeue","-h","-u","lmolfett","-o","%i|%k"])
        matches=[line.split("|",1)[0] for line in query.splitlines() if line.endswith("|cv3h:"+health["sha256"])]
        require(len(matches)==1,"health intent unresolved; reconcile accounting/comment without duplicate submission")
        return save(receipt_path,{"schema":"verification-health-submission-v3","job_id":matches[0],"health_card_sha256":health["sha256"],"reconciled":True})
    logs=private_path(root/"health/logs");logs.mkdir(parents=True,exist_ok=True)
    args=["sbatch","--test-only" if test_only else "--parsable","--account=iscrc_miosr","--partition=lrd_all_serial",
        "--qos=normal","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=4G","--time=00:10:00","--no-requeue",
        "--dependency=afterany:"+predecessor["job_id"],"--job-name=cv3-health-"+card["model"],
        "--comment=cv3h:"+health["sha256"],"--chdir="+str(REPO),
        "--output="+str(logs/"%x-%j.out"),"--error="+str(logs/"%x-%j.err"),str(SHELL),str(path)]
    if test_only:
        result=subprocess.run(args,check=True,capture_output=True,text=True)
        return {"test_only":True,"scheduler_output":result.stdout+result.stderr,"health_card":str(path)}
    snapshot=save(slot/"PREFLIGHT.json",preflight())
    save(slot/"INTENT.json",{"schema":"verification-health-intent-v3","health_card_sha256":health["sha256"],
        "preflight_sha256":snapshot["sha256"],"epoch":time.time()})
    response=command(args);job=response.split(";",1)[0];require(re.fullmatch(r"\d+",job),"invalid health sbatch response")
    observed=command(["scontrol","show","job",job])
    require("JobId="+job in observed and "Account=iscrc_miosr" in observed,"health job read-back failed")
    return save(receipt_path,{"schema":"verification-health-submission-v3","health_card_sha256":health["sha256"],
        "job_id":job,"predecessor_job_id":predecessor["job_id"],"read_back":observed,"preflight_sha256":snapshot["sha256"]})


def terminal_accounting(job):
    observed=[]
    for delay in (0,1,2,4,8,16):
        if delay:time.sleep(delay)
        try:
            raw=subprocess.run(["sacct","-nP","-j",job,"--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"],
                check=True,capture_output=True,text=True,timeout=10).stdout.strip()
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            observed.append(type(exc).__name__);continue
        rows=[line.split("|") for line in raw.splitlines() if line.split("|",1)[0]==job]
        observed.append(raw)
        if len(rows)==1 and rows[0][1].split()[0] in TERMINAL:
            row=rows[0]
            return {"state":row[1],"exit_code":row[2],"elapsed_seconds":int(row[3]),"allocated_tres":row[4],"raw":raw}
    raise ValueError("predecessor terminal accounting unavailable after bounded read-only retries: "+str(observed)[-1500:])


def inspect_artifacts(health,campaign):
    work=health["expected_work"];key=work["model"];stage=work["stage"]
    for path,sha in work["inputs"].items():require(load(path)["sha256"]==sha,"health input changed: "+path)
    model=load(campaign["models"][key]) if key in campaign["models"] else None
    committed=[];missing=[];gates=[]
    card=load(health["job_card_path"]);args=arguments(card)
    frame=load(args["--frame"]) if "--frame" in args else None
    reader=load(Path(health["campaign_path"]).parent/"READER.json") if stage=="rate" else None
    if reader:
        require(reader["sha256"]==campaign["reader_sha256"] and frame["reader_contract_sha256"]==reader["sha256"],
                "reader deployment/frame differs from campaign")
        check_files(reader["config_files"])
    for path in work["required_artifacts"]:
        if not Path(path).exists():missing.append(path);continue
        result=load(path)
        if stage=="run":
            require_generation_binding(result,campaign,model)
            manifest=load(args["--manifest"])
            roster={r["uid"]:r for r in manifest["assignments"]}
            uid=Path(path).stem
            require(uid in work["expected_uids"] and result["assignment"]==roster[uid] and
                    result["assignment_uid"]==uid and result["engineering_only"] is False and result["execution_complete"] is True,
                    "foreign or incomplete scientific health receipt")
            archive=private_path(result["capture"]["array_path"])
            require(archive==Path(path).resolve().with_suffix(".npz"),"foreign archive path")
            if not archive.exists():missing.append(str(archive));continue
            require(file_sha(archive)==result["capture"]["array_sha256"] and archive.stat().st_size==result["capture"]["array_bytes"],
                    "corrupt scientific capture archive")
        if stage=="rate":
            require(result["frame_sha256"]==frame["sha256"] and result["runtime_sha256"]==campaign["runtime_sha256"],
                    "foreign reader frame/runtime in health receipt")
            if Path(path).name=="RUNTIME.json":
                require(result["reader_sha256"]==reader["sha256"],"foreign reader runtime contract")
            else:
                rows={r["uid"]:r for r in frame["records"]};uid=result["uid"]
                require(type(result["reader"]) is int and Path(path).name==uid+f".reader-{result['reader']}.json",
                        "reader filename UID/draw mismatch")
                require(uid in rows and result["input_sha256"]==digest(rows[uid]) and
                        result["reader_contract_sha256"]==reader["sha256"] and result["checkpoint"]==reader["checkpoint"] and
                        result["reader"] in (0,1) and result["role"]==args.get("--role","eligibility"),
                        "foreign reader input/role/deployment in health receipt")
        if stage=="grade-strict" and Path(path).name!="J1_FRAME.json":
            uid=Path(path).stem;generation=load(Path(args["--source"])/(uid+".json"))
            require_generation_binding(generation,campaign,model)
            require(result["assignment_uid"]==uid and result["generation_sha256"]==generation["sha256"] and
                    result["manifest_sha256"]==load(args["--manifest"])["sha256"],"foreign grade in health receipt")
        if stage in {"qualify","recognition","calibrate","validate","j1-audit"} and result.get("status") in {"FAIL","INSUFFICIENT_SUPPORT","NO_ADMISSIBLE_POOL"}:
            gates.append({"path":path,"status":result["status"]})
        if stage=="nominate" and result.get("status")!="NOMINATED":gates.append({"path":path,"status":result.get("status")})
        if stage=="qualify":require(result["campaign_sha256"]==campaign["sha256"] and result["code_files"]==campaign["code_files"],"foreign backend qualification")
        committed.append({"path":path,"sha256":result["sha256"],"schema":result.get("schema"),"status":result.get("status")})
    return {"committed_artifacts":committed,"missing_artifacts":missing,"scientific_gate_failures":gates,
            "expected_uids_sha256":work["expected_uids_sha256"]}


def validate_reader_prices(campaign_path,campaign,root,key):
    reader=load(campaign_path.parent/"READER.json")
    require(reader["sha256"]==campaign["reader_sha256"],"reader price contract differs from campaign")
    check_files(reader["config_files"])
    for directory,name,role in (("eligibility-ratings","AUDIT_FRAME.json","eligibility"),("outcome-ratings","OUTCOME_FRAME.json","outcome")):
        base=root/key/directory
        if not base.exists():continue
        runtime_path=base/"RUNTIME.json"
        if not runtime_path.exists() and not list(base.glob("*.reader-*.json")):continue
        frame=load(root/key/name);rows={r["uid"]:r for r in frame["records"]}
        require(frame["reader_contract_sha256"]==reader["sha256"],"reader price frame deployment differs")
        runtime=load(runtime_path)
        require(runtime["reader_sha256"]==reader["sha256"] and runtime["runtime_sha256"]==campaign["runtime_sha256"] and
                runtime["frame_sha256"]==frame["sha256"],"foreign reader price runtime")
        for path in base.glob("*.reader-*.json"):
            result=load(path);uid=result["uid"]
            require(type(result["reader"]) is int and path.name==uid+f".reader-{result['reader']}.json",
                    "reader price filename UID/draw mismatch")
            require(uid in rows and result["frame_sha256"]==frame["sha256"] and result["input_sha256"]==digest(rows[uid]) and
                    result["checkpoint"]==reader["checkpoint"] and result["reader_contract_sha256"]==reader["sha256"] and
                    result["runtime_sha256"]==campaign["runtime_sha256"] and result["role"]==role and result["reader"] in (0,1),
                    "foreign reader price input/role/deployment")


def update_latest(path,event):
    path=private_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        rank=lambda r:(int(r.get("predecessor_job_id",0)),r["observed_epoch"])
        if path.exists() and rank(load(path))>rank(event):return
        save(path,event,replace=True)


def check(health_path):
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu();require(getpass.getuser()=="lmolfett","wrong callback user")
    health=load(health_path,"verification-health-card-v3")
    root=Path(health["result_root"]);event_path=root/"health/events"/(health["sha256"]+".json")
    if event_path.exists():return load(event_path)
    data={"schema":"verification-health-event-v3","health_card_sha256":health["sha256"],
        "campaign_sha256":health["campaign_sha256"],"job_card_sha256":health["job_card_sha256"],
        "predecessor_job_id":health["predecessor_job_id"],"callback_job_id":__import__("os").environ["SLURM_JOB_ID"],
        "model":health["expected_work"]["model"],"stage":health["expected_work"]["stage"],"observed_epoch":time.time(),
        "scientific_effect":"NOT_INFERRED_FROM_COMPLETION","automatic_scientific_submissions":0}
    phase="SOURCE_BINDING"
    try:
        check_files(health["observer_files"])
        campaign=load(health["campaign_path"]);check_files(campaign["code_files"])
        require(campaign["sha256"]==health["campaign_sha256"],"health campaign changed")
        require(load(health["predecessor_submission_path"])["sha256"]==health["predecessor_submission_sha256"] and
                load(health["job_card_path"])["sha256"]==health["job_card_sha256"],"health predecessor changed")
        phase="ACCOUNTING"
        accounting=terminal_accounting(health["predecessor_job_id"]);data["accounting"]=accounting
        phase="ARTIFACTS"
        inspection=inspect_artifacts(health,campaign);data.update(inspection)
        data["execution_artifacts_complete"]=not inspection["missing_artifacts"]
        data["scientific_gate_status"]="FAILED" if inspection["scientific_gate_failures"] else "NO_FAILED_GATE_OBSERVED"
        if accounting["state"]!="COMPLETED" or accounting["exit_code"]!="0:0":data["state"]="EXECUTION_FAILED"
        elif inspection["missing_artifacts"]:data["state"]="INCOMPLETE_ARTIFACTS"
        elif inspection["scientific_gate_failures"]:data["state"]="SCIENTIFIC_GATE_FAILED"
        else:data["state"]="STEP_COMPLETE"
        if data["stage"] in {"run","rate","grade-strict"} and inspection["committed_artifacts"] and data["model"] in campaign["models"]:
            phase="PRICING"
            from moe_exp.verification_study.pricing import collect_prices
            validate_reader_prices(Path(health["campaign_path"]),campaign,root,data["model"])
            price_path=root/health["expected_work"]["model"]/"prices-by-event"/(health["sha256"]+".json")
            prices=collect_prices(Path(health["campaign_path"]),data["model"],root,price_path)
            data.update(price_path=str(price_path),price_sha256=prices["sha256"],unmeasured_roles=prices["missing"])
        phase="PAPER_EXPORT"
        from moe_exp.verification_study.paper import export
        lock=private_path(root/"health/paper-export.lock");lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open("a") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            data["paper_provenance_sha256"]=export(Path(health["campaign_path"]),root)["sha256"]
    except Exception as exc:
        failure={"SOURCE_BINDING":"BINDING_FAILURE","ACCOUNTING":"ACCOUNTING_UNAVAILABLE",
                 "ARTIFACTS":"ARTIFACT_BINDING_FAILURE","PRICING":"PRICE_BINDING_FAILURE","PAPER_EXPORT":"PAPER_EXPORT_FAILED"}
        data.update(state=failure[phase],failure_phase=phase,error={"type":type(exc).__name__,"message":str(exc)[:2000]})
    result=save(event_path,data)
    update_latest(root/"health/latest"/data["model"]/(data["stage"]+".json"),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("register","check"))
    parser.add_argument("--campaign",type=Path,default=REPO/"report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--submission",type=Path)
    parser.add_argument("--health-card",type=Path)
    parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    if args.command=="check":result=check(args.health_card.resolve())
    else:
        campaign_path=args.campaign.resolve();campaign=load(campaign_path)
        result=register(campaign_path,args.submission.resolve(),private_path(args.root or campaign["result_root"]),args.test_only)
    print(json.dumps({k:v for k,v in result.items() if k in {"schema","sha256","state","job_id","predecessor_job_id","model","stage","test_only","scheduler_output"}},indent=2))


if __name__=="__main__":main()
