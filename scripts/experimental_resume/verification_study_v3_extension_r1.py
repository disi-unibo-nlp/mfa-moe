"""Source-sealed finite dispatch for compatible offline study extensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study.campaign import command,preflight,task_card
from moe_exp.verification_study.contracts import check_files,load,private_path,require,save,seal
from moe_exp.verification_study.registry import REPO,MODEL_KEYS
import verification_study_v3_j1_r1 as j1
import verification_study_v3_reader_r1 as semantic

STAGES={"j1-audit-frame":"audit-frame","j1-audit-slice":"j1-audit","j1-audit-closeout":"validate",
    "j1-freeze":"grade-strict","grade-j1-slice":"grade-j1","outcome-price-frame":"outcome-frame","j1-prices":"prices"}
GPU_KINDS={"j1-audit-slice","grade-j1-slice"}


def compile_card(campaign_path,request):
    card=task_card(campaign_path,request);campaign=load(campaign_path);root=Path(campaign["result_root"])
    extension=request["extension"];kind=extension["kind"];p=extension["parameters"];key=card["model"]
    require(kind in STAGES and card["stage"]==STAGES[kind],"extension/stage mismatch")
    require(card["resources"]["gpus"]==(2 if kind in GPU_KINDS else 0),"extension allocated resource mismatch")
    require(set(p)<= {"frame","output","source","manifest","grades","begin","count","qualification"},"unknown extension parameter")
    inputs={}
    for name in ("frame","manifest","qualification"):
        if p.get(name):inputs[p[name]]=load(p[name])["sha256"]
    executor=j1.binding(campaign);files={**executor["code_files"],**semantic.binding(campaign)["code_files"]}
    check_files(files)
    if kind=="j1-audit-frame":require(Path(p["output"])==root/"J1/AUDIT_FRAME.json","noncanonical J1 audit frame")
    elif kind=="j1-audit-closeout":
        require(Path(p["frame"])==root/"J1/AUDIT_FRAME.json" and Path(p["source"])==root/"J1/audit" and
                Path(p["output"])==root/"J1/J1_QUALIFICATION.json","noncanonical J1 qualification inputs")
    elif kind=="outcome-price-frame":
        require(key in MODEL_KEYS and Path(p["frame"])==root/key/"AUDIT_FRAME.json" and
                Path(p["output"])==root/key/"OUTCOME_PRICE_FRAME.json","noncanonical discovery outcome-price frame")
    elif kind=="j1-freeze":
        require(key in MODEL_KEYS,"J1 residuals require a generator model")
        stage=load(p["manifest"])["stage"];base=root/key
        expected_manifest=campaign_path.parent/key/"FINAL_NATIVE_ASSIGNMENTS.json" if stage=="final_native" else base/"final/UTILITY_ASSIGNMENTS.json"
        require(stage in {"final_native","utility"} and Path(p["manifest"])==expected_manifest and
                Path(p["source"])==base/stage and Path(p["grades"])==base/(stage+"-grades") and
                Path(p["output"])==base/(stage+"-grades")/"J1_FROZEN_FRAME.json","noncanonical full-roster J1 freeze")
    elif kind=="j1-prices":require(key in MODEL_KEYS,"J1 price sheet must identify a required model cell")
    if kind in GPU_KINDS:
        parent,contract,frame,bound=j1.deployment(campaign_path,Path(p["frame"]))
        rows=semantic.selected_records(frame,p["begin"],p["count"])
        shape=j1.probe(campaign_path,Path(p["frame"]),p["begin"],p["count"])
        if kind=="j1-audit-slice":
            require(Path(p["frame"])==root/"J1/AUDIT_FRAME.json" and Path(p["output"])==root/"J1/audit" and
                    not p.get("qualification") and frame["schema"]=="verification-j1-audit-frame-v3","J1 audit slice differs")
        else:
            require(not card["measurement_pilot"] and key in MODEL_KEYS and
                    Path(p["qualification"])==root/"J1/J1_QUALIFICATION.json" and
                    Path(p["frame"]).name=="J1_FROZEN_FRAME.json" and Path(p["output"])==Path(p["frame"]).parent/"j1" and
                    Path(p["frame"]).parent in {root/key/"final_native-grades",root/key/"utility-grades"},"J1 production frame differs")
            qualified=load(p["qualification"])
            require(qualified["status"]=="PASS" and qualified["j1_executor"]==executor and
                    qualified["reader_contract_sha256"]==contract["sha256"] and qualified["runtime_sha256"]==campaign["runtime_sha256"],
                    "J1 production deployment is not qualified")
        if card["measurement_pilot"]:
            require(kind=="j1-audit-slice" and p["count"]==1 and request["finite_assignments"]==1 and
                    request["population"]=="frozen_J1_audit","J1 first price must be exactly one retained original audit item")
        else:
            prices=load(request["price_path"],"verification-prices-v3")
            require(prices.get("j1_executor")==executor and request["priced_work"].get("j1_load",0)>=2 and
                    request["priced_work"].get("j1_generation",0)>=3*8192*(p["count"]+1) and
                    shape["max_input_tokens"]<=prices["components"]["j1_generation"]["max_context_input_tokens"],
                    "J1 needs own measured3-vote token/load/context prices and one bounded recovery reserve")
        require(card["resources"]["cpus"]==16 and card["resources"]["memory_gb"]==120,"qualified J1 resource shape differs")
        extension={**extension,"full_frame_sha256":frame["sha256"],"selected_uids":[r["uid"] for r in rows],"input_shape":shape}
    # Observer and worker use one declared argument vector, not two divergent views.
    args=request["arguments"];observed={args[i]:args[i+1] for i in range(0,len(args),2)}
    require(len(args)%2==0 and len(observed)==len(args)//2,"invalid extension argument pairs")
    for name,value in p.items():require(observed.get("--"+name)==str(value),"extension parameter/observer arguments differ")
    require(set(observed)<={"--"+name for name in p}|{"--model","--root"} and
            observed.get("--model",key)==key and observed.get("--root",str(root))==str(root),"extension argument overrides scope")
    return seal({**card,"extension":extension,"extension_files":files,"j1_executor":executor,
        "input_artifacts":inputs,"price_path":request.get("price_path"),
        "price_sha256":card["measured_budget"]["price_sha256"] if card["measured_budget"] else None,
        "priced_work":request.get("priced_work")})


def submit(campaign_path,card,test_only=False):
    campaign=load(campaign_path);check_files(campaign["code_files"]);check_files(card["extension_files"])
    require(card["campaign_sha256"]==campaign["sha256"] and card["j1_executor"]==j1.binding(campaign),"extension source changed")
    root=private_path(campaign["result_root"]);slot=root/"submissions"/card["sha256"];slot.mkdir(parents=True,exist_ok=True)
    receipt_path=slot/"SUBMISSION.json";comment="cv3x:"+card["sha256"]
    if receipt_path.exists() and not test_only:
        old=load(receipt_path);require(old["job_card_sha256"]==card["sha256"],"changed extension submission");return old
    if (slot/"INTENT.json").exists() and not test_only:
        queue=command(["squeue","-h","-u","lmolfett","-o","%i|%k"])
        jobs=[r.split("|",1)[0] for r in queue.splitlines() if r.endswith("|"+comment)]
        require(len(jobs)==1,"extension intent unresolved; reconcile accounting/comment before another submission")
        observed=command(["scontrol","show","job",jobs[0]])
        require("Account=iscrc_miosr" in observed and "JobId="+jobs[0] in observed,"extension reconciliation readback failed")
        return save(receipt_path,{"schema":"verification-submission-v3","job_id":jobs[0],"job_card_sha256":card["sha256"],
            "read_back":observed,"reconciled":True})
    path=slot/"JOB_CARD.json";save(path,card);logs=root/"logs";logs.mkdir(parents=True,exist_ok=True)
    r=card["resources"];gpu=r["gpus"]>0
    script=Path(__file__).with_name("verification_study_v3_extension_r1_"+("gpu" if gpu else "cpu")+".sbatch")
    argv=["sbatch","--test-only" if test_only else "--parsable","--account=iscrc_miosr","--partition="+r["partition"],
        "--qos=normal","--nodes=1","--ntasks=1","--cpus-per-task="+str(r["cpus"]),"--mem="+str(r["memory_gb"])+"G",
        "--time="+str(r["walltime_seconds"]//60),"--no-requeue","--job-name=cv3x-"+card["stage"]+"-"+card["model"],
        "--comment="+comment,"--chdir="+str(REPO),"--output="+str(logs/"%x-%j.out"),"--error="+str(logs/"%x-%j.err")]
    if gpu:argv.append("--gres=gpu:2")
    argv.extend((str(script),str(path)))
    if test_only:
        result=subprocess.run(argv,check=True,capture_output=True,text=True)
        return {"test_only":True,"scheduler_output":result.stdout+result.stderr,"job_card":str(path)}
    snapshot=save(slot/"PREFLIGHT.json",preflight())
    save(slot/"INTENT.json",{"schema":"verification-submission-intent-v3","job_card_sha256":card["sha256"],
        "preflight_sha256":snapshot["sha256"],"comment":comment,"epoch":time.time()})
    response=command(argv);job=response.split(";",1)[0];require(re.fullmatch(r"\d+",job),"invalid extension sbatch result")
    observed=command(["scontrol","show","job",job])
    require("JobId="+job in observed and "Account=iscrc_miosr" in observed,"extension submission readback failed")
    return save(receipt_path,{"schema":"verification-submission-v3","job_id":job,"job_card_sha256":card["sha256"],
        "sbatch_response":response,"read_back":observed,"preflight_sha256":snapshot["sha256"],
        "scientific_success":"NOT_YET_MEASURED"})


def execute(card_path,deadline=None):
    card=load(card_path,"verification-job-card-v3");campaign_path=Path(card["campaign_path"]);campaign=load(campaign_path)
    check_files(campaign["code_files"]);check_files(card["extension_files"])
    require(card["campaign_sha256"]==campaign["sha256"] and card["j1_executor"]==j1.binding(campaign),"extension executor changed")
    for path,sha in card["input_artifacts"].items():require(load(path)["sha256"]==sha,"extension input artifact changed")
    if card["measured_budget"]:
        from moe_exp.verification_study.pricing import shard_budget
        prices=load(card["price_path"])
        require(prices["sha256"]==card["price_sha256"] and
                shard_budget(prices,card["priced_work"],hours=card["resources"]["walltime_seconds"]/3600)==card["measured_budget"],
                "extension price/budget changed")
    p=card["extension"]["parameters"];kind=card["extension"]["kind"];root=Path(campaign["result_root"]);key=card["model"]
    require(kind in STAGES and card["stage"]==STAGES[kind],"extension kind changed")
    if kind in GPU_KINDS:
        from moe_exp.verification_study.backend import require_compute
        require_compute();require(deadline is not None,"finite extension deadline missing")
        frame=load(p["frame"])
        require(frame["sha256"]==card["extension"]["full_frame_sha256"] and
                [r["uid"] for r in semantic.selected_records(frame,p["begin"],p["count"])]==card["extension"]["selected_uids"],
                "J1 frame/slice changed")
        return j1.run_slice(campaign_path,Path(p["frame"]),Path(p["output"]),deadline,p["begin"],p["count"],
            Path(p["qualification"]) if p.get("qualification") else None)
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu()
    if kind=="j1-audit-frame":return j1.audit_frame(campaign_path,Path(p["output"]))
    if kind=="j1-audit-closeout":return j1.qualification(campaign_path,Path(p["frame"]),Path(p["source"]),Path(p["output"]))
    if kind=="j1-freeze":return j1.freeze_residuals(campaign_path,key,Path(p["manifest"]),Path(p["source"]),Path(p["grades"]),Path(p["output"]))
    if kind=="outcome-price-frame":return semantic.outcome_price_frame(campaign_path,key,root,Path(p["output"]))
    return j1.collect_prices(campaign_path,key,root,Path(p["output"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("submit","execute","probe"))
    parser.add_argument("--campaign",type=Path,default=REPO/"report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--request",type=Path);parser.add_argument("--job-card",type=Path);parser.add_argument("--frame",type=Path)
    parser.add_argument("--deadline-epoch",type=float);parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args();campaign_path=args.campaign.resolve()
    if args.command=="execute":result=execute(args.job_card.resolve(),args.deadline_epoch)
    elif args.command=="probe":result=j1.probe(campaign_path,args.frame.resolve())
    else:result=submit(campaign_path,compile_card(campaign_path,load(args.request)),args.test_only)
    print(json.dumps({k:v for k,v in result.items() if k in {"schema","sha256","status","job_id","test_only","scheduler_output","max_input_tokens","largest_record_index"}},indent=2))


if __name__=="__main__":main()
