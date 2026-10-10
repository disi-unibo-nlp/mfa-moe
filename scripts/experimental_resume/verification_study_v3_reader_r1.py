"""Finite reader slices outside the frozen r4 scientific source closure.

The pinned Reader supplies every prompt, sampler and seed. This module changes
only dispatch, resume validation, invocation accounting and finite slicing.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study.campaign import command, preflight, task_card
from moe_exp.verification_study.contracts import (check_files, digest, file_sha,
    load, private_path, require, save, seal)
from moe_exp.verification_study.registry import REPO

SCRIPT=Path(__file__).resolve()
SHELL=SCRIPT.with_suffix(".sbatch")


def binding(campaign):
    return seal({"schema":"verification-reader-executor-v3",
        "campaign_sha256":campaign["sha256"],"scientific_executor_sha256":digest(campaign["code_files"]),
        "runtime_sha256":campaign["runtime_sha256"],"reader_contract_sha256":campaign["reader_sha256"],
        "code_files":{str(p):file_sha(p) for p in (SCRIPT,SHELL)},
        "reader_behavior":"unchanged r4 Reader.rate; two draws, original full-frame identities and canonical filenames"})


def deployment(campaign_path,frame_path):
    campaign=load(campaign_path);check_files(campaign["code_files"])
    contract=load(campaign_path.parent/"READER.json","verification-reader-contract-v3")
    require(contract["sha256"]==campaign["reader_sha256"],"reader contract differs from campaign")
    check_files(contract["config_files"])
    frame=load(frame_path)
    require(frame["reader_contract_sha256"]==contract["sha256"],"foreign full reader frame")
    require(frame["records"] and len({r["uid"] for r in frame["records"]})==len(frame["records"]) and
            all(re.fullmatch(r"[a-f0-9]{64}",r["uid"]) for r in frame["records"]),"empty, duplicate or unsafe reader identities")
    return campaign,contract,frame,binding(campaign)


def selected_records(frame,begin,count):
    require(type(begin) is int and begin>=0 and type(count) is int and count>0 and begin+count<=len(frame["records"]),
            "reader slice lies outside original full frame")
    return frame["records"][begin:begin+count]


def validate_runtime(value,campaign,contract,frame,executor):
    require(value.get("schema")=="verification-reader-runtime-v3" and
            value["reader_sha256"]==contract["sha256"] and value["runtime_sha256"]==campaign["runtime_sha256"] and
            value["frame_sha256"]==frame["sha256"] and re.fullmatch(r"\d+",str(value["job_id"])) and
            type(value["load_seconds"]) in (float,int) and math.isfinite(value["load_seconds"]) and value["load_seconds"]>=0,
            "foreign or invalid reader runtime")
    require(value.get("reader_executor_sha256")==executor["sha256"] and value.get("reader_executor")==executor and
            value.get("campaign_sha256")==campaign["sha256"] and value.get("checkpoint")==contract["checkpoint"] and
            value.get("role") in {"eligibility","outcome"},"reader runtime executor/deployment binding changed")
    require(isinstance(value.get("slice"),list) and len(value["slice"])==2,"reader runtime lacks exact slice")
    rows=selected_records(frame,*value["slice"])
    require(value.get("selected_uids")==[r["uid"] for r in rows],"reader runtime slice identities changed")


def validate_rating(path,record,draw,role,campaign,contract,frame,executor):
    value=load(path,"verification-reader-receipt-v3")
    require(type(value["reader"]) is int and value["reader"]==draw and
            path.name==record["uid"]+f".reader-{draw}.json" and value["uid"]==record["uid"] and
            value["input_sha256"]==digest(record) and value["frame_sha256"]==frame["sha256"] and
            value["role"]==role and value["checkpoint"]==contract["checkpoint"] and
            value["reader_contract_sha256"]==contract["sha256"] and value["runtime_sha256"]==campaign["runtime_sha256"],
            "foreign reader UID/draw/input/frame/role/deployment")
    require(type(value["seconds"]) in (int,float) and math.isfinite(value["seconds"]) and value["seconds"]>=0 and
            type(value["input_tokens"]) is int and value["input_tokens"]>0 and
            type(value["output_tokens"]) is int and 0<=value["output_tokens"]<=1024,"invalid reader measured costs")
    require(value.get("reader_executor_sha256")==executor["sha256"],"reader receipt executor missing or changed")
    runtime_path=private_path(value["invocation_runtime_path"])
    require(runtime_path.is_relative_to(path.parent/"invocations") and runtime_path.name=="RUNTIME.json",
            "reader invocation runtime path is foreign")
    runtime=load(runtime_path);validate_runtime(runtime,campaign,contract,frame,executor)
    require(runtime["sha256"]==value["invocation_runtime_sha256"] and value["job_id"]==runtime["job_id"] and
            runtime["role"]==role and record["uid"] in runtime["selected_uids"],"reader invocation identity changed")
    return value


def validate_directory(output,campaign,contract,frame,executor,role):
    rows={r["uid"]:r for r in frame["records"]}
    paths=list(output.glob("*.reader-*.json"))
    if paths:
        require((output/"RUNTIME.json").exists(),"committed reader labels lack runtime anchor")
    if (output/"RUNTIME.json").exists():
        runtime=load(output/"RUNTIME.json");validate_runtime(runtime,campaign,contract,frame,executor)
        require(runtime["role"]==role,"reader runtime role changed")
    for path in paths:
        match=re.fullmatch(r"([a-f0-9]{64})\.reader-([01])\.json",path.name)
        require(match and match[1] in rows,"foreign reader filename")
        validate_rating(path,rows[match[1]],int(match[2]),role,campaign,contract,frame,executor)


def rate_slice(campaign_path,frame_path,output,deadline,begin,count,role,*,reader_factory=None,check_runtime=True):
    campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    if check_runtime:
        from moe_exp.verification_study.backend import require_compute
        from moe_exp.verification_study.prepare import runtime_binding
        require_compute()
        require(runtime_binding()["sha256"]==campaign["runtime_sha256"],"reader installed runtime changed")
    require(role in {"eligibility","outcome"},"unsupported reader role")
    rows=selected_records(frame,begin,count);output=private_path(output);output.mkdir(parents=True,exist_ok=True)
    validate_directory(output,campaign,contract,frame,executor,role)
    invocation=str(os.environ["SLURM_JOB_ID"])+"-"+str(time.time_ns())
    slot=output/"invocations"/invocation;slot.mkdir(parents=True,exist_ok=True)
    identity={"campaign_sha256":campaign["sha256"],"frame_sha256":frame["sha256"],"role":role,
        "job_id":os.environ["SLURM_JOB_ID"],"slice":[begin,count],"selected_uids":[r["uid"] for r in rows],
        "reader_executor_sha256":executor["sha256"],"reader_executor":executor}
    save(slot/"INTENT.json",{"schema":"verification-reader-invocation-intent-v3",**identity,"started_epoch":time.time()})
    reader=None;error=None;generated=0
    with ExitStack() as stack:
        # Disjoint slices can run concurrently; overlapping slices fail before loading.
        for row in sorted(rows,key=lambda r:r["uid"]):
            lock_path=output/"locks"/(row["uid"]+".lock");lock_path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(lock_path.open("a"))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_directory(output,campaign,contract,frame,executor,role)
        missing=[(row,draw) for row in rows for draw in (0,1) if not (output/(row["uid"]+f".reader-{draw}.json")).exists()]
        try:
            if missing:
                require(time.time()<deadline-120,"reader deadline before load")
                if reader_factory is None:
                    from moe_exp.verification_study.readers import Reader
                    reader_factory=Reader
                reader=reader_factory(contract["checkpoint"])
                runtime=save(slot/"RUNTIME.json",{"schema":"verification-reader-runtime-v3",**identity,
                    "reader_sha256":contract["sha256"],"runtime_sha256":campaign["runtime_sha256"],
                    "load_seconds":reader.load_seconds,"checkpoint":contract["checkpoint"]})
                with (output/"runtime.lock").open("a") as handle:
                    fcntl.flock(handle,fcntl.LOCK_EX)
                    if not (output/"RUNTIME.json").exists():save(output/"RUNTIME.json",runtime)
                    else:validate_runtime(load(output/"RUNTIME.json"),campaign,contract,frame,executor)
                for row,draw in missing:
                    require(time.time()<deadline-120,"reader allocation deadline")
                    receipt=reader.rate(row,role,draw)
                    path=output/(row["uid"]+f".reader-{draw}.json")
                    save(path,{**receipt,"frame_sha256":frame["sha256"],"reader_contract_sha256":contract["sha256"],
                        "runtime_sha256":campaign["runtime_sha256"],"reader_executor_sha256":executor["sha256"],
                        "job_id":os.environ["SLURM_JOB_ID"],"invocation_runtime_path":str(slot/"RUNTIME.json"),
                        "invocation_runtime_sha256":runtime["sha256"]})
                    validate_rating(path,row,draw,role,campaign,contract,frame,executor);generated+=1
        except Exception as exc:
            error={"type":type(exc).__name__,"message":str(exc)[:2000]}
        finally:
            if reader is not None:
                try:reader.close()
                except Exception as exc:
                    error=error or {"type":type(exc).__name__,"message":str(exc)[:2000]}
        complete=all((output/(r["uid"]+f".reader-{d}.json")).exists() for r in rows for d in (0,1))
        result=save(slot/"RESULT.json",{"schema":"verification-reader-slice-v3",**identity,
            "status":"FAIL" if error else "COMPLETE" if complete else "INCOMPLETE",
            "new_ratings":generated,"loaded":reader is not None,"error":error,"ended_epoch":time.time()})
    require(result["status"]=="COMPLETE","reader slice failed: "+str(error))
    return result


def collect_reader_prices(campaign_path,key,root,output):
    """Keep native prices and count every validated reader load exactly once."""
    from moe_exp.verification_study.pricing import collect_prices,price_sheet
    campaign=load(campaign_path);check_files(campaign["code_files"])
    model=load(campaign["models"][key]);executor=binding(campaign)
    reader_samples=[]
    for directory,frame_name,role,price_role in (
        ("eligibility-ratings","AUDIT_FRAME.json","eligibility","eligibility_reader"),
        ("outcome-price-ratings","OUTCOME_PRICE_FRAME.json","outcome","outcome_reader"),
        ("outcome-ratings","OUTCOME_FRAME.json","outcome","outcome_reader")):
        base=root/key/directory
        if not base.exists() or not list(base.glob("*.reader-*.json")) and not (base/"RUNTIME.json").exists():continue
        parent,contract,frame,bound=deployment(campaign_path,root/key/frame_name)
        validate_directory(base,parent,contract,frame,bound,role)
        runtimes={}
        for path in sorted(base.glob("invocations/*/RUNTIME.json")):
            runtime=load(path);validate_runtime(runtime,parent,contract,frame,bound)
            require(runtime.get("reader_executor_sha256")==executor["sha256"] and runtime["role"]==role,
                    "foreign measured invocation executor or role")
            runtimes[runtime["sha256"]]=(path,runtime)
        canonical=load(base/"RUNTIME.json")
        runtimes.setdefault(canonical["sha256"],(base/"RUNTIME.json",canonical))
        for path,runtime in runtimes.values():
            reader_samples.append({"role":"reader_load","seconds":runtime["load_seconds"],"units":1,
                "model_sha256":model["sha256"],"execution_sha256":campaign["runtime_sha256"],
                "receipt_path":str(path),"receipt_sha256":runtime["sha256"],
                "reader_executor_sha256":runtime.get("reader_executor_sha256"),"engineering_only":False})
        for path in sorted(base.glob("*.reader-*.json")):
            value=load(path)
            reader_samples.append({"role":price_role,"seconds":value["seconds"],"units":1,
                "model_sha256":model["sha256"],"execution_sha256":campaign["runtime_sha256"],
                "receipt_path":str(path),"receipt_sha256":value["sha256"],"engineering_only":False,
                "reader_executor_sha256":value.get("reader_executor_sha256"),
                "context_input_tokens":value["input_tokens"],"total_context_tokens":value["input_tokens"]+value["output_tokens"],
                "input_tokens":value["input_tokens"],"output_tokens":value["output_tokens"]})
    # The immutable external snapshot replaces only the reader samples; native
    # scientific bindings and price units retain their existing contract.
    raw_path=private_path(output.with_name(output.stem+"-native.json"))
    raw=collect_prices(campaign_path,key,root,raw_path)
    native_samples=[sample for name,component in raw["components"].items()
                    if name not in {"reader_load","eligibility_reader","outcome_reader"} for sample in component["samples"]]
    prices=price_sheet(model,native_samples+reader_samples,campaign["code_files"])
    return save(output,{**prices,"reader_executor":executor,"native_price_sha256":raw["sha256"]})


def probe_inputs(campaign_path,frame_path,role,begin=0,count=None):
    """Bounded tokenization only; no model load or generation."""
    from transformers import AutoTokenizer
    from moe_exp.verification_study.readers import normalize_prompt_ids
    from moe_exp.verification_study.semantics import reader_messages
    campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    rows=selected_records(frame,begin,len(frame["records"])-begin if count is None else count)
    tokenizer=AutoTokenizer.from_pretrained(contract["checkpoint"],local_files_only=True)
    counts=[len(normalize_prompt_ids(tokenizer.apply_chat_template(reader_messages(row,role=role),tokenize=True,
        add_generation_prompt=True,enable_thinking=True,reasoning_effort="low",return_dict=False))) for row in rows]
    require(all(n+1024<=49152 for n in counts),"reader input exceeds pinned context; truncation forbidden")
    return seal({"schema":"verification-reader-input-shape-v3","frame_sha256":frame["sha256"],
        "reader_contract_sha256":contract["sha256"],"reader_executor_sha256":executor["sha256"],"role":role,
        "begin":begin,"count":len(rows),"uids":[r["uid"] for r in rows],"input_tokens":counts,
        "max_input_tokens":max(counts),"output_cap":1024,"largest_record_index":begin+counts.index(max(counts))})


def outcome_price_frame(campaign_path,key,root,output):
    """Reuse native discovery windows only; these ratings are cost measurements."""
    from transformers import AutoTokenizer
    from moe_exp.verification_study.closeout import require_cpu
    from moe_exp.verification_study.contracts import identity,require_generation_binding
    from moe_exp.verification_study.views import reasoning_limit
    require_cpu();campaign,contract,frame,executor=deployment(campaign_path,root/key/"AUDIT_FRAME.json")
    model=load(campaign["models"][key]);tokenizer=AutoTokenizer.from_pretrained(model["local_path"],local_files_only=True)
    records=[];sources={}
    for row in frame["records"]:
        require(row["population"]=="discovery","reader cost source is not discovery")
        source=root/key/"discovery"/(row["source_uid"]+".json")
        if str(source) not in sources:sources[str(source)]=load(source)
        receipt=sources[str(source)]
        require_generation_binding(receipt,campaign,model)
        require(receipt["assignment"]["stage"]=="discovery" and receipt["assignment"]["arm"]=="native" and
                receipt["engineering_only"] is False and receipt["execution_complete"] is True and
                receipt["emitted_token_ids"][:row["position"]]==row["prefix_token_ids"] and
                digest(row["prefix_token_ids"])==row["prefix_ids_sha256"],"foreign discovery reader price prefix")
        window=receipt["emitted_token_ids"][row["position"]:min(reasoning_limit(receipt),row["position"]+1024)]
        records.append({**row,"uid":identity("outcome-price",row["uid"],receipt["sha256"]),
            "window_text":tokenizer.decode(window,skip_special_tokens=False),"window_token_count":len(window),
            "source_receipt_sha256":receipt["sha256"],"measurement_only":True})
    return save(output,{"schema":"verification-outcome-price-frame-v3","records":records,
        "reader_contract_sha256":contract["sha256"],"source_audit_frame_sha256":frame["sha256"],
        "source_receipts":{path:r["sha256"] for path,r in sources.items()},"source_population":"discovery",
        "price_only":True,"window_cap":1024,"closure_policy":"exclude first native closure marker and subsequent text",
        "scientific_effect":"NOT_ESTIMATED_FROM_THIS_PRICE_FRAME","reader_executor":executor})


def reader_card(campaign_path,request):
    card=task_card(campaign_path,request);campaign=load(campaign_path)
    require(card["stage"]=="rate","reader extension accepts only blinded semantic ratings")
    raw=request["arguments"];args={raw[i]:raw[i+1] for i in range(0,len(raw),2)}
    require(len(raw)%2==0 and set(args)<={"--model","--frame","--output","--role","--begin","--count"} and
            len(args)==len(raw)//2 and {"--frame","--output","--begin","--count","--role"}<=set(args),"invalid finite reader arguments")
    frame_path=Path(args["--frame"]);campaign,contract,frame,executor=deployment(campaign_path,frame_path)
    begin=int(args["--begin"]);count=int(args["--count"]);rows=selected_records(frame,begin,count)
    require(args["--role"] in {"eligibility","outcome"} and ("--model" not in args or args["--model"]==card["model"]),"foreign reader model/role")
    root=Path(campaign["result_root"]);base=root/card["model"]
    outcome_pilot=args["--role"]=="outcome" and card["measurement_pilot"]
    directory="eligibility-ratings" if args["--role"]=="eligibility" else "outcome-price-ratings" if outcome_pilot else "outcome-ratings"
    frame_name="AUDIT_FRAME.json" if args["--role"]=="eligibility" else "OUTCOME_PRICE_FRAME.json" if outcome_pilot else "OUTCOME_FRAME.json"
    require(private_path(args["--output"])==base/directory and frame_path.resolve()==base/frame_name,
            "reader must preserve canonical full-frame and rating directories")
    shape=probe_inputs(campaign_path,frame_path,args["--role"],begin,count)
    if card["measurement_pilot"]:
        require(count==1 and request["finite_assignments"]==1 and request["population"]=="discovery" and
                all(r.get("population")=="discovery" for r in rows),
                "first reader measurement is exactly one retained discovery record/two draws")
        if outcome_pilot:
            require(frame.get("price_only") is True and frame.get("window_cap")==1024 and
                    frame.get("source_population")=="discovery","outcome price frame must use discovery-only material")
    else:
        prices=load(request["price_path"]);role="eligibility_reader" if args["--role"]=="eligibility" else "outcome_reader"
        require(prices.get("reader_executor")==executor and request["priced_work"].get("reader_load",0)>=1 and
                request["priced_work"].get(role,0)>=2*count and
                shape["max_input_tokens"]<=prices["components"][role]["max_context_input_tokens"],
                "reader price, work count or measured context coverage is missing")
    require(card["resources"]["cpus"]==16 and card["resources"]["memory_gb"]==120,"qualified reader allocation shape differs")
    return seal({**card,"reader_executor":executor,"full_frame_sha256":frame["sha256"],
        "selected_uids":[r["uid"] for r in rows],"reader_input_shape":shape,
        "output_directory":str(base/directory),"runner_arguments":args,
        "price_path":request.get("price_path"),"price_sha256":None if card["measurement_pilot"] else prices["sha256"],
        "priced_work":request.get("priced_work"),"measurement_population":request.get("population")})


def submit_reader(campaign_path,card,root,test_only=False):
    campaign=load(campaign_path);check_files(campaign["code_files"])
    require(card["campaign_sha256"]==campaign["sha256"] and card["reader_executor"]==binding(campaign),"reader job binding differs")
    root=private_path(root);slot=root/"submissions"/card["sha256"];slot.mkdir(parents=True,exist_ok=True)
    receipt_path=slot/"SUBMISSION.json"
    if receipt_path.exists() and not test_only:
        old=load(receipt_path);require(old["job_card_sha256"]==card["sha256"],"foreign reader submission");return old
    comment="cv3r:"+card["sha256"]
    if (slot/"INTENT.json").exists() and not test_only:
        jobs=command(["squeue","-h","-u","lmolfett","-o","%i|%k"])
        matches=[row.split("|",1)[0] for row in jobs.splitlines() if row.endswith("|"+comment)]
        require(len(matches)==1,"reader intent unresolved; reconcile live/accounting comment before resubmission")
        job=matches[0];observed=command(["scontrol","show","job",job])
        return save(receipt_path,{"schema":"verification-submission-v3","job_card_sha256":card["sha256"],"job_id":job,
            "read_back":observed,"reconciled":True,"reader_executor_sha256":card["reader_executor"]["sha256"]})
    path=slot/"JOB_CARD.json";save(path,card);logs=root/"logs";logs.mkdir(parents=True,exist_ok=True)
    resources=card["resources"]
    args=["sbatch","--test-only" if test_only else "--parsable","--account=iscrc_miosr","--partition=boost_usr_prod",
        "--qos=normal","--nodes=1","--ntasks=1","--gres=gpu:2","--cpus-per-task=16","--mem=120G",
        "--time="+str(resources["walltime_seconds"]//60),"--no-requeue","--job-name=cv3-rate-"+card["model"],
        "--comment="+comment,"--chdir="+str(REPO),"--output="+str(logs/"%x-%j.out"),"--error="+str(logs/"%x-%j.err"),str(SHELL),str(path)]
    if test_only:
        result=subprocess.run(args,check=True,capture_output=True,text=True)
        return {"test_only":True,"scheduler_output":result.stdout+result.stderr,"job_card":str(path)}
    snapshot=save(slot/"PREFLIGHT.json",preflight())
    save(slot/"INTENT.json",{"schema":"verification-submission-intent-v3","job_card_sha256":card["sha256"],
        "preflight_sha256":snapshot["sha256"],"comment":comment,"epoch":time.time()})
    response=command(args);job=response.split(";",1)[0];require(re.fullmatch(r"\d+",job),"invalid reader sbatch response")
    observed=command(["scontrol","show","job",job])
    require("JobId="+job in observed and "Account=iscrc_miosr" in observed,"reader job read-back failed")
    return save(receipt_path,{"schema":"verification-submission-v3","job_card_sha256":card["sha256"],"job_id":job,
        "sbatch_response":response,"read_back":observed,"preflight_sha256":snapshot["sha256"],
        "reader_executor_sha256":card["reader_executor"]["sha256"],"scientific_success":"NOT_YET_MEASURED"})


def execute(card_path,deadline):
    card=load(card_path,"verification-job-card-v3");campaign_path=Path(card["campaign_path"]);campaign=load(campaign_path)
    require(card["campaign_sha256"]==campaign["sha256"] and card["reader_executor"]==binding(campaign),"reader executor source changed")
    if not card["measurement_pilot"]:
        from moe_exp.verification_study.pricing import shard_budget
        prices=load(card["price_path"],"verification-prices-v3")
        require(prices["sha256"]==card["price_sha256"] and prices.get("reader_executor")==card["reader_executor"],
                "reader dispatch price snapshot changed")
        require(shard_budget(prices,card["priced_work"],hours=card["resources"]["walltime_seconds"]/3600)==card["measured_budget"],
                "reader finite measured budget changed")
    args=card["runner_arguments"];frame=load(args["--frame"])
    require(frame["sha256"]==card["full_frame_sha256"] and
            [r["uid"] for r in selected_records(frame,int(args["--begin"]),int(args["--count"]))]==card["selected_uids"],
            "reader job frame or slice changed")
    return rate_slice(campaign_path,Path(args["--frame"]),Path(args["--output"]),deadline,
                      int(args["--begin"]),int(args["--count"]),args["--role"])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("submit","execute","probe","prices","outcome-price-frame"))
    parser.add_argument("--campaign",type=Path,default=REPO/"report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--request",type=Path);parser.add_argument("--job-card",type=Path);parser.add_argument("--deadline-epoch",type=float)
    parser.add_argument("--frame",type=Path);parser.add_argument("--role",choices=("eligibility","outcome"),default="eligibility")
    parser.add_argument("--model");parser.add_argument("--root",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--test-only",action="store_true");args=parser.parse_args();campaign_path=args.campaign.resolve()
    if args.command=="execute":result=execute(args.job_card.resolve(),args.deadline_epoch)
    elif args.command=="submit":
        campaign=load(campaign_path);result=submit_reader(campaign_path,reader_card(campaign_path,load(args.request)),
            args.root or Path(campaign["result_root"]),args.test_only)
    elif args.command=="probe":result=probe_inputs(campaign_path,args.frame.resolve(),args.role)
    elif args.command=="outcome-price-frame":result=outcome_price_frame(campaign_path,args.model,args.root,args.output)
    else:result=collect_reader_prices(campaign_path,args.model,args.root,args.output)
    print(json.dumps({k:v for k,v in result.items() if k in {"schema","sha256","status","job_id","new_ratings","loaded","test_only","scheduler_output","largest_record_index","max_input_tokens","count"}},indent=2))


if __name__=="__main__":main()
