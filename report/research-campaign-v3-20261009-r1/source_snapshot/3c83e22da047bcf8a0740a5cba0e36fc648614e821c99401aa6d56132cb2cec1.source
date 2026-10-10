"""Finite, idempotent Slurm submissions with paper artifacts and read-back IDs."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from .contracts import check_files, digest, load, private_path, require, save, seal
from .registry import MODEL_KEYS, REPO


def command(args):
    return subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip()


def preflight():
    require(getpass.getuser()=="lmolfett" and socket.gethostname().endswith("leonardo.local"),"wrong host/account")
    snapshots = {"hostname":socket.gethostname(),"user":getpass.getuser(),"observed_epoch":time.time()}
    for key,args in {
        "balance":["saldo","-b"],
        "association":["sacctmgr","-nP","show","assoc","where","user=lmolfett","format=Account,Partition,QOS,DefaultQOS"],
        "gpu_partition":["scontrol","show","partition","boost_usr_prod"],
        "cpu_partition":["scontrol","show","partition","lrd_all_serial"],
        "qos":["sacctmgr","-nP","show","qos","normal","format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU"],
        "queue":["squeue","-h","-u","lmolfett","-o","%i|%T|%j|%R"],
        "storage":["cinQuota"]}.items():
        snapshots[key] = command(args)
    require("iscrc_miosr" in snapshots["association"] and "normal" in snapshots["association"],"association not entitled")
    require("State=UP" in snapshots["gpu_partition"] and "State=UP" in snapshots["cpu_partition"],"required partition unavailable")
    return seal({"schema":"verification-slurm-preflight-v3",**snapshots})


def qualification_card(campaign_path,key):
    campaign = load(campaign_path); model = load(campaign["models"][key])
    require(key in MODEL_KEYS,"foreign model")
    return seal({"schema":"verification-job-card-v3","stage":"qualify","model":key,
        "campaign_path":str(campaign_path),"campaign_sha256":campaign["sha256"],"model_sha256":model["sha256"],
        "claim":"C3: valid native-k verification actuator for this required checkpoint",
        "reuse":"existing checkpoint/native selector and v2 engineering fixtures; historical scientific results preserved",
        "missing_cell":"actual architecture callback, TP parity, 256-token timing/closure and full-cap feasibility",
        "artifact":f"paper generated verification status table: {key} backend cell; engineering receipt and compact captures",
        "estimand":"engineering fidelity only; no semantic or utility effect",
        "controls":["same-state native callback","natural native repeat","inactive neighboring request","early native closure","engine usable after local invalid-action rejection; CPU worker patch rollback"],
        "positive":"qualify this exact executor; still require nomination/recognition/dose/prices",
        "negative":"record engineering failure; model remains in required roster; no causal-null interpretation",
        "resources":{"partition":"boost_usr_prod","qos":"normal","gpus":2,"cpus":16,"memory_gb":120,
            "walltime_seconds":7200,"billing_core_hours_max":32,
            "price_basis":"observed Qwen full-cap fixture2637.33s plus cold-load7-8min; finite architecture qualification to measure own price",
            "output_bound":"14 fixtures; one16384-token full-cap capture; compact arrays; no checkpoint copy"},
        "retry":{"automatic_assignment_retries":1,"automatic_new_allocations":0},
        "authorization":"user: Implement the plan; all seven; no arbitrary global cap, only relevant finite jobs",
        "code_files":campaign["code_files"],"prerequisite_mode":"engineering measurement, not production dispatch"})


def evidence_card(campaign_path):
    campaign=load(campaign_path)
    return seal({"schema":"verification-job-card-v3","stage":"evidence","model":"all-seven",
        "campaign_path":str(campaign_path),"campaign_sha256":campaign["sha256"],"code_files":campaign["code_files"],
        "claim":"C1/C2: reuse existing seven-model evidence and expose missing prediction/causal cells",
        "reuse":"wiki evidence snapshot and bound R3 native readout/anticipation artifacts; no historical dense recovery",
        "missing_cell":"current source compatibility/qualification and paper table with all-model accounting",
        "artifact":"generated all-seven evidence table, qualification matrix and machine-readable paper provenance",
        "estimand":"saved results retained under original populations and uncertainty; no new fitted outcome claim",
        "resources":{"partition":"lrd_all_serial","qos":"normal","gpus":0,"cpus":4,"memory_gb":24,
                     "walltime_seconds":1800,"billing_core_hours_max":0,"price_basis":"bounded receipt reconciliation/table export; no model load"},
        "positive":"update manuscript from observed receipts","negative":"keep explicit missing or incompatible cells",
        "retry":{"automatic_assignment_retries":0,"automatic_new_allocations":0},
        "authorization":"user: Implement the plan and iteratively add results to paper"})


def task_card(campaign_path,request):
    """Compile a concrete reviewed stage request; pricing cannot be invented."""
    from .contracts import verify
    from .pricing import shard_budget
    verify(request,"verification-task-request-v3")
    campaign=load(campaign_path);check_files(campaign["code_files"])
    stage=request["stage"];key=request["model"]
    gpu=stage in {"run","rate","calibrate","j1-audit","grade-j1"}
    allowed={"run","rate","calibrate","j1-audit","grade-j1","audit-frame","recognition","nominate","validate",
             "freeze-final","outcome-frame","grade-strict","assemble","terminal-source","analyze","evidence","prices"}
    require(stage in allowed and (key in MODEL_KEYS or key=="all-seven"),"task lies outside the adopted study")
    require(request["claim"] and request["artifact"] and request["reuse"] and request["missing_cell"] and
            request["decision_if_failure"] and request["resources"]["walltime_seconds"]>0,"incomplete concrete job card")
    args=request["arguments"]
    require(isinstance(args,list) and all(isinstance(v,str) for v in args) and "--campaign" not in args and
            "--deadline-epoch" not in args,"invalid or overriding task arguments")
    measurement=request.get("measurement_pilot",False)
    if gpu and not measurement:
        prices=load(request["price_path"],"verification-prices-v3")
        require(prices["model_sha256"]==load(campaign["models"][key])["sha256"] and
                prices["code_files"]==campaign["code_files"],"task price binding differs")
        budget=shard_budget(prices,request["priced_work"],hours=request["resources"]["walltime_seconds"]/3600)
    else:
        budget=None
        if gpu:
            # First measurement is small, outcome-blind and retained. It is not
            # an exemption for an unpriced final experiment or broad sweep.
            require(stage in {"run","rate","calibrate","j1-audit"} and request["population"] in {"discovery","validation","frozen_J1_audit"},
                    "unpriced final experiment forbidden")
            require(request["resources"]["walltime_seconds"]<=8*3600 and request.get("finite_assignments") is not None,
                    "measurement pilot lacks finite enrollment")
            if stage=="run":
                require("--count" in args and int(args[args.index("--count")+1])<=1 and
                        "--manifest" in args,"first native price pilot must be one retained discovery/validation assignment")
                m=load(args[args.index("--manifest")+1]);require(m["stage"] in ("discovery","validation"),"unpriced final stage forbidden")
    resources=request["resources"]
    require(resources["partition"]==("boost_usr_prod" if gpu else "lrd_all_serial") and resources["qos"]=="normal" and
            resources["gpus"]==(2 if gpu else 0) and resources["walltime_seconds"]<=(24 if gpu else 4)*3600,
            "task resource shape differs from qualified partitions")
    return seal({"schema":"verification-job-card-v3","stage":stage,"model":key,
        "campaign_path":str(campaign_path),"campaign_sha256":campaign["sha256"],"code_files":campaign["code_files"],
        "claim":request["claim"],"artifact":request["artifact"],"reuse":request["reuse"],"missing_cell":request["missing_cell"],
        "decision_if_failure":request["decision_if_failure"],"resources":resources,"measured_budget":budget,
        "payload_arguments":[stage,"--campaign",str(campaign_path),*args],"measurement_pilot":measurement,
        "task_request_sha256":request["sha256"],"prerequisites":request.get("prerequisites",[]),
        "authorization":"user implementation of the adopted all-seven study; exact measured/pricing cell only",
        "retry":{"automatic_assignment_retries":1,"automatic_new_allocations":0}})


def submit(campaign_path,card,root,*,test_only=False,dependencies=()):
    from .contracts import verify
    verify(card,"verification-job-card-v3")
    campaign=load(campaign_path);check_files(campaign["code_files"])
    require(card["campaign_sha256"]==campaign["sha256"],"foreign job card")
    root=private_path(root);root.mkdir(parents=True,exist_ok=True)
    identity=card["sha256"]
    slot=root/"submissions"/identity
    slot.mkdir(parents=True,exist_ok=True)
    receipt_path=slot/"SUBMISSION.json"
    if receipt_path.exists() and not test_only:
        existing=load(receipt_path)
        require(existing["job_card_sha256"]==identity,"changed submission")
        return existing
    # The intent persists before sbatch. An interrupted submission is reconciled
    # by unique Slurm comment, never blindly retried.
    if (slot/"INTENT.json").exists() and not test_only:
        jobs=command(["squeue","-h","-u","lmolfett","-o","%i|%k"])
        matches=[line.split("|",1)[0] for line in jobs.splitlines() if line.endswith("|cv3:"+identity)]
        require(len(matches)==1,"submission intent unresolved; reconcile sacct/comment before retry")
        return save(receipt_path,{"schema":"verification-submission-v3","job_card_sha256":identity,"job_id":matches[0],"reconciled":True})
    resources=card["resources"]
    jobcard_path=slot/"JOB_CARD.json";save(jobcard_path,card)
    logs=root/"logs";logs.mkdir(parents=True,exist_ok=True)
    gpu=resources["gpus"]>0
    script=REPO/"scripts/experimental_resume"/("verification_study_v3_gpu.sbatch" if gpu else "verification_study_v3_cpu.sbatch")
    args=["sbatch","--test-only" if test_only else "--parsable","--account=iscrc_miosr",
        "--partition="+resources["partition"],"--qos=normal","--nodes=1","--ntasks=1",
        "--cpus-per-task="+str(resources["cpus"]),"--mem="+str(resources["memory_gb"])+"G",
        "--time="+str(resources["walltime_seconds"]//60),"--no-requeue",
        "--job-name=cv3-"+card["stage"]+"-"+card["model"],"--comment=cv3:"+identity,
        "--chdir="+str(REPO),"--output="+str(logs/"%x-%j.out"),"--error="+str(logs/"%x-%j.err")]
    if gpu:args += ["--gres=gpu:"+str(resources["gpus"])]
    if dependencies:
        require(all(re.fullmatch(r"\d+",str(j)) for j in dependencies),"invalid dependency")
        args += ["--dependency=afterany:"+":".join(map(str,dependencies))]
    args += [str(script),str(campaign_path),card["stage"],card["model"],str(root),str(jobcard_path)]
    if test_only:
        result=subprocess.run(args,text=True,capture_output=True,check=True)
        return {"test_only":True,"scheduler_output":result.stdout+result.stderr,"job_card":str(jobcard_path)}
    snapshot=save(slot/"PREFLIGHT.json",preflight())
    save(slot/"INTENT.json",{"schema":"verification-submission-intent-v3","job_card_sha256":identity,
         "preflight_sha256":snapshot["sha256"],"comment":"cv3:"+identity,"epoch":time.time()})
    submitted=command(args);job_id=submitted.split(";",1)[0]
    require(re.fullmatch(r"\d+",job_id),"unexpected sbatch receipt")
    observed=command(["scontrol","show","job",job_id])
    require("JobId="+job_id in observed and "Account=iscrc_miosr" in observed,"submitted job read-back failed")
    return save(receipt_path,{"schema":"verification-submission-v3","job_card_sha256":identity,
        "job_id":job_id,"sbatch_response":submitted,"read_back":observed,"preflight_sha256":snapshot["sha256"],
        "scientific_success":"NOT_YET_MEASURED"})
