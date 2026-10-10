"""One bounded continuation of the shared, original100-item J1 audit.

No inference or daemon runs here. Actual health evidence and immutable prices
precede finite Slurm cards; original items, votes and audit gates are unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time

from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
from moe_exp.verification_study.pricing import shard_budget
from moe_exp.verification_study.registry import REPO
import verification_study_v3_extension_r1 as extension
import verification_study_v3_health_r4 as health
import verification_study_v3_j1_r1 as j1

TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


def work(count):
    require(type(count) is int and count>0,"finite nonempty J1 slice required")
    # One complete8192-cap three-vote item and a cold load are reserved for recovery.
    return {"j1_load":2,"j1_generation":3*8192*(count+1)}


def priced_size(prices,remaining,target_hours=2):
    require(type(remaining) is int and remaining>0 and 0<target_hours<=24,"invalid J1 shard bounds")
    best=None
    for count in range(1,remaining+1):
        try:budget=shard_budget(prices,work(count),hours=target_hours)
        except ValueError:break
        best=(count,budget)
    if best is None:
        budget=shard_budget(prices,work(1),hours=24)
        best=(1,budget)
    count,budget=best
    seconds=60*math.ceil(budget["reservation_seconds"]/60)
    require(seconds<=24*3600,"single J1 item exceeds finite production partition")
    return count,seconds,shard_budget(prices,work(count),hours=seconds/3600)


def uncovered_blocks(records,completed,covered):
    uids=[r["uid"] for r in records]
    require(set(completed)|set(covered)<=set(uids),"foreign completed or submitted J1 identity")
    blocks=[];begin=None
    for index,uid in enumerate(uids):
        available=uid not in completed and uid not in covered
        if available and begin is None:begin=index
        if begin is not None and (not available or index==len(uids)-1):
            stop=index if not available else index+1
            blocks.append((begin,stop-begin));begin=None
    return blocks


@contextmanager
def stage_lock(root):
    path=root/"J1/orchestration.lock";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def observed_jobs(campaign,root):
    jobs=[];unresolved=[]
    for path in (root/"submissions").glob("*/JOB_CARD.json"):
        card=load(path);kind=card.get("extension",{}).get("kind")
        if kind not in {"j1-audit-slice","j1-audit-closeout"}:continue
        require(card["campaign_sha256"]==campaign["sha256"] and card["j1_executor"]==j1.binding(campaign),"foreign J1 campaign/source card")
        receipt_path=path.parent/"SUBMISSION.json"
        if not receipt_path.exists():
            if (path.parent/"INTENT.json").exists():unresolved.append(str(path.parent/"INTENT.json"))
            continue
        receipt=load(receipt_path)
        require(receipt["job_card_sha256"]==card["sha256"],"foreign J1 submission")
        jobs.append({"card":card,"receipt":receipt,"submission_path":str(receipt_path)})
    ids=sorted({r["receipt"]["job_id"] for r in jobs},key=int);states={}
    if ids:
        result=subprocess.run(["sacct","-nP","-j",",".join(ids),"--format=JobIDRaw,State,ExitCode"],
            check=True,capture_output=True,text=True,timeout=20)
        for line in result.stdout.splitlines():
            fields=line.split("|")
            if fields[0] in ids:states[fields[0]]={"state":fields[1].split(" ",1)[0],"exit_code":fields[2]}
        result=subprocess.run(["squeue","-h","-u","lmolfett","-o","%i|%T"],
            check=True,capture_output=True,text=True,timeout=20)
        live={r.split("|",1)[0]:r.split("|",1)[1] for r in result.stdout.splitlines()}
        for row in jobs:
            job=row["receipt"]["job_id"];row["accounting"]=states.get(job)
            row["live_state"]=live.get(job);row["event"]=None
            for path in (root/"health/submissions").glob("*/HEALTH_CARD.json"):
                card=load(path)
                if card["predecessor_job_id"]!=job or card["job_card_sha256"]!=row["card"]["sha256"]:continue
                event_path=root/"health/events"/(card["sha256"]+".json")
                if not event_path.exists():continue
                event=load(event_path);submission=load(path.parent/"SUBMISSION.json")
                require(event["campaign_sha256"]==campaign["sha256"] and event["health_card_sha256"]==card["sha256"] and
                        event["predecessor_job_id"]==job and event["callback_job_id"]==submission["job_id"] and
                        event["job_card_sha256"]==row["card"]["sha256"],"foreign J1 health event")
                check_files(card["observer_files"])
                require(row["event"] is None,"ambiguous J1 completion events")
                row["event"]=event;row["event_path"]=str(event_path)
    return jobs,unresolved


def continue_audit(campaign_path,submit=False,target_hours=2):
    campaign=load(campaign_path);check_files(campaign["code_files"]);root=Path(campaign["result_root"])
    frame_path=root/"J1/AUDIT_FRAME.json"
    if not frame_path.exists():return {"status":"WAITING_FOR_AUDIT_FRAME","cards":[]}
    campaign,contract,frame,executor=j1.deployment(campaign_path,frame_path)
    output=root/"J1/audit";completed={}
    for item in frame["records"]:
        path=output/(item["uid"]+".json")
        if path.exists():completed[item["uid"]]=j1.validate_verdict(path,item,campaign,contract,frame,executor)
    jobs,unresolved=observed_jobs(campaign,root)
    if unresolved:return {"status":"UNRESOLVED_SUBMISSION_INTENT","intents":unresolved,"cards":[]}
    covered=set();price_events=[];waiting=[];failures=[]
    for row in jobs:
        card=row["card"];job=row["receipt"]["job_id"];kind=card["extension"]["kind"]
        state=(row["accounting"] or {}).get("state");event=row["event"]
        if row["live_state"] or state not in TERMINAL:
            if kind=="j1-audit-slice":covered.update(card["extension"]["selected_uids"])
            waiting.append(job);continue
        if state!="COMPLETED" or row["accounting"]["exit_code"]!="0:0":failures.append(job);continue
        if event is None:
            if kind=="j1-audit-slice":covered.update(card["extension"]["selected_uids"])
            waiting.append(job);continue
        if event["state"]!="STEP_COMPLETE":failures.append(job);continue
        for artifact in event["committed_artifacts"]:
            require(load(artifact["path"])["sha256"]==artifact["sha256"],"J1 health artifact changed")
        if kind=="j1-audit-slice":
            require(set(card["extension"]["selected_uids"])<=set(completed),"completed J1 job lost a verdict")
            if "price_path" in event:price_events.append(event)
    if failures:return {"status":"RECONCILE_TERMINAL_EXECUTION_OR_CALLBACK","jobs":failures,"cards":[]}
    attempts_requiring_reconciliation=[]
    known={row["receipt"]["job_id"]:set(row["card"]["extension"].get("selected_uids",[])) for row in jobs}
    for item in frame["records"]:
        uid=item["uid"]
        if uid in completed or uid in covered:continue
        paths=list(output.glob(uid+".attempt-*.json"))
        if len(paths)>=2:attempts_requiring_reconciliation.extend(str(p) for p in paths);continue
        for path in paths:
            attempt=load(path,"verification-j1-attempt-v3")
            if uid not in known.get(attempt["job_id"],set()):attempts_requiring_reconciliation.append(str(path))
    if attempts_requiring_reconciliation:
        return {"status":"RECONCILE_RETRY_BUDGET_OR_UNTRACKED_ITEM_ATTEMPTS","attempt_paths":attempts_requiring_reconciliation,"cards":[]}
    if not price_events:return {"status":"WAITING_FOR_ACTUAL_J1_PRICE_EVENT","live_or_signal_jobs":waiting,"cards":[]}
    latest=max(price_events,key=lambda r:r["observed_epoch"]);prices=load(latest["price_path"],"verification-prices-v3")
    require(prices["sha256"]==latest["price_sha256"] and prices["code_files"]==campaign["code_files"] and
            prices["model_sha256"]==load(campaign["models"]["gpt"])["sha256"] and prices.get("j1_executor")==executor,
            "J1 price deployment/model/source differs")
    for role in ("j1_load","j1_generation","j1_item"):
        require(prices["components"][role]["status"]=="MEASURED" and prices["components"][role]["samples"],"unmeasured J1 role")
    requests=[];request_root=campaign_path.parent/"job-requests-r4/J1"
    for begin,length in uncovered_blocks(frame["records"],completed,covered):
        while length:
            count,seconds,budget=priced_size(prices,length,target_hours)
            p={"frame":str(frame_path),"output":str(output),"begin":begin,"count":count};arguments=[]
            for name,value in p.items():arguments.extend(("--"+name,str(value)))
            request=save(request_root/f"AUDIT_{begin:03d}_{count:03d}_{prices['sha256'][:12]}.json",{
                "schema":"verification-task-request-v3","stage":"j1-audit","model":"gpt","population":"frozen_J1_audit",
                "extension":{"kind":"j1-audit-slice","parameters":p},"arguments":arguments,
                "claim":"C4/C5 all-seven: qualify the unchanged common offline grader against the original100-item audit",
                "artifact":"retained original audit verdicts with three fixed draws, exact invocation prices and eventual full100 deployment qualification",
                "reuse":"original full audit frame, original UID seeds, pinned r4 Reader.grade; completed verdicts and live disjoint slices reused",
                "missing_cell":"remaining original shared J1 audit items; no final residual grading until complete exact-deployment PASS",
                "decision_if_failure":"retain committed items, reconcile exact job and attempts, keep incomplete/unqualified accuracy explicit; never change labels or thresholds",
                "measurement_pilot":False,"finite_assignments":count,"price_path":latest["price_path"],"priced_work":work(count),
                "price_event_sha256":latest["sha256"],"orchestration_file_sha256":file_sha(__file__),"prerequisites":[str(frame_path)],
                "resources":{"partition":"boost_usr_prod","qos":"normal","gpus":2,"cpus":16,"memory_gb":120,
                    "walltime_seconds":seconds,"billing_core_hours_max":16*seconds/3600,
                    "price_basis":"actual shared J1 load and prefill-inclusive token prices; full3x8192 cap/item, one full-item/cold-load recovery,1.25 factor plus600seconds"}})
            card=extension.compile_card(campaign_path,request)
            path=request_root/f"AUDIT_{begin:03d}_{count:03d}_{prices['sha256'][:12]}.json"
            entry={"request_path":str(path),"card_sha256":card["sha256"],"begin":begin,"count":count,"budget":budget}
            if submit:
                extension.submit(campaign_path,card,test_only=True)
                receipt=extension.submit(campaign_path,card)
                slot=root/"submissions"/card["sha256"]/"SUBMISSION.json"
                health.register(campaign_path,slot,root,test_only=True)
                callback=health.register(campaign_path,slot,root)
                entry.update(job_id=receipt["job_id"],callback_job_id=callback["job_id"])
            requests.append(entry);begin+=count;length-=count
    return {"status":"SUBMITTED_FINITE_AUDIT_SLICES" if submit and requests else "PRICED_AUDIT_CARDS_READY" if requests else
            "WAITING_FOR_DISJOINT_AUDIT_SLICES" if waiting else "ALL100_VERDICTS_READY_FOR_CPU_QUALIFICATION",
            "completed":len(completed),"covered_by_live_or_signal":len(covered),"live_or_signal_jobs":waiting,
            "price_path":latest["price_path"],"price_sha256":prices["sha256"],"cards":requests}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,default=REPO/"report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--submit",action="store_true");parser.add_argument("--target-hours",type=float,default=2)
    args=parser.parse_args();path=args.campaign.resolve();campaign=load(path);root=Path(campaign["result_root"])
    with stage_lock(root):result=continue_audit(path,args.submit,args.target_hours)
    receipt=save(path.parent/"J1_CONTINUATION_STATUS-r1.json",{"schema":"verification-j1-continuation-status-v3",
        "campaign_sha256":campaign["sha256"],"observed_epoch":time.time(),"orchestration_file_sha256":file_sha(__file__),
        "scope":"common unchanged grader audit serves all seven generators",**result},replace=True)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
