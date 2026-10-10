"""One bounded campaign-health observation; no daemon or scientific dispatch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from moe_exp.verification_study.contracts import check_files,load,require,save
from moe_exp.verification_study.registry import REPO

TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


def observe(campaign_path):
    campaign=load(campaign_path);check_files(campaign["code_files"]);root=Path(campaign["result_root"])
    rows=[];unresolved=[];superseded=set()
    for path in campaign_path.parent.glob("HEALTH_RECONCILIATION-*.json"):
        superseded.update(str(j) for j in load(path)["targets"])
    for path in sorted((root/"health/submissions").glob("*/HEALTH_CARD.json")):
        card=load(path,"verification-health-card-v3")
        require(card["campaign_sha256"]==campaign["sha256"],"health card belongs to another campaign")
        receipt_path=path.parent/"SUBMISSION.json"
        if not receipt_path.exists():
            if (path.parent/"INTENT.json").exists():unresolved.append(str(path.parent/"INTENT.json"))
            continue
        receipt=load(receipt_path)
        require(receipt["health_card_sha256"]==card["sha256"],"foreign health submission")
        rows.append({"health_card_sha256":card["sha256"],"model":card["expected_work"]["model"],
            "stage":card["expected_work"]["stage"],"predecessor_job_id":card["predecessor_job_id"],
            "callback_job_id":receipt["job_id"],"submission_path":str(receipt_path)})
    jobs=sorted({j for row in rows for j in (row["predecessor_job_id"],row["callback_job_id"])},key=int)
    accounting={};queue={}
    if jobs:
        result=subprocess.run(["sacct","-nP","-j",",".join(jobs),"--format=JobIDRaw,State,ExitCode,ElapsedRaw"],
            check=True,capture_output=True,text=True,timeout=20)
        for line in result.stdout.splitlines():
            fields=line.split("|")
            if fields[0] in jobs:accounting[fields[0]]={"state":fields[1],"exit_code":fields[2],"elapsed_seconds":int(fields[3])}
        result=subprocess.run(["squeue","-h","-u","lmolfett","-o","%i|%T|%j|%R"],
            check=True,capture_output=True,text=True,timeout=20)
        for line in result.stdout.splitlines():
            fields=line.split("|",3)
            if fields[0] in jobs:queue[fields[0]]={"state":fields[1],"name":fields[2],"reason":fields[3]}
    for row in rows:
        callback=row["callback_job_id"];predecessor=row["predecessor_job_id"]
        row["callback_accounting"]=accounting.get(callback);row["predecessor_accounting"]=accounting.get(predecessor)
        row["callback_queue"]=queue.get(callback);row["predecessor_queue"]=queue.get(predecessor)
        event_path=root/"health/events"/(row["health_card_sha256"]+".json")
        if event_path.exists():
            event=load(event_path,"verification-health-event-v3")
            require(event["health_card_sha256"]==row["health_card_sha256"] and event["callback_job_id"]==callback and
                    event["predecessor_job_id"]==predecessor and event["campaign_sha256"]==campaign["sha256"],"foreign health event")
            row.update(signal=event["state"],event_path=str(event_path),event_sha256=event["sha256"])
            if "price_path" in event:row.update(price_path=event["price_path"],price_sha256=event["price_sha256"])
        elif callback in superseded:row["signal"]="SUPERSEDED_UNSTARTED_OBSERVER"
        elif accounting.get(callback,{}).get("state","").split(" ",1)[0] in TERMINAL:
            failure_path=root/"health/agent-events"/("callback-no-event-"+callback+".json")
            failure=load(failure_path) if failure_path.exists() else save(failure_path,{
                "schema":"verification-health-observer-failure-v3","state":"CALLBACK_FAILED_NO_EVENT",
                "campaign_sha256":campaign["sha256"],"health_card_sha256":row["health_card_sha256"],
                "callback_job_id":callback,"predecessor_job_id":predecessor,"accounting":accounting[callback],
                "observed_epoch":time.time(),"scientific_success":"UNKNOWN_FROM_CALLBACK_SILENCE"})
            row.update(signal="CALLBACK_FAILED_NO_EVENT",failure_event_path=str(failure_path),failure_event_sha256=failure["sha256"])
        elif predecessor in queue:row["signal"]="WAITING_FOR_PREDECESSOR"
        elif callback in queue:row["signal"]="SIGNAL_PENDING"
        else:row["signal"]="ACCOUNTING_OR_SIGNAL_UNAVAILABLE"
    return save(campaign_path.parent/"HEALTH_STATUS.json",{"schema":"verification-campaign-health-status-v3",
        "campaign_sha256":campaign["sha256"],"observed_epoch":time.time(),"callbacks":rows,
        "unresolved_submission_intents":unresolved,"scientific_dispatch":"persistent Goal only",
        "automatic_submissions":0},replace=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,default=REPO/"report/verification-study-v3/CAMPAIGN.json")
    args=parser.parse_args();result=observe(args.campaign.resolve())
    print(json.dumps({"observed_epoch":result["observed_epoch"],"unresolved_submission_intents":result["unresolved_submission_intents"],
        "callbacks":[{k:r[k] for k in ("model","stage","predecessor_job_id","callback_job_id","signal")} for r in result["callbacks"]]},indent=2))


if __name__=="__main__":main()
