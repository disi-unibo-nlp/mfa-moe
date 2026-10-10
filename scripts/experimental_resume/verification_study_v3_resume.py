"""On-demand all-seven qualification reconciliation and retained price pilots.

No daemon, inference, effect-based selection or automatic recovery allocation.
Later scientific stages follow the canonical runbook and measured job cards.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from moe_exp.verification_study.campaign import submit, task_card
from moe_exp.verification_study.contracts import file_sha, load, require, save
from moe_exp.verification_study.execution import prerequisites
from moe_exp.verification_study.registry import MODEL_KEYS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,default=Path("report/verification-study-v3/CAMPAIGN.json"))
    parser.add_argument("--root",type=Path,default=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4"))
    parser.add_argument("--submit",action="store_true")
    args=parser.parse_args();campaign_path=args.campaign.resolve();campaign=load(campaign_path)
    request_root=campaign_path.parent/f"job-requests-r{campaign['pre_submission_revision']}"
    rows=[]
    for key in MODEL_KEYS:
        manifest_path=campaign_path.parent/key/"DISCOVERY_ASSIGNMENTS.json"
        manifest=load(manifest_path);assignment=manifest["assignments"][0]
        qualification=args.root/key/"qualification/QUALIFICATION.json"
        output=args.root/key/"discovery"
        request=save(request_root/key/"DISCOVERY_PRICE_PILOT.json",{
            "schema":"verification-task-request-v3","stage":"run","model":key,"population":"discovery",
            "claim":"C2: measure this checkpoint's retained all-layer native discovery capture before producing its other47 traces",
            "artifact":f"{key} native discovery receipt, compact routing arrays and first all-layer role/context prices",
            "reuse":"existing frozen discovery family/question and exact qualified model/runtime; no historical dense recovery",
            "missing_cell":"full-cap native all-layer discovery runtime/storage cost on the actual paper executor",
            "decision_if_failure":"record execution/price non-identification; no effect claim and no automatic new recovery allocation",
            "measurement_pilot":True,"finite_assignments":1,"assignment_uid":assignment["uid"],
            "resources":{"partition":"boost_usr_prod","qos":"normal","gpus":2,"cpus":16,"memory_gb":120,
                         "walltime_seconds":28800,"billing_core_hours_max":128,
                         "price_basis":"finite first measurement: exactly one retained frozen discovery assignment, cap16384; qualify first"},
            "arguments":["--model",key,"--qualification",str(qualification),"--manifest",str(manifest_path),
                         "--output",str(output),"--root",str(args.root),"--begin","0","--count","1"],
            "prerequisites":[str(qualification)],"orchestration_file_sha256":file_sha(__file__)})
        row={"model":key,"request":str(request_root/key/"DISCOVERY_PRICE_PILOT.json")}
        if not qualification.exists():row["status"]="WAITING_FOR_ACTUAL_QUALIFICATION"
        elif load(qualification)["status"]!="PASS":row["status"]="QUALIFICATION_FAILED_OR_INCOMPLETE"
        elif (output/(assignment["uid"]+".json")).exists():row["status"]="PILOT_RECORDED_COLLECT_PRICES"
        elif list(output.glob(assignment["uid"]+".attempt-*.json")):
            row["status"]="PILOT_ATTEMPTED_RECONCILE_EXISTING_JOB_NO_AUTOMATIC_RECOVERY"
        else:
            prerequisites(campaign_path,key,qualification)
            row["status"]="READY_FIRST_RETAINED_MEASUREMENT"
            if args.submit:
                receipt=submit(campaign_path,task_card(campaign_path,request),args.root)
                row.update(status="SUBMITTED_OR_EXISTING",job_id=receipt["job_id"])
        rows.append(row)
    save(campaign_path.parent/"RESUME_STATUS.json",{"schema":"verification-resume-status-v3",
        "campaign_sha256":campaign["sha256"],"models":rows,"automatic_recovery_allocations":0},replace=True)
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
