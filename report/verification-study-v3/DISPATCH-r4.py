"""Finite adopted engineering wave with immutable, read-back Slurm receipts."""
import argparse
import json
from pathlib import Path

from moe_exp.verification_study.campaign import command, evidence_card, qualification_card, submit
from moe_exp.verification_study.contracts import check_files, load, require, save
from moe_exp.verification_study.registry import MODEL_KEYS


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    directory=Path(__file__).resolve().parent;campaign_path=directory/"CAMPAIGN.json"
    campaign=load(campaign_path);require(campaign["pre_submission_revision"]==4,"wrong repair binding")
    check_files(campaign["code_files"]);root=Path(campaign["result_root"])
    rows=[];tests=[];ids=[]
    cards=[qualification_card(campaign_path,key) for key in MODEL_KEYS]+[evidence_card(campaign_path)]
    for card in cards:
        receipt=submit(campaign_path,card,root,test_only=args.test_only,
                       dependencies=ids if card["stage"]=="evidence" and not args.test_only else ())
        if args.test_only:
            tests.append({"model":card["model"],**receipt})
            print(json.dumps({"model":card["model"],"test_only":"PASS"}),flush=True)
            continue
        job=receipt["job_id"]
        if card["stage"]=="qualify":ids.append(job)
        rows.append({"model":card["model"],"stage":card["stage"],"job_id":job,
                     "gpus":card["resources"]["gpus"],"walltime_seconds":card["resources"]["walltime_seconds"],
                     "job_card_sha256":card["sha256"],"submission_sha256":receipt["sha256"],
                     "receipt_path":str(root/"submissions"/card["sha256"]/"SUBMISSION.json"),
                     "observed_job":command(["scontrol","show","job",job]),
                     "current_queue":command(["squeue","-h","-j",job,"-o","%i|%T|%j|%R"])})
        save(directory/"SUBMITTED_JOBS.json",{"schema":"verification-submitted-jobs-v3",
            "campaign_sha256":campaign["sha256"],"result_root":str(root),"jobs":rows,
            "scientific_results":"NOT_MEASURED","qualification_billing_core_hours_max":224},replace=True)
        print(json.dumps({"model":card["model"],"job_id":job,"queue":rows[-1]["current_queue"]}),flush=True)
    if args.test_only:
        record={"schema":"verification-slurm-test-only-v3","campaign_sha256":campaign["sha256"],
                "shapes":tests,"actual_submissions":0}
        save(directory/"SLURM_TEST_ONLY-r4.json",record)
        save(directory/"SLURM_TEST_ONLY.json",record,replace=True)
    else:
        save(directory/"RPC_REJECTION_REPLACEMENT_RECEIPT-r4.json",{
            "schema":"verification-source-repair-replacement-v3","campaign_sha256":campaign["sha256"],
            "new_jobs":[{"model":row["model"],"job_id":row["job_id"]} for row in rows],
            "old_jobs":"RPC_REJECTION_RECONCILIATION-r4.json","source_frozen_while_pending":True,
            "scientific_scope_change":False,"new_result_root":str(root)})


if __name__=="__main__":main()
