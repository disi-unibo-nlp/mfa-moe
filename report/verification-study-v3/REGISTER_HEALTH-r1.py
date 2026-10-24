"""Register one observed qualification signal and all seven discovery signals."""
import argparse
import json
from pathlib import Path

from moe_exp.verification_study.campaign import task_card
from moe_exp.verification_study.contracts import load,require,save
from verification_study_v3_health_r1 import register


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args();directory=Path(__file__).resolve().parent
    campaign_path=directory/"CAMPAIGN.json";campaign=load(campaign_path);root=Path(campaign["result_root"])
    completed=load(directory/"SUBMITTED_JOBS.json")
    qualification=next(r for r in completed["jobs"] if r["model"]=="gpt" and r["stage"]=="qualify")
    predecessors=[("gpt","qualify",Path(qualification["receipt_path"]))]
    for row in load(directory/"RESUME_STATUS.json")["models"]:
        require(row["status"]=="SUBMITTED_OR_EXISTING","first measurement not yet submitted")
        card=task_card(campaign_path,load(row["request"]))
        path=root/"submissions"/card["sha256"]/"SUBMISSION.json"
        require(load(path)["job_id"]==row["job_id"],"measurement submission differs")
        predecessors.append((row["model"],"run",path))
    results=[]
    for model,stage,path in predecessors:
        result=register(campaign_path,path,root,args.test_only)
        results.append({"model":model,"predecessor_stage":stage,"predecessor_submission_path":str(path),**result})
        print(json.dumps({"model":model,"stage":stage,"test_only":args.test_only,
                          "job_id":result.get("job_id"),"predecessor_job_id":load(path)["job_id"]}),flush=True)
        if not args.test_only:
            save(directory/"HEALTH_CALLBACKS.json",{"schema":"verification-health-callbacks-v3",
                "campaign_sha256":campaign["sha256"],"callbacks":results,"observer_version":"r1",
                "agent_continuation":"native persistent Goal; observer emits facts, Goal alone dispatches successors",
                "resources_each":"serial normal; 1CPU4G10min; no GPU or new scientific allocation in callback"},replace=True)
    if args.test_only:
        save(directory/"HEALTH_SLURM_TEST_ONLY-r1.json",{"schema":"verification-health-slurm-test-only-v3",
            "campaign_sha256":campaign["sha256"],"results":results,"actual_submissions":0},replace=True)


if __name__=="__main__":main()
