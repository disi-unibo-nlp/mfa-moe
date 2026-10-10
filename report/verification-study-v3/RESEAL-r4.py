"""One-time source repair reconciliation; no experiment submission or inference."""
from pathlib import Path
import json
import subprocess
import time

from moe_exp.verification_study.contracts import digest, load, require, save
from moe_exp.verification_study.prepare import code_binding, runtime_binding


def command(args):
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip()


def main():
    directory=Path(__file__).resolve().parent
    campaign=load(directory/"CAMPAIGN.json")
    require(campaign["pre_submission_revision"]==3,"reseal only the explicitly reconciled revision3")
    prior=load(directory/"SUBMITTED_JOBS-r3-superseded.json")
    jobs=[r["job_id"] for r in prior["jobs"]]
    queue=command(["squeue","-h","-j",",".join(jobs),"-o","%i|%T|%R"])
    require(not queue,"affected jobs have not stopped")
    accounting=command(["sacct","-nP","-j",",".join(jobs),
                        "--format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES"])
    lines=[line.split("|") for line in accounting.splitlines() if line]
    require(all(row[1].split()[0] in {"CANCELLED","COMPLETED","FAILED","TIMEOUT"} for row in lines),
            "affected job or step still active")
    tops={row[0]:row for row in lines if row[0] in jobs}
    require(set(tops)==set(jobs),"missing terminal accounting")
    old_root=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007")
    fixtures={}
    for key in campaign["models"]:
        fixtures[key]=[{"path":str(p),"sha256":load(p)["sha256"],
                        "assignment_uid":load(p)["assignment_uid"]}
                       for p in sorted((old_root/key/"qualification/fixtures").glob("*.json"))]
    save(directory/"RPC_REJECTION_RECONCILIATION-r4.json",{
        "schema":"verification-source-repair-reconciliation-v3","previous_campaign_sha256":campaign["sha256"],
        "observed_epoch":time.time(),"queue":queue,"terminal_accounting":accounting,
        "actual_allocated_billing_core_hours":sum(int(r[2])*int(r[3])/3600 for r in tops.values()),
        "partial_engineering_receipts":fixtures,"scientific_results":"NONE",
        "preserved_result_root":str(old_root),"new_result_root":str(old_root)+"-r4",
        "repair":"Reject invalid raw singleton action locally before sampler/RPC; bind generation receipts and joins to executor source",
        "scope_change":"none; seven models, nomination, controls, semantic and utility definitions unchanged"})
    save(directory/"CAMPAIGN-pre-execution-r3.json",campaign)
    require(runtime_binding()["sha256"]==campaign["runtime_sha256"],"installed runtime changed")
    updated={**campaign,"code_files":code_binding(),"pre_submission_revision":4,
             "previous_campaign_sha256":campaign["sha256"],"result_root":str(old_root)+"-r4",
             "revision_reason":"Actual r3 invalid-expert fixture raised inside worker RPC; all affected jobs stopped and partial engineering receipts preserved. Local preflight rejects malformed actions before worker access; receipts bind executor source. Scientific protocol/resources unchanged."}
    updated=save(directory/"CAMPAIGN.json",updated,replace=True)
    print(json.dumps({"revision":4,"campaign_sha256":updated["sha256"],
                      "executor_sha256":digest(updated["code_files"]),
                      "preserved_fixtures":{key:len(rows) for key,rows in fixtures.items()}},indent=2))


if __name__=="__main__":main()
