"""Replace only the exact unstarted r1 observers; preserve all science jobs."""
import json
from pathlib import Path

from moe_exp.verification_study.campaign import command
from moe_exp.verification_study.contracts import load,require,save


def main():
    directory=Path(__file__).resolve().parent
    old=load(directory/"HEALTH_CALLBACKS.json")
    require(old["observer_version"]=="r1","wrong observer revision")
    save(directory/"HEALTH_CALLBACKS-r1.json",old)
    targets=[r for r in old["callbacks"] if r["predecessor_stage"]=="run"]
    evidence={}
    for row in targets:
        job=row["job_id"];text=command(["scontrol","show","job",job])
        require("Account=iscrc_miosr" in text and "JobName=cv3-health-" in text and
                "JobState=PENDING" in text and "RunTime=00:00:00" in text,"observer is not the exact unstarted target")
        evidence[job]=text
    save(directory/"HEALTH_REPAIR_INTENT-r2.json",{"schema":"verification-health-repair-intent-v3",
        "targets":evidence,"reason":"verify capture archives, exact reader filename identity/deployment, predecessor ordering and bounded accounting command failures",
        "scientific_jobs_affected":0,"scientific_source_changes":0})
    for job in evidence:command(["scancel",job])
    queue=command(["squeue","-h","-j",",".join(evidence),"-o","%i|%T|%R"])
    require(not queue,"cancelled observer has not cleared")
    accounting=command(["sacct","-nP","-j",",".join(evidence),"--format=JobID,State,ElapsedRaw"])
    save(directory/"HEALTH_RECONCILIATION-r2.json",{"schema":"verification-health-reconciliation-v3",
        "targets":list(evidence),"queue":queue,"accounting":accounting,
        "r1_source_and_events":"PRESERVED","scientific_jobs_affected":0,"new_observer_version":"r2"})
    print(json.dumps({"cancelled_unstarted_observers":list(evidence),"science_jobs_cancelled":0},indent=2))


if __name__=="__main__":main()
