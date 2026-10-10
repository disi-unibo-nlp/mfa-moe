"""Execute one sealed, finite study task in an allocated Slurm step."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from moe_exp.verification_study.contracts import check_files, load, private_path, require

GPU_TASKS={"run","rate","calibrate","j1-audit","grade-j1"}
CPU_TASKS={"audit-frame","recognition","nominate","validate","freeze-final","outcome-frame",
           "grade-strict","assemble","terminal-source","analyze","evidence","prices"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_card",type=Path)
    parser.add_argument("--deadline-epoch",type=float)
    args=parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID"),"task requires allocated Slurm step")
    card=load(args.job_card,"verification-job-card-v3")
    campaign=load(card["campaign_path"])
    require(card["campaign_sha256"]==campaign["sha256"] and card["code_files"]==campaign["code_files"],"task source binding changed")
    check_files(card["code_files"])
    stage=card["stage"];gpu=card["resources"]["gpus"]>0
    require(stage in (GPU_TASKS if gpu else CPU_TASKS),"task not allowed on allocated resource")
    require(os.environ["SLURM_JOB_PARTITION"]==card["resources"]["partition"] and
            os.environ["SLURM_JOB_ACCOUNT"]=="iscrc_miosr","allocated resource differs")
    payload=card["payload_arguments"]
    require(isinstance(payload,list) and payload[0]==stage and all(isinstance(v,str) for v in payload),"invalid task argument vector")
    require("--campaign" in payload and Path(payload[payload.index("--campaign")+1]).resolve()==Path(card["campaign_path"]).resolve(),
            "task belongs to another campaign")
    # GPU deadlines come from the running allocation, never caller-supplied argv.
    require("--deadline-epoch" not in payload,"task cannot override allocation deadline")
    if gpu:
        require(args.deadline_epoch is not None,"finite GPU allocation deadline required")
        payload=[*payload,"--deadline-epoch",str(args.deadline_epoch)]
    script=Path(__file__).with_name("verification_study_v3.py")
    return subprocess.run([sys.executable,"-B",str(script),*payload],check=False).returncode


if __name__=="__main__":
    sys.exit(main())
