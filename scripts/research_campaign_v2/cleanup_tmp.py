"""Remove only this completed allocation's private, non-scientific caches."""
import os
from pathlib import Path
import re
import shutil


def main():
    jid=os.environ["SLURM_JOB_ID"]
    if not re.fullmatch(r"\d+",jid):raise ValueError("invalid allocation ID")
    staging=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
    path=staging/("research-campaign-v2-"+jid)
    if path.is_symlink() or path.resolve().parent!=staging.resolve():raise ValueError("unsafe cache path")
    if path.exists():shutil.rmtree(path)


if __name__=="__main__":main()
