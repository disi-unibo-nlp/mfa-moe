#!/usr/bin/env python3
"""CPU regression entrypoint using installed environments, with no GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=('measurement','guiding'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    project=root.parent
    sys.path[:0]=[str(root),str(project/'repo/src')]
    sys.path.append(str(project/'envs/correlation-client-3.11/lib/python3.11/site-packages'))
    import moe_exp
    # Native capture uses repo/src; only intervention package is worktree-owned.
    moe_exp.__path__.append(str(root/'src/moe_exp'))
    import pytest
    tests=['tests/measurement_study'] if a.group=='measurement' else ['tests/test_available_metrics.py','tests/test_routing_study.py','tests/test_routing_study_pipeline.py','tests/test_routing_study_torch.py']
    args=['-q','-p','no:cacheprovider','--basetemp='+str(root/'results/measurement-study/campaign-v3/tests'/('pytest-'+a.group+'-'+os.environ.get('SLURM_JOB_ID','local'))),*tests]
    watched=list((root/'measurement_study').glob('*.py'))+list((root/'src/moe_exp/routing_study').glob('*.py'))+list((project/'repo/src').rglob('*.py'))+[Path(__file__).resolve(),root/'scripts/analyze_available_metrics.py',root/'scripts/analyze_pilot_metrics.py']
    for test in tests:
        path=root/test
        watched.extend(path.rglob('*.py') if path.is_dir() else [path])
    def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
    before=hashes()
    code=pytest.main(args)
    after=hashes()
    if before!=after:
        print('Source changed during CPU tests; acceptance withheld',flush=True)
        code=3
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'status':'passed' if code==0 else 'failed','pytest_exit_code':int(code),'group':a.group,'job_id':os.environ.get('SLURM_JOB_ID'),'tests':tests,'source_hashes_before':before,'source_hashes_after':after})+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
