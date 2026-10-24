#!/usr/bin/env python3
"""Small CPU verification only. Never loads a checkpoint or submits a job."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base-temp",type=Path,required=True)
    parser.add_argument("--without-tensors",action="store_true")
    args=parser.parse_args()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1"
    from moe_exp.routing_study.common import code_hashes,private_path,seal,write_json,file_hash
    import pytest
    root=Path(__file__).resolve().parents[1]
    tests=["tests/test_routing_study.py","tests/test_routing_study_pipeline.py"]
    if not args.without_tensors:
        tests.append("tests/test_routing_study_torch.py")
    class Coverage:
        passed=0
        skipped=[]
        def pytest_runtest_logreport(self,report):
            if report.skipped or getattr(report,"wasxfail",False):
                self.skipped.append(report.nodeid)
            elif report.when=="call" and report.passed:
                self.passed+=1
        def pytest_collectreport(self,report):
            if report.skipped:
                self.skipped.append(report.nodeid)
    coverage=Coverage()
    began=time.perf_counter()
    flags=["-q","-p","no:cacheprovider","--basetemp="+str(private_path(args.base_temp)),
           "--junitxml="+str(private_path(args.output.with_suffix(".xml")))]
    code=code_hashes()
    status=int(pytest.main(flags+[str(root/t) for t in tests],plugins=[coverage]))
    if code!=code_hashes() or coverage.skipped or not coverage.passed:
        status=1
    result=seal(dict(status="passed" if status==0 else "failed",exit_code=status,code=code,
        passed=coverage.passed,skipped=coverage.skipped,tensor_tests=not args.without_tensors,
        test_files={t:file_hash(root/t) for t in tests},elapsed_seconds=time.perf_counter()-began,
        scope="small CPU fixtures only; no real model/GPU validation",
        job_id=os.environ.get("SLURM_JOB_ID")))
    write_json(args.output,result)
    return status


if __name__=="__main__":
    raise SystemExit(main())
