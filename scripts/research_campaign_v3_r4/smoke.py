#!/usr/bin/env python3
"""Bounded CPU repair suite with a machine-readable immutable receipt."""
import argparse
import os
from pathlib import Path
import sys
import time
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/"src"),str(REPO/"scripts/experimental_resume")]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from moe_exp.research_campaign_v3_r4.contracts import file_sha,publish,private_path,source_binding
    output=private_path(args.output)
    before=source_binding()
    start=time.monotonic()
    cpu=time.process_time()
    suite=unittest.defaultTestLoader.discover(str(REPO/"tests/research_campaign_v3_r4"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files={**before,**{str(p):file_sha(p) for p in sorted((REPO/"tests/research_campaign_v3_r4").glob("*.py"))}}
    unchanged=all(file_sha(path)==sha for path,sha in files.items())
    receipt=publish(output,{"tests_run":result.testsRun,"passed":result.wasSuccessful() and unchanged,
        "failures":[{"test":str(t),"traceback":trace} for t,trace in result.failures],
        "errors":[{"test":str(t),"traceback":trace} for t,trace in result.errors],
        "skipped":[{"test":str(t),"reason":reason} for t,reason in result.skipped],
        "seconds":time.monotonic()-start,"cpu_seconds":time.process_time()-cpu,
        "files":files,"source_unchanged_during_tests":unchanged,"gpu_work":False,
        "host":os.uname().nodename,"user":os.getuid()})
    print("CPU smoke:",receipt["tests_run"],"tests; passed=",receipt["passed"],"receipt=",output)
    sys.exit(0 if receipt["passed"] else 1)


if __name__=="__main__":main()
