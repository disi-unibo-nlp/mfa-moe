"""Retest the isolated measured storage envelope after the full Slurm suite."""
import sys,time,unittest,os
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r4.contracts import source_binding,load,publish,require,file_sha
full_path=REPO/'report/research-campaign-v3-r4-validation-r4/READINESS.json'
full=load(full_path);require(full['passed'],'full regression must pass first')
before=source_binding()
changed=sorted(p for p in set(before)|set(full['files']) if before.get(p)!=full['files'].get(p))
expected=sorted(str(REPO/p) for p in ('src/moe_exp/research_campaign_v3_r4/storage.py','tests/research_campaign_v3_r4/test_results_first.py'))
require(changed==expected,'supplemental validation encountered an unrelated source change')
started=time.monotonic()
suite=unittest.defaultTestLoader.discover(str(REPO/'tests/research_campaign_v3_r4'),pattern='test_results_first.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
unchanged=before==source_binding()
receipt=publish(Path(__file__).with_name('READINESS.json'),{'passed':result.wasSuccessful() and unchanged,
    'tests_run':result.testsRun,'seconds':time.monotonic()-started,'files':before,
    'source_unchanged_during_tests':unchanged,'full_regression_receipt':str(full_path),
    'full_regression_sha256_file':file_sha(full_path),'full_regression_tests_run':full['tests_run'],
    'changed_files_since_full_regression':changed,'scientific_sources_changed':False,
    'full_regression_plus_focused_operational_revalidation':True,
    'errors':[{'test':str(t),'traceback':trace} for t,trace in result.errors],
    'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures],
    'job_id':os.environ['SLURM_JOB_ID']})
print('Full suite plus focused storage validation:',receipt['passed'])
sys.exit(0 if receipt['passed'] else 1)
