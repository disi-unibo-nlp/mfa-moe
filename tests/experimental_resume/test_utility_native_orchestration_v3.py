"""Clean-process no-submit tests for CPU readiness and true-login dispatch."""
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]


def probe(code):
    prefix = "import sys;sys.path[:0]=['scripts/experimental_resume','src'];import utility_native_orchestration_v3 as n;"
    return subprocess.run([sys.executable, '-B', '-c', prefix + code], cwd=REPO,
                          capture_output=True, text=True, timeout=60)


class NativeTests(unittest.TestCase):
    def check(self, code):
        result = probe(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_real_bootstrap_and_frozen_production_and_grading_source_closures(self):
        self.check("n.P.validate_config(n.P.U.sealed(n.P.DOC/'UTILITY_PRODUCTION_CONFIG_v1.json'));n.J1.validate_plan();print('PASS')")

    def test_login_guard_rejects_compute_and_slurm_environment_without_fabricating_it(self):
        self.check("""
from unittest import mock
import os
with mock.patch.object(n.socket,'gethostname',return_value='viz16.leonardo.local'):
    try:n.login_guard()
    except ValueError:pass
    else:raise AssertionError('compute must be rejected')
with mock.patch.object(n.socket,'gethostname',return_value='login05.leonardo.local'),mock.patch.dict(os.environ,{'SLURM_JOB_ID':'unit-test'}):
    try:n.login_guard()
    except ValueError:pass
    else:raise AssertionError('Slurm environment must be rejected')
print('PASS')
""")

    def test_cpu_price_prepares_on_compute_and_stops_at_readiness_without_any_job_submission(self):
        self.check("""
from unittest import mock
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary)
    ops={'sha256':'ops','config_sha256':'config','pilot_attachment_dir':str(root/'pilot'),'native_root':str(root)}
    def prepare(config,attachment,out):
        return n.P.save(out/'PRICE.json',{'status':'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION'}),n.P.save(out/'MANIFEST.json',{'assigned':384})
    pilot=root/'pilot';pilot.mkdir();n.P.save(pilot/'PILOT_ACCOUNTING.json',{'assigned':8})
    with mock.patch.object(n,'cpu_guard'),mock.patch.object(n.P,'DOC',root),mock.patch.object(n.P,'prepare',side_effect=prepare),mock.patch.object(n,'ready',return_value={'kind':'initial'}) as ready,mock.patch.object(n.shared,'submit') as submit,mock.patch.object(n.original,'gpu_submit') as gpu:
        result=n.run_cpu(ops,{'sha256':'config','dispatch_on_pass':True},root/'OPERATIONS.json','price')
        assert result['kind']=='initial'
        assert ready.call_args.args[1]=='initial'
        assert set(ready.call_args.args[2])=={'price','manifest','pilot_accounting'}
        submit.assert_not_called();gpu.assert_not_called()
print('PASS')
""")

    def test_readiness_needs_exact_successful_producer_and_unchanged_artifacts(self):
        self.check("""
from unittest import mock
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary).resolve()
    ops={'sha256':'ops','config_sha256':'config','pilot_attachment_dir':'pilot','native_root':str(root)}
    artifact=root/'PRICE.json';n.P.save(artifact,{'tokens':10})
    path=root/'INITIAL_READY.json'
    n.P.save(path,{'schema':'utility-native-readiness-v3',**n.binding(ops),'kind':'initial','status':'READY_FOR_NATIVE_LOGIN_DISPATCH','producer_job_id':'123','producer_hostname':'viz16.leonardo.local','artifacts':{'price':n.artifact(artifact)}})
    with mock.patch.object(n,'terminal_job',side_effect=ValueError('failed producer')):
        try:n.validate_ready(ops,path)
        except ValueError:pass
        else:raise AssertionError('failed producer accepted')
    with mock.patch.object(n,'terminal_job',return_value={'accounting':'123|COMPLETED|0:0'}):
        n.validate_ready(ops,path)
        artifact.unlink();n.P.save(artifact,{'tokens':11})
        try:n.validate_ready(ops,path)
        except ValueError:pass
        else:raise AssertionError('changed price accepted')
print('PASS')
""")

    def test_native_live_balance_counts_compressed_project_arrays_and_holds_before_submission(self):
        self.check("""
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile
def run(command,**kwargs):
    if command[0]=='/cineca/bin/saldo':return SimpleNamespace(stdout='IscrC_MIOSR 20260504 20270204 68000 21350 21350 31.4 7391 1251\\n')
    if command[0]=='squeue':return SimpleNamespace(stdout='9_[0-3%2]|PENDING|2:00:00|2:00:00|16|gres/gpu:2|1\\n')
    if command[0]=='sacctmgr':return SimpleNamespace(stdout='lmolfett|iscrc_miosr||normal|')
    return SimpleNamespace(stdout='PartitionName='+command[-1])
with tempfile.TemporaryDirectory(dir='tests') as temporary,mock.patch.object(n,'login_guard'),mock.patch.object(n.subprocess,'run',side_effect=run),mock.patch.object(n.shared,'submit') as submit:
    value=n.live_preflight(Path(temporary),'initial',17984)
    assert value['budget']['active_remaining_commitment_billing_core_hours']==128
    try:n.live_preflight(Path(temporary),'initial',50000)
    except ValueError:pass
    else:raise AssertionError('overspending accepted')
    submit.assert_not_called()
print('PASS')
""")

    def test_native_initial_uses_frozen_gpu_worker_and_afterany_actual_cpu_accounting(self):
        self.check("""
from unittest import mock
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary).resolve();chain=root/'chain';chain.mkdir()
    ops={'sha256':'ops','config_sha256':'config','pilot_attachment_dir':'pilot','native_root':str(root),'production_chain_dir':str(chain),'config_path':'frozen-config'}
    price=n.P.save(root/'PRICE.json',{'status':'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION','shards':[{'index':0}],'offline_measurement':{}})
    manifest=n.P.save(root/'MANIFEST.json',{'config_sha256':'config','price_sha256':price['sha256'],'shards':price['shards']})
    receipt={'kind':'initial','sha256':'ready','artifacts':{'price':n.artifact(root/'PRICE.json'),'manifest':n.artifact(root/'MANIFEST.json')}}
    with mock.patch.object(n,'login_guard'),mock.patch.object(n.P,'validate_manifest'),mock.patch.object(n.original,'gpu_submit',return_value='201') as gpu,mock.patch.object(n,'cpu_submit',return_value='202') as cpu:
        result=n.dispatch_native(ops,{'sha256':'config','pilot_proposal_sha256':'proposal'},root/'OPERATIONS.json',receipt,{'sha256':'live1'})
        assert result['array_job']=='201' and result['account_job']=='202'
        assert gpu.call_args.args[1]=='initial' and gpu.call_args.args[-1]==[0]
        assert gpu.call_args.args[4]['UTILITY_PRODUCTION_MANIFEST']==str(root/'MANIFEST.json')
        assert cpu.call_args.args[4]==['--dependency=afterany:201']
        assert 'live_preflight_sha256' not in gpu.call_args.args[5]
        gpu.reset_mock();cpu.reset_mock()
        assert n.dispatch_native(ops,{'sha256':'config'},root/'OPERATIONS.json',receipt,{'sha256':'live2'})==result
        gpu.assert_not_called();cpu.assert_not_called()
print('PASS')
""")

    def test_cpu_accounting_publishes_whole_bounded_recovery_and_never_submits_gpu(self):
        self.check("""
from unittest import mock
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary).resolve();chain=root/'chain';chain.mkdir();output=root/'output';output.mkdir()
    ops={'sha256':'ops','config_sha256':'config','pilot_attachment_dir':'pilot','native_root':str(root),'production_chain_dir':str(chain)}
    manifest=n.P.save(root/'MANIFEST.json',{'shards':[{'index':0}]})
    n.P.save(root/'PRICE.json',{'infrastructure_recovery_reserve_gpu_hour_ceiling':12})
    n.P.save(chain/'INITIAL_CHAIN.json',{'native_operations_sha256':'ops','config_sha256':'config','manifest_path':str(root/'MANIFEST.json'),'output':str(output),'array_job':'201'})
    index=n.P.save(output/'RECONCILED_INDEX_INITIAL.json',{'missing':8,'generation_errors':0,'total_generation_allocated_gpu_hours':2})
    recovery={'wave':1,'manifest_sha256':manifest['sha256'],'shards':{'0':{}},'assigned':8,'allocation_gpu_hour_ceiling':8}
    with mock.patch.object(n,'cpu_guard'),mock.patch.object(n.P,'validate_manifest'),mock.patch.object(n.P,'allocation_accounting',return_value=[]),mock.patch.object(n.P,'reconcile',return_value=index),mock.patch.object(n.P,'recovery_plan',return_value=recovery),mock.patch.object(n,'ready',side_effect=lambda operations,kind,artifacts:{'kind':kind}) as ready,mock.patch.object(n.original,'gpu_submit') as gpu,mock.patch.object(n.shared,'submit') as submit:
        result=n.run_cpu(ops,{'sha256':'config'},root/'OPERATIONS.json','account-initial')
        assert result['kind']=='recovery';assert ready.call_args.args[1]=='recovery'
        assert n.P.U.sealed(output/'RECOVERY.json')['assigned']==8
        gpu.assert_not_called();submit.assert_not_called()
        (root/'PRICE.json').unlink();n.P.save(root/'PRICE.json',{'infrastructure_recovery_reserve_gpu_hour_ceiling':4})
        result=n.run_cpu(ops,{'sha256':'config'},root/'OPERATIONS.json','account-initial')
        assert result['kind']=='final'
        final=n.P.U.sealed(chain/'FINAL_CHAIN.json')
        assert final['missing']==8 and final['offline_measurement_status']=='INCOMPLETE_ITT_WITH_EXPLICIT_UNKNOWN_ENDPOINTS'
        assert n.P.U.sealed(root/'RECOVERY_RESERVE_HOLD.json')['gpu_submitted'] is False
        gpu.assert_not_called();submit.assert_not_called()
print('PASS')
""")

    def test_partial_submission_budget_keeps_reserve_without_counting_queued_array_twice(self):
        self.check("""
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary);chain=root/'chain';chain.mkdir()
    ops={'sha256':'ops','production_chain_dir':str(chain)}
    config={'offline_grading_reserve_gpu_hours':216}
    price=n.P.save(root/'PRICE.json',{'generation_billing_core_hour_ceiling':16000,'infrastructure_recovery_reserve_gpu_hour_ceiling':10})
    receipt={'kind':'initial','sha256':'ready','artifacts':{'price':n.artifact(root/'PRICE.json')}}
    assert n.required_budget(ops,config,receipt)==17728
    n.P.save(chain/'initial.json',{'binding':{'native_operations_sha256':'ops','readiness_sha256':'ready'},'job_id':'201'})
    assert n.required_budget(ops,config,receipt)==1808
print('PASS')
""")

    def test_native_j1_uses_frozen_legacy_worker_and_actual_cpu_finalizer(self):
        self.check("""
from unittest import mock
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    root=Path(temporary).resolve();grade=root/'grade';grade.mkdir()
    ops={'sha256':'ops','config_sha256':'config','pilot_attachment_dir':'pilot','native_root':str(root),'production_chain_dir':str(root/'chain')}
    shard={'index':0,'items_path':'exact-items','verdicts_path':'exact-verdicts','items_file_sha256':'items-sha','item_ids':['item']}
    values=({'sha256':'j1-plan'},{'sha256':'prep'},{'sha256':'price','shards':[shard]},[])
    calls=[]
    def submit(*args):calls.append(args);return str(300+len(calls))
    with mock.patch.object(n,'login_guard'),mock.patch.object(n.J1,'priced',return_value=values),mock.patch.object(n.J1,'safe_environment',return_value={}),mock.patch.object(n.J1,'sealed',return_value={'tree_sha256':'tree'}),mock.patch.object(n.J1.verify,'ensure_verified'),mock.patch.object(n.J1.dependencies,'dependency',return_value=(['--dependency=afterok:301'],{'active_predecessor':'301'})),mock.patch.object(n.shared,'submit',side_effect=submit):
        result=n.dispatch_native(ops,{},root/'OPERATIONS.json',{'kind':'j1','sha256':'ready','grade_root':str(grade),'artifacts':{}},{'sha256':'live'})
    assert result['J1_jobs']==['301'] and result['finalize_job']=='302'
    assert calls[0][2]==['--time=06:00:00','--job-name=utility-j1-000',str(n.J1.LEGACY),'exact-items','exact-verdicts']
    assert calls[1][2][0]=='--dependency=afterok:301'
    assert calls[1][2][1]==str(n.J1.WRAPPERS['finalize'])
print('PASS')
""")


if __name__ == '__main__':
    unittest.main()
