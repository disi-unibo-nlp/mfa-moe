"""Real import-order regressions in clean bounded subprocesses; no inference."""
from pathlib import Path
import subprocess
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]


def probe(code):
    return subprocess.run([sys.executable,'-B','-c',"import sys;sys.path[:0]=['src','scripts/experimental_resume'];"+code],
                          cwd=REPO,capture_output=True,text=True,timeout=60)


class EntryTests(unittest.TestCase):
    def test_real_repeated_scorer_import_after_all_shared_helpers(self):
        value=probe("import utility_j1_entry_v2 as e;j=e.install();o=j.outcome_module();a=o.scoring_modules();b=o.scoring_modules();assert a==b;assert e.install() is j;assert a[0].answer_equivalence.VOTES==3;print('PASS')")
        self.assertEqual(value.returncode,0,value.stderr)
        self.assertIn('PASS',value.stdout)

    def test_foreign_cached_namespace_is_rejected(self):
        value=probe("import moe_exp;import utility_j1_entry_v2 as e;e.install()")
        self.assertNotEqual(value.returncode,0)
        self.assertIn('foreign scoring package',value.stderr)

    def test_v2_binds_new_wrappers_but_same_frozen_scientific_adapter(self):
        value=probe("import utility_j1_entry_v2 as e;j=e.install();f=j.code_files();assert str(j.SCRIPTS/'utility_j1_chain_v1.py') in f;assert all(p.name.endswith('_v2.sbatch') for p in j.WRAPPERS.values());assert j.ENVELOPE.name=='UTILITY_J1_MAX384_ENVELOPE_v2.json';assert j.binding()['outcome_measurement_plan_sha256']=='19314ebda38fb2b51bca615e931637470639128498568b841904184657964195';print('PASS')")
        self.assertEqual(value.returncode,0,value.stderr)

    def test_preproduction_follows_exact_finite_successor_receipt(self):
        code = """
import utility_j1_entry_v2 as e
j=e.install()
import tempfile,os,contextlib,io
from pathlib import Path
from unittest import mock
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    chain=Path(temporary)/('utility-production-submissions-v1-'+'a'*16)
    chain.mkdir()
    j.shared.save(chain/'follow-pilot-generation-CHAIN.json',{'schema':'utility-production-follow-pilot-v1','config_sha256':'a'*64,'next_phase':'follow-pilot-analysis','next_job':'123'})
    with mock.patch.object(j,'validate_plan',return_value={'sha256':'p'*64}), mock.patch.object(j.verify,'live_account_snapshot'), mock.patch.object(j,'cpu_once',return_value='124') as submit, mock.patch.dict(os.environ,{'SLURM_JOB_ID':'122'}), contextlib.redirect_stdout(io.StringIO()):
        e.follow_preproduction(j,chain,None)
    assert submit.call_args.args[4]=='123'
    assert '--chain-dir' in submit.call_args.args[2]
    assert submit.call_args.args[2][0].endswith('utility_j1_attach_v2.sbatch')
print('PASS')
"""
        value=probe(code)
        self.assertEqual(value.returncode,0,value.stderr)

    def test_preproduction_missing_receipt_fails_without_submission(self):
        code="""
import utility_j1_entry_v2 as e
j=e.install()
import tempfile
from pathlib import Path
from unittest import mock
with tempfile.TemporaryDirectory(dir='tests') as temporary:
    with mock.patch.object(j,'validate_plan',return_value={'sha256':'p'*64}), mock.patch.object(j,'cpu_once') as submit:
        try:e.follow_preproduction(j,Path(temporary),None)
        except ValueError:pass
        else:raise AssertionError('expected missing receipt rejection')
        submit.assert_not_called()
print('PASS')
"""
        value=probe(code)
        self.assertEqual(value.returncode,0,value.stderr)


if __name__=='__main__':unittest.main()
