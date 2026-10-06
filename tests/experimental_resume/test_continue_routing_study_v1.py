"""Finite continuation never duplicates dispatch or depends on aged jobs."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import continue_routing_study_v1 as R
C = R.C


class ContinuationTests(unittest.TestCase):
    def fixture(self, root, j1=True):
        utility, grade = root / 'utility', root / 'grade'
        utility.mkdir(); grade.mkdir()
        closeout = {'sha256': 'closeout', 'fresh_chain': str(root / 'fresh.json'), 'dense_chain': str(root / 'dense.json')}
        C.save(root / 'fresh.json', {'analysis_job': '1'})
        C.save(root / 'dense.json', {'analysis_job': '2'})
        C.save(root / 'CONTINUATION.json', {'code_files': {}, 'closeout_plan_sha256': 'closeout', 'native_operations_sha256': 'ops'})
        if j1:
            ready = C.save(utility / 'J1_READY.json', {'grade_root': str(grade)})
            chain = C.save(grade / 'dispatch/CHAIN.json', {'readiness_sha256': ready['sha256'], 'native_operations_sha256': 'ops', 'finalize_job': '3'})
            C.save(utility / 'J1_NATIVE_DISPATCH.json', {'readiness_sha256': ready['sha256'], 'result_sha256': chain['sha256']})
        n = SimpleNamespace(login_guard=lambda: None, validate_operations=lambda p: ({'sha256': 'ops', 'native_root': str(utility)}, {}))
        return closeout, n

    def patches(self, root, closeout, n, stack):
        stack.enter_context(mock.patch.object(C, 'OUT', root))
        stack.enter_context(mock.patch.object(R, 'PLAN', root / 'CONTINUATION.json'))
        stack.enter_context(mock.patch.object(C, 'native', return_value=n))
        stack.enter_context(mock.patch.object(C, 'validate_plan', return_value=closeout))
        return stack.enter_context(mock.patch.object(C, 'advance'))

    def test_no_readiness_is_one_finite_step_without_final_submission(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp, ExitStack() as stack:
            root = Path(temp); closeout, n = self.fixture(root, j1=False)
            advance = self.patches(root, closeout, n, stack)
            submit = stack.enter_context(mock.patch.object(C, 'submit_checkpoint'))
            R.step(True)
            advance.assert_called_once_with(True)
            submit.assert_not_called()

    def test_completed_aged_jobs_are_omitted_from_final_dependencies(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp, ExitStack() as stack:
            root = Path(temp); closeout, n = self.fixture(root)
            self.patches(root, closeout, n, stack)
            rows = {job: {'state': 'COMPLETED'} for job in ('1', '2', '59380077', '59380073', '59380092')}
            rows['3'] = {'state': 'PENDING'}
            stack.enter_context(mock.patch.object(C, 'accounting', return_value={'rows': rows}))
            submit = stack.enter_context(mock.patch.object(C, 'submit_checkpoint'))
            R.step(True)
            submit.assert_called_once_with('final', ['3'])

    def test_final_receipt_is_reused_before_new_submission_or_accounting(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp, ExitStack() as stack:
            root = Path(temp); closeout, n = self.fixture(root)
            self.patches(root, closeout, n, stack)
            C.save(root / 'checkpoint-final.json', {'binding': {'plan_sha256': 'closeout'}, 'job_id': '4'})
            submit = stack.enter_context(mock.patch.object(C, 'submit_checkpoint'))
            acct = stack.enter_context(mock.patch.object(C, 'accounting'))
            R.step(True)
            submit.assert_not_called(); acct.assert_not_called()

    def test_changed_j1_dispatch_seal_stops_before_final_submission(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp, ExitStack() as stack:
            root = Path(temp); closeout, n = self.fixture(root)
            self.patches(root, closeout, n, stack)
            (root / 'utility/J1_NATIVE_DISPATCH.json').unlink()
            ready = C.sealed(root / 'utility/J1_READY.json')
            C.save(root / 'utility/J1_NATIVE_DISPATCH.json', {'readiness_sha256': ready['sha256'], 'result_sha256': 'changed'})
            submit = stack.enter_context(mock.patch.object(C, 'submit_checkpoint'))
            with self.assertRaisesRegex(ValueError, 'final J1 dispatch changed'):
                R.step(True)
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
