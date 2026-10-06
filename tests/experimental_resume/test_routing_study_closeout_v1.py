"""Closeout must preserve unknowns and refuse rebound or duplicate evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('closeout', REPO / 'scripts/experimental_resume/routing_study_closeout_v1.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CloseoutTests(unittest.TestCase):
    def test_changed_json_seal_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp:
            path = Path(temp) / 'receipt.json'
            C.save(path, {'assigned': 3480})
            value = json.loads(path.read_text()); value['assigned'] -= 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'changed JSON seal'):
                C.sealed(path)

    def test_pending_analysis_is_never_reported_as_complete(self):
        value = C.semantic(Path('absent-closeout-analysis.json'), 3480, '123', {})
        self.assertEqual(value['status'], 'PENDING')
        self.assertEqual(value['assigned'], 3480)
        self.assertFalse(C.successful({'123': {'state': 'COMPLETED', 'exit_code': '1:0'}}, '123'))

    def test_semantic_file_requires_successful_terminal_producer(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp:
            path = Path(temp) / 'ANALYSIS.json'
            C.save(path, {'assigned': 1})
            with self.assertRaisesRegex(ValueError, 'terminal producer'):
                C.semantic(path, 1, '123', {'123': {'state': 'RUNNING', 'exit_code': '0:0'}})

    def test_array_job_names_are_preserved_and_allocations_counted_once(self):
        raw = '5_0|COMPLETED|0:0|1800|billing=16,cpu=16,gres/gpu=2\n5_[1-3]|PENDING|0:0|0|\n'
        with mock.patch.object(C.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
            value = C.accounting(['5'])
        self.assertEqual(value['rows']['5_0']['allocated_GPU_hours'], 1)
        self.assertEqual(value['rows']['5_0']['billing_unit_hours'], 8)
        self.assertIn('5_[1-3]', value['rows'])
        with mock.patch.object(C.subprocess, 'run', return_value=SimpleNamespace(stdout=raw + raw)):
            with self.assertRaisesRegex(ValueError, 'ambiguous accounting'):
                C.accounting(['5'])

    def test_all_absent_readers_remain_explicit_unknown_assignments(self):
        rating = SimpleNamespace()
        frame = {'records': [{'blind_id': str(i)} for i in range(3480)]}
        price = {'ratings': 6960, 'shards': [{'start': 0, 'end': 3480}]}
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp, mock.patch.dict(sys.modules, {'rate_overnight_semantics_v2': rating}):
            value = C.reader_coverage(frame, price, Path(temp))
        self.assertEqual(value['unknown_ratings'], 6960)
        self.assertEqual(value['accounted_in_complete_shards'], 0)
        self.assertEqual(len(value['unknown_assignment_ids']), 6960)
        self.assertEqual(value['malformed_or_capped'], 0)

    def test_native_advance_waits_for_producer_and_consumes_only_one_readiness(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp:
            root = Path(temp)
            C.save(root / 'ATTACH_CHAIN.json', {'schema': 'attachment'})
            C.save(root / 'INITIAL_READY.json', {'producer_job_id': '201'})
            C.save(root / 'FINAL_READY.json', {'producer_job_id': '202'})
            n = SimpleNamespace(login_guard=lambda: None, __file__='frozen-adapter.py',
                validate_operations=lambda path: ({'native_root': str(root)}, {}))
            with mock.patch.object(C, 'native', return_value=n), mock.patch.object(C, 'accounting', return_value={'rows': {'202': {'state': 'RUNNING', 'exit_code': '0:0'}}}), mock.patch.object(C.subprocess, 'run') as run:
                C.advance(True)
                run.assert_not_called()
            with mock.patch.object(C, 'native', return_value=n), mock.patch.object(C, 'accounting', return_value={'rows': {'202': {'state': 'COMPLETED', 'exit_code': '0:0'}}}), mock.patch.object(C.subprocess, 'run') as run:
                C.advance(True)
                run.assert_called_once()
                self.assertIn(str(root / 'FINAL_READY.json'), run.call_args.args[0])
                self.assertIn('--submit', run.call_args.args[0])

    def test_consumed_readiness_cannot_be_rebound_after_interruption(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temp:
            root = Path(temp)
            C.save(root / 'ATTACH_CHAIN.json', {'schema': 'attachment'})
            C.save(root / 'INITIAL_READY.json', {'producer_job_id': '201'})
            C.save(root / 'INITIAL_NATIVE_DISPATCH.json', {'readiness_sha256': 'different'})
            n = SimpleNamespace(login_guard=lambda: None,
                validate_operations=lambda path: ({'native_root': str(root)}, {}))
            with mock.patch.object(C, 'native', return_value=n), mock.patch.object(C.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'rebound consumed readiness'):
                    C.advance(True)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
