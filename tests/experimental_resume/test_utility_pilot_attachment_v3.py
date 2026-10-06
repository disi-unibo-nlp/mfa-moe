"""No-submit tests for first-two-family enrollment and one-shot dependencies."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import attach_utility_price_pilot_v3 as attachment


class AttachmentTests(unittest.TestCase):
    def test_canonical_first_two_families_keep_all_eight_uids(self):
        plan = attachment.utility.sealed(REPO / 'report/experimental-resume-v1/UTILITY_SCOUT_PLAN_v1.json')
        families, rows = attachment.first_two_assignments(plan)
        self.assertEqual(families, plan['family_order'][:2])
        self.assertEqual(rows, plan['assignments'][:8])
        self.assertEqual(len({row['uid'] for row in rows}), 8)
        self.assertTrue(all(row['uid'].startswith('utility-v1|') for row in rows))

    def test_generation_and_analysis_attach_once_to_saved_successors(self):
        with tempfile.TemporaryDirectory(prefix='.utility-attach-test-', dir=REPO) as name:
            root = Path(name)
            source, own = root / 'source', root / 'own'
            source.mkdir(); own.mkdir()
            manifest = {'sha256': 'manifest'}
            proposal = {'sha256': 'proposal', 'qualification_job': '59346566'}
            attachment.shared.save(source / 'GENERATION_CHAIN.json', {
                'schema': 'overnight-generation-chain-v2', 'binding': {'manifest_sha256': 'manifest'},
                'reader_dispatch_job': '60000002'})
            attachment.shared.save(source / 'MEASUREMENT_CHAIN.json', {
                'schema': 'overnight-measurement-chain-v2', 'binding': {'manifest_sha256': 'manifest'},
                'analysis_job': '60000003',
                'analysis_output': str(attachment.DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2')})
            calls = []
            def submit(directory, label, args, env, binding):
                calls.append((label, args, dict(env), binding))
                return str(60000010 + len(calls))
            with patch.object(attachment, 'validate_sources'), patch.object(attachment, 'active_preflight'), \
                 patch.object(attachment, 'paths', return_value=(root / 'manifest.json', manifest, source, own)), \
                 patch.object(attachment, 'submit', side_effect=submit), \
                 patch.dict(os.environ, {'UTILITY_FRESH_DISPATCH_JOB': '60000001'}):
                attachment.phase_attach(proposal, root / 'proposal.json', 'generation')
                attachment.phase_attach(proposal, root / 'proposal.json', 'analysis')
            self.assertIn('--dependency=afterok:60000002', calls[0][1])
            self.assertEqual(calls[0][2]['UTILITY_ATTACH_PHASE'], 'analysis')
            self.assertEqual(calls[0][3]['fresh_generation_dispatch_job'], '60000001')
            self.assertIn('--dependency=afterok:60000003:59346566', calls[1][1])
            self.assertEqual(calls[1][2]['UTILITY_ATTACH_PHASE'], 'select')
            self.assertTrue(all('run_utility_price_pilot_v3.sbatch' not in ' '.join(c[1]) for c in calls))

    def test_failed_qualification_cannot_reach_gpu_input_gate(self):
        with tempfile.TemporaryDirectory(prefix='.utility-attach-test-', dir=REPO) as name:
            root = Path(name)
            qual = attachment.shared.save(root / 'QUALIFICATION.json', {
                'schema': 'utility-pair-engine-qualification-v2', 'status': 'HOLD_FAILED_OR_INCOMPLETE_ENGINEERING'})
            proposal = {'qualification_result': str(root / 'QUALIFICATION.json')}
            with patch.object(attachment, 'validate_sources'):
                with self.assertRaisesRegex(ValueError, 'has not passed'):
                    attachment.pilot_inputs(proposal, root, {'sha256': 'manifest'})

    def test_missing_artifact_is_incomplete_even_with_successful_slurm_state(self):
        with tempfile.TemporaryDirectory(prefix='.utility-attach-test-', dir=REPO) as name:
            root = Path(name)
            output = root / 'output'; output.mkdir()
            attachment.shared.save(root / 'PILOT_CHAIN.json', {'pilot_job': '60000004', 'output': str(output)})
            raw = '60000004|COMPLETED|0:0|3600|32|billing=32,cpu=32,gres/gpu=4|start|end|\n'
            with patch.object(attachment, 'validate_sources'), \
                 patch.object(attachment, 'paths', return_value=(None, {}, None, root)), \
                 patch.object(attachment.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
                attachment.account({'sha256': 'proposal'})
            receipt = attachment.utility.sealed(root / 'PILOT_ACCOUNTING.json')
            self.assertEqual(receipt['status'], 'INCOMPLETE_OR_FAILED_RUNTIME_PILOT')
            self.assertEqual(receipt['actual_allocated_gpu_hours'], 4)
            self.assertFalse(receipt['automatic_gpu_retry'])


if __name__ == '__main__':
    unittest.main()
