"""Exact import, terminal accounting, blind grading and wrapper verification."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import operator_panel_orchestration_v1 as N
P, O = N.P, N.O


def fixture_output(root, manifest, rows, job):
    root.mkdir()
    for name in ('attempts', 'receipts', 'routes'): (root / name).mkdir()
    binding = P.save(root / 'BINDING.json', {'manifest_sha256': manifest['sha256'], 'assigned': rows})
    for a in rows:
        key = P.U.digest(a['uid']); route = root / 'routes' / (key + '.npz')
        route.write_bytes(b'sealed engineering route fixture')
        attempt = P.save(root / 'attempts' / (key + '-000.json'), {'assignment': a,
            'binding_sha256': binding['sha256'], 'job_id': job})
        ids = [7, P.THINK_END_ID, 8]
        P.save(root / 'receipts' / (key + '.json'), {'schema': 'panel-generation-receipt-v1',
            'assignment': a, 'binding_sha256': binding['sha256'], 'attempt_sha256': attempt['sha256'],
            'status': 'COMMITTED_GENERATION', 'routed_array_sha256': P.U.file_sha(route), 'error': None,
            'result': {'assignment': a, 'maximum_tokens': 16384, 'qualification_only': False,
                'controller': {'state': {'uid': a['uid'], 'prompt_token_ids_sha256': a['prompt_token_ids_sha256'],
                    'completion_token_ids': ids, 'token_sources': ['emitted'] * len(ids), 'finish': 'stop'},
                'episode': None, 'queries': [], 'observed_generator_tokens': len(ids),
                'side_generated_tokens': 0, 'side_prompt_tokens': 0, 'side_elapsed_seconds': 0.}}})
    return root


class OrchestrationTests(unittest.TestCase):
    def test_frozen_scientific_sources_and_measurement(self):
        plan, operations, measurement = N.validate()
        self.assertEqual(plan['assignment_count'], 768)
        self.assertEqual(measurement['inference']['primary_comparisons'], 9)
        self.assertEqual(operations['generation_sources'], P.generation_sources())
        self.assertEqual(len(P.U.sealed(P.DOC / 'UTILITY_PAIR_ENGINEERING_PLAN_v2.json')['code_files']), 208)

    def test_pilot_import_exact_cells_and_missing_cells_without_reexecution(self):
        plan, _, _ = N.validate()
        p_rows = P.assignments(plan, plan['family_order'][:2])
        manifest = P.U.seal({'schema': 'operator-panel-manifest-v1', 'plan_sha256': plan['sha256'],
                            'family_order': plan['family_order'][:2], 'rows': [{'assignment': a} for a in p_rows]})
        evidence = {'rows': {'9': {'job_id_raw': '9'}, '10': {'job_id_raw': '10'}}, 'actual_billing_core_hours': 1.}
        with tempfile.TemporaryDirectory(prefix='.panel-import-', dir=REPO) as temp:
            root = Path(temp)
            out0 = fixture_output(root / 'pilot-0', manifest, p_rows[:8], '9')
            out1 = fixture_output(root / 'pilot-1', manifest, p_rows[8:], '10')
            with patch.object(N, 'terminal', return_value=evidence):
                pilot = N.reconciliation(manifest, [out0, out1], ['9', '10'], root / 'PILOT.json')
            self.assertEqual(pilot['missing'], 0)
            self.assertEqual([r['assignment'] for r in pilot['records']], p_rows)
            original_hashes = [r['receipt_sha256'] for r in pilot['records']]
            cohort = P.U.seal({'schema': 'operator-panel-manifest-v1', 'plan_sha256': plan['sha256'],
                'family_order': plan['family_order'][:12],
                'rows': [{'assignment': a} for a in P.assignments(plan, plan['family_order'][:12])]})
            with patch.object(N, 'terminal', return_value=evidence):
                imported = N.reconciliation(cohort, [], [], root / 'COHORT.json', pilot['records'])
            self.assertEqual([r['receipt_sha256'] for r in imported['records'][:16]], original_hashes)
            self.assertEqual(imported['missing'], 80)
            self.assertEqual(len(imported['records']), 96)
            with patch.object(N, 'terminal', return_value=evidence):
                with self.assertRaises(ValueError):
                    N.reconciliation(cohort, [], [], root / 'DUPLICATE.json', pilot['records'] + pilot['records'][:1])
            route = Path(pilot['records'][0]['routed_path']); route.write_bytes(b'changed')
            with self.assertRaises(ValueError): O.extract(pilot)

    def test_cap_error_missing_and_incomplete_grading(self):
        plan, _, _ = N.validate(); a = plan['assignments'][0]
        index = {'records': [{'assignment': a, 'status': 'MISSING', 'receipt_path': None, 'receipt_sha256': None}]}
        row = O.extract(index)[0]
        self.assertIsNone(row['tokens']); self.assertIsNone(row['operational_correct'])
        error = {'assignment': a, 'sha256': 'error', 'binding_sha256': 'binding',
                 'status': 'GENERATION_ERROR', 'error': 'committed failure', 'result': None}
        index['records'][0].update(status='GENERATION_ERROR', receipt_path='error.json',
                                   receipt_sha256='error', source_binding_sha256='binding')
        row = O.extract(index, loader=lambda path: error)[0]
        self.assertEqual(row['operational_correct'], False); self.assertIsNone(row['tokens'])
        self.assertIsNone(row['reasoning_tokens'])

    def test_frozen_grader_rejects_arm_metadata(self):
        score, _ = O.scoring()
        score.assert_blind([{'uid': 'blind', 'problem': 'x', 'gold': '1', 'candidate': '1'}], 'test')
        with self.assertRaises(Exception):
            score.assert_blind([{'uid': 'blind', 'problem': 'x', 'gold': '1', 'candidate': '1', 'arm': 'bias'}], 'test')

    def test_accounting_matches_physical_array_jobs_and_preserves_failures(self):
        output = '12_0|COMPLETED|0:0|3600|480|billing=32,cpu=32,gres/gpu=4,mem=240G,node=1|13\n12_1|FAILED|1:0|1800|480|billing=32,cpu=32,gres/gpu=4,mem=240G,node=1|14\n'
        with patch.object(N.subprocess, 'run', return_value=type('Result', (), {'stdout': output})()):
            evidence = N.terminal(['12'])
        self.assertEqual(evidence['actual_billing_core_hours'], 48.)
        self.assertEqual(evidence['rows']['12_0']['job_id_raw'], '13')
        self.assertEqual(evidence['rows']['12_1']['state'], 'FAILED')

    def test_reproducible_effects_and_exported_tables_figures(self):
        import xml.etree.ElementTree as ET
        plan, _, _ = N.validate(); families = plan['family_order'][:12]
        rows = []
        for a in P.assignments(plan, families):
            i = families.index(a['family']); op = P.ARMS.index(a['arm'])
            rows.append({**a, 'operational_correct': (i + op) % 3 > 0,
                'reasoning_tokens': 100 + 10 * i - op * i, 'answer_tokens': 8,
                'tokens': 109 + 10 * i - op * i, 'execution_status': 'COMMITTED_GENERATION',
                'finish': 'stop', 'boundary_status': 'CLOSED', 'closure': True,
                'fired': op > 0, 'side_queries': int(op > 0), 'side_prompt_tokens': 100 * op,
                'side_generated_tokens': 10 * op, 'side_seconds': op / 10})
        effect = O.inference(rows, families, replicates=1000)
        self.assertEqual(effect, O.inference(rows, families, replicates=1000))
        for estimate, interval in zip(effect['point_estimates'], effect['simultaneous_95_intervals']):
            self.assertLessEqual(interval[0], estimate); self.assertLessEqual(estimate, interval[1])
        with tempfile.TemporaryDirectory(prefix='.panel-figures-', dir=REPO) as temp:
            out = Path(temp)
            artifacts = O.write_tables_figures(rows, effect, out)
            self.assertEqual(set(artifacts), {'assignments.csv', 'effects.csv', 'accuracy_length.pdf', 'accuracy_length.svg'})
            self.assertEqual(len((out / 'effects.csv').read_text().splitlines()), 10)
            self.assertEqual(len((out / 'assignments.csv').read_text().splitlines()), 97)
            self.assertTrue((out / 'accuracy_length.pdf').read_bytes().startswith(b'%PDF'))
            ET.parse(out / 'accuracy_length.svg')
            self.assertTrue(all(P.U.file_sha(out / name) == sha for name, sha in artifacts.items()))


if __name__ == '__main__': unittest.main()
