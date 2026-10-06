"""No-submit tests for canonical reuse, pricing, missingness and recovery."""
from pathlib import Path
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import utility_production_v1 as P
import run_utility_production_v1 as runner
import dispatch_utility_production_v1 as dispatcher


def plan():
    return P.U.sealed(REPO / 'report/experimental-resume-v1/UTILITY_SCOUT_PLAN_v1.json')


def result(assignment, queries=()):
    return {'assignment': assignment, 'qualification_only': False, 'maximum_tokens': 16384,
        'generation_wall_seconds': 5., 'controller': {
            'state': {'uid': assignment['uid'], 'prompt_token_ids_sha256': assignment['prompt_token_ids_sha256'],
                      'completion_token_ids': [1, 2, 3], 'token_sources': ['emitted'] * 3, 'finish': 'stop'},
            'episode': None, 'queries': list(queries), 'side_prompt_tokens': 1000 * len(queries),
            'side_generated_tokens': 100 * len(queries)}}


class ProductionTests(unittest.TestCase):
    def test_pilot_import_preserves_all_eight_canonical_ids_and_original_receipt_paths(self):
        with tempfile.TemporaryDirectory(prefix='.utility-prod-test-', dir=REPO) as temp:
            root = Path(temp)
            attachment, output = root / 'attachment', root / 'output'
            attachment.mkdir(); output.mkdir()
            assigned = plan()
            plan_path = root / 'PLAN.json'
            P.save(plan_path, {k: v for k, v in assigned.items() if k != 'sha256'})
            qual_binding = P.save(root / 'BINDING.json', {'prepared_plan_sha256': 'engineering', 'code_files': {}})
            qual = P.save(root / 'QUALIFICATION.json', {'status': 'PASS_ENGINEERING', 'binding_sha256': qual_binding['sha256']})
            proposal = P.save(root / 'PROPOSAL.json', {'utility_plan_path': str(plan_path),
                'qualification_result': str(root / 'QUALIFICATION.json'), 'engineering_plan_sha256': 'engineering'})
            policy = P.save(attachment / 'SELECTED_POLICY.json', {'selections': {'candidate_to_verify': {'arm': 'selected'}}})
            pilot = P.save(attachment / 'PILOT_MANIFEST.json', {
                'schema': 'utility-price-pilot-manifest-v2', 'plan_sha256': assigned['sha256'],
                'policy_sha256': policy['sha256'], 'policy': policy, 'qualification_sha256': qual['sha256'],
                'code_files': {}, 'family_order': assigned['family_order'][:2],
                'rows': [{'assignment': row} for row in assigned['assignments'][:8]]})
            chain = P.save(attachment / 'PILOT_CHAIN.json', {'proposal_sha256': proposal['sha256'],
                'pilot_manifest_sha256': pilot['sha256'], 'pilot_job': '7', 'output': str(output)})
            P.save(attachment / 'PILOT_ACCOUNTING.json', {'proposal_sha256': proposal['sha256'],
                'pilot_chain_sha256': chain['sha256'], 'pilot_job': '7', 'allocated_gpus': 4})
            binding = P.save(output / 'BINDING.json', {'manifest_sha256': pilot['sha256'], 'policy_sha256': policy['sha256']})
            for assignment in assigned['assignments'][:8]:
                key = P.U.digest(assignment['uid'])
                route = output / 'routes' / (key + '.npz')
                route.parent.mkdir(exist_ok=True)
                route.write_bytes(b'exact saved route bytes')
                attempt = P.save(output / 'attempts' / (key + '-000.json'), {
                    'assignment': assignment, 'binding_sha256': binding['sha256'], 'job_id': '7'})
                P.save(output / 'receipts' / (key + '.json'), {
                    'schema': 'utility-price-pilot-receipt-v2', 'assignment': assignment,
                    'binding_sha256': binding['sha256'], 'attempt_sha256': attempt['sha256'],
                    'status': 'COMMITTED_GENERATION', 'result': result(assignment), 'error': None,
                    'routed_array_sha256': P.U.file_sha(route)})
            with patch.object(P, 'validate_config'):
                loaded = P.load_pilot({'pilot_proposal_path': str(root / 'PROPOSAL.json'),
                                       'scientific_code_files': {}}, attachment)
            self.assertEqual([row['assignment'] for row in loaded['imports']], assigned['assignments'][:8])
            self.assertTrue(all(Path(row['receipt_path']).parent == output / 'receipts' for row in loaded['imports']))
            self.assertEqual(len({row['receipt_sha256'] for row in loaded['imports']}), 8)
            wrong = dict(loaded['receipts'][0], assignment=assigned['assignments'][1])
            with self.assertRaises(ValueError):
                P.validate_receipt(wrong, assigned['assignments'][0], binding['sha256'])

    def test_complete_price_partitions_376_without_reusing_pilot_and_accounts_recovery_and_measurement(self):
        assigned = plan()
        profile = {'status': 'MEASURED_STRESS_REFERENCE', 'generator_seconds_per_prompt_or_emitted_token': .01,
            'side_queries_per_16k_policy_request_stress': 2, 'side_pair_seconds_stress': 1.,
            'joint_cold_load_seconds': 100., 'shutdown_and_allocation_overhead_seconds': 20.,
            'pilot_allocated_gpu_hours': .5}
        config = {'sha256': 'config', 'stress_factor': 2., 'shutdown_margin_seconds': 600,
            'wall_seconds': 28800, 'maximum_families_per_shard': 2, 'maximum_infrastructure_recovery_waves': 1,
            'infrastructure_recovery_reserve_fraction': .25,
            'offline_measurement_plan_sha256': 'measurement', 'offline_grading_reserve_gpu_hours': 4.,
            'offline_grading_envelope_path': 'envelope.json',
            'offline_grading_envelope': {'sha256': 'envelope', 'median_full_cap_all_attempts_projected_GPU_h': 3.,
                                         'historical_item_rate_forecast_GPU_h': 2.},
            'available_generation_billing_core_hours': 100000.,
            'offline_measured_reference_sha256': 'reference'}
        imports = [{'assignment': a} for a in assigned['assignments'][:8]]
        price = P.price_stage(assigned, imports, profile, config)
        self.assertEqual(price['status'], 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION')
        ids = [uid for shard in price['shards'] for uid in shard['assigned_uids']]
        self.assertEqual(set(ids), {r['uid'] for r in assigned['assignments'][8:]})
        self.assertEqual(len(ids), 376)
        self.assertEqual(len(price['shards']), 47)
        initial_ceiling = 47 * 4 * price['requested_array_wall_seconds'] / 3600
        self.assertLess(price['requested_array_wall_seconds'], config['wall_seconds'])
        self.assertEqual(price['initial_allocation_gpu_hour_ceiling'], initial_ceiling)
        recovery_ceiling = 12 * 4 * price['requested_array_wall_seconds'] / 3600
        self.assertEqual(price['infrastructure_recovery_reserved_shards'], 12)
        self.assertEqual(price['total_generation_allocation_gpu_hour_ceiling'], initial_ceiling + recovery_ceiling)
        self.assertEqual(price['generation_and_reserved_measurement_gpu_hours'], initial_ceiling + recovery_ceiling + 4)
        self.assertFalse(price['completion_guarantee'])
        held = P.price_stage(assigned, imports, {**profile, 'side_pair_seconds_stress': 100000.}, config)
        self.assertEqual(held['status'], 'HOLD_COMPLETE_GENERATION_PRICE')
        self.assertTrue(held['oversized_families'])
        self.assertTrue(held['revised_proposal'])
        with self.assertRaises(ValueError):
            P.price_stage(assigned, imports + imports[:1], profile, config)

    def test_runtime_profile_charges_nonfire_queries_and_holds_when_side_work_is_unmeasured(self):
        assigned = plan()['assignments'][:8]
        reader = {'elapsed_seconds': .1, 'prompt_tokens': 1000, 'generated_tokens': 50}
        receipts = [{'status': 'COMMITTED_GENERATION', 'sha256': str(i), 'assignment': a,
            'result': result(a, [{'results': [reader, reader]}] if a['arm'] == 'frozen_policy' else [])}
            for i, a in enumerate(assigned)]
        pilot = {'receipts': receipts, 'loads': [{'load_wall_seconds': 10}], 'costs': [{}],
                 'accounting': {'status': 'COMPLETE_UNGRADED_RUNTIME_PILOT', 'elapsed_seconds': 100,
                    'actual_allocated_gpu_hours': 4 / 36, 'sha256': 'account'}}
        profile = P.measured_profile(pilot)
        self.assertEqual(profile['status'], 'MEASURED_STRESS_REFERENCE')
        policies = [r for r in profile['rows'] if r['arm'] == 'frozen_policy']
        self.assertTrue(all(r['nonfire'] and r['side_queries'] == 1 and r['side_seconds'] > 0 for r in policies))
        no_side = [{**r, 'result': result(r['assignment'])} for r in receipts]
        self.assertEqual(P.measured_profile({**pilot, 'receipts': no_side})['status'], 'HOLD_INCOMPLETE_MEASURED_PROFILE')

    def test_committed_errors_are_never_retried_and_uncommitted_attempt_requires_recovery(self):
        with tempfile.TemporaryDirectory(prefix='.utility-prod-test-', dir=REPO) as temp:
            root = Path(temp)
            for name in ('attempts', 'receipts'):
                (root / name).mkdir()
            assignments = plan()['assignments'][:2]
            binding = {'sha256': 'binding'}
            shard = {'index': 0, 'assigned_uids': [a['uid'] for a in assignments]}
            manifest = {'sha256': 'manifest', 'rows': [{'assignment': a} for a in assignments]}
            key = P.U.digest(assignments[0]['uid'])
            committed_attempt = P.save(root / 'attempts' / (key + '-000.json'), {
                'assignment': assignments[0], 'binding_sha256': 'binding'})
            P.save(root / 'receipts' / (key + '.json'), {'schema': 'utility-production-generation-receipt-v1',
                'assignment': assignments[0], 'binding_sha256': 'binding', 'status': 'GENERATION_ERROR',
                'result': None, 'error': 'CUDA allocation failure', 'routed_array_sha256': None,
                'attempt_sha256': committed_attempt['sha256']})
            key2 = P.U.digest(assignments[1]['uid'])
            attempt = P.save(root / 'attempts' / (key2 + '-000.json'), {'assignment': assignments[1]})
            with self.assertRaisesRegex(ValueError, 'sealed infrastructure recovery'):
                runner.pending_rows(manifest, shard, root, binding)
            recovery = {'schema': 'utility-production-recovery-v1', 'manifest_sha256': 'manifest', 'wave': 1,
                'shards': {'0': {'assigned_uids': [a['uid'] for a in assignments]}},
                'prior_attempt_sha256s': {assignments[1]['uid']: [attempt['sha256']]}}
            pending, committed = runner.pending_rows(manifest, shard, root, binding, recovery)
            self.assertEqual([r['assignment'] for r, _ in pending], assignments[1:])
            self.assertEqual([r['assignment'] for r in committed], assignments[:1])

    def test_reconciliation_keeps_all_384_missing_and_charges_failed_loads(self):
        with tempfile.TemporaryDirectory(prefix='.utility-prod-test-', dir=REPO) as temp:
            root = Path(temp)
            assigned = plan()
            pilot_account = P.save(root / 'pilot' / 'PILOT_ACCOUNTING.json', {'actual_allocated_gpu_hours': 2.})
            manifest = P.save(root / 'MANIFEST.json', {
                'plan_path': str(root / 'PLAN.json'), 'plan_sha256': assigned['sha256'],
                'policy_path': str(root / 'POLICY.json'), 'policy_sha256': 'policy', 'imported_receipts': [],
                'rows': [{'assignment': a} for a in assigned['assignments']],
                'shards': [{'index': 0, 'wall_seconds': 28800, 'assigned_uids': [a['uid'] for a in assigned['assignments']]}],
                'pilot_attachment': str(root / 'pilot'), 'pilot_accounting_sha256': pilot_account['sha256']})
            with patch.object(P, 'validate_manifest', return_value=assigned):
                index = P.reconcile(root / 'MANIFEST.json', root / 'output',
                    [{'allocated_gpu_hours': 4., 'state': 'FAILED', 'accounting_job_id': '9_0'}], final=False)
            self.assertEqual(index['assigned'], 384)
            self.assertEqual(index['missing'], 384)
            self.assertEqual(index['total_generation_allocated_gpu_hours'], 6.)
            self.assertEqual([r['assignment'] for r in index['records']], assigned['assignments'])
            self.assertTrue(all(r['receipt_path'] is None for r in index['records']))
            recovery = P.recovery_plan(manifest, index)
            self.assertEqual(recovery['assigned'], 384)

    def test_failed_and_unallocated_array_tasks_are_accounted_without_fabricated_cost(self):
        raw = '9_0|FAILED|1:0|3600|billing=32,cpu=32,gres/gpu=4|\n9_1|CANCELLED by 123|0:0|0||\n'
        with patch.object(P.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
            costs = P.allocation_accounting([{'array_job': '9', 'wave': 0}], {0: [0, 1]})
        self.assertEqual(sum(row['allocated_gpu_hours'] for row in costs), 4.)
        self.assertEqual(costs[1]['allocated_gpus'], 0)

    def test_live_budget_counts_compressed_arrays_and_gpu_billing(self):
        balance = 'IscrC_MIOSR 20260504 20270204 68000 20690 20690 30.4 7391 591\n'
        queue = '9_[0-3%2]|PENDING|2:00:00|2:00:00|16|gres/gpu:2|1\n10|RUNNING|1:00:00|30:00|32|gres/gpu:4|1\n'
        value = dispatcher.remaining_commitments(balance, queue)
        self.assertEqual(value['reported_remaining_billing_core_hours'], 47310)
        self.assertEqual(value['active_remaining_commitment_billing_core_hours'], 4 * 16 * 2 + 32 * .5)
        self.assertEqual(dispatcher.array_multiplicity('11_[1-5:2,8]'), 4)
        self.assertEqual(dispatcher.duration_seconds('1-02:03:04'), 93784)

    def test_generator_failure_commits_error_keeps_remaining_missing_and_blocks_unpriced_restart(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='.utility-prod-test-', dir=REPO) as temp:
            root = Path(temp)
            assignments = plan()['assignments'][:4]
            config = P.save(root / 'CONFIG.json', {})
            shards = [{'index': 0, 'assigned_uids': [a['uid'] for a in assignments]}]
            price = P.save(root / 'PRICE.json', {'config_sha256': config['sha256'], 'shards': shards,
                'status': 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION'})
            manifest = P.save(root / 'MANIFEST.json', {'config_sha256': config['sha256'],
                'price_sha256': price['sha256'], 'shards': shards, 'policy_sha256': 'policy',
                'policy': {'selections': {}}, 'rows': [{'assignment': a, 'original_prompt_ids': [1],
                                                       'problem': 'fixture'} for a in assignments]})
            calls = []
            class Backend:
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def generate_one(self, assignment, prompt, problem):
                    calls.append(assignment['uid'])
                    if len(calls) == 2:
                        raise RuntimeError('fixture transport failure')
                    return {**result(assignment), 'routed': np.zeros((3, 40, 8), dtype=np.uint8)}
            args = SimpleNamespace(manifest=root / 'MANIFEST.json', config=root / 'CONFIG.json',
                out=root / 'run', shard_index=0, recovery_manifest=None, deadline_epoch=time.time() + 600)
            with patch.object(P, 'validate_manifest'), patch.object(P, 'validate_config'), \
                 patch.dict(os.environ, {'SLURM_JOB_ID': '9', 'SLURM_STEP_ID': '0',
                                         'SLURM_ARRAY_JOB_ID': '9', 'SLURM_ARRAY_TASK_ID': '0'}):
                with self.assertRaises(SystemExit) as stopped:
                    runner.run(args, backend_factory=Backend)
                self.assertEqual(stopped.exception.code, 2)
                with self.assertRaisesRegex(ValueError, 'prior allocation requires sealed infrastructure recovery'):
                    runner.run(args, backend_factory=Backend)
            summary = P.U.sealed(root / 'run/shard-000/SUMMARY-9_0.json')
            self.assertEqual(calls, [a['uid'] for a in assignments[:2]])
            self.assertEqual(summary['receipts'], 2)
            self.assertEqual(summary['generation_errors'], 1)
            self.assertEqual(summary['missing_uids'], [a['uid'] for a in assignments[2:]])


if __name__ == '__main__':
    unittest.main()
