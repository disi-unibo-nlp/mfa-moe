"""High-impact ITT, binding, and budget guards for the 16k utility stage."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/utility_scout_v1.py'
SPEC = importlib.util.spec_from_file_location('utility_scout_v1', SOURCE)
U = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(U)


def receipt(row, execution, *, emitted=(7,), injected=(), finish='stop', error=None):
    completion = list(emitted) + list(injected)
    return U.seal({'schema': 'routing-utility-scout-receipt-v1',
                   'execution_sha256': execution['sha256'], 'uid': row['uid'],
                   'question': row['question'], 'seed': row['seed'], 'arm': row['arm'],
                   'prompt_token_ids_sha256': row['prompt_token_ids_sha256'],
                   'policy_sha256': (None if row['arm'] == 'native' else execution['policy_sha256']),
                   'completion_token_ids': completion,
                   'token_sources': ['emitted'] * len(emitted) + ['injected'] * len(injected),
                   'emitted_token_ids': list(emitted), 'injected_token_ids': list(injected),
                   'emitted_tokens': len(emitted), 'injected_tokens': len(injected),
                   'tokens_charged': len(emitted) + len(injected),
                   'finish': finish, 'error': error})


class UtilityScoutContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = U.build_plan(U.sealed(U.FAMILY), U.sealed(U.UNITS), U.sealed(U.PROMPTS),
                                prompt_file_sha256=U.file_sha(U.PROMPTS))
        U.validate_plan(cls.plan)
        cls.execution = U.seal({'schema': 'routing-utility-scout-execution-v1',
                                'plan_sha256': cls.plan['sha256'], 'policy_sha256': 'p',
                                'qualification_sha256': 'q', 'assignment_count': 384,
                                'max_tokens': 16384, 'status': 'BOUND_UNPRICED'})

    def test_exact_original_prompt_factorial_and_disjointness(self):
        plan = self.plan
        self.assertEqual((plan['family_count'], plan['assignment_count']), (96, 384))
        self.assertEqual(plan['maximum_decode_and_injection_tokens'], 384 * 16384)
        self.assertTrue(all(row['prompt_tokens'] > 0 for row in plan['assignments']))
        damaged = {**plan, 'assignments': plan['assignments'][:-1]}
        damaged['sha256'] = U.digest({k: v for k, v in damaged.items() if k != 'sha256'})
        with self.assertRaises(ValueError):
            U.validate_plan(damaged)

    def test_missing_and_failed_receipts_remain_in_itt(self):
        first, second = self.plan['assignments'][:2]
        one = receipt(second, self.execution, emitted=(7, 8), injected=(9,))
        audit = U.audit_receipts(self.plan, self.execution, [one])
        self.assertEqual((audit['assigned_count'], audit['missing_count']), (384, 383))
        self.assertEqual(audit['observed_tokens_charged'], 3)
        self.assertTrue(audit['rows'][1]['gradeable'])
        self.assertEqual(audit['rows'][0]['receipt_status'], 'MISSING')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            U.audit_receipts(self.plan, self.execution, [one, one])
        with self.assertRaisesRegex(ValueError, 'natural, length'):
            U.audit_receipts(self.plan, self.execution, [
                receipt(first, self.execution, emitted=(7,), finish='length')])

    def test_complete_blind_grade_contract_and_failure_defaults(self):
        receipts = [receipt(row, self.execution) for row in self.plan['assignments']]
        receipts[0] = receipt(self.plan['assignments'][0], self.execution,
                              emitted=(), finish='error', error='worker failure')
        audit = U.audit_receipts(self.plan, self.execution, receipts)
        self.assertEqual(audit['status'], 'COMPLETE_UNGRADED')
        grade = U.blind_grading_contract(audit, U.sealed(U.QUESTIONS))
        self.assertEqual((grade['assigned_count'], grade['gradeable_count']), (384, 383))
        self.assertIs(grade['map'][0]['default_operational_correct'], False)
        self.assertTrue(all('arm' not in row and 'family' not in row
                            for row in grade['blind_rows']))
        bundle = U.materialize_blind_bundle(grade, audit, receipts, lambda ids: str(ids))
        self.assertEqual(len(bundle), 383)
        self.assertEqual(set(bundle[0]), {'uid', 'question', 'dataset', 'problem', 'gold',
                                          'content_text', 'finish_reason', 'cumulative_tokens',
                                          'natural_stop'})

    def test_all_in_price_requires_measurement_and_walltime(self):
        plan, execution = self.plan, self.execution
        self.assertEqual(U.price_complete_stage(plan, None, None)['status'],
                         'HOLD_POLICY_OR_MEASURED_PROFILE')
        profile = {'schema': 'routing-utility-measured-price-profile-v1',
                   'measurement_job_ids': ['measured-job'],
                   'measurement_evidence_sha256': 'measured-evidence',
                   'gpu_count': 2, 'shards': 16,
                   'model_loads': 16, 'cold_load_seconds_per_load': 1,
                   'prefill_tokens_per_second': 1000, 'decode_tokens_per_second': 1000,
                   'request_overhead_seconds': 1, 'prefix_preparation_seconds': 1,
                   'recompute_prefill_tokens': 0, 'retry_seconds': 1,
                   'shutdown_seconds': 1, 'grading_gpu_hours': 1,
                   'nll_gpu_hours': 0, 'labeling_gpu_hours': 0,
                   'controller_gpu_hours': 0,
                   'available_gpu_hours': 100, 'walltime_seconds': 10000}
        priced = U.price_complete_stage(plan, execution, profile)
        self.assertEqual(priced['status'], 'PASS_COMPLETE_STAGE')
        self.assertEqual(priced['maximum_shard_decode_tokens'], 24 * 16384)
        profile['available_gpu_hours'] = 1
        self.assertEqual(U.price_complete_stage(plan, execution, profile)['status'],
                         'HOLD_REPRICE_OR_RESOURCE_CHANGE')
        del profile['grading_gpu_hours']
        with self.assertRaisesRegex(ValueError, 'complete measured'):
            U.price_complete_stage(plan, execution, profile)

    def test_future_policy_binding_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'missing sealed'):
            U.bind_execution(self.plan, {}, {})


if __name__ == '__main__':
    unittest.main()
