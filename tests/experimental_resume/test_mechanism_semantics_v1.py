"""CPU-only checks for the sealed, all-assigned 1024-token measurement stage."""
import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'scripts/experimental_resume'))
if importlib.util.find_spec('numpy') is None and 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')

import build_mechanism_blind_frame_v1 as frame_builder  # noqa: E402
import rate_mechanism_semantics_v1 as rating  # noqa: E402
import price_mechanism_semantics_v1 as pricing  # noqa: E402
import analyze_mechanism_semantics_v1 as analysis  # noqa: E402
import diagnose_mechanism_validation_v1 as diagnostic  # noqa: E402
import diagnose_mechanism_validation_v2 as import_recovery  # noqa: E402


class MechanismSemanticsTests(unittest.TestCase):
    def test_import_recovery_keeps_worker_overlay_after_runner_world_action_imports(self):
        overlay = (Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/') /
                   'claude-analysis-2026-09-24/steering-v1/addenda/ordered/'
                   '9727c10299b71e7a/moe_exp_src')
        base = (Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/') /
                'claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24')
        script = ("import importlib.util,json; "
                  "from diagnose_mechanism_validation_v2 import pin_qualified_worker; "
                  "pin_qualified_worker(" + repr(str(overlay)) + "); "
                  "import run_mechanism_validation_v1; "
                  "from moe_steer import manifests; "
                  "from moe_exp.routing_control import design; "
                  "import moe_exp; "
                  "print(json.dumps([moe_exp.__file__, "
                  "importlib.util.find_spec('moe_exp.routing_control.ordered_vllm').origin]))")
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join((str(overlay), str(HERE / 'src'),
            str(HERE / 'scripts/experimental_resume'), str(base), str(base / 'moe_exp_src')))
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                cwd=HERE, env=env, text=True, capture_output=True,
                                timeout=60, check=True)
        package, worker = json.loads(result.stdout.strip())
        self.assertEqual(Path(package).resolve().parent, overlay / 'moe_exp')
        self.assertEqual(Path(worker).resolve(),
                         overlay / 'moe_exp/routing_control/ordered_vllm.py')

    def test_diagnostic_wrapper_exposes_caught_exception(self):
        fake_package = types.ModuleType('moe_steer')
        fake_qualify = types.ModuleType('moe_steer.qualify')
        exits = []
        fake_qualify.finish_child = lambda code, driver=None: exits.append(code)
        fake_package.qualify = fake_qualify
        fake_runner = types.ModuleType('run_mechanism_validation_v1')

        def run():
            try:
                raise RuntimeError('synthetic model-load failure')
            except RuntimeError:
                fake_qualify.finish_child(3)

        fake_runner.main = run
        stderr = io.StringIO()
        with patch.dict(sys.modules, {'moe_steer': fake_package,
                                      'run_mechanism_validation_v1': fake_runner}):
            with contextlib.redirect_stderr(stderr):
                diagnostic.main()
        self.assertEqual(exits, [3])
        self.assertIn('synthetic model-load failure', stderr.getvalue())

    def test_frame_gate_requires_sealed_stage_and_exact_assigned_results(self):
        def save(path, body):
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {**body, 'sha256': frame_builder.base.digest(body)}
            path.write_text(json.dumps(value))
            return value

        arms = list(frame_builder.generation.ARMS)
        manifest = {'sha256': 'manifest', 'base_driver_sha256': 'base-code',
                    'rows': [{'uid': 'p', 'family': 'f',
                    'transition': 'candidate_to_verify', 'question': 'key'}],
                    'seeds': [0, 1], 'expected_requests': 8,
                    'arm_order_by_uid_seed': {'p|0': arms, 'p|1': arms[::-1]}}
        price = {'sha256': 'price', 'shards': [{'start_row': 0, 'end_row': 1}]}
        with tempfile.TemporaryDirectory(prefix='.mechanism-blind-test-', dir=HERE) as name:
            root = Path(name)
            with self.assertRaises(FileNotFoundError):
                frame_builder.complete_outputs(manifest, price, root)
            plan = frame_builder.expected_assignments(manifest)
            directory = root / 'shard-000'
            save(root / 'STAGE_COMPLETION.json', {
                'schema': 'mechanism-validation-stage-completion-v1',
                'status': 'COMPLETE_UNGRADED_GENERATION',
                'manifest_sha256': 'manifest', 'price_sha256': 'price',
                'shards': 1, 'counts': {'assigned': 8}})
            save(directory / 'BINDING.json', {
                'schema': 'routing-boundary-micro-screen-binding-v1',
                'manifest_sha256': 'manifest', 'driver_sha256': 'base-code'})
            binding = frame_builder.base.sealed(directory / 'BINDING.json')
            summary = save(directory / 'SUMMARY.json', {
                'schema': 'routing-boundary-micro-screen-summary-v1',
                'status': 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION',
                'manifest_sha256': 'manifest',
                'binding_sha256': binding['sha256'], 'requests': 8, 'batches': 1})
            save(directory / 'MECHANISM_COMPLETION.json', {
                'schema': 'mechanism-validation-completion-v1',
                'status': 'COMPLETE_UNGRADED_GENERATION',
                'manifest_sha256': 'manifest',
                'binding_sha256': binding['sha256'],
                'source_summary_sha256': summary['sha256'],
                'counts': {'assigned': 8}})
            save(directory / 'batch-000-assignment.json', {
                'schema': 'routing-boundary-micro-screen-assignment-v1',
                'batch_index': 0,
                'manifest_sha256': 'manifest', 'binding_sha256': binding['sha256'],
                'requests': plan})
            array = directory / 'batch-000.npz'
            array.write_bytes(b'synthetic routed arrays')
            save(directory / 'batch-000.json', {
                'schema': 'routing-boundary-micro-screen-batch-v1',
                'batch_index': 0,
                'manifest_sha256': 'manifest', 'binding_sha256': binding['sha256'],
                'array_sha256': frame_builder.base.file_sha(array),
                'outputs': plan})
            _, expected, outputs, receipts = frame_builder.complete_outputs(
                manifest, price, root)
            self.assertEqual(len(expected), len(outputs))
            self.assertEqual(len(receipts), 1)
            bad = dict(plan[0], uid='wrong')
            save(directory / 'batch-000.json', {
                'schema': 'routing-boundary-micro-screen-batch-v1',
                'batch_index': 0,
                'manifest_sha256': 'manifest', 'binding_sha256': binding['sha256'],
                'array_sha256': frame_builder.base.file_sha(array),
                'outputs': [bad] + plan[1:]})
            with self.assertRaisesRegex(ValueError, 'assigned batch'):
                frame_builder.complete_outputs(manifest, price, root)

    def test_assignment_plan_keeps_all_four_arms_and_both_seeds(self):
        arms = list(frame_builder.generation.ARMS)
        manifest = {'sha256': 'frozen', 'rows': [{'uid': 'p', 'family': 'f',
                    'transition': 'candidate_to_verify', 'question': 'Q'}],
                    'seeds': [0, 1], 'expected_requests': 8,
                    'arm_order_by_uid_seed': {'p|0': arms, 'p|1': arms[::-1]}}
        plan = frame_builder.expected_assignments(manifest)
        self.assertEqual(len(plan), 8)
        self.assertEqual({(r['seed'], r['arm']) for r in plan},
                         {(seed, arm) for seed in (0, 1) for arm in arms})
        self.assertEqual(len({r['uid'] for r in plan}), 8)
        manifest['arm_order_by_uid_seed']['p|1'] = arms[:-1] + arms[-2:-1]
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            frame_builder.expected_assignments(manifest)

    def test_reader_input_rejects_arm_identity_and_trigger_as_outcome(self):
        row = {'blind_id': 'opaque', 'reader_input': {
            'transition': 'candidate_to_verify', 'problem': 'Compute x.',
            'full_emitted_prefix': 'Maybe x=2. Check x=2.',
            'triggering_sentence': 'Check x=2.', 'continuation': '2+2=4.'}}
        messages = rating.messages(row)
        self.assertEqual(len(messages), 2)
        self.assertNotIn('native_duplicate', str(messages))
        row['reader_input']['arm'] = 'target_bias1'
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            rating.messages(row)
        self.assertIsNone(rating.parse_rating('{"target":1}'))
        self.assertEqual(rating.parse_rating('</think>{"target":true}'), {'target': True})

    def test_complete_stage_price_includes_two_readers_and_recovery(self):
        frame = {'schema': 'mechanism-1024-blind-frame-v1',
                 'continuation_max_tokens': 1024,
                 'sha256': 'frame', 'records': [{}, {}, {}]}
        prior = {'schema': 'mechanism-start-reader-price-v2',
                 'status': 'PASS_COMPLETE_20_GPUH', 'sha256': 'prior',
                 'bounded_decode_tps': 43.71, 'bounded_prefill_tps': 4914,
                 'components_seconds': {'two_cold_loads': 1200,
                                        'two_shutdowns': 392}}
        binding = {'schema': 'mechanism-start-reader-binding-v2',
                   'sha256': 'binding', 'price_sha256': 'prior',
                   'model_snapshot': str(rating.MODEL), 'readers': 2}
        summary = {'schema': 'mechanism-start-reader-summary-v2',
                   'sha256': 'summary', 'binding_sha256': 'binding',
                   'counts': {'rows': 454, 'ratings': 908}}
        result = pricing.price_stage(frame, [100, 200, 300], prior, binding,
                                     summary, 7200, 100)
        self.assertEqual(result['prompt_tokens_exact_twice'], 1200)
        self.assertEqual(result['max_decode_tokens'], 6144)
        self.assertEqual(result['shards'], [{'start': 0, 'end': 3,
                                              'estimated_work_seconds':
                                              result['shards'][0]['estimated_work_seconds']}])
        self.assertGreaterEqual(result['recovery_load_reserve'], 1)
        self.assertEqual(result['status'], 'PASS_COMPLETE_STAGE')
        result = pricing.price_stage(frame, [100, 200, 300], prior, binding,
                                     summary, 7200, .01)
        self.assertEqual(result['status'], 'HOLD_EXCEEDS_GPU_HOUR_CEILING')

    def test_reader_rejects_price_for_different_walltime_or_pricing_code(self):
        row = {'blind_id': 'opaque', 'reader_input': {
            'transition': 'candidate_to_verify', 'problem': 'Compute x.',
            'full_emitted_prefix': 'Maybe x=2. Check x=2.',
            'triggering_sentence': 'Check x=2.', 'continuation': '2+2=4.'}}
        frame = {'schema': 'mechanism-1024-blind-frame-v1',
                 'sha256': 'frame', 'continuation_max_tokens': 1024,
                 'rubric_sha256': rating.base.file_sha(rating.RUBRIC),
                 'reader_input_allowlist': list(frame_builder.ALLOWLIST),
                 'assigned': 1, 'records': [row]}
        price = {'schema': 'mechanism-1024-reader-price-v1',
                 'status': 'PASS_COMPLETE_STAGE', 'frame_sha256': 'frame',
                 'rating_driver_sha256': rating.base.file_sha(rating.__file__),
                 'pricing_driver_sha256': rating.base.file_sha(pricing.__file__),
                 'rubric_sha256': rating.base.file_sha(rating.RUBRIC),
                 'model_snapshot': str(rating.MODEL), 'rows': 1, 'ratings': 2,
                 'max_decode_tokens': 2048, 'batch_size': rating.BATCH,
                 'gpus': 2, 'max_wall_seconds_per_job': 7200,
                 'estimated_complete_gpu_hours': 1., 'gpu_hour_ceiling': 100.,
                 'prompt_token_lengths': [300], 'prompt_tokens_exact_twice': 600,
                 'shards': [{'start': 0, 'end': 1}]}
        rating.validate(frame, price)
        price['max_wall_seconds_per_job'] = 3600
        with self.assertRaisesRegex(ValueError, 'price differs'):
            rating.validate(frame, price)
        price['max_wall_seconds_per_job'] = 7200
        price['pricing_driver_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'price differs'):
            rating.validate(frame, price)

    def test_family_clustered_itt_retains_failed_and_duplicate_draws(self):
        records = []
        arms = ('native', 'native_duplicate', 'target_bias1', 'random_bias1')
        for transition in ('candidate_to_verify', 'approach_to_commit'):
            for family in ('f1', 'f2'):
                prefix = transition + family
                for seed in (0, 1):
                    for arm in arms:
                        positive = arm == 'target_bias1' and family == 'f1'
                        records.append({'prefix_uid': prefix, 'seed': seed,
                                        'family': family, 'transition': transition,
                                        'arm': arm, 'both_positive': positive,
                                        'at_least_one_positive': positive,
                                        'operational_status': 'failure' if arm == 'random_bias1'
                                        else 'complete'})
        result = analysis.clustered_contrasts(records, n_boot=100)
        pooled = [r for r in result['contrasts'] if r['scope'] == 'all']
        self.assertEqual(len(pooled), 4)
        self.assertEqual(pooled[0]['estimate'], .5)
        self.assertEqual(pooled[2]['estimate'], 0)
        self.assertEqual(pooled[3]['estimate'], .5)
        self.assertEqual(pooled[0]['families'], 2)
        records.pop()
        with self.assertRaisesRegex(ValueError, 'incomplete four-arm'):
            analysis.clustered_contrasts(records, n_boot=100)


if __name__ == '__main__':
    unittest.main()
