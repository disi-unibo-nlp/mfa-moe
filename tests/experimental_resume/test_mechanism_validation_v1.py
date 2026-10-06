"""CPU-only fail-closed checks for mechanism enrollment and stage accounting."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'scripts/experimental_resume'))
# The standard Python used for these pure checks has no NumPy. The generation
# worker imports NumPy only when it executes on a compute allocation.
if importlib.util.find_spec('numpy') is None and 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')

import prepare_mechanism_validation_v1 as prep  # noqa: E402
import run_mechanism_validation_v1 as runner  # noqa: E402


class MechanismValidationTests(unittest.TestCase):
    def test_stage_completion_requires_every_exact_assigned_uid(self):
        def save(path, body):
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {**body, 'sha256': prep.base.digest(body)}
            path.write_text(json.dumps(value))
            return value

        with tempfile.TemporaryDirectory(prefix='.mechanism-stage-test-', dir=HERE) as name:
            root = Path(name)
            order = list(runner.ARMS)
            manifest = {
                'sha256': 'manifest', 'rows': [{'uid': 'u'}], 'seeds': [0, 1],
                'arm_order_by_uid_seed': {'u|0': order, 'u|1': list(reversed(order))},
                'expected_requests': 8, 'accepted_family_count': 1,
                'accepted_start_count': 1,
                'registered_128_family_feasibility': 'FAIL',
                'claim_limit': 'exploratory only'}
            price = {'sha256': 'price', 'shards': [{'start_row': 0, 'end_row': 1}]}
            directory = root / 'shard-000'
            uids = ['mechanism-v1|' + prep.base.digest([
                'manifest', 'u', seed, arm])[:24]
                for seed in (0, 1)
                for arm in manifest['arm_order_by_uid_seed'][f'u|{seed}']]
            routes = directory / 'batch-000.npz'
            routes.parent.mkdir(parents=True)
            routes.write_bytes(b'synthetic routes')
            batch = save(directory / 'batch-000.json', {
                'manifest_sha256': 'manifest', 'binding_sha256': 'binding',
                'array_sha256': prep.base.file_sha(routes),
                'outputs': [{'uid': uid} for uid in uids]})
            summary = save(directory / 'SUMMARY.json', {'batches': 1, 'requests': 8})
            save(directory / 'MECHANISM_COMPLETION.json', {
                'manifest_sha256': 'manifest', 'binding_sha256': 'binding',
                'source_summary_sha256': summary['sha256'],
                'counts': {'assigned': 8}})
            runner.seal_stage(root, manifest, price)
            self.assertEqual(prep.base.sealed(root / 'STAGE_COMPLETION.json')
                             ['registered_128_family_feasibility'], 'FAIL')
            batch.pop('sha256')
            batch['outputs'][0]['uid'] = 'wrong-assignment'
            save(directory / 'batch-000.json', batch)
            with self.assertRaisesRegex(ValueError, 'sealed row'):
                runner.seal_stage(root, manifest, price)

    def test_full_horizon_qualification_gate(self):
        def save(path, body):
            path.write_text(json.dumps({**body, 'sha256': prep.base.digest(body)}))
            return prep.base.sealed(path)

        with tempfile.TemporaryDirectory(prefix='.mechanism-qual-test-', dir=HERE) as name:
            root = Path(name)
            h14 = root / 'h14.json'
            h14.write_text('{}')
            worker = save(root / 'worker.json', {'worker_code_digest': 'worker-code'})
            driver = HERE / 'scripts/experimental_resume/qualify_mechanism_engine_1024_v1.py'
            manifest = save(root / 'qual-manifest.json', {
                'schema': 'routing-mechanism-serial-eager-1024-qual-manifest-v1',
                'bias': 1.0,
                'four_arm_order': ['native', 'target', 'random', 'native_duplicate'],
                'worker_code_digest': 'worker-code',
                'source_h14': {'path': str(h14),
                               'file_sha256': prep.base.file_sha(h14)},
                'source_ordered_qualification': {'sha256': 'ordered'},
                'source_serial_qualification': {'sha256': 'serial'},
                'code_files': {str(driver): prep.base.file_sha(driver)}})
            result = {
                'schema': 'routing-mechanism-serial-eager-1024-qualification-v1',
                'pass': True,
                'qualification_manifest_sha256': manifest['sha256'],
                'qualified_worker_sha256': worker['sha256'],
                'worker_code_digest': 'worker-code',
                'base_tree_sha256': prep.base.REQUIRED_BASE_TREE,
                'engine_profile': prep.PROFILE, 'max_tokens': 1024,
                'pulse_slots': [0, 512], 'pulse_length': 256,
                'same_prefix_four_arm_pass': True,
                'native_isolation_pass': True, 'ordered_pulse_pass': True,
                'preemption_recompute_pass': True, 'closure_pass': True,
                'inherited_h14_force_pass': True,
                'qualification_driver_sha256': prep.base.file_sha(driver),
                'code_files': manifest['code_files'],
                'source_h14_file_sha256': prep.base.file_sha(h14),
                'source_ordered_qualification_sha256': 'ordered',
                'source_serial_qualification_sha256': 'serial',
                'requests': 12, 'checks': [{'pass': True}] * 12,
                'job_id': 'synthetic'}
            with patch.object(prep, 'QUAL_MANIFEST', root / 'qual-manifest.json'), \
                 patch.object(prep.base, 'WORKER_QUAL', root / 'worker.json'):
                self.assertEqual(prep.validate_qualification(result), result)
                with self.assertRaisesRegex(ValueError, 'qualification'):
                    prep.validate_qualification({**result, 'closure_pass': False})

    def test_generation_uses_same_prefix_and_one_pulse(self):
        class Request:
            def __init__(self, uid, prompt, sampling):
                self.uid, self.prompt, self.sampling = uid, prompt, sampling

        engine = types.ModuleType('moe_steer.engine')
        engine.THINK_END_ID = 999
        qualify = types.ModuleType('moe_steer.qualify')
        qualify.QReq = Request
        qualify.crn = lambda info, seed: 71 + seed
        qualify.steer_extra = lambda *args, **kwargs: {'steer': {}}
        qualify.card_params = lambda cap, seed, extra, **kwargs: {
            'max_tokens': cap, 'seed': seed, 'extra': extra}
        package = types.ModuleType('moe_steer')
        package.engine, package.qualify = engine, qualify
        row = {'uid': 'u', 'family': 'f', 'transition': 'candidate_to_verify',
               'question': 'q', 'prompt_ids': [10, 11], 'prefix_ids': [12, 13]}
        manifest = {'sha256': 'manifest', 'seeds': [0, 1],
                    'arm_order_by_uid_seed': {'u|0': list(runner.ARMS),
                                              'u|1': list(reversed(runner.ARMS))},
                    'random_set_by_family': {'f': 2}, 'expected_requests': 8}
        world = types.SimpleNamespace(infos={'q': {'prompt_token_ids': [10, 11]}})
        with patch.dict(sys.modules, {'moe_steer': package,
                                      'moe_steer.engine': engine,
                                      'moe_steer.qualify': qualify}):
            cases = runner.build_requests(manifest, [row], [], world, object())
        self.assertEqual(len(cases), 8)
        self.assertEqual({tuple(req.prompt) for req, _ in cases}, {(10, 11, 12, 13)})
        self.assertEqual([req.sampling['seed'] for req, _ in cases],
                         [71] * 4 + [72] * 4)
        self.assertEqual({req.sampling['max_tokens'] for req, _ in cases}, {1024})
        active = [(req, meta) for req, meta in cases if meta['role'] != 'native']
        self.assertTrue(all(req.sampling['extra']['steer']['meta']
                            ['routing_control']['slots'] == [0]
                            for req, _ in active))
        self.assertEqual({meta['policy'] for _, meta in cases
                          if meta['role'] == 'random'}, {'verify_random2_bias1'})
        row['prefix_ids'] = [999]
        with patch.dict(sys.modules, {'moe_steer': package,
                                      'moe_steer.engine': engine,
                                      'moe_steer.qualify': qualify}):
            with self.assertRaisesRegex(ValueError, 'reasoning closure'):
                runner.build_requests(manifest, [row], [], world, object())

    def test_v2_committed_reader_attempt_chain(self):
        def save(path, body):
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {**body, 'sha256': prep.base.digest(body)}
            path.write_text(json.dumps(value))
            return value

        with tempfile.TemporaryDirectory(prefix='.mechanism-v2-test-', dir=HERE) as name:
            root = Path(name)
            price = save(root / 'price.json', {
                'schema': 'mechanism-start-reader-price-v2',
                'status': 'PASS_COMPLETE_20_GPUH'})
            rows = [{'uid': 'u0', 'family': 'f', 'transition': 'candidate_to_verify'},
                    {'uid': 'u1', 'family': 'f', 'transition': 'candidate_to_verify'}]
            selection = {'sha256': 'selection', 'records': rows}
            frame = {'sha256': 'frame',
                     'visible_input_allowlist': [
                         'problem', 'emitted_prefix', 'triggering_sentence']}
            binding = save(root / 'BINDING.json', {
                'schema': 'mechanism-start-reader-binding-v2',
                'frame_sha256': 'frame', 'selection_sha256': 'selection',
                'price_sha256': price['sha256'],
                'driver_sha256': prep.base.file_sha(prep.rating_v2.__file__),
                'message_source_sha256': prep.base.file_sha(
                    prep.rating_v2.prior_reader.__file__),
                'rubric_sha256': prep.base.file_sha(prep.rating_v2.prior_reader.RUBRIC),
                'model_snapshot': str(prep.rating_v2.prior_reader.MODEL),
                'model_revision': prep.rating_v2.prior_reader.MODEL.name,
                'sampler': {'temperature': .2, 'top_p': .95, 'thinking': True,
                            'reasoning_effort': 'low'},
                'visible_input_allowlist': frame['visible_input_allowlist'],
                'batch_size': 16, 'readers': 2, 'max_tokens_per_rating': 1024})
            good = {'rating': {'start': True}, 'finish_reason': 'stop',
                    'generated_tokens': 12, 'raw_completion': 'yes'}
            bad = {'rating': {'start': False}, 'finish_reason': 'stop',
                   'generated_tokens': 12, 'raw_completion': 'no'}
            commits = []
            all_attempts = []
            for reader_index in (0, 1):
                history = []
                for attempt_index in range(2 if reader_index == 0 else 1):
                    attempt = save(root / 'attempts' /
                                   f'000000-reader{reader_index}-attempt{attempt_index:03d}.json', {
                        'schema': 'mechanism-start-reader-attempt-v2',
                        'binding_sha256': binding['sha256'], 'start': 0,
                        'reader': reader_index, 'attempt_index': attempt_index,
                        'recovery_of_uncommitted_attempts': [a['sha256'] for a in history],
                        'uids': ['u0', 'u1'],
                        'seeds': [prep.rating_v2.rating_seed(r['uid'], reader_index)
                                  for r in rows],
                        'job_id': 'synthetic', 'state': 'started_before_model_chat'})
                    history.append(attempt)
                    all_attempts.append(attempt['sha256'])
                committed = history[-1]
                for row in rows:
                    save(root / 'assignments' /
                         f"{row['uid']}-reader{reader_index}-attempt"
                         f"{committed['attempt_index']:03d}.json", {
                        'schema': 'mechanism-start-rating-assignment-v2',
                        'binding_sha256': binding['sha256'], 'uid': row['uid'],
                        'reader': reader_index, 'start': 0,
                        'attempt_index': committed['attempt_index'],
                        'attempt_sha256': committed['sha256'],
                        'state': 'attempted_before_model_chat', 'job_id': 'synthetic'})
                result = save(root / 'batches' / f'000000-reader{reader_index}.json', {
                    'schema': 'mechanism-start-reader-batch-v2',
                    'binding_sha256': binding['sha256'], 'start': 0,
                    'reader': reader_index, 'attempt_index': committed['attempt_index'],
                    'attempt_sha256': committed['sha256'],
                    'timing': {'reader': reader_index},
                    'records': [{'uid': 'u0', 'result': good},
                                {'uid': 'u1', 'result': bad}]})
                commits.append(result)
            save(root / 'batches' / '000000.json', {
                'schema': 'mechanism-start-rating-batch-v2',
                'binding_sha256': binding['sha256'], 'start': 0,
                'reader_timings': [r['timing'] for r in commits],
                'records': [{'uid': row['uid'], 'transition': row['transition'],
                             'readers': [commits[0]['records'][i]['result'],
                                         commits[1]['records'][i]['result']]}
                            for i, row in enumerate(rows)]})
            save(root / 'SUMMARY.json', {
                'schema': 'mechanism-start-reader-summary-v2',
                'binding_sha256': binding['sha256'],
                'counts': {'rows': 2, 'ratings': 4, 'start_agree_true': 1,
                           'physical_attempts_recorded': 3,
                           'uncommitted_attempts_recorded': 1},
                'reader_timings': [r['timing'] for r in commits],
                'attempt_sha256s': all_attempts})
            with patch.object(prep.rating_v2, 'PRICE', root / 'price.json'):
                chosen, _, _ = prep.audited_reader_results(selection, frame, root)
                self.assertEqual([r['uid'] for r in chosen], ['u0'])
                damaged = dict(commits[0])
                damaged.pop('sha256')
                damaged['attempt_index'] = 0
                save(root / 'batches' / '000000-reader0.json', damaged)
                with self.assertRaisesRegex(ValueError, 'committed reader'):
                    prep.audited_reader_results(selection, frame, root)

    def test_only_two_valid_positive_readers_are_enrolled(self):
        selected = [
            {'uid': 'a0', 'family': 'a', 'transition': 'candidate_to_verify'},
            {'uid': 'a1', 'family': 'a', 'transition': 'candidate_to_verify'},
            {'uid': 'a2', 'family': 'a', 'transition': 'approach_to_commit'},
            {'uid': 'b0', 'family': 'b', 'transition': 'candidate_to_verify'},
        ]
        good = {'rating': {'start': True}, 'finish_reason': 'stop'}
        bad = {'rating': {'start': False}, 'finish_reason': 'stop'}
        capped = {'rating': {'start': True}, 'finish_reason': 'length'}
        votes = [[good, bad], [good, good], [good, capped], [bad, bad]]
        rated = [{'uid': row['uid'], 'readers': pair}
                 for row, pair in zip(selected, votes)]
        chosen, positives = prep.first_accepted_by_family_transition(selected, rated)
        self.assertEqual([r['uid'] for r in chosen], ['a1'])
        self.assertEqual(positives, 1)
        with self.assertRaisesRegex(ValueError, 'reader rows differ'):
            prep.first_accepted_by_family_transition(selected, rated[:-1])

    def test_four_positions_balance_within_transition_and_globally(self):
        rows = ([{'uid': f'c{i}', 'family': f'family{i}',
                  'transition': 'candidate_to_verify'} for i in range(13)] +
                [{'uid': f'a{i}', 'family': f'family{i+13}',
                  'transition': 'approach_to_commit'} for i in range(5)])
        random_sets, orders = prep.schedule(rows)
        self.assertEqual(len(random_sets), 18)
        self.assertEqual(len(orders), 36)
        self.assertEqual(set(random_sets.values()), {0, 1, 2, 3})
        for transition in ('candidate_to_verify', 'approach_to_commit', 'all'):
            keys = [f"{r['uid']}|{seed}" for r in rows
                    if transition == 'all' or r['transition'] == transition
                    for seed in prep.SEEDS]
            for arm in prep.ARMS:
                counts = [sum(orders[key][position] == arm for key in keys)
                          for position in range(4)]
                self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(prep.schedule(list(reversed(rows))), (random_sets, orders))

    def test_price_charges_full_horizon_and_two_loads(self):
        rows = [{'prompt_ids': [1] * 100, 'prefix_ids': [2] * 100}
                for _ in range(10)]
        manifest = {'sha256': 'manifest', 'rows': rows,
                    'expected_requests': 80,
                    'expected_prefill_tokens': 16000,
                    'maximum_decode_tokens': 80 * 1024}
        reference = {'schema': 'routing-eligible-micro-serial-price-v4',
                     'sha256': 'reference', 'cold_load_seconds': 700,
                     'shutdown_seconds': 200, 'repeat_factor': 1.25,
                     'serial_prefill_stress_tokens_per_second': 1000,
                     'serial_decode_stress_tokens_per_second': 8, 'gpus': 2}
        priced = prep.price_stage(manifest, reference, 86400)
        expected = 2 * (700 + 200) + 1.25 * (16 + 80 * 1024 / 8)
        self.assertEqual(priced['estimated_complete_wall_seconds'], expected)
        self.assertEqual(priced['maximum_decode_tokens'], 80 * 1024)
        self.assertEqual(priced['cold_loads'], 2)
        self.assertTrue(priced['status'].startswith('HOLD_'))
        larger = dict(manifest, rows=rows * 20, expected_requests=1600,
                      expected_prefill_tokens=16000 * 20,
                      maximum_decode_tokens=1600 * 1024)
        long_price = prep.price_stage(larger, reference, 86400)
        self.assertGreater(long_price['planned_jobs'], 2)
        self.assertEqual(long_price['cold_loads'], long_price['shutdowns'])
        self.assertEqual(long_price['shards'][0]['start_row'], 0)
        self.assertEqual(long_price['shards'][-1]['end_row'], len(larger['rows']))


if __name__ == '__main__':
    unittest.main()
