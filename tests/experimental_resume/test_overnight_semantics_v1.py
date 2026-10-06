"""Checks on blinding, exact assignments, horizon-specific ITT and durable votes."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import build_overnight_blind_frame_v1 as builder
import rate_overnight_semantics_v1 as rating
import price_overnight_semantics_v1 as pricing
import analyze_overnight_semantics_v1 as analysis


def seal(path, body):
    value = {**body, 'sha256': builder.base.digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def fixture():
    arms = [{'name': name, 'role': role, 'slots': [] if role == 'native' else [0],
             'policies': {} if role == 'native' else {'candidate_to_verify': ['t' if role == 'target' else 'r{random_set}']}}
            for name, role in [('native', 'native'), ('native_duplicate', 'native'),
                               ('target', 'target'), ('random', 'random')]]
    rows, sources = [], []
    for index in range(2):
        uid, family, text = f'p{index}', f'f{index}', f'Maybe x={index}. Check this candidate.'
        r = {'uid': uid, 'family': family, 'transition': 'candidate_to_verify', 'question': 'Compute x.',
             'canonical_question': f'question|{index}', 'attempt_id': f'a{index}', 'sentence_index': 4,
             'prompt_ids': [10, 11], 'prefix_ids': [12, 13], 'prefix_tokens': 2,
             'trace_sha256': 'trace', 'tokenizer_sha256': 'tokenizer',
             'prompt_ids_sha256': builder.base.digest([10, 11]), 'prefix_ids_sha256': builder.base.digest([12, 13]),
             'prefix_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        rows.append(r)
        sources.append({k: r[k] for k in ('uid', 'family', 'transition', 'attempt_id', 'sentence_index', 'prefix_tokens')} |
            {'reader_input': {'problem': r['question'], 'emitted_prefix': text, 'triggering_sentence': 'Check this candidate.'},
             'analysis_meta': {'prefix_ids_sha256': r['prefix_ids_sha256'], 'prefix_text_sha256': r['prefix_text_sha256'],
                              'native_trace_sha256': 'trace', 'tokenizer_sha256': 'tokenizer'}})
    manifest = {'schema': 'overnight-routing-manifest-v1', 'sha256': 'manifest', 'rows': rows, 'seeds': [0, 1],
        'arms': arms, 'horizon': 256, 'expected_requests': 16, 'code_files': {},
        'random_set_by_family_seed': {'f0': [0, 1], 'f1': [2, 3]},
        'arm_order_by_uid_seed': {f'{r["uid"]}|{s}': [a['name'] for a in arms] for r in rows for s in (0, 1)},
        'shards': [{'start_row': 0, 'end_row': 2}],
        'planned_contrasts': [['target', 'native'], ['target', 'random'], ['native_duplicate', 'native']],
        'analysis_scopes': ['all', 'candidate_to_verify'], 'analysis_seed': 20261004, 'bootstrap_replicates': 50000}
    price = {'schema': 'overnight-routing-price-v1', 'sha256': 'generation-price', 'manifest_sha256': 'manifest',
             'status': 'PASS_COMPLETE_STAGE_GENERATION_ONLY', 'shards': manifest['shards']}
    source = {'schema': 'transition-v22-full-prefix-start-frame-v1',
              'sha256': '6c10499bf71c7223b5c90fa6c2a1b13e13e9fb5c9c2af584d3ba67b4272290b2', 'records': sources}
    return manifest, price, source


def generation_outputs(root, manifest):
    directory = root / 'shard-000'
    plan = builder.expected_assignments(manifest)
    binding = seal(directory / 'BINDING.json', {'schema': 'routing-boundary-micro-screen-binding-v1',
                                               'manifest_sha256': manifest['sha256']})
    summary = seal(directory / 'SUMMARY.json', {'schema': 'routing-boundary-micro-screen-summary-v1',
        'status': 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION', 'manifest_sha256': manifest['sha256'],
        'binding_sha256': binding['sha256'], 'requests': len(plan), 'batches': 1})
    completion = seal(directory / 'OVERNIGHT_COMPLETION.json', {'schema': 'overnight-routing-completion-v1',
        'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION', 'manifest_sha256': manifest['sha256'],
        'source_summary_sha256': summary['sha256'], 'shard': 0, 'counts': {'assigned': len(plan)}})
    seal(root / 'STAGE_COMPLETION.json', {'schema': 'overnight-routing-stage-completion-v1',
        'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION', 'manifest_sha256': manifest['sha256'],
        'counts': {'assigned': len(plan)}, 'shard_completion_sha256s': [completion['sha256']]})
    seal(directory / 'batch-000-assignment.json', {'schema': 'routing-boundary-micro-screen-assignment-v1',
        'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'], 'batch_index': 0, 'requests': plan})
    array = directory / 'batch-000.npz'
    array.write_bytes(b'test-route-array')
    outputs = [{**p, 'tokens': [31, 32], 'finish': 'stop', 'stop_reason': None, 'error': None,
                'routed_present': True, 'action_dose': []} for p in plan]
    seal(directory / 'batch-000.json', {'schema': 'routing-boundary-micro-screen-batch-v1',
        'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'], 'batch_index': 0,
        'array_sha256': builder.base.file_sha(array), 'outputs': outputs})
    return outputs


class OvernightSemanticsTests(unittest.TestCase):
    def test_assignment_identity_horizon_and_duplicate_detection(self):
        m, _, _ = fixture()
        self.assertEqual(len(builder.expected_assignments(m)), 16)
        m['arm_order_by_uid_seed']['p0|0'][-1] = 'target'
        with self.assertRaisesRegex(ValueError, 'schedule'):
            builder.expected_assignments(m)

    def test_complete_outputs_rejects_array_tamper_and_missing_stage(self):
        m, p, _ = fixture()
        with tempfile.TemporaryDirectory(prefix='.overnight-test-', dir=REPO) as name:
            root = Path(name)
            with self.assertRaises(FileNotFoundError):
                builder.complete_outputs(m, p, root)
            generation_outputs(root, m)
            self.assertEqual(len(builder.complete_outputs(m, p, root)[2]), 16)
            (root / 'shard-000/batch-000.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'batch assignments'):
                builder.complete_outputs(m, p, root)

    def test_frame_blinds_arms_preserves_all_assignments_and_resumes_same_salt(self):
        m, p, s = fixture()
        class Tokenizer:
            @staticmethod
            def decode(ids, **kwargs):
                return '2+2=4.'
        with tempfile.TemporaryDirectory(prefix='.overnight-test-', dir=REPO) as name:
            root = Path(name)
            generation_outputs(root / 'generation', m)
            arm_map, frame = builder.build(m, p, s, root / 'generation', root / 'measurement', Tokenizer(), b'a' * 32)
            self.assertEqual(frame['continuation_max_tokens'], 256)
            self.assertEqual(len(arm_map['records']), 16)
            self.assertEqual(len(frame['records']), 16)
            for row in frame['records']:
                self.assertEqual(set(row), {'blind_id', 'reader_input'})
                self.assertEqual(set(row['reader_input']), rating.ALLOWLIST)
            self.assertEqual(builder.build(m, p, s, root / 'generation', root / 'measurement', Tokenizer())[1], frame)
            s['records'][0]['reader_input']['emitted_prefix'] += 'future text'
            with self.assertRaisesRegex(ValueError, 'pretreatment'):
                builder.build(m, p, s, root / 'generation', root / 'other', Tokenizer())

    def test_reader_rejects_forbidden_fields_and_nonboolean_json(self):
        row = {'blind_id': 'id', 'reader_input': {'transition': 'candidate_to_verify', 'problem': 'Q',
               'full_emitted_prefix': 'Candidate.', 'triggering_sentence': 'Candidate.', 'continuation': 'Check.'}}
        self.assertIn('context, not outcome evidence', rating.messages(row)[0]['content'])
        row['reader_input']['arm'] = 'target'
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            rating.messages(row)
        self.assertIsNone(rating.parse_rating('{"target":1}'))
        self.assertIsNone(rating.parse_rating('{"target":true,"arm":"target"}'))
        self.assertEqual(rating.parse_rating('</think>{"target":false}'), {'target': False})

    def test_prices_both_horizons_with_same_reader_output_cap(self):
        prior = {'schema': 'mechanism-start-reader-price-v2', 'status': 'PASS_COMPLETE_20_GPUH', 'sha256': 'p',
                 'bounded_decode_tps': 43.71, 'bounded_prefill_tps': 4914,
                 'components_seconds': {'two_cold_loads': 1200, 'two_shutdowns': 392}}
        binding = {'schema': 'mechanism-start-reader-binding-v2', 'sha256': 'b', 'price_sha256': 'p',
                   'model_snapshot': str(rating.MODEL), 'readers': 2}
        summary = {'schema': 'mechanism-start-reader-summary-v2', 'sha256': 's', 'binding_sha256': 'b',
                   'counts': {'rows': 454, 'ratings': 908}}
        for horizon in (256, 1024):
            frame = {'schema': 'overnight-semantic-blind-frame-v1', 'continuation_max_tokens': horizon,
                     'sha256': 'f', 'records': [{}, {}]}
            price = pricing.price_stage(frame, [100, 200], prior, binding, summary, 7200, 100)
            self.assertEqual(price['max_decode_tokens'], 4096)
            self.assertEqual(price['prompt_tokens_exact_twice'], 600)
            self.assertEqual(price['status'], 'PASS_COMPLETE_STAGE')
        empty = {'schema': 'overnight-semantic-blind-frame-v1', 'continuation_max_tokens': 256,
                 'sha256': 'f', 'records': [], 'assigned': 16,
                 'rubric_sha256': rating.base.file_sha(rating.RUBRIC),
                 'reader_input_allowlist': sorted(rating.ALLOWLIST)}
        price = pricing.price_stage(empty, [], prior, binding, summary, 7200, 100)
        rating.validate(empty, price)
        self.assertEqual(price['shards'], [])
        self.assertEqual(price['estimated_complete_gpu_hours'], 0)

    def test_itt_join_retains_invalid_readers_caps_and_failed_generation(self):
        m, p, s = fixture()
        class Tokenizer:
            @staticmethod
            def decode(ids, **kwargs):
                return 'Check.'
        with tempfile.TemporaryDirectory(prefix='.overnight-test-', dir=REPO) as name:
            root = Path(name)
            generation_outputs(root / 'generation', m)
            amap, frame = builder.build(m, p, s, root / 'generation', root / 'measurement', Tokenizer())
            votes = {r['blind_id']: {j: {'rating': {'target': True}, 'finish_reason': 'stop', 'generated_tokens': 5}
                                    for j in (0, 1)} for r in frame['records']}
            bid = amap['records'][0]['blind_id']
            votes[bid][1]['finish_reason'] = 'length'
            rows = analysis.join(m, amap, frame, votes)
            self.assertEqual(len(rows), 16)
            self.assertFalse(rows[0]['both_positive'])
            self.assertTrue(rows[0]['at_least_one_positive'])
            self.assertEqual(rows[0]['operational_status'], 'unscored')
            frame['continuation_max_tokens'] = 1024
            with self.assertRaisesRegex(ValueError, 'horizon'):
                analysis.join(m, amap, frame, votes)

    def test_family_itt_and_missing_sensitivity_keep_every_arm_seed(self):
        m, _, _ = fixture()
        records = []
        for r in builder.expected_assignments(m):
            positive = r['arm'] == 'target' and r['family'] == 'f0'
            records.append({**r, 'both_positive': positive, 'at_least_one_positive': positive,
                            'emitted_tokens': 256, 'measurement_unknown': r['arm'] == 'random'})
        result = analysis.clustered_contrasts(m, records, n_boot=200)
        self.assertEqual(result['scope_aliases'], {'candidate_to_verify': 'all'})
        self.assertEqual(result['multiplicity'], 6)
        effect = next(r for r in result['contrasts'] if r['endpoint'] == 'both_positive' and r['reference'] == 'native')
        self.assertEqual(effect['estimate'], .5)
        unknown = next(r for r in analysis.missing_sensitivity(m, records)['contrasts'] if r['reference'] == 'random')
        self.assertEqual(unknown['unknown_outcome_identification_bounds'], [-.5, .5])
        with self.assertRaisesRegex(ValueError, 'missing assigned arm'):
            analysis.clustered_contrasts(m, records[:-1], n_boot=200)


if __name__ == '__main__':
    unittest.main()
