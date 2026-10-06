"""Fresh-pool explicit lineage and unequal transition-arm coverage checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_overnight_semantics_v1 as fixtures
import build_overnight_blind_frame_v2 as builder
import analyze_overnight_semantics_v2 as analysis
import price_overnight_semantics_v2 as pricing
import rate_overnight_semantics_v2 as rating


def fixture():
    m, p, source = fixtures.fixture()
    m.update(schema='overnight-routing-manifest-v2',
             arms_by_transition={'candidate_to_verify': m['arms']},
             source_frame_path='/explicit/frozen/frame.json', source_frame_sha256=source['sha256'],
             source_enrollment_path='/explicit/frozen/enrollment.json', claim_limit='exploratory')
    m['random_set_by_family_transition_seed'] = {
        f'{family}|candidate_to_verify|{seed}': values[seed]
        for family, values in m['random_set_by_family_seed'].items() for seed in (0, 1)}
    m['planned_contrasts_scoped'] = [{'scope': scope, 'left': pair[0], 'right': pair[1]}
        for scope in m['analysis_scopes'] for pair in m['planned_contrasts']]
    p['schema'] = 'overnight-routing-price-v2'
    return m, p, source


class OvernightV2Tests(unittest.TestCase):
    def test_v2_identity_stage_and_source_hash(self):
        m, p, source = fixture()
        class Tokenizer:
            @staticmethod
            def decode(ids, **kwargs):
                return 'Check.'
        with tempfile.TemporaryDirectory(prefix='.overnight-v2-test-', dir=fixtures.REPO) as name:
            root = Path(name)
            with patch.object(fixtures, 'builder', builder):
                fixtures.generation_outputs(root / 'generation', m)
            stage_path = root / 'generation/STAGE_COMPLETION.json'
            stage = json.loads(stage_path.read_text())
            stage.pop('sha256')
            stage['schema'] = 'overnight-routing-stage-completion-v2'
            fixtures.seal(stage_path, stage)
            amap, frame = builder.build(m, p, source, root / 'generation', root / 'measurement', Tokenizer())
            self.assertEqual(len(frame['records']), 16)
            self.assertTrue(all(r['uid'].startswith('overnight-v2|') for r in amap['records']))
            bad = {**source, 'sha256': 'another-frame'}
            with self.assertRaisesRegex(ValueError, 'source frame'):
                builder.build(m, p, bad, root / 'generation', root / 'other', Tokenizer())

    def test_fresh_context_binds_exact_tokens_even_when_question_is_canonical_id(self):
        m, _, source = fixture()
        native = copy.deepcopy(m['rows'][0])
        original = copy.deepcopy(source['records'][0])
        native['question'] = native['canonical_question']
        original['analysis_meta']['prompt_ids_sha256'] = native['prompt_ids_sha256']
        original['analysis_meta']['trace_sha256'] = 'a' * 64
        original['analysis_meta']['tokenizer_sha256'] = 'b' * 64
        builder.reader_context(original, native, 'mechanism-extension-start-frame-v1')
        original['analysis_meta']['prompt_ids_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source replay'):
            builder.reader_context(original, native, 'mechanism-extension-start-frame-v1')

    def test_scoped_contrasts_do_not_require_unassigned_singletons(self):
        m, _, _ = fixture()
        originals = copy.deepcopy(m['rows'])
        extra = []
        for r in originals:
            r.update(uid='approach-' + r['uid'], transition='approach_to_commit')
            extra.append(r)
        approach_arms = [a for a in m['arms'] if a['name'] != 'random']
        m['arms_by_transition']['approach_to_commit'] = approach_arms
        m['rows'] += extra
        m['analysis_scopes'] = ['all', 'candidate_to_verify', 'approach_to_commit']
        m['planned_contrasts_scoped'] = [
            {'scope': 'candidate_to_verify', 'left': 'target', 'right': 'random'},
            {'scope': 'approach_to_commit', 'left': 'target', 'right': 'native'}]
        records = []
        for r in m['rows']:
            for seed in (0, 1):
                for arm in m['arms_by_transition'][r['transition']]:
                    records.append({'prefix_uid': r['uid'], 'family': r['family'], 'transition': r['transition'],
                        'seed': seed, 'arm': arm['name'], 'both_positive': arm['name'] == 'target',
                        'emitted_tokens': 256, 'measurement_unknown': False})
        result = analysis.clustered_contrasts(m, records, n_boot=100)
        self.assertEqual(result['multiplicity'], 2)
        self.assertEqual(len(result['contrasts']), 2)
        token_secondary = analysis.clustered_contrasts(m, records, ('emitted_tokens',), n_boot=100)
        self.assertEqual(token_secondary['multiplicity'], 2)
        self.assertEqual(token_secondary['endpoints'], ['emitted_tokens'])
        m['planned_contrasts_scoped'].append({'scope': 'all', 'left': 'target', 'right': 'random'})
        with self.assertRaisesRegex(ValueError, 'unassigned arm'):
            analysis.clustered_contrasts(m, records, n_boot=100)

    def test_empty_strict_subset_reports_insufficient_families(self):
        m, _, _ = fixture()
        for row in m['rows']:
            row['strict_veto_sensitivity_eligible'] = False
        result = analysis.strict_veto_sensitivity(m, [])
        self.assertEqual(result['starts'], 0)
        self.assertTrue(all(r['precision_status'] == 'INSUFFICIENT_FAMILIES'
                            for r in result['result']['contrasts']))

    def test_v2_reader_prices_bind_new_driver_and_rubric(self):
        prior = {'schema': 'mechanism-start-reader-price-v2', 'status': 'PASS_COMPLETE_20_GPUH', 'sha256': 'p',
                 'bounded_decode_tps': 43.71, 'bounded_prefill_tps': 4914,
                 'components_seconds': {'two_cold_loads': 1200, 'two_shutdowns': 392}}
        binding = {'schema': 'mechanism-start-reader-binding-v2', 'sha256': 'b', 'price_sha256': 'p',
                   'model_snapshot': str(rating.MODEL), 'readers': 2}
        summary = {'schema': 'mechanism-start-reader-summary-v2', 'sha256': 's', 'binding_sha256': 'b',
                   'counts': {'rows': 454, 'ratings': 908}}
        frame = {'schema': 'overnight-semantic-blind-frame-v2', 'continuation_max_tokens': 1024,
                 'sha256': 'f', 'assigned': 16, 'records': [{'blind_id': 'id', 'reader_input': {
                     'transition': 'candidate_to_verify', 'problem': 'Q', 'full_emitted_prefix': 'Candidate.',
                     'triggering_sentence': 'Candidate.', 'continuation': 'Check.'}}],
                 'rubric_sha256': rating.base.file_sha(rating.RUBRIC),
                 'reader_input_allowlist': sorted(rating.ALLOWLIST)}
        price = pricing.price_stage(frame, [100], prior, binding, summary, 7200, 100)
        rating.validate(frame, price)
        price['rating_driver_sha256'] = 'old-version'
        with self.assertRaisesRegex(ValueError, 'reader frame, code'):
            rating.validate(frame, price)

    def test_pooled_estimand_weights_families_equally(self):
        m, _, _ = fixture()
        m['rows'].append({**m['rows'][0], 'uid': 'second-start-same-family'})
        m['planned_contrasts_scoped'] = [{'scope': 'all', 'left': 'target', 'right': 'native'}]
        records = []
        first_family = m['rows'][0]['family']
        for row in m['rows']:
            for seed in (0, 1):
                for arm in m['arms']:
                    records.append({'prefix_uid': row['uid'], 'family': row['family'],
                        'transition': row['transition'], 'seed': seed, 'arm': arm['name'],
                        'both_positive': row['family'] == first_family and arm['name'] == 'target',
                        'measurement_unknown': False})
        result = analysis.clustered_contrasts(m, records, n_boot=100)
        self.assertEqual(result['contrasts'][0]['estimate'], .5)
        self.assertEqual(analysis.missing_sensitivity(m, records)['contrasts'][0]
                         ['unknown_outcome_identification_bounds'], [.5, .5])


if __name__ == '__main__':
    unittest.main()
