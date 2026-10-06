"""Conservative format-only recovery and complete replay price checks."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/experimental_resume')]
import adjudicate_extension_veto_format_v1 as a


def vote(text, finish='stop'):
    return {'raw_completion': text, 'already_completed': a.contract.veto.parse_veto(text),
            'finish_reason': finish, 'generated_tokens': 1024 if finish == 'length' else 50}


class FormatTests(unittest.TestCase):
    def test_plain_boolean_and_thinking_suffix(self):
        for value in ('true', 'false'):
            raw = 'reasoning </think>  {"already_completed": '+value+'} '
            parsed, form = a.parse_final(raw)
            self.assertEqual(parsed, value == 'true')
            self.assertEqual(form, 'bare_json')
            self.assertTrue(a.adjudicate_vote(vote(raw))['original_valid'])

    def test_sole_json_or_untyped_fence_recovers_both_values(self):
        for language in ('json', ''):
            for value in ('true', 'false'):
                raw = 'reasoning </think>\n```'+language+'\n{"already_completed": '+value+'}\n```\n'
                result = a.adjudicate_vote(vote(raw))
                self.assertTrue(result['format_recovered'])
                self.assertEqual(result['adjudicated_value'], value == 'true')

    def test_extra_prose_multiple_objects_and_fences_rejected(self):
        obj = '{"already_completed":false}'
        for raw in ('Answer: '+obj, obj+' trailing', obj+' '+obj,
                    '```json\n'+obj+'\n```\n'+obj,
                    '```json\n'+obj+'\n```\n```json\n'+obj+'\n```',
                    '```javascript\n'+obj+'\n```', '```json '+obj+'```'):
            self.assertIsNone(a.parse_final(raw)[0], raw)

    def test_duplicate_key_wrong_schema_type_or_malformed_rejected(self):
        for raw in ('{"already_completed":true,"already_completed":false}',
                    '{"already_completed":"false"}', '{"already_completed":0}',
                    '{"already_completed":null}', '{"already_completed":false,"note":"x"}',
                    '[false]', '{"already_completed":false,}', None):
            self.assertIsNone(a.parse_final(raw)[0])

    def test_length_result_stays_invalid_and_original_result_must_match(self):
        raw = '```json\n{"already_completed":false}\n```'
        result = a.adjudicate_vote(vote(raw, 'length'))
        self.assertFalse(result['adjudicated_valid'])
        self.assertEqual(result['residual_reason'], 'length')
        inconsistent = vote('{"already_completed":false}')
        inconsistent['already_completed'] = True
        with self.assertRaisesRegex(ValueError, 'saved raw text'):
            a.adjudicate_vote(inconsistent)

    def test_pricing_includes_full_population_qualification_and_recovery(self):
        rows = [{'key': str(i), 'prompt_tokens': 1000, 'transition': 'candidate_to_verify'} for i in range(18)]
        source = {'sha256':'source','bounded_prefill_tokens_per_second':1000.,
                  'bounded_decode_tokens_per_second':50.,'retry_factor':1.25}
        previous = {'sha256':'previous','components_seconds': {'two_cold_loads':1000.,'two_shutdowns':60.}}
        result = a.price_option(rows, 4096, source, previous)
        self.assertEqual(result['ratings'], 18)
        self.assertEqual(result['max_decode_tokens'],18*4096)
        self.assertEqual(sum(len(s['rating_keys']) for s in result['shards']),18)
        self.assertGreater(result['qualification_GPU_h'],0)
        self.assertEqual(result['recovery_loads_and_shutdowns'],1)
        self.assertAlmostEqual(result['complete_GPU_h'],result['qualification_GPU_h']+result['production_and_recovery_GPU_h'])
        self.assertTrue(all(s['complete_wall_seconds']<=7200 for s in result['shards']))

    def test_zero_residuals_require_zero_GPU_hours(self):
        self.assertEqual(a.price_option([],2048,{}, {})['complete_GPU_h'],0.)


if __name__ == '__main__':
    unittest.main()
