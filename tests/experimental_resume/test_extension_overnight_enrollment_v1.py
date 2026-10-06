"""Eligibility selection must preserve the frozen order and strict-veto sensitivity."""
import sys
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import prepare_extension_overnight_enrollment_v1 as prep


class EnrollmentTests(unittest.TestCase):
    def source_rows(self):
        rows = [{'uid': str(i), 'family': 'family-b', 'transition': 'candidate_to_verify',
                 'attempt_id': 'attempt-b', 'sentence_index': i, 'segment': 0,
                 'prefix_tokens': 100 + i} for i in range(3)]
        return sorted(rows, key=lambda r: prep.extension.event_key(r['transition'], r))

    def rating(self, row, accept, veto=False, finish='stop'):
        return {'uid': row['uid'],
                'primary_readers': [{'rating': {'start': accept}, 'finish_reason': finish}] * 2,
                'veto_readers': [{'already_completed': veto, 'finish_reason': 'stop'}] * 2}

    def test_first_primary_accept_survives_veto_and_keeps_nonfire_families(self):
        rows = self.source_rows()
        ratings = [self.rating(rows[0], False), self.rating(rows[1], True, veto=True),
                   self.rating(rows[2], True)]
        chosen, accounting = prep.choose(rows, ratings, ['family-b', 'family-a'])
        self.assertEqual([r['uid'] for r in chosen], [rows[1]['uid']])
        self.assertFalse(chosen[0]['strict_veto_sensitivity_eligible'])
        self.assertEqual(len(accounting), 4)
        self.assertEqual(accounting[-1]['family'], 'family-a')
        self.assertEqual(accounting[-1]['proposed_start_uids'], [])

    def test_length_stopped_or_missing_consensus_start_is_not_accepted(self):
        rows = self.source_rows()
        ratings = [self.rating(rows[0], True, finish='length'),
                   self.rating(rows[1], True), self.rating(rows[2], False)]
        ratings[1]['primary_readers'] = [{'rating': {'start': True}, 'finish_reason': 'stop'},
                                         {'rating': None, 'finish_reason': 'stop'}]
        chosen, _ = prep.choose(rows, ratings, ['family-b'])
        self.assertEqual(chosen, [])

    def test_reordered_or_duplicate_reader_rows_fail(self):
        rows = self.source_rows()
        ratings = [self.rating(row, True) for row in rows]
        with self.assertRaisesRegex(ValueError, 'frozen proposal order'):
            prep.choose(rows, ratings[::-1], ['family-b'])
        with self.assertRaisesRegex(ValueError, 'event hashes'):
            prep.choose(rows[::-1], ratings[::-1], ['family-b'])

    def test_plan_fixes_fresh_population_before_reader_results(self):
        value = prep.plan_body()
        self.assertEqual(len(value['families']), 220)
        self.assertEqual(value['generation_horizon'], 1024)
        self.assertEqual(value['generation_arm_counts'], {'candidate_to_verify': 14,
                                                         'approach_to_commit': 10})
        self.assertEqual(value['generation_seeds'], [0, 1])
        self.assertNotIn('reader_stage_completion_sha256', value)


if __name__ == '__main__':
    unittest.main()
