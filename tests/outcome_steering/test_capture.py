"""The capture roster is outcome-free and identical for all models; rows carry the native-id capture."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moe_exp.outcome_steering import capture


def pool():
    """A small question pool: two questions of one benchmark share a family."""
    questions = [f'{b}|{b}_{i}' for b, n in (('aime24', 4), ('math500', 6)) for i in range(n)]
    families = {q: 'f-'+q for q in questions}
    families['math500|math500_1'] = families['math500|math500_0']
    return questions, families


class RosterTests(unittest.TestCase):
    def setUp(self):
        questions, families = pool()
        self.folder = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        folds = Path(self.folder.name)/'FOLDS.json'
        folds.write_text(json.dumps({'question_families': families, 'folds': {f: 1 for f in families.values()}}))
        table = {q: {'question': q, 'attempt_id': 'a-'+q} for q in questions}
        self.patches = [mock.patch.object(capture, 'FOLDS', folds),
                        mock.patch.object(capture, 'PER_BENCHMARK', {'aime24': 2, 'math500': 4}),
                        mock.patch.object(capture, 'sources', lambda model: table)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.folder.cleanup()

    def test_roster_is_fixed_stratified_and_one_question_per_family(self):
        first, second = capture.roster(['gpt', 'glm']), capture.roster(['glm'])
        self.assertEqual(first, second)
        self.assertEqual(sum(q.startswith('aime24') for q in first), 2)
        self.assertEqual(sum(q.startswith('math500') for q in first), 4)
        families = json.loads(Path(capture.FOLDS).read_text())['question_families']
        self.assertEqual(len({families[q] for q in first}), len(first))

    def test_rows_have_distinct_seeds_capped_allowance_and_native_id_capture(self):
        questions = capture.roster(['gpt'])
        profile = {'profile_sha256': 'p', 'layers': [0, 1],
                   'card': {'max_model_len': 131072, 'max_tokens': None}}
        spec = {'questions': questions, 'attempts': 6, 'cap_tokens': 32768}
        rows = capture.capture_rows('gpt', spec, profile, {q: [1, 2, 3] for q in questions})
        self.assertEqual(len(rows), 6*len(questions))
        self.assertEqual(len({r['seed'] for r in rows}), len(rows))
        self.assertEqual(len({r['uid'] for r in rows}), len(rows))
        self.assertTrue(all(r['capture_mode'] == 'native_ids' and not r['audit_capture'] and r['arm'] == 'native'
                            and r['total_allowance'] == r['cap_tokens'] == r['horizon'] == 32768 for r in rows))


if __name__ == '__main__':
    unittest.main()
