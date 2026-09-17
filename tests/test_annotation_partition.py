import importlib
import unittest
import json
import tempfile
from pathlib import Path
from moe_exp.schemas import TraceRecord


def fixture(root, reverse=False, duplicate=False, legacy=False):
    paths = {}
    for dataset in ['a', 'b']:
        directory = root / dataset
        directory.mkdir(parents=True)
        rows = [TraceRecord(dataset=dataset, problem_id='same', source_problem_id='source',
                            sample_id=i, prompt='Question?', gold_answer='1', model_id='test',
                            model_answer='1', is_correct=False if i == 0 else None,
                            cot_text='First. Second. Third.',
                            metadata={'sentence_selection': {}} if legacy else {}).model_dump()
                for i in range(2)]
        if duplicate:
            rows.append(rows[0])
        path = directory / 'traces.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in (rows[::-1] if reverse else rows)))
        (directory / 'manifest.json').write_text('{}')
        paths[dataset] = path
    return paths


def api():
    return importlib.import_module('moe_exp.correlation_pipeline.annotation_partition')


class PopulationTests(unittest.TestCase):
    def test_capped_quotas(self):
        quotas, rounds = api().capped_quotas({'a': 1, 'b': 10, 'c': 10}, {'a': 8, 'b': 1, 'c': 1}, 8)
        self.assertEqual(quotas, {'a': 1, 'b': 4, 'c': 3})
        self.assertTrue(rounds)
        self.assertEqual(api().capped_quotas({'a': 9, 'b': 9}, {'a': 0, 'b': 1}, 4)[0], {'a': 0, 'b': 4})
        for capacities, weights in [({'a': 1}, {'a': 1}), ({'a': 10}, {'a': 0})]:
            with self.assertRaises(ValueError):
                api().capped_quotas(capacities, weights, 4)

    def test_partition_items_are_exact_and_disjoint(self):
        # Production parts are tuples of (identity, trace, unit, inputs); the digest binds item[0].
        items = [({'id': i}, None, None, None) for i in range(8)]
        parts = api().partition_items(items, part_size=2, parts=4)
        self.assertEqual([len(part) for part in parts], [2, 2, 2, 2])
        self.assertEqual(
            [item[0]['id'] for part in parts for item in part],
            list(range(8)),
        )
        with self.assertRaises(ValueError):
            api().partition_items(items[:-1], part_size=2, parts=4)
        with self.assertRaises(ValueError):
            api().partition_items(items + [items[0]], part_size=2, parts=4)


if __name__ == '__main__':
    unittest.main()
