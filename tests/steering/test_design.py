import unittest
from collections import Counter
from moe_exp.io import BENCHMARKS, CLASSES
from moe_exp.steering.allocation import allocate, assignments, strengths


def slots():
    return [{'benchmark': b, 'family': f'{b}-{i:03}', 'fold': i % 5}
            for j, b in enumerate(BENCHMARKS) for i in range(22 if j < 3 else 21)]


class DesignTests(unittest.TestCase):
    def test_fallback_preserves_primary_population_and_known_probabilities(self):
        a = allocate(slots(), 42)
        self.assertEqual(a, allocate(list(reversed(slots())), 42))
        self.assertEqual(Counter(r['fallback_outer'] for r in a['records']), {None: 97, 'low': 16, 'high': 16})
        for b in BENCHMARKS:
            rows = [r for r in a['records'] if r['benchmark'] == b]
            n = sum(r['fallback_outer'] == 'low' for r in rows)
            self.assertGreaterEqual(n, 2)
            self.assertEqual(n, sum(r['fallback_outer'] == 'high' for r in rows))
            self.assertAlmostEqual(rows[0]['fallback_dose_probability'], n/len(rows))

    def test_exact_execution_counts_and_shared_fresh_seed(self):
        a = allocate(slots(), 42)
        prefixes = [{**r, 'status': 'enrolled', 'prefix': {'attempt_id': r['family'],
                     'question': r['family'], 'prefix_tokens': 50, 'sentence_index': 4}} for r in slots()]
        ns = [{'fold': f, 'class': c, 'depth_third': d, 'status': 'nominated', 'layer': d,
               'pulse_tokens': 16, 'horizon': 320, 'expert_sets': [[0], [0, 1], [0, 1, 2]],
               'control_sets': [[4], [4, 5], [4, 5, 6]]}
              for f in range(5) for c in CLASSES for d in range(3)]
        full = assignments('test', 'profile', prefixes, ns, a, 42, 'full')
        fallback = assignments('test', 'profile', prefixes, ns, a, 42, 'fallback')
        self.assertEqual(len(full), 129*169)
        self.assertEqual(len(fallback), 129*85+32*84)
        central = lambda rows: {r['uid'] for r in rows if r['dose'] == 'central'}
        self.assertEqual(central(full), central(fallback))
        for family in ('aime24-000', 'math500-001'):
            self.assertEqual(len({r['seed'] for r in full if r['family'] == family}), 1)
        self.assertNotEqual(full[0]['seed'], 42)

    def test_unavailable_slots_are_retained_without_replacement(self):
        a = allocate(slots(), 42)
        prefixes = [{**r, 'status': 'upstream_unavailable', 'prefix': None} for r in slots()]
        rows = assignments('test', 'profile', prefixes, [], a, 42, 'fallback')
        self.assertEqual(len({r['family'] for r in rows}), 129)
        self.assertTrue(all(r['prefix_status'] == 'upstream_unavailable' for r in rows))

    def test_native_counts(self):
        self.assertEqual(strengths(4), [1, 2, 3])
        self.assertEqual(strengths(6), [2, 3, 5])
        self.assertEqual(strengths(8), [2, 4, 6])


if __name__ == '__main__':
    unittest.main()
