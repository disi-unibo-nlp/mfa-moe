from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r6 import resume
from moe_exp.research_campaign_v3_r6.contracts import publish
from moe_exp.research_campaign_v3_r6_lookup_r1 import scoped_lookup, implementation, read_only_epoch


class ScopedLookupContracts(unittest.TestCase):
    def test_shared_ancestors_are_read_once_and_missing_is_not_cached_between_calls(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            roots = [Path(temporary) / str(i) / 'measurements' for i in range(7)]
            for i, root in enumerate(roots):
                root.mkdir(parents=True)
                publish(root.parent / 'INHERITED_UNITS.json', {
                    'directories': {'measurements': [str(p) for p in roots[:i]]},
                    'compatible_statuses': ['complete'],
                })
            original = implementation
            reads = []
            native_load = resume.load
            def read(path):
                reads.append(str(path))
                return native_load(path)
            wrapped = scoped_lookup(original)
            with patch.object(resume, 'inherited_state', wrapped), patch.object(resume, 'load', read), read_only_epoch():
                self.assertIsNone(wrapped(roots[-1], 'a' * 64))
                self.assertLessEqual(len(reads), len(roots))
            with patch.object(resume, 'inherited_state', wrapped):
                publish(roots[0] / (('a' * 64) + '.result.json'), {'status': 'complete', 'value': 17})
                self.assertEqual(wrapped(roots[-1], 'a' * 64)['value'], 17)
                uid = 'e' * 64
                self.assertIsNone(wrapped(roots[-1], uid))
                publish(roots[0] / (uid + '.start-0.json'), {'status': 'started'})
                self.assertEqual(wrapped(roots[-1], uid)['status'], 'started')

    def test_first_depth_first_result_and_started_precedence_match_original(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            roots = [Path(temporary) / str(i) / 'measurements' for i in range(4)]
            for root in roots:
                root.mkdir(parents=True)
            for index, parents in ((0, []), (1, [0]), (2, []), (3, [1, 2])):
                publish(roots[index].parent / 'INHERITED_UNITS.json', {
                    'directories': {'measurements': [str(roots[i]) for i in parents]},
                    'compatible_statuses': ['complete'],
                })
            uid = 'b' * 64
            publish(roots[0] / (uid + '.result.json'), {'status': 'complete', 'value': 'deep-first'})
            publish(roots[2] / (uid + '.result.json'), {'status': 'complete', 'value': 'later'})
            expected = resume.inherited_state(roots[3], uid)
            wrapped = scoped_lookup(implementation)
            with patch.object(resume, 'inherited_state', wrapped):
                self.assertEqual(wrapped(roots[3], uid), expected)
            uid = 'c' * 64
            publish(roots[1] / (uid + '.start-0.json'), {'status': 'started'})
            expected = resume.inherited_state(roots[3], uid)
            with patch.object(resume, 'inherited_state', wrapped):
                self.assertEqual(wrapped(roots[3], uid), expected)

    def test_cycle_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary) / 'measurements'
            root.mkdir()
            publish(root.parent / 'INHERITED_UNITS.json', {
                'directories': {'measurements': [str(root)]}, 'compatible_statuses': ['complete']})
            wrapped = scoped_lookup(implementation)
            with patch.object(resume, 'inherited_state', wrapped):
                with self.assertRaises(ValueError):
                    wrapped(root, 'd' * 64)

    def test_changed_ancestor_epoch_rejected_before_return(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            parent = Path(temporary) / 'parent' / 'measurements'
            current = Path(temporary) / 'current' / 'measurements'
            parent.mkdir(parents=True); current.mkdir(parents=True)
            publish(current.parent / 'INHERITED_UNITS.json', {
                'directories': {'measurements': [str(parent)]}, 'compatible_statuses': ['complete']})
            wrapped = scoped_lookup(implementation); uid = 'f' * 64
            with patch.object(resume, 'inherited_state', wrapped):
                with self.assertRaisesRegex(ValueError, 'directory changed'):
                    with read_only_epoch():
                        self.assertIsNone(wrapped(current, uid))
                        publish(parent / (uid + '.start-0.json'), {'status': 'started'})
                self.assertEqual(wrapped(current, uid)['status'], 'started')


if __name__ == '__main__':
    unittest.main()
