"""Complement releases: the suppress cells a base release left out, joined to its rows by family."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.steering import pulse as driver, whole, whole_effects
from moe_exp.io import CLASSES, WORK, load, publish

TMP = WORK/'tools/tmp'
SUPPRESSED = ('Plan', 'Implement')  # classes the synthetic base release already suppresses


def native(family: str) -> dict:
    return {'model': 'gemma', 'family': family, 'benchmark': 'aime', 'fold': 0, 'seed': 7, 'total_allowance': 100,
            'arm': 'native', 'class': None, 'operation': 'native', 'actions': [], 'nomination_status': 'nominated',
            'audit_capture': False, 'uid': 'native-'+family}


def cell(nomination, profile, model, dose, operation):
    """A fake cell_actions result; Monitor has no available experts."""
    return {'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation, 'dose': dose,
            'qualified': nomination['class'] != 'Monitor', 'target': [{'layer': 0}], 'random': [{'layer': 1}],
            'mean_native_frequency': {'target': .2, 'random': .3}}


class ComplementManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        self.parent, self.root = Path(self.directory.name)/'parent', Path(self.directory.name)/'child'
        self.nominations = publish(self.parent/'gemma/WHOLE_NOMINATIONS.json', {
            'records': [{'fold': 0, 'class': c} for c in CLASSES]})
        self.natives = [native('f1'), native('f2')]
        steered = [{**native('f1'), 'arm': 'target_suppress', 'class': 'Plan', 'operation': 'suppress',
                    'uid': 'steered'}]
        self.base = publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'scope': 'prompt-and-generated-reasoning',
            'nominations_sha256': self.nominations['sha256'], 'cells': [], 'records': self.natives+steered})
        self.patches = [patch.object(whole, 'ROOT', self.root), patch.object(whole, 'SWEEP', {'gemma': 'main'}),
                        patch.object(whole, 'directions', return_value={'suppress': SUPPRESSED}),
                        patch.object(whole, 'cell_actions', side_effect=cell),
                        patch.object(driver, 'binding', return_value={'models': {'gemma': {}}})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_only_the_missing_suppress_cells_are_planned_on_the_native_rows(self):
        result = whole.complement_manifest('gemma', {'parent': str(self.parent)})
        value = load(whole.manifest_path('gemma', 'main'))
        classes = [c for c in CLASSES if c not in SUPPRESSED]
        self.assertEqual(value['complement_classes'], classes)
        self.assertEqual(result['planned'], 2*2*len(classes))
        self.assertEqual({(r['family'], r['class'], r['arm']) for r in value['records']},
                         {(f, c, a) for f in ('f1', 'f2') for c in classes
                          for a in ('target_suppress', 'random_suppress')})
        self.assertTrue(all(r['seed'] == 7 and r['total_allowance'] == 100 and r['complement']
                            and not r['audit_capture'] for r in value['records']))
        self.assertEqual(len({r['uid'] for r in value['records']}), result['planned'])
        dropped = {r['class'] for r in value['records'] if r['nomination_status'] != 'nominated'}
        self.assertEqual(dropped, {'Monitor'})
        self.assertEqual(value['parent_manifest_sha256'], self.base['sha256'])
        self.assertTrue(load(self.root/'gemma/SELECTED.json')['complement'])

    def test_nominations_of_another_manifest_are_refused(self):
        other = Path(self.directory.name)/'other'
        publish(other/'gemma/WHOLE_NOMINATIONS.json', {'records': []})
        publish(whole.manifest_path('gemma', 'main', other), {**self.base, 'sha256': None})
        with self.assertRaisesRegex(ValueError, 'parent nominations differ'):
            whole.complement_manifest('gemma', {'parent': str(other)})

    def test_prepare_dispatches_a_complement_release(self):
        publish(self.root/'COMPLEMENT.json', {'models': {'gemma': {'parent': str(self.parent)}}})
        result = whole.prepare_model({}, 'gemma', True)
        self.assertEqual(result['gemma:main']['planned'], 20)

    def test_the_combined_table_joins_complement_rows_and_refuses_a_foreign_complement(self):
        whole.complement_manifest('gemma', {'parent': str(self.parent)})
        source = {'model': 'gemma', 'dose': 'main', 'root': str(self.parent), 'complements': [str(self.root)]}
        table = whole_effects._source_table(source)
        self.assertEqual(len(table), 3+20)
        self.assertEqual(int(table.complement.eq(True).sum()), 20)
        foreign = Path(self.directory.name)/'foreign'
        publish(whole.manifest_path('gemma', 'main', foreign), {'design': 'complement', 'records': [],
                                                                'parent_manifest_sha256': 'other'})
        with self.assertRaisesRegex(ValueError, 'complement belongs to another manifest'):
            whole_effects._source_table({**source, 'complements': [str(foreign)]})


if __name__ == '__main__':
    unittest.main()
