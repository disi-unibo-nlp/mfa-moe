from contextlib import contextmanager
import io
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

from routing import lexical
from routing.readout_text import TextTransform
from store import WORK


@contextmanager
def lives_in(module, *classes):
    """Make the classes look as if they lived in `module`, so that pickle writes that module name.

    Pickle imports every parent of a dotted name, so missing parents are registered too.
    """
    parts = module.split('.')
    parents = [name for name in ('.'.join(parts[:i]) for i in range(1, len(parts))) if name not in sys.modules]
    for name in parents:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    created = module not in sys.modules
    host = types.ModuleType(module) if created else sys.modules[module]
    sys.modules[module] = host
    homes = [cls.__module__ for cls in classes]
    for cls in classes:
        cls.__module__ = module
        setattr(host, cls.__name__, cls)
    try:
        yield
    finally:
        for cls, home in zip(classes, homes):
            cls.__module__ = home
            if not created:
                delattr(host, cls.__name__)
        if created:
            del sys.modules[module]
        for name in parents:
            del sys.modules[name]


def text_transform(n_active, scale):
    return TextTransform(idf=np.ones(n_active), keep=np.ones(n_active, bool), remap=np.arange(n_active),
                         n_active=n_active, scale=scale)


class FoldPickleTests(unittest.TestCase):
    def test_pickles_with_the_stored_module_names_load(self):
        featurizer = lexical.Featurizer()
        featurizer.tc = text_transform(3, 0.5)
        model = lexical.LR()
        model.x, model.d = np.arange(6.), 2
        with lives_in('__main__', lexical.Featurizer, lexical.LR), \
                lives_in('moe_exp.study.readout_text', TextTransform):
            stored = pickle.dumps({'featurizer': featurizer, 'model': model})
        self.assertIn(b'moe_exp.study.readout_text', stored)
        self.assertIn(b'__main__', stored)
        with self.assertRaises((ModuleNotFoundError, AttributeError)):
            pickle.loads(stored)
        loaded = lexical._Unpickler(io.BytesIO(stored)).load()
        self.assertIsInstance(loaded['featurizer'], lexical.Featurizer)
        self.assertIsInstance(loaded['featurizer'].tc, TextTransform)
        self.assertEqual(loaded['featurizer'].tc.scale, 0.5)
        self.assertIsInstance(loaded['model'], lexical.LR)
        np.testing.assert_array_equal(loaded['model'].x, np.arange(6.))

    @unittest.skipUnless((lexical.LEXICAL/'readout-v1/fold-0.pkl').exists(), 'the deployed fold pickles are not here')
    def test_a_deployed_fold_pickle_loads(self):
        with (lexical.LEXICAL/'readout-v1/fold-0.pkl').open('rb') as stream:
            saved = lexical._Unpickler(stream).load()
        self.assertEqual((saved['fold'], saved['spec']), (0, lexical.SPEC))
        self.assertIsInstance(saved['featurizer'], lexical.Featurizer)
        self.assertIsInstance(saved['featurizer'].tc, TextTransform)
        self.assertIsInstance(saved['model'], lexical.LR)

    def test_pickles_with_the_current_module_names_load(self):
        loaded = lexical._Unpickler(io.BytesIO(pickle.dumps(text_transform(2, 1.0)))).load()
        self.assertIsInstance(loaded, TextTransform)
        self.assertEqual(loaded.n_active, 2)


class TrainingTableTests(unittest.TestCase):
    def test_the_training_table_reads_the_native_tables_of_the_folder_it_is_given(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            pd.DataFrame({'model': ['gpt', 'gpt'], 'context_hash': ['a', 'b'], 'n_rows_with_context': [1, 1],
                          'label': ['Read', 'Plan'], 'sentence_text': ['x', 'y'], 'prev_sentence': ['', 'x'],
                          'next_sentence': ['y', ''], 'family_id': ['f', 'g'], 'fold_campaign': [0, 1]}
                         ).to_parquet(Path(tmp)/'native_gpt.parquet')
            table = lexical.training_table(['gpt'], Path(tmp))
        self.assertEqual(sorted(table.context_hash), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
