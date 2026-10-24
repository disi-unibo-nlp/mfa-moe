from contextlib import contextmanager
import io
import pickle
import sys
import types
import unittest

import numpy as np

from routing import lexical
from routing.readout_text import TextTransform


@contextmanager
def lives_in(module, *classes):
    """Make the classes look as if they lived in `module`, so that pickle writes that module name."""
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


def text_transform(n_active, scale):
    return TextTransform(idf=np.ones(n_active), keep=np.ones(n_active, bool), remap=np.arange(n_active),
                         n_active=n_active, scale=scale)


class FoldPickleTests(unittest.TestCase):
    def test_pickles_with_the_old_module_names_still_load(self):
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

    def test_pickles_with_the_current_module_names_load(self):
        loaded = lexical._Unpickler(io.BytesIO(pickle.dumps(text_transform(2, 1.0)))).load()
        self.assertIsInstance(loaded, TextTransform)
        self.assertEqual(loaded.n_active, 2)


if __name__ == '__main__':
    unittest.main()
