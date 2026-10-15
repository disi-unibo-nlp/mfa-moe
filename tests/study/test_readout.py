import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from moe_exp.study import readout_fit as new
from moe_exp.study.readout_analysis import family_folds
from moe_exp.study.readouts import HISTORICAL

MODELS = ('gemma', 'glm', 'gpt', 'nemotron', 'qwen330b', 'qwen35', 'qwen36', 'lfm25', 'lfm25-engineering')


class ReadoutTests(unittest.TestCase):
    def test_actual_frozen_null_function_matches_all_explicit_indices(self):
        source = HISTORICAL / 'code/dp/s3_readout.py'
        node = next(n for n in ast.walk(ast.parse(source.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == 'null_paths')
        namespace = {'np': np, 'ModelData': SimpleNamespace, 'SEED': new.SEED,
                     'cm': SimpleNamespace(MODELS=MODELS)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
        md = SimpleNamespace(tokid=np.repeat(np.arange(8), 16), tok_abs=np.tile(np.arange(4), 32),
                             top1=np.arange(384).reshape(3, 128) % 64)
        for index, model in enumerate(MODELS):
            md.model = model; md.seed_index = index
            for draw in range(5):
                np.testing.assert_array_equal(new.null_paths(md, draw), namespace['null_paths'](md, draw))

    def test_family_folds_never_split_alias_questions(self):
        questions = np.array(['a', 'a-alias', 'b', 'c', 'd'])
        families = {'a': 'f1', 'a-alias': 'f1', 'b': 'f2', 'c': 'f3', 'd': 'f4'}
        folds = family_folds(questions, 3, 20260929, families)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(dict(zip(questions, folds)), dict(zip(questions[::-1], family_folds(questions[::-1], 3, 20260929, families))))

    def test_numerical_helpers_are_frozen_function_bodies(self):
        original = ast.parse((HISTORICAL / 'code/dp/s3_readout.py').read_text())
        current = ast.parse(Path(new.__file__).read_text())
        originals = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        currents = {n.name: n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ('bucket_of', 'abs_bin', 'path_variant', 'randomized_pca', 'BagTransform',
                     'fit_softmax', 'predict_log_proba', 'weighted_nll', 'question_bootstrap'):
            self.assertEqual(ast.dump(originals[name], include_attributes=False), ast.dump(currents[name], include_attributes=False), name)

    def test_analysis_references_only_available_numerical_helpers(self):
        from moe_exp.study import readout_analysis
        tree = ast.parse(Path(readout_analysis.__file__).read_text())
        names = {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'sr'}
        for name in names:
            self.assertTrue(hasattr(new, name), name)
