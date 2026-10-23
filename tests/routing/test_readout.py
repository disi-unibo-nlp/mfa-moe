import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from routing import readout_fit as new
from routing.readout_analysis import check_first_run, family_folds
from store import WORK
from routing.lfm_readout import HISTORICAL

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
        from routing import readout_analysis
        tree = ast.parse(Path(readout_analysis.__file__).read_text())
        names = {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'sr'}
        for name in names:
            self.assertTrue(hasattr(new, name), name)

    def test_family_folds_equal_the_copy_the_seven_model_driver_had(self):
        def original(questions, n, seed, question_family):
            digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            families = [question_family[str(q)] for q in questions]
            order = sorted(set(families), key=lambda f: digest(f'forum-v1|{seed}|{f}'))
            by_family = {f: i % n for i, f in enumerate(order)}
            return np.array([by_family[f] for f in families], int)
        rng = np.random.default_rng(3)
        questions = np.array([f'q{i}|caf\u00e9' for i in range(200)])
        families = {q: f'family-{rng.integers(60)}' for q in questions}
        for n, seed in [(5, 20260929), (3, 20260929 * 1000 + 2), (5, 20260930)]:
            np.testing.assert_array_equal(family_folds(questions, n, seed, families), original(questions, n, seed, families))

    def test_a_rerun_may_change_the_code_hashes_but_not_the_inputs(self):
        first = {'ids': {'gpt': ['a', 'b']}, 'inputs': {'/x': 'h1'}, 'code': {'/old.py': 'c1'}, 'sha256': 's1'}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            path = Path(tmp)/'FROZEN.json'
            check_first_run(path, first)  # no first run recorded: nothing to compare
            path.write_text(json.dumps(first))
            check_first_run(path, {**first, 'code': {'/new.py': 'c2'}, 'sha256': 's2'})
            with self.assertRaises(ValueError):
                check_first_run(path, {**first, 'inputs': {'/x': 'changed'}})

