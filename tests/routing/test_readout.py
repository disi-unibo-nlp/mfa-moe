import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import store
from routing import readout_analysis, readout_fit as new
from routing.readout_analysis import CACHE_READERS, PUBLISHED_RUN, code_hashes, family_folds, freeze, use_analysis_dir
from store import WORK, file_sha, publish

MODELS = ('gemma', 'glm', 'gpt', 'nemotron', 'qwen330b', 'qwen35', 'qwen36', 'lfm25', 'lfm25-engineering')


def without_docstrings(node):
    """ast.dump of a definition with the docstrings of it and of what it contains left out."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.body and isinstance(child.body[0], ast.Expr) \
                and isinstance(getattr(child.body[0], 'value', None), ast.Constant) \
                and isinstance(child.body[0].value.value, str):
            child.body = child.body[1:] or [ast.Pass()]
    return ast.dump(node, include_attributes=False)


class ReadoutTests(unittest.TestCase):
    def test_null_function_matches_the_published_run_for_every_model_and_draw(self):
        source = PUBLISHED_RUN / 'code/dp/s3_readout.py'
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

    def test_numerical_helpers_have_the_bodies_the_published_run_recorded(self):
        original = ast.parse((PUBLISHED_RUN / 'code/dp/s3_readout.py').read_text())
        current = ast.parse(Path(new.__file__).read_text())
        originals = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        currents = {n.name: n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ('bucket_of', 'abs_bin', 'path_variant', 'randomized_pca', 'BagTransform',
                     'fit_softmax', 'predict_log_proba', 'weighted_nll', 'question_bootstrap'):
            self.assertEqual(without_docstrings(originals[name]), without_docstrings(currents[name]), name)

    def test_analysis_references_only_available_numerical_helpers(self):
        from routing import readout_analysis
        tree = ast.parse(Path(readout_analysis.__file__).read_text())
        names = {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'sr'}
        for name in names:
            self.assertTrue(hasattr(new, name), name)

    def test_family_folds_follow_the_published_order_rule(self):
        def reference(questions, n, seed, question_family):
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
            np.testing.assert_array_equal(family_folds(questions, n, seed, families), reference(questions, n, seed, families))

    def test_a_rerun_may_change_the_code_hashes_but_not_the_inputs(self):
        first = {'ids': {'gpt': ['a', 'b']}, 'inputs': {'/x': 'h1'}, 'code': {'/before.py': 'c1'}, 'sha256': 's1'}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            tmp = Path(tmp)
            (tmp/'first').mkdir()
            with mock.patch.object(readout_analysis, 'PUBLISHED_RUN', tmp/'first'):
                with self.assertRaisesRegex(ValueError, 'published run'):
                    freeze(tmp/'a', 'FROZEN.json', first)  # no published run recorded: a hard error
                freeze(tmp/'a', 'FROZEN.json', first, check_published_run=False)  # only tests may skip the comparison
                (tmp/'first'/'FROZEN.json').write_text(json.dumps(first))
                rerun = {**first, 'code': {'/after.py': 'c2'}, 'sha256': 's2'}
                freeze(tmp/'b', 'FROZEN.json', rerun)
                freeze(tmp/'b', 'FROZEN.json', rerun)  # a resumed run writes the same spec again
                with self.assertRaises(ValueError):
                    freeze(tmp/'b', 'FROZEN.json', {**rerun, 'sha256': 's3'})
                with self.assertRaises(ValueError):
                    freeze(tmp/'c', 'FROZEN.json', {**first, 'inputs': {'/x': 'changed'}})

    def test_the_fits_refuse_an_interactive_shell_but_not_a_batch_job_on_the_serial_partition(self):
        from routing import next_state_anticipation, present_state_readout
        for run in (present_state_readout.run, next_state_anticipation.run):
            for environment in ({}, {'SLURM_JOB_ID': '1'}):  # nothing, or the shell salloc starts on a login node
                with mock.patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(ValueError, 'batch job'):
                    run(1, WORK/'tools/tmp/never-created')

    def test_the_analysis_folder_is_set_before_the_helper_packages_read_it(self):
        elsewhere = Path('/elsewhere/analysis')
        with mock.patch.object(store, 'ANALYSIS', store.ANALYSIS), \
                mock.patch.object(readout_analysis, 'PUBLISHED_RUN', PUBLISHED_RUN), \
                mock.patch.dict(sys.modules):
            default = store.ANALYSIS
            for name in CACHE_READERS:
                sys.modules.pop(name, None)
            self.assertEqual(use_analysis_dir(default), default)  # the default folder: nothing changes
            self.assertEqual(readout_analysis.PUBLISHED_RUN, PUBLISHED_RUN)
            self.assertEqual(use_analysis_dir(elsewhere), elsewhere)  # the helpers have not been imported yet
            self.assertEqual((store.ANALYSIS, readout_analysis.PUBLISHED_RUN),
                             (elsewhere, elsewhere/'forum/tests/r3_context'))
            sys.modules[CACHE_READERS[0]] = types.ModuleType(CACHE_READERS[0])  # now one has read the folder
            self.assertEqual(use_analysis_dir(elsewhere), elsewhere)
            with self.assertRaisesRegex(ValueError, 'imported before'):
                use_analysis_dir(default)

    def test_the_helper_packages_name_the_captured_models_in_the_orders_the_frozen_runs_used(self):
        from models.roster import CAPTURED, READOUT_ORDER
        from routing import benchmark_inventory
        from routing.depth_paths import common
        # The null seeds of the readout index into this alphabetical list; the inventory walks the study order.
        self.assertEqual(common.MODELS, ('gemma', 'glm', 'gpt', 'nemotron', 'qwen330b', 'qwen35', 'qwen36'))
        self.assertEqual(benchmark_inventory.MODELS, ['gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b'])
        self.assertEqual(benchmark_inventory.MODELS, list(CAPTURED))
        self.assertEqual(sorted(READOUT_ORDER), sorted(common.MODELS))

    def test_the_lfm_readout_compares_the_inputs_but_not_the_code_with_its_published_receipt(self):
        from routing import lfm_readout
        provenance = {'code': {'after.py': 'c2'}, 'inputs_sha256': 'i', 'folds_sha256': 'f', 'seed_index': 7}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            binding = {'root': tmp, 'config': {'bulk_root': tmp}}
            lfm_readout.check_published(binding, provenance)  # nothing published yet
            publish(Path(tmp)/'lfm25/READOUT_RESULTS.json', {'provenance': {**provenance, 'code': {'before.py': 'c1'}}})
            lfm_readout.check_published(binding, provenance)
            with self.assertRaisesRegex(ValueError, 'inputs differ'):
                lfm_readout.check_published(binding, {**provenance, 'folds_sha256': 'changed'})
            publish(Path(tmp)/'out/READOUT_RESULTS.json', {})
            with self.assertRaisesRegex(ValueError, 'already holds'):  # the published file is never rewritten
                lfm_readout.lfm(binding, Path(tmp)/'out')

    def test_code_hashes_cover_files_and_the_python_files_of_folders(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            tmp = Path(tmp)
            (tmp/'pkg').mkdir()
            (tmp/'pkg/a.py').write_text('a = 1\n')
            (tmp/'pkg/notes.txt').write_text('skip')
            (tmp/'driver.py').write_text('b = 2\n')
            self.assertEqual(code_hashes(tmp/'driver.py', tmp/'pkg'),
                             {str(tmp/'driver.py'): file_sha(tmp/'driver.py'), str(tmp/'pkg/a.py'): file_sha(tmp/'pkg/a.py')})
