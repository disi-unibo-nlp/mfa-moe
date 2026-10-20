"""Cohort identity, ownership, reader joins and six-contrast integration contracts."""
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from moe_exp.study import cohort, effects, lexical, ownership, rescue_reader
from moe_exp.study.common import CLASSES, WORK, digest, file_sha, publish, verified


class ConstantFeatures:
    """Small persisted classifier fixture, independent of production featurization."""
    def transform(self, blocks, indices):
        return np.zeros((len(indices), 1))


class ConstantModel:
    """A valid seven-probability model with a known lexical class."""
    def proba(self, features):
        return np.tile([1., 0., 0., 0., 0., 0., 0.], (len(features), 1))


class CohortContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_execution_identity_never_depends_on_pending_work(self):
        value = {'sha256': 'cohort', 'source': {'sha256': 'source'}, 'generation_uids': ['b', 'a']}
        before = cohort.identity(value, 'qwen36', 'generation', value['generation_uids'])
        value['completed'] = ['a']
        value['pending'] = ['b']
        self.assertEqual(before, cohort.identity(value, 'qwen36', 'generation', value['generation_uids']))

    def test_nonblocking_store_lock_rejects_concurrent_owner(self):
        with ownership.store_lock(self.root/'store'):
            with self.assertRaisesRegex(ValueError, 'concurrent store ownership'):
                with ownership.store_lock(self.root/'store'):
                    self.fail('second lock acquired')

    def test_successor_cannot_own_live_predecessor_store(self):
        execution = {'cohort': 'fixed'}
        publish(self.root/'owners'/'11.json', {'job': '11', 'execution': execution})
        raw = '11|RUNNING|0:0|60|cpu=16,gres/gpu=2|'
        with self.assertRaisesRegex(ValueError, 'predecessor owner is not terminal'):
            ownership.accept(self.root, execution, '12', {}, run=lambda arguments: raw)

    def test_first_owner_cannot_adopt_unbound_legacy_state(self):
        folder = self.root/'units'/('a'*64)
        folder.mkdir(parents=True)
        path = folder/'OUTPUT.json'
        path.write_text('legacy tokens')
        with self.assertRaisesRegex(ValueError, 'unbound prior state'):
            ownership.accept(self.root, {}, '12', {}, run=lambda args: self.fail('allocation before fresh-store guard'))
        self.assertEqual(path.read_text(), 'legacy tokens')

    def test_exact_worker_job_and_roster_are_checked_before_allocation(self):
        publish(self.root/'owners'/'12.json', {'job': '11', 'execution': {}, 'store': str(self.root)})
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12'}):
            with self.assertRaisesRegex(ValueError, 'worker job/store/cohort/frozen roster'):
                with ownership.worker(self.root, {}, run=lambda arguments: self.fail('native call before identity check')):
                    self.fail('wrong owner admitted')

    def test_deadline_rejection_precedes_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'deadline expired'):
            ownership.check_time({'science_cutoff': '2000-01-01T00:00:00+00:00'})

    def test_task_staging_pins_cached_default_to_private_work(self):
        with patch.dict(os.environ, {'TMPDIR': str(self.root)}), patch.object(tempfile, 'tempdir', '/tmp'):
            self.assertEqual(ownership.task_staging(), self.root)
            with tempfile.TemporaryDirectory() as directory:
                self.assertTrue(Path(directory).is_relative_to(self.root))

    def test_task_staging_rejects_system_tmp_before_creating_files(self):
        with patch.dict(os.environ, {'TMPDIR': '/tmp'}):
            with self.assertRaisesRegex(ValueError, 'user HOME/WORK'):
                ownership.task_staging()

    def test_admitted_without_intent_or_output_stays_unknown(self):
        from moe_exp.study.common import Units
        units = Units(self.root/'annotations')
        uid = 'a'*64
        publish(units.folder(uid)/'ADMITTED.json', {'uid': uid})
        self.assertEqual(units.state(uid)['status'], 'unknown')
        self.assertFalse(units.intent({'uid': uid}, '12'))

    def test_empty_and_expired_generation_do_not_load_engine(self):
        from moe_exp.study import driver
        store = driver.Store(self.root/'units')
        with patch.object(driver, 'hooked_llm') as model, patch.object(driver, 'code_hashes', return_value={}):
            result = driver.execute_rows({'config': {'bulk_root': str(self.root)}}, 'qwen36', [], store, 0,
                                         max_inflight=8, design='rescue', execution={}, pending_uids=set())
            self.assertEqual(result['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'insufficient time'):
                driver.execute_rows({}, 'gpt', [{'uid': 'a'*64}], store, 0,
                                    max_inflight=8, design='main')
        model.assert_not_called()

    def classifier(self):
        root = self.root/'classifier'
        root.mkdir()
        folds = []
        for fold in range(5):
            path = root/f'fold-{fold}.pkl'
            with path.open('wb') as stream:
                pickle.dump({'fold': fold, 'spec': lexical.SPEC,
                             'featurizer': ConstantFeatures(), 'model': ConstantModel()}, stream)
            folds.append({'fold': fold, 'sha256': file_sha(path)})
        publish(root/'READOUT.json', {'spec': lexical.SPEC, 'folds': folds})
        return lexical.artifact_binding(root)

    def test_changed_fold_or_readout_is_rejected(self):
        binding = self.classifier()
        root = Path(binding['root'])
        (root/'fold-3.pkl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'fold bytes changed'):
            lexical.artifact_binding(root)

    def test_invalid_or_uncovered_fold_is_not_read(self):
        binding = self.classifier()
        row = {'fold': 5, 'sentence_text': 'Check.', 'prev_sentence': '', 'next_sentence': ''}
        with self.assertRaisesRegex(ValueError, 'not covered'):
            lexical.predict(pd.DataFrame([row]), Path(binding['root']))

    def test_invalid_probability_is_not_read(self):
        binding = self.classifier()
        row = {'fold': 0, 'sentence_text': 'Check.', 'prev_sentence': '', 'next_sentence': ''}
        with patch.object(ConstantModel, 'proba', return_value=np.zeros((1, 7))):
            with self.assertRaisesRegex(ValueError, 'invalid classifier probabilities'):
                lexical.predict(pd.DataFrame([row]), Path(binding['root']))

    def test_semantic_no_event_censoring_and_missing_label_differ(self):
        for annotation, width in (({'status': 'no_event'}, 0), ({'status': 'unknown'}, 7),
                                  ({'status': 'reader_task', 'label': None}, 7)):
            lower, upper = effects.semantic_endpoint(annotation)
            self.assertEqual(float((upper-lower).sum()), width)
        lower, upper = effects.semantic_endpoint({'status': 'reader_task', 'label': 'Verify'})
        self.assertTrue(np.array_equal(lower, upper))
        self.assertEqual(lower[CLASSES.index('Verify')], 1)

    def test_prefix_straddlers_and_boundary_do_not_supply_semantic_first(self):
        from types import SimpleNamespace
        from moe_exp.study.outcomes import continuation_sentences
        trace = SimpleNamespace(prompt='problem')
        layout = {'units': [{'text': 'straddle.', 'start': 0}, {'text': 'at boundary.', 'start': 10}],
                  'unit_tokens': [[0, 1, 2], [514]], 'reasoning_tokens': [0, 1, 2, 514]}
        with patch('moe_exp.correlation_pipeline.dynamics.classes.saved_layout', return_value=layout), \
                patch('moe_exp.correlation_pipeline.spans.reasoning_ranges', return_value=[(0, 100)]):
            rows, _ = continuation_sentences(trace, {}, 2, 512, 'stop')
        self.assertTrue(rows[0]['straddles_prefix'])
        self.assertFalse(any(row['first_complete'] for row in rows))

    def test_mass_conservation_includes_nonreasoning_ended_and_unknown(self):
        outcome = {'pulse_tokens': 512, 'horizon': 512, 'reasoning_end': 4, 'generated_tokens': 4,
                   'finish_reason': 'stop', 'nonreasoning': '[1]'}
        sentence = {'token_start': 0, 'token_end': 1, 'complete': True, 'lexical_label': 'Read',
                    **{'p_'+label: float(label == 'Read') for label in CLASSES}}
        result = effects.endpoints(outcome, [sentence])
        total = result['full_occupancy_hard'].sum()+result['full_ended']+result['full_unknown']+result['full_nonreasoning']
        self.assertAlmostEqual(float(total), 1.)
        self.assertEqual(result['full_unknown'], 2/512)

    def test_lexical_overlap_keeps_a_truncated_unit_while_complete_event_is_unknown(self):
        outcome = {'pulse_tokens': 512, 'horizon': 512, 'reasoning_end': None, 'generated_tokens': 512,
                   'finish_reason': 'length', 'nonreasoning': '[]'}
        sentence = {'token_start': 0, 'token_end': 512, 'complete': False, 'straddles_prefix': True,
                    'lexical_label': 'Plan', **{'p_'+label: float(label == 'Plan') for label in CLASSES}}
        result = effects.endpoints(outcome, [sentence])
        self.assertEqual(result['first_overlap_lower'][CLASSES.index('Plan')], 1)
        self.assertTrue((result['first_upper']-result['first_lower'] == 1).all())

    def test_joint_six_family_shares_critical_value_and_bound_components(self):
        families = [f'f{index}' for index in range(64)]
        benchmarks = {family: effects.BENCHMARKS[index % 6] for index, family in enumerate(families)}
        boot = effects.Bootstrap(families, benchmarks)
        frame = pd.DataFrame({'family': families})
        rng = np.random.default_rng(5)
        for column in range(6):
            frame[str(column)] = rng.normal(0, .1, 64)
            frame[str(column)+':lower'] = frame[str(column)]-.2
            frame[str(column)+':upper'] = frame[str(column)]+.2
        result = effects.bounded_summary(boot, frame, list(map(str, range(6))), 'joint-6')
        self.assertEqual(len({row['critical'] for row in result.values()}), 1)
        self.assertTrue(all(row['bound_components'] == 3 and row['point_role'] == 'compatibility_only'
                            for row in result.values()))
        self.assertEqual(boot.weights.shape, (10000, 64))

    def test_engineering_selection_covers_actual_frozen_metadata(self):
        from moe_exp.study.engineering import select_blocks
        from moe_exp.study.driver import DRIVER
        rows = verified(DRIVER/'gpt'/'RESCUE.json')['records']
        selected = select_blocks(rows, 20261020)
        self.assertEqual({row['class'] for row in selected}, set(CLASSES))
        self.assertEqual({row['benchmark'] for row in selected}, set(effects.BENCHMARKS))
        self.assertEqual({row['fold'] for row in selected}, set(range(5)))
        self.assertEqual(len({row['family'] for row in selected}), 8)

    def test_ht_uses_actual_coefficients_and_retains_repeated_contexts(self):
        from moe_exp.study.measurement import disagreement, sample_occurrences
        rows = [{'occurrence_uid': f'{index:064x}', 'model': 'gpt', 'layer_regime': 'single',
                 'direction': 'activate', 'selector': 'target', 'target_class': 'Verify',
                 'coefficient': .01, 'lexical_label': 'Read', 'annotation_uid': 'same'} for index in range(40)]
        sample = sample_occurrences({'selection': {'seed': 20261020, 'audit_purpose': 'audit', 'audit_per_stratum': 32}}, rows)
        self.assertEqual(len(sample['selected']), 32)
        self.assertTrue(all(row['inclusion_probability'] == .8 for row in sample['selected']))
        result = disagreement(sample, {'same': 'Verify'})['directions']['activate']
        self.assertAlmostEqual(result['estimate'], .4)
        self.assertAlmostEqual(result['finite_population_variance'], 0)
        unknown = disagreement(sample, {})['directions']['activate']
        self.assertEqual(unknown['unknown_sampled_occurrences'], 32)
        self.assertGreater(unknown['upper'], unknown['lower'])


def _model_replays(model: str) -> tuple:
    """Construct demo historical, authorized and incompatible assignments."""
    records = []
    rows, saved = [], []
    for regime in ('native', 'single', 'triple'):
        arms = ('native', 'native_repeat') if regime == 'native' else (
            'target_activate', 'random_activate', 'target_deactivate', 'random_deactivate')
        for arm in arms:
            row = {'uid': digest([model, regime, arm]), 'model': model, 'family': 'fixture-family',
                   'fold': 0, 'benchmark': 'math500', 'class': None if regime == 'native' else 'Read',
                   'layer_regime': regime, 'arm': arm, 'horizon': 512, 'pulse_tokens': 512,
                   'prefix_status': 'enrolled', 'source': {'prefix_tokens': 2}, 'action': None,
                   'actions': None if regime == 'native' else [{'layer': 0, 'operator': 'score_override'}]}
            rows.append(row)
            unknown = model == 'qwen36' and arm == 'target_deactivate' and regime == 'triple'
            records.append({'assignment': row, 'disposition': 'incompatible_unknown' if unknown else
                            'authorized_generation' if arm == 'random_activate' else 'historical_candidate'})
            if unknown:
                continue
            inputs = {'problem_statement': 'Problem', 'previous_sentence': '',
                      'sentence': model+' '+regime+' '+arm+'.', 'next_sentence': ''}
            replay = {'sentences': [{'sentence_index': 1, 'token_start': 0, 'token_end': 4,
                                    'complete': True, 'first_complete': True, 'straddles_prefix': False,
                                    'inputs': inputs}], 'generated_tokens': 4, 'prefix_tokens': 2,
                      'reasoning_tokens': 4, 'reasoning_end': 4, 'finish_reason': 'stop',
                      'strict_correct': True, 'accuracy_bounds': [1, 1], 'nonreasoning': []}
            saved.append({'uid': row['uid'], 'replay_json': json.dumps(replay)})
    return rows, records, saved


class CohortIntegration(unittest.TestCase):
    """Trace nondefault roots and mixed evidence through actual exported endpoints."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self):
        """Persist the demo cohort, extraction and exact source joins."""
        classifier = CohortContracts.classifier(self)
        source = {'sha256': 'fixed-source'}
        model_rows, records, replays = {}, [], {}
        for model in ('gpt', 'qwen36'):
            rows, model_records, saved = _model_replays(model)
            model_rows[model], replays[model] = rows, saved
            records.extend(model_records)
        manifests = {}
        for model, rows in model_rows.items():
            path = self.root/model/'manifest.json'
            manifest = publish(path, {'records': rows})
            manifests[model] = {'path': str(path), 'sha256': manifest['sha256']}
        paths = {name: str(self.root/'nondefault'/name) for name in ('bulk', 'control', 'extraction', 'reader', 'analysis')}
        paths['generation_index'] = str(Path(paths['control'])/'GENERATION_INDEX.json')
        program = self.root/'program.json'
        program.write_text('{"prompt": "frozen"}')
        binding_path = self.root/'BINDING.json'
        publish(binding_path, {'config': {'bulk_root': paths['bulk']}, 'control_root': paths['control']})
        source['study_binding_path'] = str(binding_path)
        source['files'] = {manifest['path']: file_sha(Path(manifest['path'])) for manifest in manifests.values()}
        value = publish(self.root/'COHORT.json', {
            'schema': 'causal-rescue-cohort-v1', 'source': {'path': str(self.root/'source.json'), **source},
            'records': records, 'generation_uids': sorted(row['assignment']['uid'] for row in records
                    if row['disposition'] == 'authorized_generation'),
            'unknown_uids': sorted(row['assignment']['uid'] for row in records if row['disposition'] == 'incompatible_unknown'),
            'paths': paths, 'classifier': classifier, 'manifests': manifests, 'class_mapping': list(CLASSES),
            'study_binding_path': str(binding_path),
            'annotator': {'revision': 'fixed-reader', 'program': str(program),
                          'config': {'program_sha256': file_sha(program)}},
            'selection': {'seed': 20261020, 'audit_purpose': 'audit', 'audit_per_stratum': 32},
            'reader_runtime_files': {}, 'reuse_indexes': {}, 'historical_files': {},
        })
        extracted = {}
        for model, saved in replays.items():
            folder = Path(paths['extraction'])/model
            folder.mkdir(parents=True)
            table = folder/'terminal-replay.parquet'
            pd.DataFrame(saved).to_parquet(table, index=False)
            receipt = publish(folder/'EXTRACTION.json', {'cohort_sha256': value['sha256'],
                                                        'files': {str(table): file_sha(table)}})
            extracted[model] = receipt['sha256']
        publish(Path(paths['generation_index']), {'cohort_sha256': value['sha256'], 'quiescent': True,
            'records': [{'uid': row['assignment']['uid'], 'status': row['disposition']} for row in records],
            'extraction_sha256': extracted})
        return source, value, paths

    def test_semantic_labels_change_exported_endpoint_without_dropping_unknown(self):
        source, value, paths = self.fixture()
        with patch('moe_exp.study.rescue_source.verify', return_value=source):
            tasks = rescue_reader.cohort_tasks(self.root/'COHORT.json')
            publish(Path(paths['control'])/'REUSED_LABELS.json', {'cohort_sha256': value['sha256'],
                                                                 'tasks_sha256': tasks['sha256'], 'records': []})
            (self.root/'ALLOCATION.json').write_text(json.dumps({'records': [
                {'family': 'fixture-family', 'benchmark': 'math500'}]}))
            with patch.object(effects, 'ROOT', self.root), patch.object(effects, 'null_diagnostic', return_value={'role': 'fixture'}):
                before = effects.analyze(['gpt', 'qwen36'], self.root/'before-labels', 'rescue', cohort=self.root/'COHORT.json')
            for task in tasks['records']:
                label = 'Read' if 'target_activate' in task['inputs']['sentence'] else 'Verify'
                publish(Path(paths['reader'])/'annotations'/task['uid']/'RESULT.json', {
                    'uid': task['uid'], 'assignment_sha256': digest(task), 'status': 'complete', 'label': label,
                    'annotation_binding': rescue_reader.annotation_binding(value, task)})
            table = effects.continuation_table(['gpt', 'qwen36'], 'rescue', cohort=self.root/'COHORT.json')
            self.assertEqual(len(table), 20)
            unknown = table[table.outcome_status != 'observed']
            self.assertEqual(len(unknown), 1)
            pairs = effects.paired(table)
            self.assertTrue((pairs['activate:semantic_first'] == 1).all())
            self.assertTrue((pairs['activate:full_occupancy'] == 0).all())
            # Native is broadcast to both intervention regimes by the maintained pairing code.
            self.assertTrue(np.isfinite(pairs['activate:semantic_first:vs_native']).all())
            boot = effects.Bootstrap(['fixture-family'], {'fixture-family': 'math500'})
            exported = effects.bounded_summary(boot, pairs, ['activate:semantic_first', 'deactivate:semantic_first'], 'semantic')
            receipt = publish(self.root/'export'/'SEMANTIC.json', exported)
            self.assertEqual(receipt['activate:semantic_first']['estimate'], 1.)
            self.assertGreater(receipt['deactivate:semantic_first']['upper_bound'],
                               receipt['deactivate:semantic_first']['lower_bound'])
            with patch.object(effects, 'ROOT', self.root), patch.object(effects, 'null_diagnostic', return_value={'role': 'fixture'}):
                after = effects.analyze(['gpt', 'qwen36'], self.root/'after-labels', 'rescue', cohort=self.root/'COHORT.json')
            self.assertEqual(after['joint_six_contrasts']['activate:semantic_first']['estimate'], 1.)
            self.assertEqual(before['joint_six_contrasts']['activate:semantic_first']['estimate'], 0.)
            reused_path = Path(paths['control'])/'REUSED_LABELS.json'
            reused_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt hash differs'):
                rescue_reader.context_labels(value, tasks)


if __name__ == '__main__':
    unittest.main()
