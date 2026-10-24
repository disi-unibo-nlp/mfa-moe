"""Next-sentence labels for Fig. 6: the labelled-sentence table of Qwen3.6 and the pair-completion estimator."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

import pandas as pd

import store
from routing import labelled_sentence_table, next_sentence_labels
from store import WORK, Units, publish


def layout(n, segments=None, tokenless=()):
    """A trace of n sentences owning two reasoning tokens each (none for `tokenless`)."""
    units = [{'index': i, 'start': 10*i, 'end': 10*i+9, 'text': f's{i}'} for i in range(n)]
    tokens = [[] if i in tokenless else [2*i, 2*i+1] for i in range(n)]
    return {'value': {'prompt': 'Q?', 'generation_messages': [{'role': 'user', 'content': 'Q?'}], 'metadata': {}},
            'units': units, 'tokens': tokens, 'ranks': tokens, 'segments': segments or [0]*n, 'n_reasoning': 2*n}


def record(index, name, attempt='a', segment=0):
    return {'attempt_id': attempt, 'question': 'q', 'benchmark': 'math500', 'family': 'f', 'class': name,
            'sentence_index': index, 'segment': segment, 'n_tokens': 3, 'pair_completion': False}


class JudgeRecords(unittest.TestCase):
    def test_only_records_at_the_current_span_count(self):
        units = layout(5)['units']
        records = [(0, 0, 9, 'Read', None), (1, 10, 19, 'Plan', None),
                   (2, 20, 99, 'Plan', None),   # the sentence at this index has another span
                   (7, 70, 79, 'Plan', None),   # no such sentence
                   (3, 30, 39, None, None),     # the judge saw it and could not label it
                   (4, 40, 49, 'Verify', None), (4, 40, 49, 'Monitor', None)]  # conflicting labels
        labels, judged = labelled_sentence_table.usable_labels(units, records)
        self.assertEqual(labels, {0: 'Read', 1: 'Plan'})
        self.assertEqual(sorted(judged), [0, 1, 3, 4])
        self.assertEqual(judged[3], (30, 39))

    def test_a_repeated_identical_label_is_not_a_conflict(self):
        labels, _ = labelled_sentence_table.usable_labels(layout(2)['units'],
                                                          [(0, 0, 9, 'Read', None), (0, 0, 9, 'Read', None)])
        self.assertEqual(labels, {0: 'Read'})


class TableRows(unittest.TestCase):
    def test_columns_flags_and_extents_of_one_attempt(self):
        source = {'attempt_id': 'a', 'question': 'q', 'benchmark': 'aime24'}
        trace = layout(6, segments=[0, 0, 0, 1, 1, 1], tokenless={2})
        rows = labelled_sentence_table.table_rows(
            source, trace, {0: 'Read', 1: 'Analyze', 2: 'Plan', 3: 'Verify', 5: 'Monitor'})
        self.assertEqual([r['sentence_index'] for r in rows], [0, 1, 3, 5])  # sentence 2 owns no token
        first, second, third, last = rows
        self.assertEqual((first['tok_start'], first['tok_end'], first['n_tokens']), (0, 2, 2))
        self.assertEqual(first['rel_pos'], 0.0)
        self.assertEqual(second['rel_pos'], 2/12)
        self.assertTrue(first['has_next_labelled'] and second['has_prev_labelled'])
        self.assertFalse(second['has_next_labelled'])  # its successor is not a table row
        self.assertEqual((third['segment'], third['cls_name'], third['cls']), (1, 'Verify', 5))
        self.assertFalse(third['prev_same_segment'] or third['next_same_segment'])
        self.assertFalse(last['has_prev_labelled'])
        self.assertEqual(labelled_sentence_table.as_frame(rows).dtypes['cls'], 'int8')

    def test_extent_drops_tokens_beyond_the_captured_prefix(self):
        trace = layout(3)
        self.assertEqual(labelled_sentence_table.sentence_extent(trace, 1, limit=3),
                         {'n_tokens': 1, 'tok_start': 2, 'tok_end': 3})
        self.assertIsNone(labelled_sentence_table.sentence_extent(trace, 2, limit=3))


class PairCandidates(unittest.TestCase):
    def test_each_skip_reason_and_the_candidate(self):
        trace = layout(13, segments=[0]*7+[1]*6, tokenless={11})
        trace['segments'][5], trace['segments'][6] = 0, 1  # sentence 6 starts the second segment
        sentences = [record(0, 'Read'), record(1, 'Analyze'),   # Read with a labelled successor
                     record(3, 'Verify'),                         # unlabelled successor 4: a candidate
                     record(5, 'Read'),                           # successor 6 lies in the next segment
                     record(8, 'Verify'),                         # successor 9 was judged without a label
                     record(10, 'Read'),                          # successor 11 owns no token
                     record(12, 'Verify'),                        # last sentence
                     record(2, 'Plan')]                           # not a Read or Verify source
        candidates, skipped = next_sentence_labels.pair_candidates(sentences, trace, {9: (90, 99)})
        self.assertEqual([c['sentence_index'] for c in candidates], [4])
        self.assertEqual(dict(skipped), {'successor_labelled': 1, 'next_segment': 1, 'already_judged': 1,
                                         'no_reasoning_tokens': 1, 'last_sentence': 1})
        candidate = candidates[0]
        self.assertEqual((candidate['source_index'], candidate['source_class']), (3, 'Verify'))
        self.assertEqual(candidate['inputs'], {'problem_statement': 'Q?', 'previous_sentence': 's3', 'sentence': 's4',
                                               'next_sentence': 's5'})
        self.assertEqual((candidate['tok_start'], candidate['tok_end']), (8, 10))

    def test_the_source_classes_are_a_parameter_with_their_own_folder(self):
        sentences = [record(0, 'Read'), record(2, 'Plan')]
        default, _ = next_sentence_labels.pair_candidates(sentences, layout(5), {})
        widened, _ = next_sentence_labels.pair_candidates(sentences, layout(5), {}, classes=('Plan',))
        self.assertEqual([c['sentence_index'] for c in default], [1])
        self.assertEqual([c['sentence_index'] for c in widened], [3])
        self.assertEqual(next_sentence_labels.variant(('Read', 'Verify')), 'read-verify')
        self.assertEqual(next_sentence_labels.variant(('Read', 'Verify', 'Plan')), 'read-verify-plan')

    def test_context_edges_use_the_original_sentinels(self):
        candidates, _ = next_sentence_labels.pair_candidates([record(0, 'Read')], layout(2), {})
        self.assertEqual(candidates[0]['inputs']['next_sentence'], '<END OF RESPONSE>')
        self.assertEqual(candidates[0]['inputs']['previous_sentence'], 's0')

    def test_context_parity_counts_every_difference(self):
        trace = layout(3)
        recorded = {0: {'problem_statement': 'Q?', 'previous_sentence': '<START OF RESPONSE>', 'sentence': 's0',
                        'next_sentence': 's1'}, 2: {'problem_statement': 'other'}}
        sentences = [record(0, 'Read'), record(2, 'Verify'), record(1, 'Analyze')]
        self.assertEqual(next_sentence_labels.context_parity(sentences, trace, recorded),
                         {'checked': 2, 'mismatched': 1})


class Estimands(unittest.TestCase):
    SOURCES = [{'attempt_id': 'a', 'benchmark': 'math500', 'family': 'f', 'completion_tokens': 100}]
    ORIGINAL = [record(0, 'Read'), record(1, 'Analyze'), record(3, 'Read'), record(5, 'Verify'), record(6, 'Implement')]

    def added(self):
        return [{**record(4, 'Monitor'), 'pair_completion': True}]

    def test_pairs_start_at_original_labels_only(self):
        pairs = next_sentence_labels.completed_transitions(self.ORIGINAL, self.added())
        self.assertEqual(sorted((t['source'], t['destination']) for t in pairs),
                         [('Read', 'Analyze'), ('Read', 'Monitor'), ('Verify', 'Implement')])
        # the added Monitor at 4 sits before the original Verify at 5; a pair that starts at it is not a sampled pair
        self.assertNotIn(('Monitor', 'Verify'), [(t['source'], t['destination']) for t in pairs])

    def test_added_labels_change_only_next_state_rates(self):
        without = next_sentence_labels.native_dynamics_rows(self.SOURCES, self.ORIGINAL)[0]
        completed = next_sentence_labels.native_dynamics_rows(self.SOURCES, self.ORIGINAL, self.added())[0]
        self.assertEqual(without['Read:next:Analyze'], 1.0)
        self.assertEqual((completed['Read:next:Analyze'], completed['Read:next:Monitor']), (.5, .5))
        self.assertIsNone(without['Monitor:next:Verify'])
        self.assertIsNone(completed['Monitor:next:Verify'])
        self.assertEqual(completed['Verify:next:Implement'], 1.0)
        unchanged = [k for k in without if ':next:' not in k]
        self.assertEqual({k: completed[k] for k in unchanged}, {k: without[k] for k in unchanged})
        self.assertEqual(completed['observed_labelled_tokens'], 15)
        self.assertEqual(completed['Read:occupancy'], 6/15)

    def test_reproduction_flags_a_value_or_a_gap_that_differs(self):
        rows = next_sentence_labels.native_dynamics_rows(self.SOURCES, self.ORIGINAL)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'published.parquet'
            pd.DataFrame(rows).to_parquet(path)
            self.assertTrue(next_sentence_labels.reproduction(rows, path)['reproduced'])
            changed = [{**rows[0], 'Read:next:Analyze': .75, 'Monitor:next:Verify': .5}]
            result = next_sentence_labels.reproduction(changed, path)
            self.assertFalse(result['reproduced'])
            self.assertEqual((result['gaps_differ'], round(result['largest_difference'], 2)), (1, 0.25))

    def test_a_different_table_is_refused_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as folder:
            path = Path(folder)/'table.parquet'
            next_sentence_labels.write_table(pd.DataFrame({'a': [1, 2]}), path)
            next_sentence_labels.write_table(pd.DataFrame({'a': [1, 2]}), path)
            with self.assertRaises(ValueError):
                next_sentence_labels.write_table(pd.DataFrame({'a': [1, 3]}), path)
            self.assertEqual(pd.read_parquet(path).a.tolist(), [1, 2])


class SerialPool:
    """A stand-in for multiprocessing's pool that maps in this process."""
    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *error):
        pass

    def imap_unordered(self, function, tasks, chunksize=1):
        return map(function, tasks)


class Rebuild(unittest.TestCase):
    """A relaunched table rebuild finishes when what is on disk is what it would write."""

    def rebuild(self, root, label):
        source = {'attempt_id': 'a', 'question': 'q', 'benchmark': 'math500', 'problem_id': 'p', 'sample_id': 0,
                  'trace_sha256': 't', 'family': 'f'}
        records = {labelled_sentence_table.trace_key(source):
                   [(i, 10*i, 10*i+9, label if i == 1 else 'Read', None) for i in range(4)]}
        counts = {'attempts_in_existing_table': 0, 'rows_in_existing_table': 0, 'rows_rebuilt_for_those_attempts': 0,
                  'identical': True}
        table = labelled_sentence_table
        patches = (patch.object(table, 'campaign_dir', return_value=root),
                   patch.object(table, 'campaign_sources', return_value=[source]),
                   patch.object(table, 'original_label_records', return_value=records),
                   patch.object(table, 'attempt_layout', return_value=layout(4)),
                   patch.object(table, 'label_files', return_value=[]),
                   patch.object(table, 'campaign', return_value={'sha256': 'c'}),
                   patch.object(table, 'check_class_counts', return_value=1),
                   patch.object(table, 'compare_with_existing', return_value=counts),
                   patch.object(table, 'pairs_dir', return_value=root/'bulk'),
                   patch.object(table, 'receipt_dir', return_value=root/'receipts'),
                   patch.object(table.multiprocessing, 'get_context', return_value=SimpleNamespace(Pool=SerialPool)))
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return labelled_sentence_table.rebuild('gpt', 1)

    def test_a_relaunch_finishes_even_if_the_module_changed_and_a_changed_result_is_refused(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as folder:
            root = Path(folder)
            (root/'gpt').mkdir()
            (root/'gpt/SOURCES.json').write_text('{}')
            first = self.rebuild(root, 'Plan')
            table = root/'bulk/sentences.parquet'
            before = table.read_bytes()
            # The receipt on disk was written by another version of the module.
            written = store.seal({**{k: v for k, v in first.items() if k != 'sha256'}, 'module_sha256': 'other'})
            (root/'receipts/SENTENCES.json').write_bytes(store._dump(written)+b'\n')
            self.assertEqual(self.rebuild(root, 'Plan'), written)
            self.assertEqual(table.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'conflicting immutable data file'):
                self.rebuild(root, 'Verify')
            self.assertEqual(table.read_bytes(), before)


class Merge(unittest.TestCase):
    def test_labels_unknowns_and_unfinished_contexts(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as folder:
            root = Path(folder)
            (root/'read-verify').mkdir()
            successors = pd.DataFrame([
                {'attempt_id': 'a', 'question': 'q', 'dataset': 'math500', 'sentence_index': 4, 'source_index': 3,
                 'source_class': 'Verify', 'segment': 0, 'n_tokens': 2, 'tok_start': 8, 'tok_end': 10,
                 'n_reasoning': 20, 'annotation_uid': uid * 64} for uid in 'abc'])
            successors['sentence_index'] = [4, 7, 9]
            next_sentence_labels.write_table(successors, root/'successors.parquet')
            tasks = [{'uid': uid * 64, 'inputs': {}, 'model': 'gpt'} for uid in 'abc']
            publish(root/'read-verify/PAIR_TASKS.json', {'records': tasks})
            publish(root/'read-verify/PAIR_PLAN.json', {
                'successors': str(root/'successors.parquet'),
                'successors_sha256': next_sentence_labels.file_sha(root/'successors.parquet')})
            units = Units(root/'annotations')
            outcomes = ({'status': 'complete', 'label': 'Plan'}, {'status': 'unknown', 'label': None})
            for task, outcome in zip(tasks[:2], outcomes):
                units.intent(task, 'job')
                units.output(task, {'token_ids': []})
                units.commit(task, outcome)
            with patch.object(next_sentence_labels, 'pairs_dir', return_value=root), \
                    patch.object(next_sentence_labels, 'receipt_dir', return_value=root):
                with self.assertRaisesRegex(ValueError, '1 contexts without a terminal'):
                    next_sentence_labels.completion_records('gpt', {'a': 'f'})
                task = tasks[2]
                units.intent(task, 'job')
                units.output(task, {'token_ids': []})
                units.commit(task, {'status': 'complete', 'label': 'Monitor'})
                records, body = next_sentence_labels.completion_records('gpt', {'a': 'f'})
        self.assertEqual([(r['sentence_index'], r['class'], r['pair_completion']) for r in records],
                         [(4, 'Plan', True), (9, 'Monitor', True)])
        self.assertEqual((body['successors'], body['labelled'], body['unknown']), (3, 2, 1))
        self.assertEqual(body['destination_by_source_class'], {'Verify:Monitor': 1, 'Verify:Plan': 1})


class Commands(unittest.TestCase):
    """The command lines name the study folder and the model by what they are, and the study folder is the one read."""

    def test_variant_prints_the_receipt_folder_of_the_rule(self):
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            next_sentence_labels.main(['variant'])
            next_sentence_labels.main(['variant', '--source-classes', 'Read,Verify,Plan,Explore,Monitor'])
        self.assertEqual(shown.getvalue().split(), ['read-verify', 'read-verify-plan-explore-monitor'])

    def test_a_step_gets_the_short_key_of_the_hugging_face_id_and_the_study_folder(self):
        with patch.object(store, 'STUDY', store.STUDY), patch.object(store, 'DRIVER', store.DRIVER), \
                patch.object(next_sentence_labels, 'plan', return_value={}) as plan, \
                patch.object(next_sentence_labels, 'log'):
            next_sentence_labels.main(['plan', '--study', '/some/study', '--model', 'openai/gpt-oss-20b'])
            self.assertEqual((store.STUDY, store.DRIVER), (Path('/some/study'), Path('/some/study/driver')))
        plan.assert_called_once_with('gpt', ANY, None, ('Read', 'Verify'))

    def test_the_table_rebuild_reads_the_campaign_next_to_its_study_folder(self):
        with patch.object(store, 'STUDY', store.STUDY), patch.object(store, 'DRIVER', store.DRIVER), \
                patch.object(labelled_sentence_table, 'rebuild', return_value={}) as rebuild, \
                contextlib.redirect_stdout(io.StringIO()):
            labelled_sentence_table.main(['rebuild', '--study', '/where/study', '--model', 'Qwen/Qwen3.6-35B-A3B-FP8'])
            self.assertEqual(labelled_sentence_table.campaign_dir(),
                             Path('/where/research-campaign-v3-20261014-r5-r2'))
        rebuild.assert_called_once_with('qwen36', ANY, None, None)

    def test_a_step_without_its_model_is_refused(self):
        with patch.object(store, 'STUDY', store.STUDY), patch.object(store, 'DRIVER', store.DRIVER), \
                self.assertRaisesRegex(ValueError, 'summary needs --model'):
            next_sentence_labels.main(['summary'])


if __name__ == '__main__':
    unittest.main()
