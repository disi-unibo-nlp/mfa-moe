"""The follow-up of a complement release: its read-only plan and the release that seals it.

Built on a synthetic complement release with one finished row, one partial row, one output-only row and
never-started rows, whose shard jobs ended in different ways.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from steering import plan_continuation as continuation, release, whole_reasoning as whole
from steering import pulsed_steering as pulse
from store import MODELS, WORK, digest, file_sha, load, publish

TMP = WORK/'tools/tmp'


def native(family: str) -> dict:
    return {'model': 'gemma', 'family': family, 'benchmark': 'aime', 'fold': 0, 'seed': 7, 'total_allowance': 100,
            'arm': 'native', 'class': None, 'operation': 'native', 'actions': [], 'nomination_status': 'nominated',
            'audit_capture': False, 'uid': 'native-'+family}


class SyntheticComplement(unittest.TestCase):
    """report/parent: 4 families x 2 arms, 4 shards; row 0 finished, row 1 partial, row 2 output only."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        self.reports = Path(self.directory.name)/'report'
        self.parent = self.reports/'parent'
        self.rows = [{**native(f'f{i}'), 'arm': arm, 'class': 'Verify', 'operation': 'suppress',
                      'uid': digest(['row', i, arm])}
                     for i in range(4) for arm in ('target_suppress', 'random_suppress')]
        publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'records': self.rows, 'planned': 8, 'executable': 8})
        self.input_file = self.reports.parent/'input.json'
        self.input_file.write_text('{}')
        self.release = publish(self.parent/'RELEASE.json', {
            'models': ['gemma'], 'sweep': {'gemma': 'main'}, 'shards': {'gemma': 4}, 'complement_sha256': 'x',
            'cutoff': 1000., 'release_root': '/old/snapshot', 'inputs': {str(self.input_file): file_sha(self.input_file)},
            'classifier': {}, 'nomination_reference': 'n', 'template_release_sha256': 't'})
        store = pulse.Store(self.parent/'stores/gemma/main/units')
        publish(store.folder(self.rows[0]['uid'])/'RESULT.json', {'status': 'complete'})
        pulse.checkpoint(store.folder(self.rows[1]['uid'])/'PARTIAL.json', {'uid': self.rows[1]['uid']})
        publish(store.folder(self.rows[2]['uid'])/'OUTPUT.json', {'token_ids': []})
        # Shard i takes family f<i>: shard 0 finished, shard 1 failed after starting f1 (row 2), shards 2 and 3
        # (f2, f3) ended without touching a row.
        (self.parent/'JOBS.txt').write_text(
            ''.join(f'whole-run-gemma {i} sbatch -J whole-run-gemma steering.whole_reasoning run --release R --model gemma '
                    f'--shard {i}/4\n' for i in range(4))
            + 'whole-tests 90 sbatch -J whole-tests steering.whole_reasoning validate --release R\n'
            + 'whole-extract-gemma 7 sbatch -J whole-extract-gemma steering.whole_reasoning extract --release R --model gemma\n')
        self.states = dict(zip('0123', ('COMPLETED', 'FAILED', 'FAILED', 'CANCELLED')))
        patcher = patch.object(continuation, 'job_states', side_effect=lambda ids: {i: self.states[i] for i in ids})
        patcher.start()
        self.addCleanup(patcher.stop)


class FollowupPlanTests(SyntheticComplement):
    def test_rows_by_state_and_shards_follow_the_parent_capacity(self):
        plan = continuation.plan_continuation(self.parent)
        value = plan['models']['gemma']
        self.assertEqual({k: value[k] for k in ('never_started', 'partial', 'complete', 'output', 'unfinished')},
                         {'never_started': 5, 'partial': 1, 'complete': 1, 'output': 1, 'unfinished': 6})
        # The parent ran 8 rows in 4 shards: 2 rows per shard, so 6 rows need 3 shards (4 families cover them).
        self.assertEqual((value['rows_per_shard'], value['shards_needed'], plan['shards']), (2., 3, {'gemma': 3}))
        self.assertEqual((plan['gpu_ceiling'], plan['rows_beyond_shard_capacity']), (34., 0))
        self.assertEqual(value['parent_shards'], {'jobs': 4, 'failed': 3, 'failed_before_start': 2,
                                                   'states': {'COMPLETED': 1, 'FAILED': 2, 'CANCELLED': 1}})
        self.assertEqual((plan['live_parent_jobs'], plan['parent_extraction_jobs']), ([], ['7']))
        table = continuation.format_plan(plan)
        self.assertIn('gemma', table)
        self.assertIn('3 (  2) of   4', table)
        self.assertIn('parent extraction jobs', table)

    def test_jobs_that_are_not_terminal_are_live(self):
        self.states = dict(zip('0123', ('RUNNING', 'PENDING', 'UNKNOWN', 'TIMEOUT')))
        self.assertEqual(continuation.plan_continuation(self.parent)['live_parent_jobs'], ['0', '1', '2'])

    def test_the_gpu_limit_scales_shards_down_and_names_the_rows_left_over(self):
        plan = continuation.plan_continuation(self.parent, max_gpu_hours=18.)
        self.assertEqual((plan['shards'], plan['gpu_ceiling'], plan['rows_beyond_shard_capacity']),
                         ({'gemma': 1}, 18., 4))
        with self.assertRaisesRegex(ValueError, 'exceed'):
            continuation.plan_continuation(self.parent, max_gpu_hours=18., shards={'gemma': 2})
        with self.assertRaisesRegex(ValueError, 'exactly the models'):
            continuation.plan_continuation(self.parent, shards={'glm': 2})

    def test_only_a_complement_with_unfinished_rows_has_a_follow_up(self):
        publish(self.reports/'base/RELEASE.json', {k: v for k, v in self.release.items()
                                                   if k not in ('sha256', 'complement_sha256')})
        with self.assertRaisesRegex(ValueError, 'not a complement release'):
            continuation.plan_continuation(self.reports/'base')
        store = pulse.Store(self.parent/'stores/gemma/main/units')
        for row in self.rows:
            folder = store.folder(row['uid'])
            for name in ('OUTPUT.json', 'PARTIAL.json'):
                (folder/name).unlink(missing_ok=True)
            publish(folder/'RESULT.json', {'status': 'complete'})
        with self.assertRaisesRegex(ValueError, 'no unfinished rows'):
            continuation.plan_continuation(self.parent)

    def test_fit_shards_keeps_one_per_model_and_never_exceeds_the_limit(self):
        needed = {'a': 27, 'b': 12, 'c': 3, 'd': 1}
        self.assertEqual(continuation.fit_shards(needed, 43), needed)
        for limit in (4, 5, 12, 30, 42):
            fitted = continuation.fit_shards(needed, limit)
            self.assertEqual(sum(fitted.values()), limit)
            self.assertTrue(all(1 <= fitted[m] <= needed[m] for m in needed), fitted)
        with self.assertRaisesRegex(ValueError, 'allows 3 shards for 4 models'):
            continuation.fit_shards(needed, 3)

    def test_shard_counts_are_parsed_from_model_colon_count(self):
        self.assertEqual(continuation.parse_shards('gemma:3,glm:12'), {'gemma': 3, 'glm': 12})


class SubmittedJobsTests(unittest.TestCase):
    def test_both_job_records_give_the_shard_and_extraction_jobs(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            text, record = Path(temporary)/'text', Path(temporary)/'record'
            text.mkdir()
            record.mkdir()
            (text/'JOBS.txt').write_text(
                'whole-tests 1 sbatch -J whole-tests steering.whole_reasoning validate --release R\n'
                'whole-run-glm 2 sbatch -J whole-run-glm run_module_gpu.sbatch steering.whole_reasoning run --release R --model glm --shard 3/27\n'
                'whole-extract-glm 3 sbatch -J whole-extract-glm run_module_cpu.sbatch steering.whole_reasoning extract --release R --model glm\n')
            publish(record/'JOBS.json', {'production': {'glm': [{'id': '2', 'model': 'glm', 'shard': [3, 27]}]},
                                      'extraction': {'glm': {'id': '3'}}})
            expected = {'run': [{'id': '2', 'model': 'glm', 'shard': [3, 27]}],
                        'extract': [{'id': '3', 'model': 'glm', 'shard': None}]}
            self.assertEqual(continuation.submitted_jobs(text), expected)
            self.assertEqual(continuation.submitted_jobs(record), expected)

    def test_releases_launched_before_the_module_renames_are_read_too(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            (Path(temporary)/'JOBS.txt').write_text(  # the JOBS.txt of earlier releases names the module steering.whole
                'whole-run-glm 2 sbatch -J whole-run-glm run_module_gpu.sbatch steering.whole run --release R --model glm --shard 3/27\n'
                'whole-extract-glm 3 sbatch -J whole-extract-glm run_module_cpu.sbatch steering.whole extract --release R --model glm\n')
            self.assertEqual(continuation.submitted_jobs(Path(temporary)),
                             {'run': [{'id': '2', 'model': 'glm', 'shard': [3, 27]}],
                              'extract': [{'id': '3', 'model': 'glm', 'shard': None}]})

    def test_a_job_slurm_no_longer_knows_is_unknown_and_a_cancelled_one_keeps_its_first_word(self):
        answer = SimpleNamespace(stdout='5|COMPLETED\n6|CANCELLED by 133943\n')
        with patch.object(continuation.subprocess, 'run', return_value=answer):
            self.assertEqual(continuation.job_states(['5', '6', '7']),
                             {'5': 'COMPLETED', '6': 'CANCELLED', '7': 'UNKNOWN'})
        self.assertEqual(continuation.job_states([]), {})


class ContinuationSealTests(SyntheticComplement):
    """steering.release continuation, with everything it reads outside the report folder replaced by stand-ins."""

    def setUp(self):
        super().setUp()
        self.link = self.reports.parent/'LINK.json'
        publish(self.link, {'models': list(MODELS)})
        chain = [{'model': 'gemma', 'root': str(self.reports/'base'), 'dose': 'main', 'topups': [],
                  'complements': [str(self.parent)]},
                 {'model': 'glm', 'root': str(self.reports/'base'), 'dose': 'half', 'complements': [str(self.parent)]}]
        publish(self.reports/'chain/SOURCES.json', {'sources': chain})
        snapshot = self.reports.parent/'snapshot'
        (snapshot/'src').mkdir(parents=True)
        (snapshot/'src/code.py').write_text('x = 1\n')
        self.cutoff = float(int(time.time())+86400)
        self.args = SimpleNamespace(models=None, shards=None, cutoff=str(int(self.cutoff)), parent='parent',
                                    chain='chain', link=self.link, max_gpu_hours=None)

        def command(arguments, **kwargs):
            return SimpleNamespace(stdout='' if arguments[1] == 'status' else 'abc123\n')

        def make_snapshot(root, commit):
            (root/'SOURCE.tar.gz').write_text('archive')
            return root/'SOURCE.tar.gz', snapshot
        for patcher in (patch.object(release, 'REPORTS', self.reports), patch.object(release, 'snapshot', make_snapshot),
                        patch.object(release.subprocess, 'run', command)):
            self.enterContext(patcher)

    def test_a_continuation_is_sealed_with_the_chain_the_link_and_the_parent_manifests(self):
        release.seal('continuation', 'child', 12, self.args)
        root = self.reports/'child'
        sealed = load(root/'RELEASE.json')
        self.assertEqual((sealed['cutoff'], sealed['shards'], sealed['sweep']),
                         (self.cutoff, {'gemma': 3}, {'gemma': 'main'}))
        self.assertEqual((sealed['new_gpu_hour_ceiling'], sealed['cpu_core_hour_ceiling']), (34., 200.))
        self.assertEqual((sealed['parent_release_sha256'], sealed['template_release_sha256']),
                         (self.release['sha256'], 't'))
        self.assertEqual(sealed['continuation_sha256'], load(root/'CONTINUATION.json')['sha256'])
        self.assertIn(str(self.link), sealed['inputs'])
        self.assertIn(str(self.input_file), sealed['inputs'])
        self.assertIn(str(whole.manifest_path('gemma', 'main', self.parent)), sealed['inputs'])
        kind = load(root/'CONTINUATION.json')
        self.assertEqual(kind['models'], {'gemma': {'parent': str(self.parent)}})
        self.assertEqual((kind['link'], kind['plan']['shards']), (str(self.link), {'gemma': 3}))
        # The stages read the link from the release, and the combine runs the source chain plus this release.
        self.assertEqual(whole.link_receipt(root), self.link)
        self.assertEqual(whole.sweep_doses(root), {'gemma': 'main'})
        sources = load(root/'SOURCES.json')['sources']
        self.assertEqual([s['complements'] for s in sources if s['model'] == 'gemma'], [[str(self.parent), str(root)]])
        self.assertEqual(next(s for s in sources if s['model'] == 'glm')['complements'], [str(self.parent)])
        # The chain release itself is untouched.
        self.assertEqual(load(self.reports/'chain/SOURCES.json')['sources'][0]['complements'], [str(self.parent)])

    def test_the_chain_names_the_parent_by_another_path_to_the_same_folder(self):
        alias = self.reports/'alias'
        alias.symlink_to(self.parent, target_is_directory=True)
        chain = [{'model': 'gemma', 'root': str(self.reports/'base'), 'dose': 'main', 'complements': [str(alias)]}]
        publish(self.reports/'aliased/SOURCES.json', {'sources': chain})
        release.seal('continuation', 'child', 12, SimpleNamespace(**{**vars(self.args), 'chain': 'aliased'}))
        root = self.reports/'child'
        sources = load(root/'SOURCES.json')['sources']
        self.assertEqual(sources[0]['complements'], [str(alias), str(root.resolve())])

    def test_a_continuation_waits_for_the_parent_and_for_a_later_cutoff_and_leaves_nothing_behind(self):
        self.states = dict(zip('0123', ('COMPLETED', 'FAILED', 'FAILED', 'RUNNING')))
        with self.assertRaisesRegex(ValueError, 'not ended'):
            release.seal('continuation', 'child', 12, self.args)
        self.states = dict(zip('0123', ('COMPLETED', 'FAILED', 'FAILED', 'CANCELLED')))
        with self.assertRaisesRegex(ValueError, 'cutoff must be later'):
            release.seal('continuation', 'child', 12, SimpleNamespace(**{**vars(self.args), 'cutoff': str(int(time.time())-5)}))
        with self.assertRaisesRegex(ValueError, '--parent is required'):
            release.seal('continuation', 'child', 12, SimpleNamespace(**{**vars(self.args), 'parent': None}))
        self.assertFalse((self.reports/'child').exists())

    def test_the_chain_must_hold_the_parent_for_every_model_with_unfinished_rows(self):
        publish(self.reports/'other/SOURCES.json', {'sources': [
            {'model': 'gemma', 'root': 'r', 'dose': 'main', 'complements': []}]})
        with self.assertRaisesRegex(ValueError, 'no gemma source with parent'):
            release.seal('continuation', 'child', 12, SimpleNamespace(**{**vars(self.args), 'chain': 'other'}))
        self.assertFalse((self.reports/'child').exists())


if __name__ == '__main__':
    unittest.main()
