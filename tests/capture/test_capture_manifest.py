"""The checks around a capture run, on a tiny synthetic shard: manifest, resume markers, verification receipt, gate."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from capture import buffers, gate, run
from capture.manifest import (
    atomic, capture_folders, digest, load_manifest, read_location, resume_records, validate_verification, write_new,
)
from store import file_sha

COHORT_CORE = ('schema_version', 'model', 'seed', 'label_total', 'per_b', 'floor', 'traces_sha256')
LAYERS = [0, 1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Shard:
    """Two traces captured in one shard of model `source`, laid out like the real manifests and capture folders.

    With `repeat` the capture and its verification are in a repeat tree (`tree/0`, `tree/verify`) while the manifest
    still names the paper's folders, which do not exist."""

    def __init__(self, root, source='qwen36', repeat=False):
        self.root, self.source = Path(root), source
        self.tree = self.root / 'tree'
        cohort = dict(schema_version=1, corpus='card-v3', model=source, seed=7, label_total=2, per_b=1, floor=1,
                      traces_sha256='0' * 64, rows=[])
        cohort['manifest_sha256'] = digest({k: cohort[k] for k in COHORT_CORE})
        self.cohort_path = self.root / 'cohorts.json'
        write_json(self.cohort_path, cohort)
        paper = self.root / 'capture' / source
        self.out = self.tree / '0' if repeat else paper / 'A' / '0'
        self.out.mkdir(parents=True)
        rows = [dict(id=f'attempt{i}', trace_sha256=str(i) * 64, completion_tokens=10 * i,
                     total_input_tokens=30 * i) for i in (1, 2)]
        manifest = dict(
            schema_version=3, source=source, cohort='A', shard=0, spec=dict(capture_version=2, layers=LAYERS),
            storage_root=str(self.root), cohort_model=source, cohort_manifest_path=str(self.cohort_path.resolve()),
            cohort_binding=dict(schema_version=1, manifest_sha256=cohort['manifest_sha256'],
                                file_sha256=file_sha(self.cohort_path), path=str(self.cohort_path.resolve())),
            code_binding='c' * 64, validation_binding='v' * 64, annotation_binding=None, maximum_input_tokens=60,
            attempts=rows, expected_attempts=2, expected_tokens=sum(r['completion_tokens'] for r in rows),
            capture_dir=str(paper / 'A' / '0'), verification_path=str(paper / 'verify' / 'verification.json'))
        manifest['binding'] = digest(manifest)
        self.manifest = manifest
        self.manifest_dir = self.root / 'manifests' / source / 'A'
        write_json(self.manifest_dir / '0.json', manifest)
        write_json(self.tree / 'verify' / 'verification.json' if repeat else Path(manifest['verification_path']), dict(
            code=dict(source_sha256='s' * 64),
            status='passed', validation_binding='v' * 64, maximum_input_tokens=60, all_routed_layers=LAYERS,
            exact_saved_token_coverage=True, unchanged_resume_verified=True, differential=dict(status='passed')))
        write_json(self.out / 'run-binding.json', {'manifest_binding': manifest['binding'], 'source_sha256': 's' * 64})
        self.records = [self.capture(row) for row in rows]
        used = sum(f['bytes'] for r in self.records for f in r['files'])
        write_json(self.out / 'result.json', dict(
            status='complete_native_collection', manifest_binding=manifest['binding'], records=self.records,
            tokens=manifest['expected_tokens'], artifact_bytes=used))
        ready = {k: manifest[k] for k in gate.SHARED_KEYS}
        ready.update(status='ready', attempts=2, tokens=manifest['expected_tokens'],
                     shards=[dict(index=0, binding=manifest['binding'])])
        write_json(self.manifest_dir / 'ready.json', ready)

    def capture(self, row):
        files = []
        for suffix in ('.pt.gz', '.json.gz'):
            path = self.out / (row['id'] + suffix)
            path.write_bytes(gzip.compress(row['id'].encode() + suffix.encode()))
            files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path)))
        record = dict(id=row['id'], manifest_binding=self.manifest['binding'], trace_sha256=row['trace_sha256'],
                      files=files, tokens=row['completion_tokens'])
        write_json(self.out / (row['id'] + '.complete.json'), record)
        return record


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.shard = Shard(self.tmp.name)
        self.path = self.shard.manifest_dir / '0.json'

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite(self, **changes):
        manifest = {**self.shard.manifest, **changes}
        manifest['binding'] = digest({k: v for k, v in manifest.items() if k != 'binding'})
        write_json(self.path, manifest)

    def test_a_consistent_manifest_loads(self):
        self.assertEqual(load_manifest(self.path)['binding'], self.shard.manifest['binding'])

    def test_a_changed_field_breaks_the_binding(self):
        write_json(self.path, {**self.shard.manifest, 'expected_tokens': 1})
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            load_manifest(self.path)

    def test_a_changed_cohort_file_is_refused(self):
        cohort = json.loads(self.shard.cohort_path.read_text())
        write_json(self.shard.cohort_path, {**cohort, 'seed': 8})
        with self.assertRaisesRegex(ValueError, 'Cohort manifest changed'):
            load_manifest(self.path)

    def test_only_the_native_capture_spec_is_accepted(self):
        self.rewrite(spec=dict(capture_version=1, layers=LAYERS))
        with self.assertRaisesRegex(ValueError, 'Native capture spec'):
            load_manifest(self.path)

    def test_outcomes_never_enter_a_manifest(self):
        self.rewrite(outcome_binding={})
        with self.assertRaisesRegex(ValueError, 'must not bind outcomes'):
            load_manifest(self.path)

    def test_duplicate_attempt_ids_are_refused(self):
        rows = self.shard.manifest['attempts']
        self.rewrite(attempts=[rows[0], rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate attempt'):
            load_manifest(self.path)

    def test_a_changed_annotation_file_is_refused(self):
        labels = Path(self.tmp.name) / 'labels.json'
        labels.write_text('[1]')
        binding = dict(inputs=[dict(path=str(labels), sha256=file_sha(labels))], annotations_sha256='a' * 64)
        self.rewrite(annotation_binding=binding)
        load_manifest(self.path)
        labels.write_text('[2]')
        with self.assertRaisesRegex(ValueError, 'Annotation input changed'):
            load_manifest(self.path)


class SourceLineTests(unittest.TestCase):
    def test_a_saved_trace_line_is_read_by_position_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traces.jsonl'
            lines = [json.dumps({'problem_id': name, 'metadata': {}}).encode() + b'\n' for name in ('a', 'b', 'c')]
            path.write_bytes(b''.join(lines))
            info = dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                        byte_offset=len(lines[0]), line_bytes=len(lines[1]),
                        line_sha256=hashlib.sha256(lines[1]).hexdigest(), sentence_schema=1)
            trace = read_location(info)
            self.assertEqual((trace['problem_id'], trace['metadata']), ('b', {'measurement_sentence_schema': 1}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_location({**info, 'line_sha256': '0' * 64})
            path.write_bytes(b''.join(lines) + b'x')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                read_location(info)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.shard = Shard(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finished_traces_are_skipped_and_the_rest_remain(self):
        (self.shard.out / 'attempt2.complete.json').unlink()
        done, remaining, used = resume_records(self.shard.manifest, self.shard.out)
        self.assertEqual([r['id'] for r in done], ['attempt1'])
        self.assertEqual([r['id'] for r in remaining], ['attempt2'])
        self.assertEqual(used, sum(f['bytes'] for f in done[0]['files']))

    def test_a_corrupted_artifact_is_refused(self):
        Path(self.shard.records[0]['files'][1]['path']).write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'Corrupt accepted capture'):
            resume_records(self.shard.manifest, self.shard.out)

    def test_a_marker_of_another_manifest_is_refused(self):
        marker = self.shard.out / 'attempt1.complete.json'
        write_json(marker, {**json.loads(marker.read_text()), 'manifest_binding': 'other'})
        with self.assertRaisesRegex(ValueError, 'Incompatible capture resume'):
            resume_records(self.shard.manifest, self.shard.out)

    def test_an_artifact_outside_the_output_folder_is_refused(self):
        elsewhere = Path(self.tmp.name) / 'elsewhere'
        elsewhere.mkdir()
        for marker in self.shard.out.glob('*.complete.json'):
            (elsewhere / marker.name).write_bytes(marker.read_bytes())
        with self.assertRaisesRegex(ValueError, 'escapes'):
            resume_records(self.shard.manifest, elsewhere)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.shard = Shard(self.tmp.name)
        self.receipt = json.loads(Path(self.shard.manifest['verification_path']).read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, source='qwen36', **changes):
        validate_verification({**self.shard.manifest, 'source': source}, {**self.receipt, **changes})

    def test_a_passed_verification_is_accepted(self):
        self.check()

    def test_a_failed_or_foreign_verification_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'No accepted verify-only receipt'):
            self.check(status='failed')
        with self.assertRaisesRegex(ValueError, 'No accepted verify-only receipt'):
            self.check(validation_binding='other')

    def test_the_longest_input_and_every_layer_must_be_covered(self):
        with self.assertRaisesRegex(ValueError, 'Longest input'):
            self.check(maximum_input_tokens=59)
        with self.assertRaisesRegex(ValueError, 'Incomplete native verification coverage'):
            self.check(all_routed_layers=[0])

    def test_gpt_and_glm_need_their_differential(self):
        self.check('qwen36', differential=None)
        for source in ('gpt', 'glm'):
            with self.assertRaisesRegex(ValueError, 'differential verification required'):
                self.check(source, differential=dict(status='failed'))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.shard = Shard(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_complete_capture_is_accepted(self):
        result = gate.gate(self.shard.manifest_dir)
        self.assertEqual((result['status'], result['attempts'], result['tokens']), ('complete', 2, 30))
        self.assertEqual(list(result['results']), [str(self.shard.out / 'result.json')])

    def test_a_repeat_capture_is_gated_in_its_own_tree(self):
        with tempfile.TemporaryDirectory() as other:
            repeat = Shard(other, repeat=True)
            result = gate.gate(repeat.manifest_dir, repeat.tree)
            self.assertEqual((result['status'], result['attempts']), ('complete', 2))
            with self.assertRaises(FileNotFoundError):
                gate.gate(repeat.manifest_dir)

    def test_a_missing_trace_is_refused(self):
        (self.shard.out / 'attempt2.complete.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete capture shard'):
            gate.gate(self.shard.manifest_dir)

    def test_a_damaged_file_is_refused(self):
        Path(self.shard.records[1]['files'][0]['path']).write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'Corrupt accepted capture'):
            gate.gate(self.shard.manifest_dir)

    def test_a_result_that_disagrees_with_the_markers_is_refused(self):
        result_path = self.shard.out / 'result.json'
        write_json(result_path, {**json.loads(result_path.read_text()), 'tokens': 31})
        with self.assertRaisesRegex(ValueError, 'Stale/inconsistent shard result'):
            gate.gate(self.shard.manifest_dir)

    def test_an_unplanned_manifest_is_refused(self):
        write_json(self.shard.manifest_dir / '1.json', self.shard.manifest)
        with self.assertRaisesRegex(ValueError, 'Unexpected/missing capture manifests'):
            gate.gate(self.shard.manifest_dir)

    def test_a_shard_that_differs_from_the_plan_is_refused(self):
        ready_path = self.shard.manifest_dir / 'ready.json'
        write_json(ready_path, {**json.loads(ready_path.read_text()), 'shards': [dict(index=0, binding='other')]})
        with self.assertRaisesRegex(ValueError, 'Ready shard binding differs'):
            gate.gate(self.shard.manifest_dir)

    def test_the_acceptance_is_written_once_and_must_not_change(self):
        output = Path(self.tmp.name) / 'acceptance.json'
        env = {'SLURM_JOB_ID': '1'}
        with mock.patch.dict(os.environ, env), mock.patch('sys.argv', ['gate', str(self.shard.manifest_dir), str(output)]):
            gate.main()
            first = json.loads(output.read_text())
            gate.main()
            self.assertEqual(json.loads(output.read_text())['results'], first['results'])
            write_json(output, {**first, 'tokens': 1})
            with self.assertRaisesRegex(ValueError, 'Existing acceptance differs'):
                gate.main()
        self.assertEqual(set(first), {'status', 'attempts', 'tokens', 'ready_sha256', 'results', 'code'})


class RunHelperTests(unittest.TestCase):
    CODE = {'source_sha256': 's' * 64}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # the temporary folder stands in for the approved capture storage
        patch = mock.patch.object(buffers, 'CAPTURE_ROOTS', (self.root,))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_output_folder_belongs_to_the_first_manifest_and_code_that_wrote_to_it(self):
        manifest = dict(binding='a' * 64)
        out = self.root / 'out'
        out.mkdir()
        run.claim_capture_output(manifest, out, self.CODE)
        run.claim_capture_output(manifest, out, self.CODE)
        with self.assertRaisesRegex(ValueError, 'other inputs or by other code'):
            run.claim_capture_output({**manifest, 'binding': 'b' * 64}, out, self.CODE)
        with self.assertRaisesRegex(ValueError, 'other inputs or by other code'):
            run.claim_capture_output(manifest, out, {'source_sha256': 'edited'})
        other = self.root / 'other'
        other.mkdir()
        (other / 'stray.txt').write_text('x')
        with self.assertRaisesRegex(ValueError, 'Unbound existing capture output'):
            run.claim_capture_output(manifest, other, self.CODE)

    def test_a_repeat_capture_goes_to_a_folder_named_like_its_shard(self):
        shard = Shard(self.root, repeat=True)
        out, repeat = run.prepare_output(shard.manifest, self.root / 'fresh' / '0', self.CODE)
        self.assertEqual((out.name, repeat, sorted(os.listdir(out))), ('0', True, ['run-binding.json']))
        with self.assertRaisesRegex(ValueError, 'repeat capture goes to ROOT/0'):
            run.prepare_output(shard.manifest, self.root / 'fresh' / 'elsewhere', self.CODE)

    def test_a_finished_capture_is_not_run_or_overwritten_again(self):
        shard = Shard(self.root, repeat=True)
        with self.assertRaisesRegex(ValueError, 'already holds result.json'):
            run.prepare_output(shard.manifest, shard.out, self.CODE)
        with self.assertRaisesRegex(ValueError, 'other inputs or by other code'):
            run.prepare_output(shard.manifest, shard.out, {'source_sha256': 'edited'})
        (shard.out / 'result.json').unlink()
        run.prepare_output(shard.manifest, shard.out, self.CODE)  # an interrupted run resumes
        write_json(shard.out / 'verification.json', {'status': 'passed'})
        with self.assertRaisesRegex(ValueError, 'already holds verification.json'):
            run.prepare_output(shard.manifest, shard.out, self.CODE)

    def test_the_manifests_own_capture_dir_is_not_a_repeat(self):
        shard = Shard(self.root)
        (shard.out / 'result.json').unlink()
        out, repeat = run.prepare_output(shard.manifest, shard.manifest['capture_dir'], self.CODE)
        self.assertEqual((out, repeat), (shard.out.resolve(), False))

    def test_a_shard_in_a_repeat_tree_needs_a_verification_written_by_this_code(self):
        shard = Shard(self.root, repeat=True)
        run.check_verification(shard.manifest, shard.out, True, self.CODE)
        with self.assertRaisesRegex(ValueError, 'written by other code'):
            run.check_verification(shard.manifest, shard.out, True, {'source_sha256': 'edited'})
        paper = Shard(Path(self.tmp.name) / 'paper')
        run.check_verification(paper.manifest, paper.out, False, {'source_sha256': 'edited'})
        (shard.tree / 'verify' / 'verification.json').unlink()
        with self.assertRaises(FileNotFoundError):
            run.check_verification(shard.manifest, shard.out, True, self.CODE)

    def test_no_trace_is_started_in_the_last_half_hour_of_the_job(self):
        now = time.time()
        self.assertFalse(run.must_stop(None))
        self.assertFalse(run.must_stop(now + 3600))
        self.assertTrue(run.must_stop(now + 1700))
        self.assertTrue(run.must_stop(now - 5))

    def test_packed_files_are_lossless_reproducible_and_capped(self):
        raw = b'router' * 5000
        first = run.pack_bytes(self.root / 'a.pt.gz', raw, 10 ** 6)
        second = run.pack_bytes(self.root / 'b.pt.gz', raw, 10 ** 6)
        self.assertEqual(gzip.decompress((self.root / 'a.pt.gz').read_bytes()), raw)
        self.assertEqual((first['sha256'], first['raw_sha256']), (second['sha256'], hashlib.sha256(raw).hexdigest()))
        with self.assertRaisesRegex(RuntimeError, 'storage cap'):
            run.pack_bytes(self.root / 'c.pt.gz', raw, 10)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a.pt.gz', 'b.pt.gz'])

    def test_the_differential_receipt_is_bound_to_its_verification(self):
        path, binding = self.root / 'differential.json', {'short_id': 'x'}
        self.assertIsNone(run.saved_differential(path, binding))
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            run.save_differential(path, binding, {'status': 'failed'})
        run.save_differential(path, binding, {'status': 'passed', 'flips': 0})
        self.assertEqual(run.saved_differential(path, binding), {'status': 'passed', 'flips': 0})
        with self.assertRaisesRegex(ValueError, 'Stale or failed'):
            run.saved_differential(path, {'short_id': 'y'})
        with self.assertRaisesRegex(ValueError, 'Existing differential differs'):
            run.save_differential(path, binding, {'status': 'passed', 'flips': 1})


class CapturePathTests(unittest.TestCase):
    def test_captures_go_only_under_the_capture_roots(self):
        mfa = '/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe'
        for allowed in ('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/capture/gpt/A/0',
                        mfa + '/r/some-job/capture', mfa + '/campaign-v3/x'):
            self.assertEqual(str(buffers.approved_capture_path(allowed)), allowed)
        for refused in ('/tmp/capture', '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/capture',
                        '/leonardo/home/userexternal/lmolfett/capture', mfa + '/repo/results', mfa + '/r/../repo',
                        '/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/x'):
            with self.assertRaisesRegex(ValueError, 'Unapproved capture path'):
                buffers.approved_capture_path(refused)

    def test_a_git_checkout_is_never_a_capture_folder(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(buffers, 'CAPTURE_ROOTS', (Path(tmp),)):
            (Path(tmp) / 'checkout' / '.git').mkdir(parents=True)
            self.assertEqual(buffers.approved_capture_path(Path(tmp) / 'fresh'), (Path(tmp) / 'fresh').resolve())
            with self.assertRaisesRegex(ValueError, 'inside a git checkout'):
                buffers.approved_capture_path(Path(tmp) / 'checkout' / 'capture')


class AtomicTests(unittest.TestCase):
    def test_atomic_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sub' / 'x.json'
            atomic(path, {'a': 1})
            self.assertEqual(json.loads(path.read_text()), {'a': 1})
            self.assertEqual(os.listdir(path.parent), ['x.json'])
            with self.assertRaises(ValueError):
                atomic(path, {'a': float('nan')})

    def test_a_receipt_is_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'verification.json'
            write_new(path, {'status': 'passed'})
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                write_new(path, {'status': 'passed'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'passed'})

    def test_the_folders_of_a_shard_are_the_papers_or_those_of_a_repeat_tree(self):
        manifest = {'capture_dir': '/p/capture/m/A/3', 'verification_path': '/p/capture/m/verify/verification.json', 'shard': 3}
        self.assertEqual(capture_folders(manifest), (Path('/p/capture/m/A/3'), Path('/p/capture/m/verify/verification.json')))
        self.assertEqual(capture_folders(manifest, '/r/tree'), (Path('/r/tree/3'), Path('/r/tree/verify/verification.json')))


if __name__ == '__main__':
    unittest.main()
