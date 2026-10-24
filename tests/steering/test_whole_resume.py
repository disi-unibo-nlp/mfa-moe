"""Resumption of interrupted whole-reasoning continuations, independent of model loading.

A resume release re-prefills prompt + saved tokens through the same stream hooks, so these tests pin
that the hooks treat a re-prefilled request exactly like an uninterrupted one, and that the manifest,
state gate and seeding that surround the engine keep unknown outcomes unknown.
"""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from steering import pulse as driver, whole, whole_effects
from store import WORK, digest, file_sha, load, publish
from steering.stream import Clock, StreamCapture

TMP = WORK/'tools/tmp'
# Step counters differ legitimately between chunkings; every other counter must match.
STEP_COUNTERS = ('finite', 'observations')
POSITION_COUNTERS = ('active_positions', 'active_prompt_positions', 'active_generated_positions',
                     'changed_membership', 'membership_failures')


def feed(clock, tokens, start, stop, size):
    """Observe tokens[start:stop] as consecutive chunks of `size`, like prefill chunks or decode steps."""
    contexts = []
    for first in range(start, stop, size):
        last = min(first+size, stop)
        contexts += clock.observe(tokens[first:last], list(range(first, last)))
    return contexts


class ClockEquivalenceTests(unittest.TestCase):
    """A fresh Clock fed prompt + history as prefill agrees with the Clock that watched it decode."""
    CLOSURE = [(7, 8, 9), (12,)]

    def check(self, prompt, generated, expected_closed):
        full = prompt+generated
        original = Clock(prompt, [], self.CLOSURE)
        reference = feed(original, full, 0, len(prompt), len(prompt)) + feed(original, full, len(prompt),
                                                                             len(full), 1)
        self.assertEqual(original.closed, expected_closed)
        for history in range(len(generated)+1):
            for size in (1, 2, 3, 4, 5, 7):
                with self.subTest(history=history, size=size):
                    resumed = Clock(prompt, [], self.CLOSURE)
                    stop = len(prompt)+history
                    contexts = feed(resumed, full, 0, stop, size) + feed(resumed, full, stop, len(full), 1)
                    # Same prompt flag, reasoning flag, generated position and closure for every position.
                    self.assertEqual(contexts, reference)
                    self.assertEqual((resumed.closed, resumed.history), (original.closed, original.history))
        return reference

    def test_multi_token_closure_inside_the_history_and_split_across_every_boundary(self):
        reference = self.check([1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 3, 4, 5, 6, 1], True)
        reasoning = [c['reasoning'] for c in reference[6:]]
        self.assertEqual(reasoning, [True]*4+[False]*6)
        self.assertEqual([c['prompt'] for c in reference], [True]*6+[False]*10)
        self.assertEqual([c['generated_position'] for c in reference[6:]], list(range(10)))

    def test_single_token_closure_and_no_closure(self):
        self.assertTrue(all(not c['reasoning'] for c in self.check([1, 2, 3], [4, 5, 6, 12, 4, 4], True)[-3:]))
        self.assertTrue(all(c['reasoning'] for c in self.check([1, 2, 3], [4, 5, 6, 4, 4, 4], False)))

    def test_prompt_suffix_never_completes_a_closure_with_generated_tokens(self):
        # The prompt ends with the first two closure tokens; only generated tokens count toward closure.
        reference = self.check([1, 2, 3, 7, 8], [9, 4, 4, 4], False)
        self.assertTrue(all(c['reasoning'] for c in reference))

    def test_recompute_from_position_zero_replays_the_same_contexts(self):
        full = [1, 2, 3]+[1, 7, 8, 9, 4]
        clock = Clock([1, 2, 3], [], self.CLOSURE)
        first = feed(clock, full, 0, len(full), 2)
        second = feed(clock, full, 0, len(full), 3)
        self.assertEqual((clock.passes, second), (2, first))


def attach(prompt, prompt_scope, closure=((9, 9),)):
    """The real clock, masks and dual operator on a tiny native router whose routing depends on the token.

    Mirrors test_whole.PromptScopeTests.capture, with token-dependent logits so that executed expert IDs
    differ per position, and a parameterized prompt scope. Expert 31 is natively selected and expert 0
    natively unselected, so steering changes the membership of every steered position.
    """
    table = torch.randn(16, 32, generator=torch.Generator().manual_seed(7))
    table[:, 31] += 4.
    table[:, 0] -= 4.

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            return router._compute_routing(router_logits=table[input_ids])

    def native(router_logits):
        q = router_logits.softmax(-1)
        ids = q.topk(4, -1).indices
        weights = q.gather(1, ids)
        return weights/weights.sum(-1, keepdim=True), ids
    router = SimpleNamespace(top_k=4, global_num_experts=32, _compute_routing=native)
    model = Model()
    runner = SimpleNamespace(get_model=lambda: model, _prepare_inputs=lambda: None,
                             input_batch=SimpleNamespace(req_ids=['internal']),
                             query_start_loc=SimpleNamespace(np=np.array([0, 0])))
    with patch('steering.stream.discover', return_value={0: ('router', router, None)}), \
         patch('steering.stream.policy', return_value={'scoring': 'softmax', 'renormalize': True}):
        capture = StreamCapture(runner, {'layers': [0], 'hook_layers': [0], 'native_k': 4, 'experts': 32})
    capture.add([{'uid': 'uid', 'prompt': prompt, 'prefix': [], 'closure': [list(s) for s in closure],
                  'actions': [{'layer': 0, 'activate': [0], 'deactivate': [31], 'pulse_tokens': 0}],
                  'capture_layers': [0], 'window': 50, 'whole': True, 'prompt_scope': prompt_scope,
                  'audit_capture': True}], {'internal': 'uid'})
    return capture, runner, model


def trace(prompt, generated, prompt_scope, plan):
    """Drive one request through fresh hooks in the given chunk sizes; return executed IDs and its release."""
    capture, runner, model = attach(prompt, prompt_scope)
    try:
        tokens, executed, position = prompt+generated, [], 0
        assert sum(plan) == len(tokens), (plan, len(tokens))
        for size in plan:
            chunk = tokens[position:position+size]
            runner.query_start_loc.np = np.array([0, len(chunk)])
            runner._prepare_inputs()
            executed.append(model(torch.tensor(chunk), torch.tensor(range(position, position+len(chunk))))[1])
            position += len(chunk)
        return torch.cat(executed), capture.release('uid')
    finally:
        capture.restore()


class CaptureEquivalenceTests(unittest.TestCase):
    """Executed routing and dose counters agree between an uninterrupted request and a re-prefilled one."""

    def compare(self, prompt, generated, prompt_scope, history, size):
        uninterrupted = trace(prompt, generated, prompt_scope, [len(prompt)]+[1]*len(generated))
        stop = len(prompt)+history
        plan = [min(size, stop-first) for first in range(0, stop, size)]+[1]*(len(generated)-history)
        resumed = trace(prompt, generated, prompt_scope, plan)
        self.assertTrue(torch.equal(uninterrupted[0], resumed[0]))
        a, b = uninterrupted[1], resumed[1]
        self.assertEqual((a['closed'], a['passes'], a['observed_tokens']), (b['closed'], b['passes'], b['observed_tokens']))
        log_a, log_b = a['capture']['0'], b['capture']['0']
        for key in ('positions', 'native_ids', 'executed_ids', 'active'):
            self.assertEqual(log_a[key], log_b[key], key)
        np.testing.assert_allclose(log_a['executed_weights'], log_b['executed_weights'], rtol=1e-6)
        dose_a, dose_b = a['stats']['dose']['0'], b['stats']['dose']['0']
        for key, value in dose_a.items():
            if key in STEP_COUNTERS:  # one per hooked step: the chunking differs, finite == observations holds
                continue
            if key in POSITION_COUNTERS:
                self.assertEqual(value, dose_b[key], key)
            else:
                self.assertAlmostEqual(value, dose_b[key], places=4, msg=key)
        for dose in (dose_a, dose_b):
            self.assertEqual(dose['finite'], dose['observations'])
        self.assertEqual(a['stats']['active_positions'], b['stats']['active_positions'])
        return dose_a, uninterrupted[0]

    def test_resumed_request_matches_uninterrupted_for_both_scopes_and_every_closure_placement(self):
        generated = [1, 2, 9, 9, 3, 4, 5, 6]  # the closure sits at generated positions 2-3
        for prompt in ([1, 2, 3, 4, 5], [1, 2, 3, 9, 9]):  # the second ends with a Harmony-like marker
            for scope in (True, False):
                for history in (0, 2, 3, 4, 6, 8):  # split at the closure, ending on it, or inside the history
                    for size in (2, 3):
                        with self.subTest(prompt=prompt[-2:], scope=scope, history=history, size=size):
                            dose, executed = self.compare(prompt, generated, scope, history, size)
                            self.assertEqual(dose['active_prompt_positions'], len(prompt) if scope else 0)
                            # Steering stops at the closure: generated positions 0-2 precede its last token.
                            self.assertEqual(dose['active_generated_positions'], 3)
                            self.assertEqual(dose['changed_membership'], dose['active_positions'])
                            generated_ids = executed[len(prompt):]
                            self.assertTrue(all(0 in row and 31 not in row for row in generated_ids[:3].tolist()))
                            self.assertTrue(all(31 in row for row in generated_ids[3:].tolist()))


def partial(row, tokens, segments=None, **extra):
    """A PARTIAL.json as Engine.admit/drain write it."""
    segments = segments if segments is not None else [{'job': '1', 'history': 0, 'seed': row.get('seed', 0),
                                                       'max_tokens': 100, 'drained': 1., 'generated': len(tokens)}]
    return {'uid': row['uid'], 'assignment_sha256': digest(row), 'tokens': tokens, 'segments': segments,
            'total_allowance': 100, **extra}


def synthetic_rows():
    return [{'uid': digest([family, arm]), 'family': family, 'arm': arm, 'class': None, 'operation': 'native',
             'model': 'gemma', 'seed': index, 'nomination_status': 'nominated', 'total_allowance': 100}
            for index, (family, arm) in enumerate((f, a) for f in ('f0', 'f1', 'f2') for a in ('native', 'repeat'))]


class ResumeManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.parent, self.topup = base/'parent', base/'topup'
        self.rows = synthetic_rows()
        self.base = publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'scope': 'prompt-and-generated-reasoning', 'records': self.rows})
        publish(whole.manifest_path('gemma', 'main', self.topup), {
            'records': self.rows[4:], 'parent_manifest_sha256': self.base['sha256'], 'design': 'top-up'})
        self.stores = {'parent': self.parent/'stores/gemma/main/units', 'topup': self.topup/'stores/gemma/main/units'}
        self.n = 0

    def put(self, where, index, kind, **changes):
        row = self.rows[index]
        folder = self.stores[where]/row['uid']
        if kind == 'partial':
            driver.checkpoint(folder/'PARTIAL.json', {**partial(row, [1, 2, 3]), **changes})
        elif kind == 'complete':
            publish(folder/'RESULT.json', {'status': 'complete'})
        elif kind == 'output':
            publish(folder/'OUTPUT.json', {'token_ids': [1]})

    def resume(self, sources=None, parent=None):
        self.n += 1
        spec = {'parent': str(parent or self.parent), 'sources': [str(s) for s in (sources or (self.parent, self.topup))]}
        root = Path(self.directory.name)/f'child{self.n}'
        publish(root/'RELEASE.json', {'models': ['gemma'], 'sweep': {'gemma': 'main'}})
        result = whole.resume_manifest(root, 'gemma', spec)
        return result, load(whole.manifest_path('gemma', 'main', root)), load(root/'gemma/SELECTED.json')

    def test_only_partial_rows_are_selected_each_with_its_source_store(self):
        self.put('parent', 0, 'complete')
        self.put('parent', 1, 'partial')          # interrupted in the parent
        # row 2 never started; row 3 crashed between output and result
        self.put('parent', 3, 'output')
        self.put('topup', 4, 'partial')           # interrupted in a top-up release
        self.put('topup', 5, 'complete')
        result, value, selected = self.resume()
        self.assertEqual([r['uid'] for r in value['records']], [self.rows[1]['uid'], self.rows[4]['uid']])
        self.assertEqual(value['resume_sources'], {self.rows[1]['uid']: str(self.parent),
                                                   self.rows[4]['uid']: str(self.topup)})
        self.assertEqual((value['design'], value['parent_root'], value['parent_manifest_sha256']),
                         ('resume', str(self.parent), self.base['sha256']))
        self.assertEqual((value['planned'], value['executable'], result['planned']), (2, 2, 2))
        # Parent keys survive; the parent's own counts and hash are replaced.
        self.assertEqual((value['model'], value['dose'], value['scope']), ('gemma', 'main', self.base['scope']))
        self.assertNotEqual(value['sha256'], self.base['sha256'])
        self.assertEqual(selected, {'dose': 'main', 'manifest_sha256': value['sha256'], 'canary_job': None,
                                    'sweep': True, 'resume': True, 'sha256': selected['sha256']})
        # Rows keep the parent's records unchanged, so their digests and seeds are the parent's.
        self.assertEqual(value['records'], [self.rows[1], self.rows[4]])

    def test_a_row_with_state_in_two_stores_is_refused(self):
        self.put('parent', 4, 'partial')
        self.put('topup', 4, 'partial')
        with self.assertRaisesRegex(ValueError, 'several stores'):
            self.resume()

    def test_a_partial_whose_result_exists_elsewhere_is_refused(self):
        self.put('parent', 4, 'partial')
        self.put('topup', 4, 'complete')
        with self.assertRaisesRegex(ValueError, 'several stores'):
            self.resume()

    def test_a_partial_of_another_assignment_is_refused(self):
        self.put('parent', 1, 'partial', assignment_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'another assignment'):
            self.resume()

    def test_a_source_of_another_parent_or_without_the_parent_is_refused(self):
        self.put('parent', 1, 'partial')
        publish(whole.manifest_path('gemma', 'main', self.topup.parent/'foreign'),
                {'records': [], 'parent_manifest_sha256': 'someone else'})
        with self.assertRaisesRegex(ValueError, 'another manifest'):
            self.resume(sources=(self.parent, self.topup.parent/'foreign'))
        with self.assertRaisesRegex(ValueError, 'include the parent'):
            self.resume(sources=(self.topup,))

    def test_nothing_to_resume_is_refused(self):
        self.put('parent', 0, 'complete')
        with self.assertRaisesRegex(ValueError, 'no interrupted rows'):
            self.resume()

    def test_prepare_dispatches_a_resume_release_and_never_both(self):
        self.put('parent', 1, 'partial')
        root = Path(self.directory.name)/'release'
        spec = {'models': {'gemma': {'parent': str(self.parent), 'sources': [str(self.parent)]}}}
        publish(root/'RESUME.json', spec)
        publish(root/'RELEASE.json', {'models': ['gemma'], 'sweep': {'gemma': 'main'}})
        result = whole.prepare_model(root, {}, 'gemma')
        self.assertEqual(result['gemma:main']['planned'], 1)
        publish(root/'TOPUP.json', {'models': {}})
        with self.assertRaisesRegex(ValueError, 'either a top-up, a resume or an outcome-steering release'):
            whole.prepare_model(root, {}, 'gemma')


class SavedStateGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        self.store = driver.Store(Path(self.directory.name)/'units')
        self.rows = synthetic_rows()
        self.resume = {'design': 'resume', 'records': self.rows}

    def put(self, index, kind, **changes):
        row = self.rows[index]
        folder = self.store.folder(row['uid'])
        if kind == 'partial':
            driver.checkpoint(folder/'PARTIAL.json', {**partial(row, [1, 2]), **changes})
        elif kind == 'complete':
            publish(folder/'RESULT.json', {'status': 'complete'})
        elif kind == 'output':
            publish(folder/'OUTPUT.json', {'token_ids': [1]})

    def test_fresh_stores_pass_for_every_design(self):
        for value in (self.resume, {'records': self.rows}, {'design': 'top-up', 'records': self.rows}):
            whole.check_saved_state(value, self.rows, self.store)

    def test_resume_continues_its_own_partial_rows_only(self):
        self.put(0, 'partial')
        self.put(1, 'partial')
        whole.check_saved_state(self.resume, self.rows, self.store)

    def test_a_partial_of_any_other_design_is_refused(self):
        self.put(0, 'partial')
        for design in (None, 'top-up', 'efficiency-sweep'):
            with self.subTest(design=design), self.assertRaisesRegex(ValueError, 'fresh only'):
                whole.check_saved_state({'design': design, 'records': self.rows}, self.rows, self.store)

    def test_a_foreign_assignment_is_refused(self):
        self.put(0, 'partial', assignment_sha256='f'*64)
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state(self.resume, self.rows, self.store)

    def test_finished_or_half_finished_rows_are_never_resumed(self):
        # A row this resume release already finished passes: a restarted shard skips it.
        store = driver.Store(Path(self.directory.name)/'complete')
        publish(store.folder(self.rows[2]['uid'])/'RESULT.json', {'status': 'complete'})
        whole.check_saved_state(self.resume, self.rows, store)
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state({'records': self.rows}, self.rows, store)
        store = driver.Store(Path(self.directory.name)/'output')
        publish(store.folder(self.rows[2]['uid'])/'OUTPUT.json', {'status': 'output'})
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state(self.resume, self.rows, store)
        # A PARTIAL that coexists with an OUTPUT is the output's, never a continuation.
        store = driver.Store(Path(self.directory.name)/'both')
        folder = store.folder(self.rows[3]['uid'])
        driver.checkpoint(folder/'PARTIAL.json', partial(self.rows[3], [1]))
        publish(folder/'OUTPUT.json', {'token_ids': [1]})
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state(self.resume, self.rows, store)

    def test_a_row_outside_the_manifest_and_canary_shards_are_refused(self):
        self.put(0, 'partial')
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state({'design': 'resume', 'records': self.rows[1:]}, self.rows[:1], self.store)
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state(self.resume, self.rows, self.store, canary=True)


class SeedResumeTests(unittest.TestCase):
    """Seeding copies partials into the release's own store and never touches the sources."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.parent, self.topup, self.root = base/'parent', base/'topup', base/'release'
        self.rows = synthetic_rows()[:3]
        self.sources = {self.rows[0]['uid']: self.parent, self.rows[1]['uid']: self.parent,
                        self.rows[2]['uid']: self.topup}
        self.states = {
            self.rows[0]['uid']: partial(self.rows[0], [5, 6, 7]),
            # A crash: the last segment was never drained.
            self.rows[1]['uid']: partial(self.rows[1], [8], [{'job': '2', 'history': 0, 'seed': 1, 'max_tokens': 100}]),
            self.rows[2]['uid']: partial(self.rows[2], [], [{'job': '3', 'history': 0, 'seed': 2, 'max_tokens': 100}])}
        for uid, root in self.sources.items():
            driver.checkpoint(root/'stores/gemma/main/units'/uid/'PARTIAL.json', self.states[uid])
        self.value = {'design': 'resume', 'dose': 'main', 'sha256': 'manifest', 'records': self.rows,
                      'resume_sources': {uid: str(root) for uid, root in self.sources.items()}}
        self.store = driver.Store(self.root/'stores/gemma/main/units')

    def source_files(self):
        paths = [root/'stores/gemma/main/units'/uid/'PARTIAL.json' for uid, root in self.sources.items()]
        return {str(p): (file_sha(p), p.stat().st_mtime_ns) for p in paths}

    def seed(self, rows=None, shard=(1, 4)):
        return whole.seed_resume(self.root, 'gemma', self.value, rows or self.rows, self.store, shard)

    def test_partials_are_copied_with_a_receipt_and_sources_stay_untouched(self):
        before = self.source_files()
        receipt = self.seed()
        self.assertEqual(self.source_files(), before)
        for row in self.rows:
            self.assertEqual(self.store.state(row['uid']), {'status': 'partial', **self.states[row['uid']]})
        entries = {e['uid']: e for e in receipt['rows']}
        for uid, root in self.sources.items():
            path = root/'stores/gemma/main/units'/uid/'PARTIAL.json'
            self.assertEqual(entries[uid], {'uid': uid, 'source': str(path), 'source_sha256': file_sha(path),
                                            'history': len(self.states[uid]['tokens']),
                                            'drained': uid == self.rows[0]['uid']})
        self.assertEqual(load(self.root/'gemma/resume/shard-1-4.json'), receipt)
        self.assertEqual((receipt['shard'], receipt['manifest_sha256']), ([1, 4], 'manifest'))

    def test_restart_is_idempotent_and_keeps_progress_a_later_job_made(self):
        first = self.seed()
        uid = self.rows[0]['uid']
        state = self.states[uid]
        advanced = {**state, 'tokens': state['tokens']+[9, 9], 'segments': state['segments']+[
            {'job': '4', 'history': 3, 'seed': 99, 'max_tokens': 97}]}
        driver.checkpoint(self.store.folder(uid)/'PARTIAL.json', advanced)
        self.assertEqual(self.seed(), first)
        self.assertEqual(load(self.store.folder(uid)/'PARTIAL.json'), advanced)

    def test_a_target_that_diverged_from_its_source_is_refused(self):
        self.seed()
        uid = self.rows[0]['uid']
        diverged = {**self.states[uid], 'tokens': [5, 6, 0]}
        driver.checkpoint(self.store.folder(uid)/'PARTIAL.json', diverged)
        with self.assertRaisesRegex(ValueError, 'diverged'):
            self.seed()
        self.assertEqual(load(self.store.folder(uid)/'PARTIAL.json'), diverged)

    def test_a_restart_over_finished_rows_leaves_their_result_and_output_alone(self):
        first = self.seed()
        done, crashed, partial_row = (row['uid'] for row in self.rows)
        publish(self.store.folder(done)/'RESULT.json', {'status': 'complete'})
        publish(self.store.folder(crashed)/'OUTPUT.json', {'token_ids': [1]})
        self.assertEqual(self.seed(), first)
        self.assertEqual(self.store.state(done)['status'], 'complete')
        self.assertEqual(self.store.state(crashed)['status'], 'output')
        self.assertEqual(self.store.state(partial_row), {'status': 'partial', **self.states[partial_row]})

    def test_a_source_of_another_assignment_or_a_non_resume_manifest_is_refused(self):
        row = self.rows[0]
        driver.checkpoint(self.parent/'stores/gemma/main/units'/row['uid']/'PARTIAL.json',
                          {**self.states[row['uid']], 'assignment_sha256': '1'*64})
        with self.assertRaisesRegex(ValueError, 'not this assignment'):
            self.seed(rows=[row])
        with self.assertRaisesRegex(ValueError, 'only a resume release'):
            whole.seed_resume(self.root, 'gemma', {**self.value, 'design': 'top-up'}, self.rows, self.store, (0, 1))


class RunWiringTests(unittest.TestCase):
    """run() gates and seeds before the engine would load (the load itself is made to fail here)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.parent, self.root = base/'parent', base/'release'
        self.rows = synthetic_rows()
        self.base = publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'scope': 'x', 'records': self.rows})
        self.units = self.parent/'stores/gemma/main/units'
        driver.checkpoint(self.units/self.rows[1]['uid']/'PARTIAL.json', partial(self.rows[1], [4, 5, 6]))
        publish(self.root/'RELEASE.json', {'models': ['gemma'], 'sweep': {'gemma': 'main'}})
        publish(self.root/'PREPARED.json', {})
        self.patches = [patch.object(whole, 'verify_release', return_value={'cutoff': time.time()+1e6}),
                        patch.object(whole, 'bound_store', side_effect=self.bound),
                        patch.object(driver, 'binding', return_value={}),
                        patch.object(driver, 'hooked_llm', side_effect=RuntimeError('engine reached'))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def bound(self, root, model, dose, canary=False):
        return driver.Store(root/'stores'/model/dose/'units')

    def test_resume_shard_seeds_its_rows_then_reaches_the_engine(self):
        whole.resume_manifest(self.root, 'gemma', {'parent': str(self.parent), 'sources': [str(self.parent)]})
        with self.assertRaisesRegex(RuntimeError, 'engine reached'):
            whole.run(self.root, 'gemma', (0, 1), 0.)
        uid = self.rows[1]['uid']
        self.assertEqual(self.bound(self.root, 'gemma', 'main').state(uid)['tokens'], [4, 5, 6])
        receipt = load(self.root/'gemma/resume/shard-0-1.json')
        self.assertEqual([(r['uid'], r['history']) for r in receipt['rows']], [(uid, 3)])
        # A restarted shard passes the gate again with the seeded partial in place.
        with self.assertRaisesRegex(RuntimeError, 'engine reached'):
            whole.run(self.root, 'gemma', (0, 1), 0.)

    def test_non_resume_release_refuses_a_partial_before_the_engine(self):
        publish(whole.manifest_path('gemma', 'main', self.root), {**self.base, 'design': 'top-up'})
        publish(self.root/'gemma/SELECTED.json', {'dose': 'main', 'manifest_sha256': load(
            whole.manifest_path('gemma', 'main', self.root))['sha256'], 'canary_job': None, 'sweep': True})
        driver.checkpoint(self.bound(self.root, 'gemma', 'main').folder(self.rows[1]['uid'])/'PARTIAL.json',
                          partial(self.rows[1], [1]))
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.run(self.root, 'gemma', (0, 1), 0.)

    def test_after_the_cutoff_nothing_is_seeded(self):
        whole.resume_manifest(self.root, 'gemma', {'parent': str(self.parent), 'sources': [str(self.parent)]})
        with patch.object(whole, 'verify_release', return_value={'cutoff': 1.}):
            self.assertIsNone(whole.run(self.root, 'gemma', (0, 1), 0.))
        self.assertIsNone(self.bound(self.root, 'gemma', 'main').state(self.rows[1]['uid']))


class ResumeAnalysisJoinTests(unittest.TestCase):
    def test_resume_outcomes_join_the_parent_by_uid_and_the_parent_adds_nothing_for_partials(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            base, resume = Path(temporary)/'base', Path(temporary)/'resume'
            rows = [{'uid': digest([name]), 'family': name, 'model': 'gpt', 'dose': 'half', 'total_allowance': 100,
                     'nomination_status': 'nominated', 'arm': 'native', 'operation': 'native', 'class': None}
                    for name in ('done', 'cut')]
            planned = publish(whole.manifest_path('gpt', 'half', base), {'records': rows})
            follow_up = publish(whole.manifest_path('gpt', 'half', resume), {
                'records': rows[1:], 'parent_manifest_sha256': planned['sha256'], 'design': 'resume'})
            store = driver.Store(base/'stores/gpt/half/units')
            publish(store.folder(rows[0]['uid'])/'RESULT.json', {'status': 'complete'})
            driver.checkpoint(store.folder(rows[1]['uid'])/'PARTIAL.json', partial(rows[1], [1, 2]))
            # The parent's extraction skips the interrupted row: it stays missing there.
            self.assertIsNone(whole_effects._extract_one(rows[1], b={}, model='gpt', tokenizer=None, close=None,
                                                         store=store, inputs={}))
            for root, row, manifest in ((base, rows[0], planned), (resume, rows[1], follow_up)):
                (root/'gpt/extraction').mkdir(parents=True)
                publish(root/'gpt/extraction/EXTRACTION.json', {'manifest_sha256': manifest['sha256']})
                pd.DataFrame([{'uid': row['uid'], 'outcome_status': 'observed', 'reasoning_tokens': 5,
                               'reasoning_end': 5, 'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                    root/'gpt/extraction/outcomes.parquet')
            source = {'model': 'gpt', 'root': str(base), 'dose': 'half', 'topups': [str(resume)]}
            table = whole_effects.load_table([source]).set_index('uid')
            self.assertEqual(set(table.outcome_status), {'observed'})
            self.assertEqual(set(table.index), {row['uid'] for row in rows})
            # Observed in both releases would mean the row ran twice: refused.
            pd.DataFrame([{'uid': rows[0]['uid'], 'outcome_status': 'observed', 'reasoning_tokens': 5,
                           'reasoning_end': 5, 'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                resume/'gpt/extraction/outcomes.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate whole extracted UID'):
                whole_effects.load_table([source])


if __name__ == '__main__':
    unittest.main()
