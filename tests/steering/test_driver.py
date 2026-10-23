import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from steering import pulse as driver
from store import WORK, verified
from steering.stream import Clock


class Selection(unittest.TestCase):
    def test_one_layer_by_association_then_stability_then_layer(self):
        noms = [{'status': 'nominated', 'fold': 0, 'class': 'Read', 'depth_third': d, 'layer': l,
                 'association': a, 'stability': s}
                for d, l, a, s in ((0, 3, .2, .5), (1, 9, .3, .1), (2, 15, .3, .4))]
        noms.append({'status': 'insufficient_support', 'fold': 0, 'class': 'Plan', 'depth_third': 0, 'layer': None,
                     'association': None, 'stability': None})
        chosen, evidence = driver.select_layers(noms)
        self.assertEqual(chosen, {(0, 'Read'): {'depth_third': 2, 'layer': 15}})
        self.assertEqual(len(evidence), 3)

    def test_published_manifests_match_reduced_design(self):
        total = 0
        for model in ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b'):
            value = verified(driver.DRIVER/model/'ASSIGNMENTS.json')
            rows = value['records']
            self.assertFalse(value['outcomes_used'])
            self.assertEqual(len(value['selection']), 35)
            self.assertTrue(all(r['dose'] == 'central' for r in rows))
            layers = {(s['fold'], s['class']): s['layer'] for s in value['selection']}
            for r in rows:
                if r['arm'] == 'native':
                    self.assertIsNone(r['action'])
                    self.assertEqual(r['capture_layers'], sorted({l for (f, _), l in layers.items() if f == r['fold']}))
                else:
                    self.assertEqual(r['action']['layer'], layers[(r['fold'], r['class'])])
            self.assertEqual(len({r['uid'] for r in rows}), len(rows))
            total += len(rows)
        self.assertEqual(total, 26129)

    def test_shards_partition_families(self):
        rows = verified(driver.DRIVER/'gpt'/'ASSIGNMENTS.json')['records']
        parts = [driver.shard_rows('gpt', (i, 3)) for i in range(3)]
        self.assertEqual(sum(map(len, parts)), len(rows))
        fams = [{r['family'] for r in p} for p in parts]
        self.assertFalse(fams[0] & fams[1] or fams[1] & fams[2] or fams[0] & fams[2])


class ClockTest(unittest.TestCase):
    def test_recompute_replays_same_history_and_closure(self):
        c = Clock([1, 2, 3], [4, 5], [[9, 9]])
        a = c.observe([1, 2, 3, 4], [0, 1, 2, 3])
        self.assertEqual([x['output_position'] for x in a], [-2, -1, 0, 1])
        c.observe([5], [4])
        c.observe([7], [5])
        b = c.observe([9], [6])
        self.assertTrue(b[0]['reasoning'])
        b = c.observe([9], [7])
        self.assertFalse(b[0]['reasoning'])
        # preemption/resume: same tokens from position 0
        r = c.observe([1, 2, 3, 4, 5, 7, 9, 9], list(range(8)))
        self.assertEqual(c.passes, 2)
        self.assertFalse(r[-1]['reasoning'])
        self.assertTrue(r[5]['reasoning'])
        with self.assertRaises(Exception):
            c.observe([1, 2, 3, 4, 5, 8], list(range(6)))

    def test_gap_and_frozen_input_are_rejected(self):
        c = Clock([1, 2], [3], [[9]])
        with self.assertRaises(Exception):
            c.observe([1, 2], [0, 2])
        c = Clock([1, 2], [3], [[9]])
        with self.assertRaises(Exception):
            c.observe([1, 2, 4], [0, 1, 2])


class FakeEngine:
    def __init__(self, lengths):
        self.lengths, self.live, self.n = lengths, {}, 0

    def has_unfinished_requests(self):
        return bool(self.live)

    def add_request(self, uid, prompt, params):
        self.n += 1
        self.live[uid] = {'params': params, 'done': 0, 'start': len(prompt['prompt_token_ids'])}
        return uid+'-x%d' % self.n

    def step(self):
        out = []
        for uid, st in list(self.live.items()):
            k = min(3, st['params'].max_tokens-st['done'])
            position = st['start']+st['done']
            tokens = [position+i for i in range(k)]
            st['done'] += k
            eos = position+k >= self.lengths[uid]
            finished = eos or st['done'] >= st['params'].max_tokens
            out.append(SimpleNamespace(request_id=uid, finished=finished, outputs=[SimpleNamespace(
                token_ids=tokens, logprobs=None, finish_reason='stop' if eos else ('length' if finished else None),
                stop_reason=None)]))
            if finished:
                del self.live[uid]
        time.sleep(.01)
        return out

    def abort_request(self, ids, internal=False):
        for uid in ids:
            self.live.pop(uid, None)


class EngineTest(unittest.TestCase):
    def test_cleanup_error_preserves_the_original_engine_error(self):
        uid = '%064x' % 414

        class Broken(driver.Engine):
            def step(self, finish):
                raise ValueError('original capture disagreement')

            def drain(self):
                raise RuntimeError('secondary cleanup failure')

        def build(row, history, segment):
            return {'prompt_ids': [1], 'params': SimpleNamespace(seed=1, max_tokens=10), 'total': 10}

        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            engine = Broken(SimpleNamespace(llm_engine=FakeEngine({uid: 10})), driver.Store(Path(temporary)),
                            time.time()+30, hooked=False, max_inflight=2, drain_seconds=0)
            with patch.object(driver, 'log') as log:
                with self.assertRaisesRegex(ValueError, 'original capture disagreement'):
                    engine.run([{'uid': uid}], build, lambda *args: None)
            self.assertEqual(log.call_args.args[0]['error'], 'secondary cleanup failure')

    def test_rank_disagreement_is_retained_and_drain_does_not_release_twice(self):
        from store import digest
        uid = '%064x' % 413
        calls = []
        record = {'positions': [0], 'native_ids': [[1, 2]], 'executed_ids': [[1, 2]],
                  'native_weights': [[.6, .4]], 'executed_weights': [[.6, .4]], 'active': [False]}
        reports = []
        for value in (.5, .50000006):
            capture = {'capture': {'0': record}, 'stats': {'l1': value},
                       'closed': False, 'passes': 1, 'observed_tokens': 1}
            reports.append({uid: {**capture, 'sha256': digest(capture)}})

        def rpc(method, *, kwargs):
            self.assertEqual(method, 'fs_release')
            calls.append(kwargs['uids'])
            self.assertEqual(len(calls), 1)
            return reports

        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            store = driver.Store(Path(tmp))
            fake = FakeEngine({uid: 100})
            engine = driver.Engine(SimpleNamespace(llm_engine=fake, collective_rpc=rpc), store,
                                   time.time()+30, hooked=True, max_inflight=2)
            engine.inflight[uid] = {'progress': {'tokens': [1], 'segments': [{}]}, 'new': 1}
            with patch('steering.whole.work_remaining_bytes', return_value=100*2**30):
                with self.assertRaisesRegex(ValueError, 'tensor-parallel captures disagree'):
                    engine.release([uid])
            for rank, expected in enumerate((.5, .50000006)):
                folder = store.folder(uid)/'TP_DISAGREEMENT'/str(rank)
                saved = verified(folder/'CAPTURE.json')
                self.assertEqual(saved['stats']['l1'], expected)
                self.assertTrue((folder/'AUDIT.npz').is_file())
                self.assertEqual(saved['routing_capture_sha256'], reports[rank][uid]['sha256'])
            engine.drain()
            self.assertEqual(calls, [[uid]])
            self.assertEqual(store.state(uid)['tokens'], [1])
            self.assertFalse(engine.inflight)

    def test_drain_and_resume_continue_the_same_trajectory(self):
        uids = [('%064x' % i) for i in range(5)]
        lengths = {u: 40+30*i for i, u in enumerate(uids)}
        rows = [{'uid': u} for u in uids]
        fake = FakeEngine(lengths)
        llm = SimpleNamespace(llm_engine=fake)

        def build(row, history, segment):
            prompt = [0]*10
            left = 400-len(history)
            return {'prompt_ids': prompt+history, 'params': SimpleNamespace(seed=segment, max_tokens=left),
                    'request': None, 'total': 400}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            store = driver.Store(Path(tmp))
            done = {}

            def finish(row, progress, reason, stop, capture):
                done[row['uid']] = (list(progress['tokens']), reason, len(progress['segments']))
            e = driver.Engine(llm, store, time.time()+0.25, hooked=False, max_inflight=2, drain_seconds=0)
            e.run(rows, build, finish)
            self.assertTrue(e.metrics['drained'] >= 1)
            e = driver.Engine(llm, store, time.time()+60, hooked=False, max_inflight=2, drain_seconds=0)
            e.run(rows, build, finish)
            self.assertEqual(set(done), set(uids))
            for u in uids:
                tokens, reason, segments = done[u]
                # tokens are absolute positions after the 10-token prompt: no gaps, no repeats
                self.assertEqual(tokens, list(range(10, 10+len(tokens))))
                self.assertEqual(reason, 'stop')
                self.assertGreaterEqual(10+len(tokens), lengths[u])
            self.assertTrue(any(s > 1 for _, _, s in done.values()))


if __name__ == '__main__':
    unittest.main()


class ReviewFixes(unittest.TestCase):
    """Regressions from the 2026-10-17 drain/resume review (P1 seed, P2 drain order, P3 crash recovery)."""

    def _engine(self, store, fake, deadline):
        return driver.Engine(SimpleNamespace(llm_engine=fake), store, deadline, hooked=False, max_inflight=2,
                             drain_seconds=0)

    def test_zero_token_drain_keeps_seed_index(self):
        seen = []

        def build(row, history, segment):
            seen.append((len(history), segment))
            return {'prompt_ids': [0]*4+history, 'params': SimpleNamespace(seed=segment, max_tokens=50),
                    'request': None, 'total': 50}
        uid = '%064x' % 7
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            store = driver.Store(Path(tmp))
            folder = store.folder(uid)
            driver.checkpoint(folder/'PARTIAL.json', {'uid': uid, 'tokens': [], 'segments': [
                {'history': 0}, {'history': 0}], 'total_allowance': 50})
            e = self._engine(store, FakeEngine({uid: 20}), time.time()+30)
            e.admit({'uid': uid}, build)
            driver.checkpoint(folder/'PARTIAL.json', {'uid': uid, 'tokens': [1, 2, 3], 'segments': [
                {'history': 0}, {'history': 0}], 'total_allowance': 50})
            e2 = self._engine(store, FakeEngine({uid: 20}), time.time()+30)
            e2.admit({'uid': uid}, build)
        self.assertEqual(seen, [(0, 0), (3, 1)])

    def test_drain_persists_tokens_before_release_can_fail(self):
        uid = '%064x' % 8

        def build(row, history, segment):
            return {'prompt_ids': [0]*4+history, 'params': SimpleNamespace(seed=0, max_tokens=500),
                    'request': None, 'total': 500}

        class Failing(driver.Engine):
            def release(self, uids):
                raise RuntimeError('release failed')
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            store = driver.Store(Path(tmp))
            e = Failing(SimpleNamespace(llm_engine=FakeEngine({uid: 10**6})), store, time.time()+30,
                        hooked=False, max_inflight=2, drain_seconds=0)
            e.admit({'uid': uid}, build)
            for _ in range(3):
                e.step(lambda *a: None)
            with self.assertRaises(RuntimeError):
                e.drain()
            self.assertEqual(len(store.state(uid)['tokens']), 9)

    def test_output_without_result_is_finalized_not_readmitted(self):
        from store import publish
        uid = '%064x' % 9
        row = {'uid': uid}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            store = driver.Store(Path(tmp))
            folder = store.folder(uid)
            driver.checkpoint(folder/'PARTIAL.json', {'uid': uid, 'tokens': [1], 'segments': [{'history': 0}],
                                                      'total_allowance': 9})
            publish(folder/'OUTPUT.json', {'uid': uid, 'token_ids': [1, 2], 'finish_reason': 'stop',
                                           'segments': [{'history': 0}]})
            self.assertEqual(store.state(uid)['status'], 'output')
            self.assertEqual(driver.recover_outputs(store, [row]), 1)
            self.assertEqual(store.state(uid)['status'], 'unknown')
            self.assertTrue(store.state(uid)['recovered_after_crash'])


class SubShards(unittest.TestCase):
    def test_sub_shards_partition_one_shard(self):
        whole = {r['uid'] for r in driver.shard_rows('glm', (2, 4))}
        parts = [{r['uid'] for r in driver.shard_rows('glm', (2, 4), sub=(j, 3))} for j in range(3)]
        self.assertEqual(set().union(*parts), whole)
        self.assertEqual(sum(map(len, parts)), len(whole))

    def test_nested_sub_shards_partition_a_sub_shard(self):
        whole = {r['uid'] for r in driver.shard_rows('glm', (2, 4), sub=[(2, 3)])}
        parts = [{r['uid'] for r in driver.shard_rows('glm', (2, 4), sub=[(2, 3), (j, 3)])} for j in range(3)]
        self.assertEqual(set().union(*parts), whole)
        self.assertEqual(sum(map(len, parts)), len(whole))
        self.assertEqual({r['uid'] for r in driver.shard_rows('glm', (2, 4), sub=(2, 3))}, whole)
