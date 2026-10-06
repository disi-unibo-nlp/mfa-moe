"""Token timing, information boundary and sparse dynamic pulse tests."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import utility_controller_interface_v1 as boundary
import utility_controller_v2 as controller
from utility_routing_worker_v2 import rows_for_episode, THINK_END_ID


def selections():
    return {transition: {'arm': 'bias', 'slots': [0, 512], 'pulse_width': 256,
                         'action_names': ['bias', 'bias']} for transition in controller.TRANSITION_ORDER}


class ControllerTests(unittest.TestCase):
    def make(self, arm='frozen_policy', accept=True):
        def screen(request):
            return [boundary.SideResult(request.request_id, i, 'stop',
                    '{"start":true}' if accept else '{"start":false}', 100, 5, .2) for i in (0, 1)]
        return controller.TokenController(uid='test', arm=arm, problem='Find x', prompt_sha256='a' * 64,
            selections=selections(), decode=lambda ids: ''.join(map(chr, ids)), screen=screen)

    def feed(self, control, text):
        episodes = []
        for token in map(ord, text):
            episode = control.observe_cumulative((*control.state.completion_token_ids, token))
            if episode:
                episodes.append(episode)
        return episodes

    def test_one_accepted_episode_only_and_no_lookahead(self):
        c = self.make()
        episodes = self.feed(c, 'The answer is 3. The answer is 4. ')
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(c.queries), 1)
        emitted = c.queries[0]['request']['reader_input']['emitted_prefix']
        self.assertNotIn('4', emitted)
        self.assertEqual(episodes[0]['absolute_slots'], [len(emitted), len(emitted) + 512])
        self.assertEqual(c.audit()['side_readers'], 2)

    def test_native_shadow_never_calls_semantic_reader_or_edits_routing(self):
        c = self.make('native')
        c.screen = lambda _: self.fail('native called semantic reader')
        self.assertEqual(self.feed(c, 'The answer is 3. '), [])
        self.assertEqual(c.queries, [])
        self.assertTrue(c.proposals)

    def test_truncation_closure_and_coarse_chunks(self):
        c = self.make()
        self.assertEqual(self.feed(c, 'The answer is 3 +'), [])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            c.observe_cumulative((*c.state.completion_token_ids, ord('1'), ord('.')))
        c = self.make()
        self.assertEqual(self.feed(c, '</think>The answer is 3. '), [])
        c = self.make(accept=False)
        self.feed(c, 'The answer is 3. The answer is 4. ')
        self.assertIsNone(c.episode)
        self.assertEqual(len(c.queries), 2)

    def test_row_boundaries_recompute_and_closure(self):
        class Table:
            @staticmethod
            def index_of(name): return 1
        episode = {'absolute_slots': [7, 519], 'action_names': ['bias', 'bias']}
        positions = [6, 7, 262, 263, 518, 519, 774, 775]
        active, _ = rows_for_episode(episode, Table(), positions, [])
        self.assertEqual(active.tolist(), [False, True, True, False, False, True, True, False])
        # Recomputing an earlier active token uses the historical pulse. Rows
        # predicting tokens after emitted closure return to native routing.
        active, _ = rows_for_episode(episode, Table(), [7, 8, 9, 519], [0] * 8 + [THINK_END_ID])
        self.assertEqual(active.tolist(), [True, True, False, False])

    def test_native_stream_buffers_split_utf8_and_repeated_reads(self):
        from tokenizers import Tokenizer, models, decoders
        tokenizer = Tokenizer(models.WordLevel({'<0xE2>': 0, '<0x82>': 1, '<0xAC>': 2, '.': 3}, unk_token='.'))
        tokenizer.decoder = decoders.ByteFallback()
        self.assertEqual(tokenizer.decode([0]), '\ufffd')
        stream = controller.NativeDecodeStream(tokenizer)
        self.assertEqual(stream([0]), '')
        self.assertEqual(stream([0, 1]), '')
        self.assertEqual(stream([0, 1, 2]), '\u20ac')
        self.assertEqual(stream([0, 1, 2]), '\u20ac')
        self.assertEqual(stream([0, 1, 2, 3]), '\u20ac.')
        with self.assertRaisesRegex(ValueError, 'history changed'):
            stream([1, 0])


if __name__ == '__main__':
    unittest.main()
