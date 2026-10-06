"""One-episode, token-level, prefix-only original-prompt utility controller.

Every newly emitted token is observed before the next generator forward pass.
All proposals from the discovery-frozen lexical detector receive the existing
two-reader starting-condition screen until one is accepted. There is no hidden
query cap, no transition re-entry, no injection and no overlap of episodes.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence

from moe_exp.routing_control.live_gate_v2 import LivePrefixGate
import utility_controller_interface_v1 as boundary

TRANSITION_ORDER = ('candidate_to_verify', 'approach_to_commit')
VERSION = 'utility-token-controller-v2-first-accepted-episode'


class NativeDecodeStream:
    """Append stable UTF-8 chunks, buffering incomplete native byte tokens.

    Repeated reads of the same emitted IDs return cached text for the side
    request. Every token remains in the history even when no text is ready.
    """
    def __init__(self, tokenizer):
        from tokenizers.decoders import DecodeStream
        self.tokenizer = getattr(tokenizer, 'backend_tokenizer', tokenizer)
        self.decoder = DecodeStream(skip_special_tokens=False)
        self.ids, self.text = (), ''

    def __call__(self, token_ids):
        ids = tuple(token_ids)
        if ids[:len(self.ids)] != self.ids or len(ids) < len(self.ids):
            raise ValueError('native decoder token history changed')
        for token in ids[len(self.ids):]:
            chunk = self.decoder.step(self.tokenizer, token)
            if chunk is not None:
                self.text += chunk
        self.ids = ids
        return self.text


class TokenController:
    def __init__(self, *, uid: str, arm: str, problem: str, prompt_sha256: str,
                 selections: dict, decode: Callable[[Sequence[int]], str],
                 screen: Callable[[boundary.SideRequest], Sequence[boundary.SideResult]]):
        if (arm not in ('native', 'frozen_policy') or not selections or
                not set(selections) <= set(TRANSITION_ORDER)):
            raise ValueError('unknown utility arm or selected transition set')
        self.state = boundary.RequestState(uid, problem, prompt_sha256)
        self.arm, self.selections, self.decode, self.screen = arm, selections, decode, screen
        # This is not a newly fitted cheap classifier: every proposal of the
        # frozen v2.4 lexical detector proceeds to the frozen two-reader rubric.
        self.gate = LivePrefixGate(tuple(t for t in TRANSITION_ORDER if t in selections), screen=lambda _: True)
        self.episode = None
        self.queries, self.proposals = [], []
        self.next_engine_step = 0

    def observe_cumulative(self, emitted_ids: Sequence[int], finish_reason=None):
        ids = tuple(emitted_ids)
        old = self.state.completion_token_ids
        if len(ids) != len(old) + 1 or ids[:-1] != old:
            raise ValueError('generator must expose exactly one new token before every observation')
        if finish_reason not in (None, 'stop', 'length', 'error'):
            raise ValueError('unrecognized generator stop reason')
        self.state = self.state.append(ids[-1:], source='emitted')
        self.next_engine_step += 1
        if finish_reason is not None:
            self.state = self.state.close(finish_reason)
            return None
        if not self.state.remaining_tokens:
            raise ValueError('sampler failed to signal the 16384-token length stop')
        if self.episode is not None:
            return None
        opportunities = self.gate.observe_tokens(self.state.problem, ids, self.decode)
        opportunities.sort(key=lambda q: TRANSITION_ORDER.index(q.transition))
        for query in opportunities:
            self.proposals.append({'transition': query.transition,
                'emitted_tokens': len(ids), 'emitted_ids_sha256': boundary.digest(ids),
                'triggering_sentence': query.triggering_sentence,
                'shadow_only': self.arm == 'native'})
            if self.arm == 'native':
                continue
            request = boundary.make_side_request(self.state, transition=query.transition,
                        triggering_sentence=query.triggering_sentence, decode=self.decode)
            results = list(self.screen(request))
            accepted = boundary.two_reader_accept(request, results)
            self.queries.append({'request': asdict(request),
                'results': [asdict(r) for r in results], 'accepted': accepted,
                'decision': 'accepted' if accepted else 'abstain_or_veto'})
            if accepted:
                selected = self.selections[query.transition]
                if selected['slots'] not in ([0], [0, 512]) or selected['pulse_width'] != 256:
                    raise ValueError('unqualified selected pulse geometry')
                self.episode = {'transition': query.transition, 'selected_arm': selected['arm'],
                    'trigger_request_id': request.request_id, 'trigger_emitted_tokens': len(ids),
                    'trigger_emitted_ids_sha256': boundary.digest(ids),
                    'absolute_slots': [len(ids) + slot for slot in selected['slots']],
                    'relative_slots': selected['slots'], 'action_names': selected['action_names'],
                    'pulse_width': 256, 'stop_at_reasoning_closure': True}
                return self.episode
        return None

    def audit(self):
        return {'controller_version': VERSION, 'state': asdict(self.state),
                'episode': self.episode, 'proposals': self.proposals, 'queries': self.queries,
                'observed_generator_tokens': self.next_engine_step,
                'side_readers': sum(len(q['results']) for q in self.queries),
                'side_prompt_tokens': sum(r['prompt_tokens'] for q in self.queries for r in q['results']),
                'side_generated_tokens': sum(r['generated_tokens'] for q in self.queries for r in q['results']),
                'side_elapsed_seconds': sum(r['elapsed_seconds'] for q in self.queries for r in q['results'])}


def hard_side_work_bound(assignments=192):
    """Explicit loose hard bound; no empirical mean is called a stage ceiling.

    There are at most two supported proposals per emitted token. Two readers
    each receive at most 49152 context+output tokens, of which at most 1024 are
    generated. A lower practical query budget would change the policy and must
    be explicitly frozen before a price pilot or production run.
    """
    if type(assignments) is not int or assignments < 1:
        raise ValueError('positive policy-assignment count required')
    queries = assignments * boundary.MAX_OUTPUT_TOKENS * len(TRANSITION_ORDER)
    return {'policy_assignments': assignments, 'maximum_side_queries': queries,
            'maximum_side_ratings': 2 * queries,
            'maximum_side_decode_tokens': 2 * queries * 1024,
            'maximum_side_prefill_tokens': 2 * queries * (49152 - 1024),
            'query_cap': None, 'practical_limit': 'None; every lexical opportunity before first acceptance is screened.'}
