"""Versioned arm/seed adapter over the unchanged qualified four-GPU engine."""
from dataclasses import replace
import time

import numpy as np

from utility_controller_v2 import TokenController, NativeDecodeStream
from utility_pair_backend_v2 import FourGPUBackend
from utility_pair_backend_v2 import _side_model as qualified_side_model
import operator_panel_v1 as P


def paired_reader_seed(request_id, reader, native_seed):
    return native_seed(request_id.split('~', 1)[0], reader)


def panel_side_model(pipe, devices, output):
    # Physical reader IDs stay unique, while their sampler seeds use the common
    # prefix key. Only this invocation's module function is adapted; source and
    # the frozen reader prompt/parser/model/sampler parameters are unchanged.
    import rate_transition_v22_fullprefix_starts_v2 as reader_module
    original = reader_module.rating_seed
    def seed(uid, reader):
        from pathlib import Path
        value = paired_reader_seed(uid, reader, original)
        P.save(Path(output) / f'PANEL_READER_SEED_{uid}_{reader}.json', {
            'schema': 'panel-reader-seed-v1', 'physical_request_id': uid,
            'seed_request_id': uid.split('~', 1)[0], 'reader': reader, 'sampler_seed': value})
        return value
    reader_module.rating_seed = seed
    qualified_side_model(pipe, devices, output)


class PanelController(TokenController):
    def __init__(self, assignment, problem, policies, decode, screen):
        self.assignment_uid = assignment['uid']
        self.screening_uid = assignment['screening_uid']
        self.operator = assignment['arm']
        self.actual_screen = screen
        self.physical_screening_ids = []
        selected = policies['bias' if self.operator == 'native' else self.operator]
        super().__init__(uid=self.assignment_uid,
            arm='native' if self.operator == 'native' else 'frozen_policy', problem=problem,
            prompt_sha256=assignment['prompt_token_ids_sha256'], selections=selected, decode=decode,
            screen=self.paired_screen)

    def paired_screen(self, request):
        physical_id = request.request_id + '~' + P.U.digest(self.assignment_uid)[:16]
        physical = replace(request, request_id=physical_id, assignment_uid=self.assignment_uid)
        self.physical_screening_ids.append({'seed_request_id': request.request_id, 'physical_request_id': physical_id})
        return [replace(result, request_id=request.request_id) for result in self.actual_screen(physical)]

    def observe_cumulative(self, emitted_ids, finish_reason=None):
        # Only the reader-request seed key is common. Worker/sampler UIDs remain
        # distinct canonical assignments. Restore identity even on an error.
        self.state = replace(self.state, uid=self.screening_uid)
        try:
            return super().observe_cumulative(emitted_ids, finish_reason)
        finally:
            self.state = replace(self.state, uid=self.assignment_uid)

    def audit(self):
        return {**super().audit(), 'panel_operator': self.operator,
                'screening_uid': self.screening_uid, 'physical_screening_ids': self.physical_screening_ids,
                'adapter_version': 'operator-panel-controller-v1'}


class PanelBackend(FourGPUBackend):
    def __init__(self, policies, output, *, deadline_epoch):
        # The qualified constructor registers all actions in one table. This
        # shared table and engine profile stay fixed across every scientific arm.
        self.policies = policies
        combined = {t: {**policies['bias'][t], 'actions':
                    [a for op in P.ARMS[1:] for a in policies[op][t]['actions']]}
                    for t in P.TRANSITIONS}
        # The qualified constructor spawns its side target from a module global.
        # Bind the separately versioned target for this synchronous constructor,
        # then restore it immediately. No source file or engine is modified.
        import utility_pair_backend_v2 as qualified
        original_target = qualified._side_model
        qualified._side_model = panel_side_model
        try:
            super().__init__(combined, output, deadline_epoch=deadline_epoch)
        finally:
            qualified._side_model = original_target

    def generate_one(self, assignment, prompt_ids, problem, *, max_tokens=16384, controller=None,
                     qualification=False, qualification_preempt_at=None, qualification_token_ids=None):
        from vllm import SamplingParams
        P.require(P.U.digest(prompt_ids) == assignment['prompt_token_ids_sha256'] and
                  len(prompt_ids) == assignment['prompt_tokens'] and
                  (qualification or (max_tokens == 16384 and controller is None and
                      qualification_preempt_at is None and qualification_token_ids is None)),
                  'panel sampler or original prompt differs')
        ctl = controller or PanelController(assignment, problem, self.policies,
                                            NativeDecodeStream(self.tokenizer), self.screen)
        extra = {'steer': {'uid': assignment['uid'], 'policy_index': 0, 'prefix_len': 0}}
        if qualification_token_ids is not None:
            P.require(qualification and len(qualification_token_ids) == max_tokens, 'invalid engineering sequence')
            extra['utility_engineering_token_sequence'] = qualification_token_ids
        params = SamplingParams(**self.engine_module.card_sampling(max_tokens, assignment['seed'],
            extra_args=extra, routed_experts_prompt_start=len(prompt_ids) - 1))
        engine = self.model.llm_engine
        rid = engine.add_request(assignment['uid'], {'prompt_token_ids': prompt_ids}, params)
        started = time.monotonic(); routes = None; preemptions = []
        try:
            while engine.has_unfinished_requests():
                if time.time() >= self.deadline_epoch:
                    raise TimeoutError('panel generator allocation deadline')
                for output in engine.step():
                    P.require(output.request_id == assignment['uid'] and len(output.outputs) == 1,
                              'unexpected generator neighbor')
                    completion = output.outputs[0]; ids = list(completion.token_ids)
                    episode = ctl.observe_cumulative(ids, completion.finish_reason if output.finished else None)
                    if episode:
                        acknowledgements = self.model.collective_rpc('utility_set_episode',
                            args=(assignment['uid'], episode, ids))
                        P.require(len(acknowledgements) == 2 and {r['rank'] for r in acknowledgements} == {0, 1} and
                                  all(r['episode_sha256'] == P.U.digest(episode) for r in acknowledgements),
                                  'episode acknowledgement differs')
                    if output.finished:
                        routes = np.asarray(completion.routed_experts)
                    elif qualification_preempt_at is not None and len(ids) == qualification_preempt_at:
                        reset = self.model.reset_prefix_cache(reset_running_requests=True)
                        preemptions.append({'emitted_tokens': len(ids), 'reset_ok': bool(reset)})
            self.model.collective_rpc('steer_flush')
            P.require(ctl.state.finish in ('stop', 'length') and routes is not None and
                      routes.shape == (len(ctl.state.completion_token_ids), 40, 8),
                      'generator stop or route coverage differs')
            return {'assignment': assignment, 'controller': ctl.audit(), 'routed': routes,
                    'qualification_only': qualification, 'maximum_tokens': max_tokens,
                    'forced_preemption_receipts': preemptions,
                    'generation_wall_seconds': time.monotonic() - started}
        except BaseException as error:
            error.utility_partial = {'assignment': assignment, 'controller': ctl.audit(),
                'generation_wall_seconds': time.monotonic() - started,
                'forced_preemption_receipts': preemptions}
            try:
                engine.abort_request([rid], internal=True)
            except BaseException:
                pass
            raise
