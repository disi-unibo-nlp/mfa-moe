"""Execution-local, synchronous all-layer capture over the frozen routing kernels.

Only this worker process is patched. Historical workers and installed vLLM stay
unchanged. One acknowledged command selects the next prediction row; no command
can select a row belonging to another request or a different cursor.
"""
from __future__ import annotations

from types import MethodType
import numpy as np

VERSION = 'routing-local-worker-v2.1'
THINK_END_ID = 248069


def prediction_coordinates(engine_row, original_prompt_tokens, prefix_tokens):
    """Return reasoning position and continuation offset for an input row."""
    position = engine_row - original_prompt_tokens + 1
    return position, position - prefix_tokens


def validate_command(command, previous, table):
    from moe_exp.routing_control.study_v1 import require
    require(command['uid'] and type(command['offset']) is int and command['offset'] >= 0,
            'invalid request cursor')
    require(command['position'] == command['prefix_tokens'] + command['offset'] and
            command['original_prompt_tokens'] > 0, 'prediction coordinates differ')
    require((command['policy'] == 'native') == (command['action'] is None), 'policy/action mismatch')
    require(command['arm'] != 'native' or command['action'] is None, 'native assignment cannot intervene')
    if command['action'] is not None:
        index = table.index_of(command['policy'])
        policy = table.policies[index]
        action = command['action']
        require(action['name'] == policy.name and action['kind'] == policy.operator.kind and
                action['sign'] == policy.operator.sign and action['magnitude'] == policy.operator.magnitude and
                tuple((l, tuple(sorted(ids))) for l, ids in action['experts']) == policy.targets.experts,
                'action differs from frozen policy table')
    if previous is not None:
        require(command['uid'] == previous['uid'] and command['offset'] == previous['offset'] + 1 and
                all(command[k] == previous[k] for k in
                    ('arm', 'original_prompt_tokens', 'prefix_tokens', 'input_sha256')),
                'stale request or skipped prediction cursor')


def install():
    from moe_steer import vllm_ext as E
    if getattr(E.install_steering, 'local_version', None) == VERSION:
        return E
    original_install, original_accumulate = E.install_steering, E._accumulate
    # Compile the existing sparse operators against all 40 layers. Untargeted
    # layers are exact zero edits and use the same native routing implementation.
    E.telemetry_layers = lambda table: tuple(range(40))
    captures = {}

    def accumulate(tele, slot, h, act, pidx, member, nw, ni, aw, ai):
        original_accumulate(tele, slot, h, act, pidx, member, nw, ni, aw, ai)
        ctl = captures.get(id(tele))
        if ctl is None or ctl.local_capture_row is None:
            return
        row = ctl.local_capture_row
        # Eager execution only; one prediction row, even on a chunked prefill.
        ctl.local_ids[0, h].copy_(ni[row])
        ctl.local_ids[1, h].copy_(ai[row])
        ctl.local_weights[0, h].copy_(nw[row].float())
        ctl.local_weights[1, h].copy_(aw[row].float())
        ctl.local_seen[h] += 1

    E._accumulate = accumulate

    def installed(*args, **kwargs):
        ctl = original_install(*args, **kwargs)
        import torch
        with torch.inference_mode(False):
            ctl.local_ids = torch.empty((2, 40, 8), dtype=torch.int64, device=ctl.tele.device)
            ctl.local_weights = torch.empty((2, 40, 8), dtype=torch.float32, device=ctl.tele.device)
        ctl.local_command = ctl.local_previous = None
        ctl.local_capture_row = ctl.local_engine_row = None
        ctl.local_seen = np.zeros(40, np.int64)
        ctl.local_processed = 0
        captures[id(ctl.tele)] = ctl
        new_state, plan = ctl._new_state, ctl._plan_rows

        def new(self, rid, request, start):
            state = new_state(rid, request, start)
            if state.policy_index != 0 or state.fsm is not None:
                raise ValueError('local request must begin as native without a worker FSM')
            state.needs_slot = True
            return state

        def planned(self, state, request, a, b, start, mask, indices, slot):
            plan(state, request, a, b, start, mask, indices, slot)
            command = self.local_command
            if command is None or state.uid != command['uid']:
                raise ValueError('forward without an acknowledged isolated command')
            prompt_len = command['original_prompt_tokens'] + command['prefix_tokens']
            if state.prompt_len != prompt_len or state.prefix_len != command['prefix_tokens']:
                raise ValueError('worker prompt/prefix coordinates differ')
            outputs = list(request.output_token_ids)
            if len(outputs) != command['offset'] or (outputs and outputs[-1] != command['last_token_id']):
                raise ValueError('worker emitted history differs from controller cursor')
            if THINK_END_ID in outputs and command['policy'] != 'native':
                raise ValueError('intervention after reasoning closure')
            target = prompt_len - 1 + command['offset']
            self.local_capture_row = None
            self.local_processed += b - a
            if start <= target < start + b - a:
                row = a + target - start
                position, offset = prediction_coordinates(target, command['original_prompt_tokens'], command['prefix_tokens'])
                if (position, offset) != (command['position'], command['offset']):
                    raise ValueError('worker prediction mapping differs')
                self.local_capture_row, self.local_engine_row = row, target
                self.local_seen.fill(0)
                if command['policy'] != 'native':
                    mask[row] = True
                    indices[row] = self.table.index_of(command['policy'])

        ctl._new_state = MethodType(new, ctl)
        ctl._plan_rows = MethodType(planned, ctl)
        return ctl

    installed.local_version = VERSION
    E.install_steering = installed
    return E


def worker_extension_class():
    E = install()

    class LocalWorkerExtension(E.SteerWorkerExtension):
        def routing_local_command(self, command):
            ctl = self.model_runner.steer_ctl
            with ctl.lock:
                validate_command(command, ctl.local_previous, ctl.table)
                if ctl.local_command is not None:
                    raise ValueError('previous command has not been consumed')
                ctl.local_command = dict(command)
                ctl.local_capture_row = None
                ctl.local_processed = 0
                ctl.local_seen.fill(0)
                return {'rank': ctl.rank, 'uid': command['uid'], 'position': command['position'], 'version': VERSION}

        def routing_local_capture(self, uid, position):
            ctl = self.model_runner.steer_ctl
            with ctl.lock:
                command = ctl.local_command
                if command is None or (command['uid'], command['position']) != (uid, position):
                    raise ValueError('stale capture request')
                if not np.all(ctl.local_seen == 1):
                    raise ValueError('missing or duplicated all-layer capture: ' + repr(ctl.local_seen))
                with E._sync_allowed():
                    ids = ctl.local_ids.cpu().numpy().astype(np.uint16)
                    weights = ctl.local_weights.cpu().numpy()
                ctl.local_previous, ctl.local_command = command, None
                return {'uid': uid, 'position': position, 'rank': ctl.rank, 'engine_input_row': ctl.local_engine_row,
                        'ids': ids, 'weights': weights, 'processed_input_tokens': ctl.local_processed}

        def routing_local_release(self, uid):
            ctl = self.model_runner.steer_ctl
            with ctl.lock:
                active = ctl.local_command or ctl.local_previous
                if active is not None and active['uid'] != uid:
                    raise ValueError('release belongs to another assignment')
                ctl.local_previous = ctl.local_command = None
                ctl.local_capture_row = None
                ctl.local_seen.fill(0)
                return {'uid': uid, 'rank': ctl.rank, 'released': True}

        def routing_local_state(self):
            ctl = self.model_runner.steer_ctl
            with ctl.lock:
                import hashlib
                import torch
                batch = self.model_runner.input_batch
                rng = {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                       'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                       'request_generators': {str(k): hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest()
                                              for k, g in batch.generators.items()}}
                return {'rank': ctl.rank, 'version': VERSION, 'command': ctl.local_command,
                        'previous': ctl.local_previous, 'rng': rng, 'states': [
                            {'uid': s.uid, 'hwm': s.hwm, 'prompt_len': s.prompt_len, 'prefix_len': s.prefix_len}
                            for s in ctl.states.values()]}

    return LocalWorkerExtension


def __getattr__(name):
    if name == 'LocalWorkerExtension':
        return worker_extension_class()
    raise AttributeError(name)
