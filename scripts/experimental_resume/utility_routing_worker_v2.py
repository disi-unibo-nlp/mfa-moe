"""Qualification-only dynamic first-episode pulses on the immutable worker.

No installed runtime or frozen worker file is modified. This process-local
adapter adds an RPC that binds one selected episode at an observed token edge.
Production utility use requires its own exact GPU qualification receipt.
"""
from __future__ import annotations

from types import MethodType
import numpy as np

from moe_exp.routing_control import worker_adapter as ordered
from moe_exp.routing_control.design import digest
from overnight_routing_worker_v2 import OperatorPulse

VERSION = 'utility-routing-worker-v2-first-episode'
THINK_END_ID = 248069


def rows_for_episode(episode, table, positions, output_ids):
    positions = np.asarray(positions)
    mask = np.zeros(len(positions), dtype=np.bool_)
    indices = np.zeros(len(positions), dtype=np.int32)
    if episode is None:
        return mask, indices
    closed = list(output_ids).index(THINK_END_ID) if THINK_END_ID in output_ids else None
    for slot, name in zip(episode['absolute_slots'], episode['action_names'], strict=True):
        take = (positions >= slot) & (positions < slot + 256)
        if closed is not None:
            take &= positions <= closed
        mask[take] = True
        indices[take] = table.index_of(name)
    return mask, indices


def install():
    ordered.OrderedPulse = OperatorPulse
    extension = ordered.install()
    if getattr(extension.install_steering, 'utility_version', None) == VERSION:
        return extension
    previous = extension.install_steering

    def installed(*args, **kwargs):
        ctl = previous(*args, **kwargs)
        ctl.utility_episodes = {}
        new_state, plan_rows, record = ctl._new_state, ctl._plan_rows, ctl._record

        def new(self, rid, request, start):
            state = new_state(rid, request, start)
            if state.policy_index != 0 or state.fsm is not None or state.ordered is not None:
                raise ValueError('utility request must start with native routing and no preset episode')
            state.utility_counts = {}
            return state

        def plan(self, state, request, a, b, start, mask, indices, slot):
            old_hwm = state.hwm
            plan_rows(state, request, a, b, start, mask, indices, slot)
            episode = self.utility_episodes.get(state.uid)
            positions = np.arange(start, start + b - a, dtype=np.int64) - state.prompt_len + 1
            active, policy_indices = rows_for_episode(episode, self.table, positions, request.output_token_ids)
            mask[a:b], indices[a:b] = active, policy_indices
            if episode is None:
                return
            fresh = np.arange(start, start + b - a) >= old_hwm
            total = int(np.count_nonzero(active & fresh))
            state.cpu_pulse_rows += total
            state.cpu_active_rows += total
            for name in set(episode['action_names']):
                count = int(np.count_nonzero(active & fresh & (policy_indices == self.table.index_of(name))))
                state.utility_counts[name] = state.utility_counts.get(name, 0) + count
                self.policy_active_rows[name] += count

        def observed(self, state, *args, **kwargs):
            value = record(state, *args, **kwargs)
            episode = self.utility_episodes.get(state.uid)
            value.update(utility_worker_version=VERSION, utility_episode=episode,
                         utility_action_rows=dict(state.utility_counts))
            if episode:
                value['utility_action_dose'] = {name: {str(layer): dict(zip(extension.TELE_FIELDS,
                    [float(x) for x in state.ordered_dose_acc[self.table.index_of(name), h]]))
                    for h, layer in enumerate(self.layers)} for name in set(episode['action_names'])}
            return value

        ctl._new_state = MethodType(new, ctl)
        ctl._plan_rows = MethodType(plan, ctl)
        ctl._record = MethodType(observed, ctl)
        return ctl

    installed.routing_control_version = ordered.VERSION
    installed.utility_version = VERSION
    extension.install_steering = installed
    return extension


def worker_extension_class():
    extension = install()

    class UtilityWorkerExtension(extension.SteerWorkerExtension):
        def utility_status(self):
            return {'version': VERSION, 'qualification': 'REQUIRES_EXACT_SEPARATE_GPU_RESULT',
                    'status': self.steer_status()}

        def utility_set_episode(self, uid, episode, emitted_ids):
            ctl = getattr(getattr(self, 'model_runner', None), 'steer_ctl', None)
            if ctl is None:
                raise ValueError('utility routing controller absent')
            with ctl.lock:
                states = [s for s in ctl.states.values() if s.uid == uid]
                if len(states) != 1:
                    raise ValueError('utility RPC must bind exactly one live request')
                state = states[0]
                ids = list(emitted_ids)
                if (not ids or any(type(i) is not int or i < 0 for i in ids) or THINK_END_ID in ids or
                        episode['trigger_emitted_tokens'] != len(ids) or
                        episode['trigger_emitted_ids_sha256'] != digest(ids) or
                        state.hwm != state.prompt_len + len(ids) - 1):
                    raise ValueError('worker ran ahead, prefix changed, or reasoning already closed')
                known = list(state.ref.output_token_ids)
                if len(known) not in (len(ids) - 1, len(ids)) or known != ids[:len(known)]:
                    raise ValueError('worker emitted-token history differs from controller prefix')
                value = {'action_policy_names': episode['action_names'],
                         'slots': episode['relative_slots'], 'horizon': 1024}
                OperatorPulse.load({**value, 'sha256': digest(value)}, ctl.table, episode['action_names'][0])
                if (episode['absolute_slots'] != [len(ids) + x for x in episode['relative_slots']] or
                        episode['pulse_width'] != 256 or episode['stop_at_reasoning_closure'] is not True):
                    raise ValueError('online episode differs from selected qualified local pulse')
                prior = ctl.utility_episodes.get(uid)
                if prior is not None and prior != episode:
                    raise ValueError('one first accepted episode per answer; re-entry forbidden')
                ctl.utility_episodes[uid] = dict(episode)
                return {'uid': uid, 'rank': ctl.rank, 'episode_sha256': digest(episode),
                        'next_output_index': len(ids), 'worker_hwm': state.hwm}

    return UtilityWorkerExtension


def __getattr__(name):
    if name == 'UtilityWorkerExtension':
        return worker_extension_class()
    raise AttributeError(name)
