"""Qualified-after-test operator pulses over the immutable ordered-worker row planner.

The sole change to the inherited mask/telemetry implementation is the policy
allowlist: force-in and selected-weight reweighting may occupy the first pulse.
This module is imported only in an allocated vLLM worker process.
"""
from moe_exp.routing_control import worker_adapter as ordered
from moe_exp.routing_control.design import digest


class OperatorPulse(ordered.OrderedPulse):
    @classmethod
    def load(cls, value, table, first_policy):
        if (set(value) != ordered.FIELDS or value['horizon'] != 1024 or
                value['sha256'] != digest({k:v for k,v in value.items() if k != 'sha256'})):
            raise ValueError('operator pulse fields, horizon or hash differs')
        names, slots = tuple(value['action_policy_names']), tuple(value['slots'])
        if (not 1 <= len(names) <= 2 or slots != (0,512)[:len(names)] or
                names[0] != first_policy or any(type(s) is not int for s in slots)):
            raise ValueError('operator pulses require first slot0 and optional slot512')
        indices = []
        for name in names:
            index = table.index_of(name)
            policy = table.policies[index]
            op, targets = policy.operator, policy.targets
            allowed = (op.kind == 'bias' and op.sign == 1 and op.magnitude in (.5,1.) or
                       op.kind == 'force' and op.sign == 1 and op.magnitude == 0 or
                       op.kind == 'reweight' and op.sign == 1 and op.magnitude == 1.)
            if not allowed or policy.schedule.kind != 'always' or targets is None:
                raise ValueError('unqualified operator, magnitude or schedule')
            layers = [layer for layer,_ in targets.experts]
            if (not 1 <= len(layers) <= 4 or layers != list(range(min(layers),max(layers)+1)) or
                    any(not 1 <= len(ids) <= 2 for _,ids in targets.experts)):
                raise ValueError('operator targets exceed qualified sparse geometry')
            if len(names) == 2 and (op.kind != 'bias' or names[0] != names[1]):
                raise ValueError('second pulse is qualified only for repeated identical positive bias')
            indices.append(index)
        return cls(names, tuple(indices), slots, value['sha256'])


def install():
    ordered.OrderedPulse = OperatorPulse
    return ordered.install()


def worker_extension_class():
    E = install()
    class OvernightWorkerExtension(E.SteerWorkerExtension):
        def overnight_operator_status(self):
            return {'version': 'overnight-routing-operator-pulse-v2',
                    'status': self.steer_status()}
    return OvernightWorkerExtension


# Importing on the CPU to test OperatorPulse must not import torch or install hooks.
def __getattr__(name):
    if name == 'OvernightWorkerExtension':
        return worker_extension_class()
    raise AttributeError(name)
