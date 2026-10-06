"""Prospective enrollment and digest-bound request construction for the separate new study.

This schema is intentionally not a legacy moe_steer manifest. A qualified worker adapter
must consume it; an old runner cannot silently execute a new trajectory policy.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib

from .design import Action, STAGES, Template, balanced_random_assignment, digest

EXPECTED = {'discovery': 48, 'mechanism': 128, 'utility': 96}


def verify(value):
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('manifest content digest differs')
    return value


def build(stage, *, family_freeze, enrollment, code_digest, qualification,
          price, actions=(), template=None, random_templates=(), detector_digest=None):
    """No outcome-driven replacements, unsupported-transition search, or unpriced launch."""
    if stage not in EXPECTED:
        raise ValueError('unregistered experiment stage')
    verify(family_freeze)
    if len(code_digest) != 64 or set(code_digest) - set('0123456789abcdef'):
        raise ValueError('explicit code digest required')
    pools = family_freeze['new_parent_pools']
    parents = pools['parent_pools'][stage]
    if len(parents) != EXPECTED[stage] or set(enrollment) != set(parents):
        raise ValueError('every frozen parent requires an eligibility receipt; no replacements')
    if not qualification.get('pass') or not qualification.get('sha256'):
        raise ValueError('actual worker/prefix qualification is required before a manifest')
    verify(qualification)
    if qualification.get('worker_code_digest') != code_digest:
        raise ValueError('worker qualification applies to another code version')
    if price.get('stage') != stage or price.get('status') != 'PASS' or price.get('ceiling') != STAGES[stage]:
        raise ValueError('complete qualified stage price must pass its own ceiling')
    verify(price)
    if not detector_digest or len(detector_digest) != 64 or set(detector_digest) - set('0123456789abcdef'):
        raise ValueError('freeze a qualified discovery detector before enrollment')
    # The protocol's maximum targets are a feasibility gate, not an invitation to top up.
    eligible = [f for f in parents if enrollment[f].get('eligible') is True]
    if len(eligible) != EXPECTED[stage]:
        raise ValueError('FAIL_INSUFFICIENT_ELIGIBLE_FAMILIES; revise resources/design prospectively')
    actions = tuple(actions)
    if stage == 'discovery':
        if not 1 <= len(actions) <= 6 or len({a.name for a in actions}) != len(actions):
            raise ValueError('one to six distinct supported actions required')
        for action in actions:
            action.validate()
        for transition in {a.transition for a in actions}:
            subset = [a for a in actions if a.transition == transition]
            if len({a.experts for a in subset}) != 1 or len({a.bias for a in subset}) != len(subset):
                raise ValueError('only one expert set and each registered bias per transition')
        arms = ('native', *(a.name for a in actions))
        templates = {a.name: asdict(Template(a.transition, (a,), (0,)).validate()) for a in actions}
        n_random = 0
    else:
        if not isinstance(template, Template):
            raise ValueError('freeze one discovery-selected template')
        template.validate()
        arms = ('native', 'policy', 'random', 'reversed') if stage == 'mechanism' else ('native', 'policy')
        templates = {'policy': asdict(template), 'reversed': asdict(template.reversed())}
        n_random = len(random_templates)
        if stage == 'mechanism':
            if n_random != 4:
                raise ValueError('four frozen matched random template sets required')
            for other in random_templates:
                other.validate()
                if other.slots != template.slots or other.starting_condition != template.starting_condition:
                    raise ValueError('random template timing differs')
                for action, control in zip(template.actions, other.actions):
                    if (len(other.actions) != len(template.actions) or action.bias != control.bias
                            or [(l, len(e)) for l, e in action.experts] != [(l, len(e)) for l, e in control.experts]):
                        raise ValueError('random expert support or action multiset differs')
    assignments = {(r['family'], r['seed']): r['random_set'] for r in
                   balanced_random_assignment(parents, n_sets=4)}
    requests = []
    for family in parents:
        receipt = enrollment[family]
        question = pools['representative_questions'][family]
        if receipt.get('question') != question:
            raise ValueError('enrollment question differs from the frozen representative')
        prompt = receipt['original_prompt_token_ids']
        prefix = [] if stage == 'utility' else receipt['emitted_prefix_token_ids']
        if not prompt or any(type(t) is not int or t < 0 for t in (*prompt, *prefix)):
            raise ValueError('invalid prompt/prefix tokens')
        if stage != 'utility' and (not prefix or receipt.get('detector_digest') != detector_digest):
            raise ValueError('prefix must be selected by the frozen prefix-only detector')
        for seed in (0, 1):
            for arm in arms:
                request = {'family': family, 'question': question, 'seed': seed, 'arm': arm,
                    'prompt_token_ids': list(prompt), 'emitted_prefix_token_ids': list(prefix),
                    'max_tokens': 16384 if stage == 'utility' else 1024,
                    'random_set': assignments[family, seed] if arm == 'random' else None,
                    'detector_digest': detector_digest, 'code_digest': code_digest}
                # Unlike legacy UIDs, this identity includes both caps and code.
                request['uid'] = digest({'stage': stage, 'family_freeze': family_freeze['sha256'], **request})
                requests.append(request)
    requests.sort(key=lambda r: hashlib.sha256(('execution-v1|' + r['uid']).encode()).hexdigest())
    manifest = {'schema': 'routing-control-manifest-v1', 'stage': stage,
        'family_freeze': family_freeze['sha256'], 'detector_digest': detector_digest,
        'qualification_digest': qualification['sha256'], 'price_digest': price['sha256'],
        'code_digest': code_digest, 'enrollment': enrollment, 'templates': templates,
        'random_templates': [asdict(t) for t in random_templates], 'requests': requests,
        'native_top_k': 8, 'shared_expert': 'unchanged',
        'gpu_hour_ceiling': STAGES[stage], 'decode_token_cap': sum(r['max_tokens'] for r in requests)}
    return {**manifest, 'sha256': digest(manifest)}
