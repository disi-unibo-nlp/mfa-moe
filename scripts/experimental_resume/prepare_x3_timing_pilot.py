"""CPU-only preparation of a frozen, high-cost legacy X3 32k timing pilot.

No GPU inference, label, grade, NLL or X3 enrollment change is performed here.
The pilot's future analysis may use timing/token counts only, never semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket

from x3_build_v2 import verified, assert_registered_x3_population, pinned_order
from x2_build import e_policy

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
SELECTION = REPORT / 'X3_G3_SELECTION_v2.json'
ELIGIBILITY = REPORT / 'X3_ELIGIBILITY_v1.json'
PRICE_OUT = REPORT / 'X3_TIMING_PILOT_PRICE_v1.json'
NAME = 'x3-timing-pilot-v1'
SNAPSHOT = S / 'code/s1-9a61e32f48c04c24'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'previous timing pilot artifact differs: {path}')
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    temp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    temp.replace(path)
    return value


def choose(rows):
    # Native length is a preexisting cost indicator. Select only registered
    # long-endpoint firing rows likely to exercise the expensive 32k tail.
    candidates = sorted((r for r in rows if r['eligible'] and r['long_endpoint']
                         and r['trace_completion_tokens'] >= 32768
                         and not r['confirm_connected_family']),
                        key=lambda r: (r['prefix_len'], r['question']))
    if len(candidates) < 6:
        raise ValueError('fewer than six native-long, clean-family timing candidates')
    positions = (0, len(candidates)//3, 2*len(candidates)//3, len(candidates)-1)
    chosen = [candidates[i] for i in positions]
    if len({r['question'] for r in chosen}) != 4 or len({r['family'] for r in chosen}) != 4:
        raise ValueError('timing pilot must use four disjoint families')
    return chosen


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('X3 timing pilot preparation requires CPU Slurm')
    from moe_steer import engine, manifests as M, policies as P
    from moe_steer.spec import seal
    selected, eligibility = verified(SELECTION), verified(ELIGIBILITY)
    split, world = M.load_split(), M.load_world()
    order = pinned_order(split)
    assert_registered_x3_population(order, split, world.infos)
    if (eligibility['split_sha256'] != split['sha256'] or
        [r['question'] for r in eligibility['rows']] != order or
        selected['version'] != 'v2-registered-dose-support' or
        len(selected['selected']) != 2 or
        {c['sign'] for c in selected['selected']} != {-1, 1}):
        raise ValueError('registered X3 enrollment or G3 selection differs')
    if engine.code_tree_sha256() != '9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430':
        raise ValueError('X3 timing pilot requires frozen capped sampler')
    chosen = choose(eligibility['rows'])
    cells = selected['selected']
    target, random = [], []
    for c in cells:
        target.append(e_policy(world.inputs, c['scope'],
                               (c['operator'], c['sign'], c['magnitude']),
                               P.landmark_schedule(256)))
        for k in (0, 1):
            base = e_policy(world.inputs, c['scope'],
                            (c['operator'], c['sign'], c['random_magnitudes'][str(k)]),
                            P.landmark_schedule(256))
            random.append(P.matched_random(base, world.inputs, k))
    sham = P.sham(world.inputs.scope('ALL'), P.landmark_schedule(256))
    table = P.build_table([*target, *random, sham])
    requests = []
    for row in chosen:
        q, prefix = row['question'], row['prefix_len']
        cap = 32768 - prefix
        for k in (0, 1):
            arms = [('N', sham.name)]
            for c, policy in zip(cells, target):
                sign = '+' if c['sign'] > 0 else '-'
                base = e_policy(world.inputs, c['scope'],
                                (c['operator'], c['sign'], c['random_magnitudes'][str(k)]),
                                P.landmark_schedule(256))
                arms.extend([('E' + sign, policy.name),
                             ('M' + sign, P.matched_random(base, world.inputs, k).name)])
            for arm, name in arms:
                r = M.make_request(NAME, table, world.infos[q], arm=arm,
                                   policy_name=name, seed_k=k,
                                   parent={'trace_ref': row['trace_ref'], 'prefix_len': prefix},
                                   landmark=prefix, expected_len=cap, max_new_tokens=cap)
                if r['max_tokens'] != cap:
                    raise ValueError('timing pilot cap failed to reach sampler request')
                requests.append(r)
    manifest = M.build_manifest(NAME, 'x3', table, requests, world.infos,
                                n_shards=1, seals=world.seals,
                                code_tree=engine.code_tree_sha256(),
                                lexicon_path=world.lexicon_path,
                                vocab_path=world.vocab_path,
                                routed={'target_policies': None, 'window': 64,
                                        'after_pulse': 1024}, expect_fingerprint=None)
    M.validate_manifest(manifest)
    if len(requests) != 40 or len({r['uid'] for r in requests}) != 40:
        raise ValueError('X3 timing pilot must contain exactly 4x2x5 requests')
    pilot_dir = S / 'runs/resume-v1/x3-timing-pilot-v1'
    manifest_path = pilot_dir / f'{NAME}-{manifest["sha256"][:16]}.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('timing pilot manifest changed under the same digest')
    else:
        pilot_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    cost = json.loads((S / 'runs/x2prep/cost_model.json').read_text())
    prefill = sum(r['prompt_tokens'] + r['parent']['prefix_len'] for r in requests)
    decode = sum(r['max_tokens'] for r in requests)
    historical_rate = cost['inputs']['x1']['steady_tok_per_s'] * cost['constants']['pessimistic_derate']
    context_stress_rate = historical_rate / 2
    prefill_rate = cost['inputs']['q10']['prefill_tps'] / 2
    components = {'two_cold_loads_seconds': 2 * 838.0,
                  'generation_seconds': decode / context_stress_rate * 1.25,
                  'prefill_seconds': prefill / prefill_rate * 1.25,
                  'shutdown_seconds': 196.0}
    seconds = sum(components.values())
    ceiling_GPUh = math.ceil(2 * seconds / 3600 * 2) / 2
    body = {'schema': 'x3-32k-timing-pilot-price-v1',
            'status': 'CPU_PREPARED_GPU_HOLD',
            'job_id': os.environ['SLURM_JOB_ID'],
            'driver_sha256': sha(__file__),
            'manifest_path': str(manifest_path),
            'manifest_sha256': manifest['sha256'],
            'selection_sha256': selected['sha256'],
            'eligibility_sha256': eligibility['sha256'],
            'selected_questions': [{'question': r['question'], 'family': r['family'],
                                    'prefix_len': r['prefix_len'],
                                    'native_completion_tokens': r['trace_completion_tokens']}
                                   for r in chosen],
            'selection_rule': 'four distinct non-confirm-connected families among registered first64 firing rows with native trace >=32768 tokens; positions 0, floor(n/3), floor(2n/3), n-1 after ascending prefix length',
            'requests': len(requests), 'max_decode_tokens': decode,
            'prefill_tokens_without_cache_reuse': prefill,
            'stress_assumptions': {'historical_generation_derate': .75,
                                   'additional_context_slowdown': 2.,
                                   'prefill_rate_slowdown': 2.,
                                   'repeat_work': 1.25, 'cold_loads': 2},
            'components_wall_seconds': components,
            'projected_wall_seconds': seconds,
            'projected_GPUh_two_A100': 2 * seconds / 3600,
            'suggested_rounded_GPUh_ceiling': ceiling_GPUh,
            'measurement_outputs_allowed': ['elapsed', 'load', 'prefill_tokens',
                                            'decoded_tokens', 'finish_reason',
                                            'per-request wall_seconds', 'batch_shape'],
            'measurement_outputs_embargoed': ['text', 'answer', 'correctness',
                                             'semantic_labels', 'routing_effects'],
            'interpretation': 'Separate X3 cost-only pilot. Model generations are not X3 causal estimates and are not added to registered 96-question analysis. No GPU job has been submitted.'}
    priced = write_once(PRICE_OUT, body)
    print(json.dumps({'status': priced['status'], 'manifest': str(manifest_path),
                      'requests': len(requests), 'projected_GPUh': priced['projected_GPUh_two_A100'],
                      'ceiling_GPUh': priced['suggested_rounded_GPUh_ceiling']}), flush=True)


if __name__ == '__main__':
    main()
