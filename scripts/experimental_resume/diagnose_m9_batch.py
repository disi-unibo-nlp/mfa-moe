"""Diagnose greedy M9 batch divergence without upgrading qualification.

This is an engineering check, not a semantic or accuracy result. The same model
session runs fixed requests twice in isolation and once in a mixed batch,
recording the first divergence and top-two logprob gap. Both prior failed
qualifications remain failures regardless of this diagnostic's exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
MANIFEST = R / 'steering-v1/runs/m9-resume-v1/MANIFEST.json'
EXPECTED_MANIFEST = '63772b263e855692ecc5a37b10f634f96b30fa51e7e54bf22f5026f16e43024b'
PRIOR = R / 'steering-v1/runs/m9-resume-v1/replay-deterministic-63772b26-ea7f3a93/REPLAY.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def select(value):
    rows = [r for r in value['rows'] if r['prepared']['status'] == 'generate_closure']
    rows.sort(key=lambda r: digest(['M9-replay-qualification-v1', r['uid']]))
    picked = rows[:4]
    if len(picked) != 4 or len({r['question'] for r in picked}) != 4:
        raise ValueError('four deterministic, distinct-question M9 branches required')
    return picked


def source_checks(value, picked, tokenizer):
    from moe_steer import engine
    if value['sha256'] != EXPECTED_MANIFEST or digest({k: v for k, v in value.items() if k != 'sha256'}) != value['sha256']:
        raise ValueError('frozen M9 manifest differs')
    if engine.code_tree_sha256() != value['code_tree']:
        raise ValueError('frozen sampler tree differs')
    for path, expected in value['code'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('M9 preparation code differs: ' + path)
    for row in picked:
        p = row['prepared']
        start = p['prefix_presence_start']
        if p['prompt_token_ids'][start + value['cut']: ] != p['injection_token_ids']:
            raise ValueError('closure injection not at the fixed cut')
        if tokenizer.encode(value['closure_text'], add_special_tokens=False) != p['injection_token_ids']:
            raise ValueError('literal closure retokenized differently')
        if p['max_new_tokens'] + value['cut'] + len(p['injection_token_ids']) != value['total_budget']:
            raise ValueError('injection and generation exceed the 16k endpoint')
        if len(p['prompt_token_ids']) != start + p['prefix_len']:
            raise ValueError('branch prompt length differs')


def numerical_check(picked):
    """Compare the frozen mask against the explicit presence-penalty formula."""
    import torch
    from types import SimpleNamespace
    from vllm import SamplingParams
    from moe_steer.logits import PrefixPresencePenalty
    p = picked[0]['prepared']
    prompt = p['prompt_token_ids']
    start = p['prefix_presence_start']
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        model_config=SimpleNamespace(get_vocab_size=lambda: max(prompt) + 12),
    )
    processor = PrefixPresencePenalty(config, torch.device('cpu'), False)
    params = SamplingParams(max_tokens=2, presence_penalty=0.,
                            extra_args={'steer': {'prefix_presence': {'penalty': 1.5, 'prefix_start': start}}})
    out = []
    processor._add(0, params, prompt, out)
    width = max(prompt) + 12
    logits = torch.zeros((2, width), dtype=torch.float32)
    actual = processor.apply(logits).clone()
    expected = torch.zeros_like(actual)
    expected[0, torch.tensor(sorted(set(prompt[start:])), dtype=torch.long)] = -1.5
    if not torch.equal(actual, expected):
        raise ValueError('prefix-presence numerical mask mismatch')
    new_token = next(t for t in range(width) if t not in set(prompt[start:]))
    out.append(new_token)
    actual_next = processor.apply(torch.zeros_like(logits))
    expected[0, new_token] = -1.5
    if not torch.equal(actual_next, expected):
        raise ValueError('generated-token presence update mismatch')
    processor._drop(0)
    if torch.count_nonzero(processor.apply(torch.zeros_like(logits))):
        raise ValueError('removed request left a penalty on a neighbor')
    return {'prefix_unique_tokens': len(set(prompt[start:])), 'new_token': new_token,
            'penalty': 1.5, 'mask_exact': True, 'removed_row_zero': True}


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('M9 replay qualification requires GPU Slurm')
    if args.out.exists():
        raise FileExistsError('digest-bound replay output already exists')
    args.out.mkdir(parents=True)
    started = time.monotonic()
    value = json.loads(MANIFEST.read_text())
    prior = json.loads(PRIOR.read_text())
    if (digest({k: v for k, v in prior.items() if k != 'sha256'}) != prior.get('sha256') or
        prior.get('job_id') != '59118998' or prior.get('pass') is not False or
        prior.get('batch_isolation') is not False or
        not all(c['isolated_equals_mixed'] for c in prior['checks'] if c['mode'] == 'native')):
        raise ValueError('raw deterministic batch-identity failure changed')
    picked = select(value)
    from moe_steer import engine
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    source_checks(value, picked, tokenizer)
    numeric = numerical_check(picked)
    result = {'schema': 'M9-batch-divergence-diagnostic-v1', 'manifest_sha256': value['sha256'],
              'job_id': os.environ['SLURM_JOB_ID'], 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'code_tree': value['code_tree'], 'fixture_uids': [r['uid'] for r in picked],
              'numerical_check': numeric, 'pass': False,
              'sampling_mode': 'deterministic_greedy_logprobs2',
              'prior_deterministic_failure_sha256': prior['sha256'],
              'qualification_status': 'HOLD_ENGINE_ISOLATION'}
    model = None
    try:
        for key in tuple(os.environ):
            if key.startswith('STEER_'):
                os.environ.pop(key)
        kwargs = engine.engine_kwargs(plugin=False, max_num_seqs=12)
        kwargs['logits_processors'] = ['moe_steer.logits:PrefixPresencePenalty']
        kwargs.update(gpu_memory_utilization=.80, long_prefill_token_threshold=1024)
        engine.apply_env(engine.engine_env(None, None, plugin=False))
        model = engine.build_llm(kwargs)
        from vllm import SamplingParams
        cases = []
        for row in picked:
            p = row['prepared']
            for mode in ('edited', 'native', 'large_dose'):
                if mode == 'native':
                    settings = engine.card_sampling(24, row['seed'], presence_penalty=1.5)
                else:
                    dose = 1.5 if mode == 'edited' else 20.
                    settings = engine.card_sampling(24, row['seed'], presence_penalty=0.,
                        extra_args={'steer': {'prefix_presence': {
                            'penalty': dose, 'prefix_start': p['prefix_presence_start']}}})
                settings.update(temperature=0., top_p=1., top_k=-1, min_p=0., logprobs=2)
                cases.append({'uid': row['uid'], 'mode': mode, 'prompt': p['prompt_token_ids'],
                              'sampling': SamplingParams(**settings)})
        def selected(output):
            choice = output.outputs[0]
            gaps = []
            for row in choice.logprobs or []:
                values = sorted((float(v.logprob) for v in row.values()), reverse=True)
                gaps.append(values[0] - values[1] if len(values) > 1 else None)
            return {'tokens': list(choice.token_ids), 'top_two_logprob_gaps': gaps,
                    'finish_reason': choice.finish_reason}

        by_key = {}
        for case in cases:
            outputs = model.generate([{'prompt_token_ids': case['prompt']}], case['sampling'], use_tqdm=False)
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise ValueError('isolated replay output count differs')
            output = outputs[0]
            if list(output.prompt_token_ids) != case['prompt']:
                raise ValueError('isolated replay prefix differs')
            by_key[(case['uid'], case['mode'])] = selected(output)
        outputs = model.generate([{'prompt_token_ids': c['prompt']} for c in cases],
                                 [c['sampling'] for c in cases], use_tqdm=False)
        if len(outputs) != len(cases):
            raise ValueError('mixed-batch replay output count differs')
        checks = []
        for case, output in zip(cases, outputs, strict=True):
            key = (case['uid'], case['mode'])
            mixed = selected(output)
            single = by_key[key]
            n = min(len(single['tokens']), len(mixed['tokens']))
            divergence = next((i for i in range(n) if single['tokens'][i] != mixed['tokens'][i]),
                              n if len(single['tokens']) != len(mixed['tokens']) else None)
            checks.append({'uid': case['uid'], 'mode': case['mode'],
                           'prompt_exact': list(output.prompt_token_ids) == case['prompt'],
                           'isolated_equals_mixed': divergence is None,
                           'first_divergence': divergence,
                           'isolated': single, 'mixed': mixed,
                           'first_divergence_logprob_gaps': None if divergence is None else {
                               name: (series['top_two_logprob_gaps'][divergence]
                                      if divergence < len(series['top_two_logprob_gaps']) else None)
                               for name, series in (('isolated', single), ('mixed', mixed))}})
        repeats = []
        for case in cases:
            if case['mode'] == 'native':
                continue
            output = model.generate([{'prompt_token_ids': case['prompt']}], case['sampling'], use_tqdm=False)[0]
            repeat = selected(output)
            repeats.append({'uid': case['uid'], 'mode': case['mode'],
                            'identical_to_first_isolated': repeat['tokens'] == by_key[(case['uid'], case['mode'])]['tokens'],
                            'repeat': repeat})
        large_changed = sum(by_key[(r['uid'], 'large_dose')]['tokens'] != by_key[(r['uid'], 'native')]['tokens'] for r in picked)
        isolation = all(c['prompt_exact'] and c['isolated_equals_mixed'] for c in checks)
        result.update(identical_prefix=all(c['prompt_exact'] for c in checks),
                      prefix_presence=numeric['mask_exact'] and large_changed >= 1,
                      closure_injection=True, batch_isolation=isolation,
                      large_dose_changed_vs_native=large_changed, checks=checks,
                      repeated_isolated=repeats,
                      model_load_and_replay_seconds=time.monotonic() - started)
        result['diagnostic_complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()[-3500:]
    result['elapsed_seconds'] = time.monotonic() - started
    result['sha256'] = digest(result)
    atomic_json(args.out / 'REPLAY.json', result)
    print(json.dumps({'diagnostic_complete': result.get('diagnostic_complete'), 'error': result.get('error'),
                      'result': str(args.out / 'REPLAY.json')}), flush=True)
    from moe_steer.qualify import finish_child
    from types import SimpleNamespace
    finish_child(0 if result.get('diagnostic_complete') else 3, SimpleNamespace(llm=model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
