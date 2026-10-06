"""Checkpointed, arm-blind two-vote LLM audit of frozen discovery transitions.

The two ratings are independent model draws with separate seeds from the same
Qwen judge. They are LLM audits of visible text, not human ground truth. The
online detector never reads these outputs or any future sentence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
INPUT = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_AUDIT_CANDIDATES.json'
RUBRIC = REPO / 'report/experimental-resume-v1/TRANSITION_RUBRIC_v0.1.md'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
TRANSITIONS = {
    'candidate_to_verify': ('The triggering sentence states a complete proposed value or answer that has not '
                            'already been substantively checked. The later sentence evaluates that candidate '
                            'or an intermediate result against an original constraint or independent computation. '
                            'Verification words alone do not qualify.'),
    'approach_to_commit': ('The triggering sentence tentatively names a specific approach and the operation it '
                           'would perform. The later sentence commits to that approach with a concrete next '
                           'operation or executes its first step. A generic intention or restatement does not qualify.'),
    'failed_check_to_revise': ('The triggering sentence visibly computes a contradiction, invalid value, or '
                               'violated original condition. The later sentence changes an assumption, method, '
                               'branch, or next operation in response. Mere uncertainty or continued unchanged '
                               'calculation does not qualify.'),
}
BATCH = 32
MAX_TOKENS = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if digest({k: v for k, v in value.items() if k != 'sha256'}) != value.get('sha256'):
        raise ValueError('changed transition audit seal: ' + str(path))
    return value


def atomic_json(path, body):
    value = {**body, 'sha256': digest(body)}
    temp = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(temp, path)
    return value


def messages(row):
    if set(row['reader_input']) != {'problem', 'previous_sentence', 'triggering_sentence', 'later_sentence'}:
        raise ValueError('semantic reader input allowlist changed')
    definition = TRANSITIONS[row['transition']]
    system = ('You independently audit a local transition in visible mathematical reasoning. '
              'Use only the supplied text. Do not solve the problem or infer hidden thoughts. '
              'Return exactly one JSON object with boolean keys "start" and "target". '
              'Start means the triggering sentence satisfies the stated starting condition. '
              'Target means the later sentence performs the specified response to that trigger. '
              'If evidence is ambiguous, set the relevant field false. No explanation.\n'
              'Criterion: ' + definition)
    x = row['reader_input']
    user = ('Original problem:\n' + x['problem'] + '\n\nPrevious sentence:\n' + x['previous_sentence'] +
            '\n\nTriggering sentence:\n' + x['triggering_sentence'] +
            '\n\nLater sentence:\n' + x['later_sentence'] + '\n\nJSON:')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_rating(text):
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    starts = re.findall(r'"start"\s*:\s*(true|false)', text, re.I)
    targets = re.findall(r'"target"\s*:\s*(true|false)', text, re.I)
    if len(starts) != 1 or len(targets) != 1:
        return None
    return {'start': starts[0].lower() == 'true', 'target': targets[0].lower() == 'true'}


def rating_seed(uid, reader):
    return int(digest(['transition-rating-v1', uid, reader])[:8], 16) % 2_000_000_000


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('transition LLM audit requires GPU Slurm')
    fixture = sealed(INPUT)
    if fixture['schema'] != 'transition-audit-candidates-unlabeled-v1' or fixture['families'] != 48:
        raise ValueError('not the frozen discovery candidate frame')
    if fixture['reader_input_allowlist'] != ['problem','previous_sentence','triggering_sentence','later_sentence']:
        raise ValueError('reader allowlist changed')
    rows = fixture['records']
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate UID')
    binding_body = {'schema': 'transition-LLM-audit-binding-v1',
                    'fixtures_sha256': fixture['sha256'],
                    'driver_sha256': file_sha(Path(__file__)),
                    'rubric_sha256': file_sha(RUBRIC),
                    'model_snapshot': str(MODEL),
                    'model_revision': MODEL.name,
                    'batch_size': BATCH, 'max_tokens_per_rating': MAX_TOKENS,
                    'readers': 2, 'sampler': {'temperature': .2, 'top_p': .95,
                                              'thinking': True, 'reasoning_effort': 'low'},
                    'scope': 'discovery-only arm-blind same-model independent draws; LLM audit, not human ground truth'}
    args.out.mkdir(parents=True, exist_ok=True)
    binding_path = args.out / 'BINDING.json'
    if binding_path.exists():
        binding = sealed(binding_path)
        if binding != {**binding_body, 'sha256': digest(binding_body)}:
            raise ValueError('rating output directory is bound to different inputs/code')
    else:
        binding = atomic_json(binding_path, binding_body)
    (args.out / 'batches').mkdir(exist_ok=True)
    pending = []
    for start in range(0, len(rows), BATCH):
        path = args.out / 'batches' / f'{start:06d}.json'
        if path.exists():
            part = sealed(path)
            if (part['binding_sha256'] != binding['sha256'] or part['start'] != start or
                [r['uid'] for r in part['records']] != [r['uid'] for r in rows[start:start+BATCH]]):
                raise ValueError('cached semantic rating batch differs')
        else:
            pending.append((start, path))
    if pending:
        from vllm import LLM, SamplingParams
        model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                    dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=49152,
                    max_num_seqs=32, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config='vllm', language_model_only=True,
                    attention_config={'backend': 'FLASH_ATTN'})
        started = time.monotonic()
        for start, path in pending:
            block = rows[start:start+BATCH]
            conversations = [messages(r) for r in block]
            per_reader = []
            for reader in (0, 1):
                sampling = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                           seed=rating_seed(r['uid'], reader)) for r in block]
                outputs = model.chat(conversations, sampling_params=sampling,
                                     chat_template_kwargs={'enable_thinking': True,
                                                           'reasoning_effort': 'low'}, use_tqdm=False)
                if len(outputs) != len(block):
                    raise ValueError('semantic judge output count differs')
                per_reader.append([{'rating': parse_rating(o.outputs[0].text),
                                    'finish_reason': o.outputs[0].finish_reason,
                                    'generated_tokens': len(o.outputs[0].token_ids),
                                    'raw_completion': o.outputs[0].text}
                                   for o in outputs])
            part_rows = [{'uid': r['uid'], 'transition': r['transition'],
                          'readers': [per_reader[0][i], per_reader[1][i]]}
                         for i, r in enumerate(block)]
            atomic_json(path, {'schema': 'transition-LLM-audit-batch-v1',
                               'binding_sha256': binding['sha256'], 'start': start,
                               'records': part_rows})
            print(json.dumps({'complete': min(start+BATCH, len(rows)), 'of': len(rows),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
    complete = []
    for start in range(0, len(rows), BATCH):
        part = sealed(args.out / 'batches' / f'{start:06d}.json')
        complete.extend(part['records'])
    if [r['uid'] for r in complete] != [r['uid'] for r in rows]:
        raise ValueError('semantic audit rows incomplete')
    counts = Counter()
    for r in complete:
        for reader in r['readers']:
            counts['ratings'] += 1
            counts['generated_tokens'] += reader['generated_tokens']
            if reader['rating'] is not None and reader['finish_reason'] == 'stop':
                counts['parsed_stop'] += 1
        a, b = r['readers']
        if all(x['rating'] is not None and x['finish_reason'] == 'stop' for x in (a,b)):
            counts['pair_covered'] += 1
            counts['start_agree'] += int(a['rating']['start'] == b['rating']['start'])
            counts['target_agree'] += int(a['rating']['target'] == b['rating']['target'])
    summary = {'schema': 'transition-LLM-audit-summary-v1',
               'binding_sha256': binding['sha256'], 'rows': len(rows),
               'counts': dict(counts),
               'scope': 'discovery-only, arm-blind same-model independent draws; no human semantic validation',
               'interpretation': 'Raw ratings only; detector confusion and support gates require a separate frozen CPU analysis'}
    atomic_json(args.out / 'SUMMARY.json', summary)
    print(json.dumps({'summary': str(args.out / 'SUMMARY.json'), 'rows': len(rows),
                      'parsed_stop': counts['parsed_stop'], 'of_ratings': counts['ratings']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
