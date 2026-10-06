"""Bounded GPU parity audit for a new direct seven-class sentence judge.

This is an LLM measurement check, not independent human validation and not the
online semantic trigger detector. Historical labels are excluded from prompts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FIXTURES = ROOT / 'steering-v1/runs/routing-control-v1/dense-judge-parity/FIXTURES.json'
PROGRAM = REPO / 'results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json'
PROGRAM_SHA = '467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def parse_label(text):
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    text = text.strip().replace('**', '')
    match = re.fullmatch(r'(?:Label\s*:\s*)?(Read|Analyze|Plan|Implement|Explore|Verify|Monitor)[.!\s]*', text, re.I)
    return next((name for name in CLASSES if name.lower() == match.group(1).lower()), None) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('dense judge audit requires GPU Slurm')
    if sha(PROGRAM) != PROGRAM_SHA:
        raise ValueError('pinned historical judge program changed')
    fixture = json.loads(FIXTURES.read_text())
    if fixture['sha256'] != digest({k: v for k, v in fixture.items() if k != 'sha256'}):
        raise ValueError('fixture seal mismatch')
    if len(fixture['fixtures']) != 200:
        raise ValueError('expected exactly 200 frozen discovery fixtures')
    if args.out.exists():
        raise FileExistsError('judge result directory already exists')
    args.out.mkdir(parents=True)
    instruction = json.loads(PROGRAM.read_text())['classify']['signature']['instructions']
    instruction += '\nReturn only the one label. Do not include an explanation.'
    inputs = []
    for row in fixture['fixtures']:
        values = row['inputs']
        if set(values) != {'problem_statement', 'previous_sentence', 'sentence', 'next_sentence'}:
            raise ValueError('judge input allowlist differs')
        content = '\n'.join((
            'Problem Statement: ' + values['problem_statement'],
            'Previous Sentence: ' + values['previous_sentence'],
            'Sentence: ' + values['sentence'],
            'Next Sentence: ' + values['next_sentence'],
            'Label:',
        ))
        inputs.append([{'role': 'system', 'content': instruction},
                       {'role': 'user', 'content': content}])
    started = time.monotonic()
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
              dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=49152,
              max_num_seqs=32, max_num_batched_tokens=8192,
              gpu_memory_utilization=.85, enforce_eager=True,
              generation_config='vllm', language_model_only=True,
              attention_config={'backend': 'FLASH_ATTN'})
    load_seconds = time.monotonic() - started
    sampling = SamplingParams(temperature=0., max_tokens=1024)
    records = []
    generation_start = time.monotonic()
    for offset in range(0, len(inputs), 32):
        outputs = llm.chat(inputs[offset:offset + 32], sampling_params=sampling,
                           chat_template_kwargs={'enable_thinking': True,
                                                 'reasoning_effort': 'low'}, use_tqdm=False)
        if len(outputs) != len(inputs[offset:offset + 32]):
            raise ValueError('judge output count mismatch')
        for row, output in zip(fixture['fixtures'][offset:offset + 32], outputs, strict=True):
            if len(output.outputs) != 1:
                raise ValueError('judge returned multiple completions')
            choice = output.outputs[0]
            records.append({'family': row['family'], 'identity': row['identity'],
                            'historical_label': row['historical_label'],
                            'new_label': parse_label(choice.text),
                            'finish_reason': choice.finish_reason,
                            'generated_tokens': len(choice.token_ids),
                            'raw_completion': choice.text})
        print(json.dumps({'completed': len(records), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    generation_seconds = time.monotonic() - generation_start
    covered = [r for r in records if r['new_label'] is not None and r['finish_reason'] == 'stop']
    confusion = defaultdict(Counter)
    for row in covered:
        confusion[row['historical_label']][row['new_label']] += 1
    result = {'schema': 'dense-judge-parity-v1', 'job_id': os.environ['SLURM_JOB_ID'],
              'fixtures_sha256': fixture['sha256'], 'program_sha256': PROGRAM_SHA,
              'driver_sha256': sha(__file__), 'model_path': str(MODEL),
              'protocol': 'direct offline vLLM chat; pinned program instructions; temperature 0; 1024 max; thinking low; new measurement, not DSPy prompt parity',
              'load_seconds': load_seconds, 'generation_seconds': generation_seconds,
              'total_seconds': time.monotonic() - started,
              'coverage': len(covered) / len(records),
              'agreement_among_covered': sum(r['new_label'] == r['historical_label'] for r in covered) / len(covered) if covered else None,
              'confusion': {a: dict(b) for a, b in confusion.items()},
              'total_generated_tokens': sum(r['generated_tokens'] for r in records),
              'records': records,
              'interpretation': 'LLM audit agreement with historical sparse labels; neither judge is human ground truth or a substantive-check rating.'}
    result['sha256'] = digest(result)
    (args.out / 'PARITY.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('coverage', 'agreement_among_covered', 'total_generated_tokens', 'total_seconds')}), flush=True)


if __name__ == '__main__':
    main()
