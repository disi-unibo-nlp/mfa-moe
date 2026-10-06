"""Checkpointed seven-class labels for frozen, contiguous discovery sentences.

This is an offline LLM audit. The lookahead field is never used by the online
controller. The parity gate is checked before loading the model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from qualify_dense_judge import MODEL, PROGRAM, PROGRAM_SHA, CLASSES, parse_label, sha, digest


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
UNITS = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'
PARITY = ROOT / ('steering-v1/runs/routing-control-v1/dense-judge-parity/'
                 'results-8e939bee-2341bac1/PARITY.json')
BATCH = 32


def load_sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'seal mismatch: {path}')
    return value


def messages(row, instruction):
    values = row['inputs']
    if set(values) != {'problem_statement', 'previous_sentence', 'sentence', 'next_sentence'}:
        raise ValueError('offline judge input fields changed')
    content = '\n'.join((
        'Problem Statement: ' + values['problem_statement'],
        'Previous Sentence: ' + values['previous_sentence'],
        'Sentence: ' + values['sentence'],
        'Next Sentence: ' + values['next_sentence'],
        'Label:',
    ))
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': content}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('dense labels require GPU Slurm')
    if sha(PROGRAM) != PROGRAM_SHA or not MODEL.is_dir():
        raise ValueError('judge model/program changed')
    parity = load_sealed(PARITY)
    if parity.get('schema') != 'dense-judge-parity-v1' or parity.get('program_sha256') != PROGRAM_SHA:
        raise ValueError('judge parity binding changed')
    covered = [r for r in parity['records'] if r['new_label'] in CLASSES and r['finish_reason'] == 'stop']
    if len(parity['records']) != 200 or len(covered) < 190 or sum(
            r['new_label'] == r['historical_label'] for r in parity['records']) < 140:
        raise ValueError('prospective direct-judge qualification failed')
    units = load_sealed(UNITS)
    if units.get('schema') != 'dense-discovery-units-v1' or units['sentences'] != len(units['records']):
        raise ValueError('dense input schema/count mismatch')
    if units['attempts'] != 48 or len({r['family'] for r in units['records']}) != 48:
        raise ValueError('discovery parent families differ')
    binding = {'schema': 'dense-discovery-label-binding-v1',
               'units_sha256': units['sha256'], 'parity_sha256': parity['sha256'],
               'judge_program_sha256': PROGRAM_SHA, 'driver_sha256': sha(__file__),
               'model_path': str(MODEL), 'batch_size': BATCH,
               'prompt': 'same direct offline chat formatting as parity v1'}
    binding['sha256'] = digest(binding)
    expected = f"results-{units['sha256'][:8]}-{binding['driver_sha256'][:8]}"
    if args.out.name != expected or args.out.parent != UNITS.parent:
        raise ValueError('output directory must bind units and driver digests')
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / 'BINDING.json'
    if meta.exists():
        if load_sealed(meta) != binding:
            raise ValueError('existing output binding differs')
    else:
        meta.write_text(json.dumps(binding, indent=1) + '\n')
    instruction = json.loads(PROGRAM.read_text())['classify']['signature']['instructions']
    instruction += '\nReturn only the one label. Do not include an explanation.'
    n = len(units['records'])
    checkpoints = args.out / 'batches'
    checkpoints.mkdir(exist_ok=True)
    missing = []
    for start in range(0, n, BATCH):
        path = checkpoints / f'{start:06d}.json'
        if not path.exists():
            missing.append(start)
            continue
        value = load_sealed(path)
        if value['binding_sha256'] != binding['sha256'] or value['start'] != start or value['stop'] != min(start+BATCH, n):
            raise ValueError(f'checkpoint mismatch at {start}')
        if len(value['records']) != value['stop']-value['start']:
            raise ValueError(f'checkpoint length mismatch at {start}')
    started = time.monotonic()
    if missing:
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                  dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=49152,
                  max_num_seqs=32, max_num_batched_tokens=8192,
                  gpu_memory_utilization=.85, enforce_eager=True,
                  generation_config='vllm', language_model_only=True,
                  attention_config={'backend': 'FLASH_ATTN'})
        print(json.dumps({'load_seconds': time.monotonic()-started,
                          'pending_batches': len(missing), 'total_batches': (n+BATCH-1)//BATCH}), flush=True)
        sampling = SamplingParams(temperature=0., max_tokens=1024)
        for start in missing:
            rows = units['records'][start:start+BATCH]
            outputs = llm.chat([messages(row, instruction) for row in rows],
                               sampling_params=sampling,
                               chat_template_kwargs={'enable_thinking': True,
                                                     'reasoning_effort': 'low'}, use_tqdm=False)
            if len(outputs) != len(rows):
                raise ValueError('output count mismatch')
            records = []
            for row, output in zip(rows, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise ValueError('multiple judge completions')
                choice = output.outputs[0]
                records.append({'family': row['family'], 'attempt_id': row['attempt_id'],
                                'sentence_index': row['sentence_index'],
                                'label': parse_label(choice.text),
                                'finish_reason': choice.finish_reason,
                                'generated_tokens': len(choice.token_ids),
                                'raw_completion': choice.text})
            value = {'schema': 'dense-discovery-label-batch-v1',
                     'binding_sha256': binding['sha256'], 'start': start,
                     'stop': start + len(rows), 'job_id': os.environ['SLURM_JOB_ID'],
                     'records': records}
            value['sha256'] = digest(value)
            path = checkpoints / f'{start:06d}.json'
            partial = path.with_suffix('.pending')
            partial.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
            partial.replace(path)
            print(json.dumps({'completed': start+len(rows), 'of': n,
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
    result = []
    for start in range(0, n, BATCH):
        batch = load_sealed(checkpoints / f'{start:06d}.json')
        result.extend(batch['records'])
    if len(result) != n or any((a['family'], a['attempt_id'], a['sentence_index']) !=
        (b['family'], b['attempt_id'], b['sentence_index']) for a,b in zip(result, units['records'], strict=True)):
        raise ValueError('final label/offset ownership mismatch')
    labels = Counter(row['label'] or 'UNPARSED' for row in result)
    family_coverage = defaultdict(lambda: [0,0])
    for row in result:
        family_coverage[row['family']][0] += 1
        family_coverage[row['family']][1] += int(row['label'] in CLASSES and row['finish_reason']=='stop')
    summary = {'schema': 'dense-discovery-labels-v1', 'binding_sha256': binding['sha256'],
               'job_id': os.environ['SLURM_JOB_ID'], 'sentences': n,
               'class_counts': dict(labels), 'family_coverage': dict(family_coverage),
               'parsed_stop_coverage': sum(row['label'] in CLASSES and row['finish_reason']=='stop' for row in result)/n,
               'generated_tokens': sum(row['generated_tokens'] for row in result),
               'elapsed_seconds': time.monotonic()-started,
               'status': 'COMPLETE_LLM_AUDIT',
               'limitations': 'Direct Qwen class audit, not independent semantic validation or online detection'}
    summary['sha256'] = digest(summary)
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({'complete': True, 'sentences': n, 'coverage': summary['parsed_stop_coverage']}), flush=True)


if __name__ == '__main__':
    main()
