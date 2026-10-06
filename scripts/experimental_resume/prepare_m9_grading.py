"""Prepare frozen M9 strict scoring and an arm-blind J1 bundle on CPU Slurm.

Native/capped endpoints reuse the sealed X1 score table when the identical
native answer stopped by 16k. Only newly generated closure answers and fixed
integer emissions enter a new strict/J1 bundle. No policy result is published
until every required J1 item has a verified verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = R / 'steering-v1'
MANIFEST = S / 'runs/m9-resume-v1/MANIFEST.json'
QUESTIONS = S / 'manifests/question-table-v1.json'
X1_LONG = S / 'runs/x1/score/long.parquet'
X1_LONG_MANIFEST = X1_LONG.with_name(X1_LONG.name + '.manifest.json')
EXPECTED_M9 = '63772b263e855692ecc5a37b10f634f96b30fa51e7e54bf22f5026f16e43024b'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed M9 grading input seal: ' + str(path))
    return value


def native_16k(row, x1):
    return bool(x1['acc32']) if row['native_endpoint_natural_stop'] else False


def comparator_native_or_fail(row, x1):
    status = row['integer_comparator']['status']
    if status in ('retain_native_finished', 'fallback_native_no_integer'):
        return native_16k(row, x1)
    if status in ('failed_source_prefix', 'failed_emission_budget'):
        return False
    if status == 'integer_emission':
        return None  # New answer; strict/J1 bundle supplies its outcome.
    raise ValueError('unknown integer comparator status: ' + status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True, help='sealed closure-results.json')
    parser.add_argument('--out', type=Path, required=True, help='new digest-bound grading directory')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('M9 strict grading requires CPU Slurm')
    if args.out.exists():
        raise FileExistsError('M9 grading output directory already exists')

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from moe_steer import engine, score

    manifest = sealed(MANIFEST)
    if manifest['sha256'] != EXPECTED_M9:
        raise ValueError('frozen M9 manifest differs')
    source = json.loads(args.generation.read_text())
    binding = source.get('binding', {})
    if binding.get('manifest_sha256') != manifest['sha256'] or len(source.get('records', [])) != 400:
        raise ValueError('M9 generation incomplete or rebound')
    expected = {row['uid'] for row in manifest['rows']}
    generated = {row['uid']: row for row in source['records']}
    if len(generated) != 400 or set(generated) != expected:
        raise ValueError('M9 generation lacks exactly all 400 assignments')
    if source.get('grading_status') != 'PENDING blinded strict+J1; no utility result before complete grading':
        raise ValueError('M9 generation result status differs')

    qtable = score.load_question_table(QUESTIONS)
    if qtable['split_sha256'] != manifest['split_sha256']:
        raise ValueError('question/gold table belongs to another split')
    long_meta = sealed(X1_LONG_MANIFEST)
    if long_meta['parquet_sha256'] != file_sha(X1_LONG):
        raise ValueError('native X1 scoring table bytes changed')
    x1_rows = pq.read_table(X1_LONG, columns=['uid', 'acc32', 'acc131']).to_pylist()
    x1 = {row['uid']: row for row in x1_rows}
    if len(x1) != len(x1_rows) or any(row['native_uid'] not in x1 for row in manifest['rows']):
        raise ValueError('X1 score table lacks native M9 references')
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)

    blind, map_rows = [], []
    for row in manifest['rows']:
        uid = row['uid']
        output = generated[uid]
        q = qtable['questions'][row['question']]
        native = x1[row['native_uid']]
        if q['dataset'] != row['question'].split('|', 1)[0]:
            raise ValueError('question dataset differs')
        closure_status = output['status']
        if row['prepared']['status'] == 'generate_closure':
            if closure_status != 'GENERATED':
                raise ValueError('generated M9 branch lacks a result: ' + uid)
            if (output.get('injection_tokens') != row['prepared']['injection_tokens'] or
                output['tokens_charged'] != len(output['completion_token_ids']) or
                output['tokens_charged'] > manifest['total_budget']):
                raise ValueError('closure token accounting differs: ' + uid)
            grade_uid = digest(['M9-blind-grade-v1', uid, 'closure'])
            blind.append({'uid': grade_uid, 'dataset': q['dataset'], 'problem': q['problem'],
                          'gold': q['gold'], 'content_text': tokenizer.decode(
                              output['completion_token_ids'], skip_special_tokens=False),
                          'finish_reason': output['finish_reason'],
                          'natural_stop': output['natural_stop']})
            closure_score = None
            map_rows.append({'grade_uid': grade_uid, 'M9_uid': uid, 'branch': 'closure'})
        elif row['prepared']['status'].startswith('retain_native'):
            if closure_status != row['prepared']['status']:
                raise ValueError('retained native status differs: ' + uid)
            closure_score = native_16k(row, native)
        elif row['prepared']['status'] == 'incomplete_prefix':
            if closure_status != 'FAILED_SOURCE_PREFIX':
                raise ValueError('missing-prefix status differs: ' + uid)
            closure_score = False
        else:
            raise ValueError('unknown closure preparation status')
        comparator_score = comparator_native_or_fail(row, native)
        if row['integer_comparator']['status'] == 'integer_emission':
            p = row['integer_comparator']
            if p['tokens_charged'] != len(p['completion_token_ids']) or p['tokens_charged'] > manifest['total_budget']:
                raise ValueError('integer emission token accounting differs')
            grade_uid = digest(['M9-blind-grade-v1', uid, 'integer'])
            blind.append({'uid': grade_uid, 'dataset': q['dataset'], 'problem': q['problem'],
                          'gold': q['gold'], 'content_text': tokenizer.decode(
                              p['completion_token_ids'], skip_special_tokens=False),
                          'finish_reason': 'stop', 'natural_stop': True})
            map_rows.append({'grade_uid': grade_uid, 'M9_uid': uid, 'branch': 'integer'})
        map_rows.append({'M9_uid': uid, 'native_uid': row['native_uid'],
                         'native16k_correct': native_16k(row, native),
                         'native_full_correct': bool(native['acc131']),
                         'closure_reused_correct': closure_score,
                         'integer_reused_correct': comparator_score,
                         'closure_status': closure_status,
                         'integer_status': row['integer_comparator']['status'],
                         'closure_tokens': output['tokens_charged'],
                         'integer_tokens': row['integer_comparator']['tokens_charged'],
                         'native16k_tokens': row['native_tokens_charged']})
    if len(blind) != (manifest['counts']['generate_closure'] +
                      manifest['integer_comparator_counts']['integer_emission']):
        raise ValueError('new blind grading count differs from frozen maximum')
    if len({r['uid'] for r in blind}) != len(blind):
        raise ValueError('duplicate blind scoring UID')
    score.assert_blind(blind, 'M9 new-answer grading input')
    strict = score.strict_rows(blind, workers=4)
    strict_by_uid = {row['uid']: row for row in strict}
    if len(strict_by_uid) != len(blind):
        raise ValueError('strict grade UIDs differ')
    items = score.build_j1_items(blind, strict_by_uid, corpus='M9-single-cut-v1')
    if len(items) > 267:
        raise ValueError('M9 J1 items exceed complete-stage maximum')
    score.assert_blind(items, 'M9 J1 items')

    args.out.mkdir(parents=True)
    input_info = score.write_jsonl(args.out / 'blind.jsonl.gz', blind)
    strict_info = score.write_jsonl(args.out / 'strict.jsonl.gz', strict)
    items_info = score.write_jsonl(args.out / 'items.jsonl', items)
    map_info = score.write_jsonl(args.out / 'arm-map.jsonl.gz', map_rows)
    body = {'schema': 'M9-blind-grading-preparation-v1', 'job_id': os.environ['SLURM_JOB_ID'],
            'M9_manifest_sha256': manifest['sha256'], 'generation_binding': binding,
            'generation_file_sha256': file_sha(args.generation),
            'question_table_sha256': qtable['sha256'],
            'X1_score_manifest_sha256': long_meta['sha256'],
            'X1_score_file_sha256': long_meta['parquet_sha256'],
            'code_sha256': file_sha(Path(__file__)),
            'strict_score_code_sha256': file_sha(Path(score.__file__)),
            'rows': 400, 'new_blind_answers': len(blind), 'J1_items': len(items),
            'counts': dict(Counter(row['closure_status'] for row in map_rows if 'closure_status' in row)),
            'files': {'blind': input_info, 'strict': strict_info, 'items': items_info,
                      'arm_map': map_info},
            'status': 'PENDING_BLIND_J1_AND_COMPLETE_POSTHOC_ANALYSIS',
            'policy': 'capped/failure answers count wrong; retained natural native answers reuse sealed X1 score',
            'integer_emission_rule': 'fixed complete answer emission is gradeable despite no generated EOS'}
    receipt = {**body, 'sha256': digest(body)}
    (args.out / 'GRADE_PREP.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'path': str(args.out / 'GRADE_PREP.json'), 'new_answers': len(blind),
                      'J1_items': len(items), 'status': receipt['status']}))


if __name__ == '__main__':
    main()
