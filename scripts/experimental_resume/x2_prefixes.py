"""Prepare exactly the existing X2 branch prefixes on CPU, before a native NLL load."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('trace indexing and prefix preparation require CPU Slurm')
    from moe_steer import engine, manifests as M, results as RS
    frozen = json.loads((S/'runs/x2-resume-v1/FROZEN.json').read_text())
    manifest = M.load_manifest(frozen['manifest'])
    records = list(RS.read_shard_records(frozen['output'], 0))
    by_uid = {r['uid']: r for r in records}
    if len(by_uid) != 6630 or set(by_uid) != {r['uid'] for r in manifest['requests']}:
        raise ValueError('prefix preparation requires complete unique X2 generation')
    out = S/'runs/x2-resume-v1/measurement-prefixes'
    out.mkdir(exist_ok=True)
    traces = M.TraceStore(cache_dir=out/'trace-offsets')
    questions = {}
    positions = 0
    for request in manifest['requests']:
        record = by_uid[request['uid']]
        RS.validate_result(record, request)
        if record['code_tree'] != manifest['code_tree'] or record['manifest_sha256'] != manifest['sha256']:
            raise ValueError('generation input binding differs')
        question, parent = request['question'], request['parent']
        base = manifest['questions'][question]['prompt_token_ids']
        if question not in questions:
            ref = parent['trace_ref']
            parent_ids = traces.completion_ids(ref['dataset'], ref['problem_id'])
            prefix = M.branch_prompt(base, parent_ids, parent['prefix_len'])
            questions[question] = {'parent': parent, 'original_prompt_token_ids': base,
                'prefix_token_ids': prefix,
                'parent_completion_token_ids_sha256': hashlib.sha256(json.dumps(parent_ids,
                    separators=(',', ':')).encode()).hexdigest()}
        if questions[question]['parent'] != parent or questions[question]['original_prompt_token_ids'] != base:
            raise ValueError('X2 question uses inconsistent prefixes')
        if request['arm'] in ('N', 'E+', 'E-'):
            positions += len(questions[question]['prefix_token_ids']) + min(256, len(record['completion_token_ids']))
    value = {'schema': 'x2-measurement-prefixes-v1', 'manifest_sha256': manifest['sha256'],
        'code_tree': engine.code_tree_sha256(), 'job_id': os.environ['SLURM_JOB_ID'],
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'questions': questions, 'teacher_forced_positions': positions, 'assigned_N_E_sequences': 3354,
        'method_change': False, 'future_completion_fields_used_for_prefix': False}
    destination = out/'PREFIXES.json'
    if destination.exists():
        raise ValueError('prepared prefix artifact exists; inspect it instead of overwriting')
    destination.write_text(json.dumps(value, separators=(',', ':'))+'\n')
    print(json.dumps({'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'questions': len(questions), 'teacher_forced_positions': positions}))


if __name__ == '__main__':
    run()
