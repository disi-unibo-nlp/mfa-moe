"""Count native X3 512-token window label units; no intervention results or judge calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
PRICE = REPORT / 'X3_LABEL_DENSITY_PRICE_v1.json'
ELIGIBILITY = REPORT / 'X3_ELIGIBILITY_v1.json'
SELECTION = REPORT / 'X3_G3_SELECTION_v2.json'
OUT = REPORT / 'X3_LABEL_DENSITY_PROJECTION_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('X3 label density projection requires owner CPU Slurm')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '1':
        raise ValueError('X3 label density CPU shape differs')
    price, eligible, selection = sealed(PRICE), sealed(ELIGIBILITY), sealed(SELECTION)
    if price['source_sha256'] != sha(__file__) or price['eligibility_sha256'] != eligible['sha256']:
        raise ValueError('changed X3 label density source or eligibility')
    if price['selection_sha256'] != selection['sha256']:
        raise ValueError('changed G3 selection')
    for path, expected in price['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'changed input: {path}')
    from moe_steer import engine, manifests as M
    from moe_steer.label_units import HFTokenizer, THINK_END_ID, build_branch, problem_statement
    if engine.code_tree_sha256() != price['sampler_tree_sha256']:
        raise ValueError('capped sampler tree changed')
    world = M.load_world()
    tokenizer = HFTokenizer(Path(engine.snapshot_path()) / 'tokenizer.json')
    if tokenizer.sha256 != price['tokenizer_sha256']:
        raise ValueError('tokenizer changed')
    trace_store = M.TraceStore(cache_dir=S / 'runs/resume-v1/x3-cpu/trace-offsets')
    counts = []
    for row in eligible['rows']:
        if not row['eligible']:
            continue
        q = row['question']
        ref = world.infos[q]['prompt_ref']
        trace = trace_store.trace(ref['dataset'], ref['problem_id'])
        replay = trace['metadata']['token_replay']
        if digest(replay) != row['parent_replay_sha256']:
            raise ValueError(f'parent replay changed: {q}')
        ids = [int(t) for t in replay['completion_token_ids']]
        p = row['prefix_len']
        cap = (32768 - p) if row['long_endpoint'] else 1024
        if not 4097 <= p < len(ids) or cap < 1:
            raise ValueError(f'invalid native prefix or cap: {q}')
        selected = ids[:min(len(ids), p + cap)]
        think = selected.index(THINK_END_ID) if THINK_END_ID in selected else None
        fake = {'uid': digest({'native_x3_density': q}), 'question': q,
                'dataset': ref['dataset'], 'prefix_len': p,
                'completion_token_ids': selected[p:], 'reasoning_tokens': think,
                'finish_reason': 'length' if len(selected) == p + cap else 'stop'}
        units, roster = build_branch(fake, tokenizer=tokenizer,
                                     problem=problem_statement(trace),
                                     prefix_ids=selected[:p], segmentation='cumulative',
                                     window=512)
        if len(units) != roster['units'] or roster['window_reasoning_tokens'] > 512:
            raise ValueError(f'native label window mismatch: {q}')
        counts.append({'question': q, 'long_endpoint': row['long_endpoint'],
                       'native_512_units': len(units),
                       'native_window_reasoning_tokens': roster['window_reasoning_tokens'],
                       'native_suffix_tokens_used': len(selected) - p,
                       'parent_replay_sha256': row['parent_replay_sha256']})
    if len(counts) != 77 or sum(x['long_endpoint'] for x in counts) != 50:
        raise ValueError('X3 eligible native label inventory changed')
    n = sum(x['native_512_units'] for x in counts)
    body = {'schema': 'legacy-X3-native-label-density-projection-v1',
            'status': 'COMPLETE_NATIVE_PROXY', 'job_id': os.environ['SLURM_JOB_ID'],
            'price_sha256': price['sha256'], 'eligibility_sha256': eligible['sha256'],
            'selection_sha256': selection['sha256'],
            'tokenizer_sha256': tokenizer.sha256, 'window': 512,
            'native_sources': 77, 'native_units': n,
            'two_seed_five_arm_proxy_units': n * 10,
            'rows': counts,
            'interpretation': 'Native sample_00 sentence counts in the exact registered X3 scoring window; 10x multiplication is a planning proxy for two seeds and five arms, not treatment-invariant observed unit count. Reprice actual GEPA units after intervention generation.'}
    body['sha256'] = digest(body)
    if OUT.exists():
        if json.loads(OUT.read_text()) != body:
            raise ValueError('existing X3 label density version differs')
    else:
        tmp = OUT.with_name(OUT.name + '.part-' + os.environ['SLURM_JOB_ID'])
        tmp.write_text(json.dumps(body, indent=1) + '\n')
        tmp.replace(OUT)
    print(json.dumps({'output': str(OUT), 'native_sources': 77,
                      'native_units': n, 'two_seed_five_arm_proxy_units': n * 10}), flush=True)


if __name__ == '__main__':
    main()
