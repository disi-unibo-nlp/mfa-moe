"""Prepare the same four discovery-family engineering fixtures on CPU."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('qualification prefix indexing requires CPU Slurm')
    from moe_steer import manifests as M, qualify as Q
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    world = M.load_world()
    fixtures = Q.Fixtures(world)
    family = json.loads((REPO/'report/experimental-resume-v1/family-freeze.json').read_text())
    allowed_families = set(family['new_parent_pools']['parent_pools']['discovery'])
    qfamily = {q: f for f, qs in family['new_parent_pools']['families'].items() for q in qs}
    candidates = [a for a in fixtures.attempts() if qfamily.get(a['question']) in allowed_families and a['n_reasoning'] >= 8192]
    candidates.sort(key=lambda a: hashlib.sha256(('ordered-qual-v1|'+str(a['attempt_id'])).encode()).hexdigest())
    picked, seen = [], set()
    for a in candidates:
        if qfamily[a['question']] in seen:
            continue
        trace = fixtures.load_trace(a)
        picked.append({'question': trace.question, 'family': qfamily[trace.question],
            'prompt_ids': trace.prompt_ids, 'completion_ids': trace.completion_ids[:2048]})
        seen.add(qfamily[a['question']])
        if len(picked) == 4:
            break
    if len(picked) != 4:
        raise ValueError('fewer than four fixed discovery-family fixtures; no replacement pool')
    prefill = sum((len(r['prompt_ids'])+2048)*4 for r in picked) + len(picked[0]['prompt_ids'])+2049
    value = {'schema': 'ordered-engineering-prefixes-v1', 'family_freeze_sha256': family['sha256'],
        'rows': picked, 'maximum_requests': 17, 'maximum_decode_tokens': 16512,
        'prefill_tokens': prefill, 'job_id': os.environ['SLURM_JOB_ID'],
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection': 'identical fixed order and >=8192 native-reasoning evaluability screen as original qualification; no future fields in fixture inputs',
        'semantic_eligibility': 'UNQUALIFIED; engineering fixture selection is not study enrollment'}
    value['sha256'] = digest(value)
    out = R/'steering-v1/runs/ordered-qualification-v1/CPU_PREFIXES.json'
    if out.exists():
        raise ValueError('CPU qualification prefixes already exist; never overwrite silently')
    out.write_text(json.dumps(value, separators=(',', ':'))+'\n')
    print(json.dumps({'path': str(out), 'sha256': value['sha256'], 'prefill_tokens': prefill, 'maximum_decode_tokens': 16512}))


if __name__ == '__main__':
    run()
