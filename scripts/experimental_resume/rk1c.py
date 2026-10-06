"""Clean, outcome-free RK1 rerun using a separately copied historical implementation.

Run under the frozen CPU sbatch wrapper, with one core and a four-core-hour ceiling.
No B cohort or outcome file is read. Input hashing and fits require a Slurm allocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
OUT = ROOT / 'reasoning-kinematics/rk1c'
HIST = ROOT / 'reasoning-kinematics/rk1'
POP = ROOT / 'forum/tests/null/manifests/population.json'
MODELS = ('gpt', 'qwen36', 'gemma', 'glm', 'nemotron', 'qwen330b', 'qwen35')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    """Bounded development only: copy reusable source, never change historical code."""
    dest = OUT / 'code/rk1'
    dest.mkdir(parents=True, exist_ok=True)
    for source in sorted((HIST / 'code/rk1').glob('*.py')):
        target = dest / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f'copied source changed: {target}')
        if not target.exists():
            shutil.copy2(source, target)
    runner = OUT / 'code/rk1c.py'
    if Path(__file__).resolve() != runner.resolve():
        shutil.copy2(__file__, runner)
    return {'copied_code': str(dest), 'driver': str(runner), 'submission': 'NOT SUBMITTED'}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('clean population analysis requires a CPU Slurm allocation')
    if int(os.environ.get('OMP_NUM_THREADS', '0')) != 1:
        raise ValueError('registered analysis uses one CPU thread')
    started = time.monotonic()
    sys.path.insert(0, str(OUT / 'code'))
    import numpy as np
    from rk1 import common as C, describe as D, sparse as S

    C.OUT = OUT  # This driver invokes no historical run/assemble writer.
    population = C.verify_json(POP)
    allowed = set(population['gpt']['A']['questions'])
    if len(allowed) != 522 or any(C.load_split()[q] not in ('dev', 'tune') for q in allowed):
        raise ValueError('clean canonical population or confirm exclusion differs')
    for model in MODELS:
        if set(population[model]['A']['questions']) != allowed:
            raise ValueError('model aliases disagree on the registered clean IDs')
    paths = [POP, C.SPLIT_PATH]
    for model in MODELS:
        paths.extend([C.V3R2 / model / 'A/attempts.parquet',
                      C.DYN / model / 'A/sentences.parquet',
                      C.DYN / model / 'A/extract_meta.csv'])
    dense = [HIST / 'results/dense.json', HIST / 'results/dense_sensitivity.json']
    paths.extend(dense)
    inputs = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}
    frozen = {'version': 'rk1c-v1', 'population': sorted(allowed), 'inputs': inputs,
        'code': {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT / 'code').rglob('*.py'))},
        'seed': C.SEED, 'n_boot': 1000, 'n_reversible_null': 200,
        'cohort': 'A only, exact-ID-clean dev+tune across all seven models',
        'outcomes': 'excluded', 'cpu_core_hour_ceiling': 4,
        'dense_reuse': 'unchanged non-confirm 16-trace outputs'}
    binding = OUT / 'FROZEN.json'
    if binding.exists():
        previous = C.verify_json(binding)
        if {k: v for k, v in previous.items() if k != 'sha256'} != frozen:
            raise ValueError('rk1c frozen inputs/code changed; requires a new version')
    else:
        C.write_json(binding, frozen)
    results = OUT / 'results'
    results.mkdir(exist_ok=True)
    for source in dense:
        C.verify_json(source)
        target = results / source.name
        if target.exists() and sha(target) != sha(source):
            raise ValueError('dense result reuse mismatch')
        if not target.exists():
            shutil.copy2(source, target)
    # Filtering load_attempts happens before sentences, profiles, bootstraps or any fit.
    historical_load = S.load_attempts
    def clean_attempts(model, cohort):
        if cohort != 'A':
            raise ValueError('B cohorts are excluded from rk1c')
        att, checks = historical_load(model, cohort)
        att = att[att['question'].isin(allowed)].reset_index(drop=True)
        if set(att['question']) != allowed:
            raise ValueError('clean population absent from a model input')
        return att, {**checks, 'all_models_confirm_excluded': True}
    S.load_attempts = clean_attempts
    out = {'cohorts': {}}
    for i, model in enumerate(MODELS):
        path = results / f'{model}-descriptive.json'
        if path.exists():
            item = C.verify_json(path)
            if item['frozen_sha256'] != C.verify_json(binding)['sha256']:
                raise ValueError('completed model result has a stale freeze binding')
            out['cohorts'][model + '/A'] = item['result']
            continue
        coh = S.load_cohort(model, 'A')
        rng = np.random.default_rng(C.SEED + i)
        result = D.describe_cohort(coh, rng, 1000)
        kept = coh.pairs[~coh.attempts['capped'].to_numpy(bool)[coh.pairs['attempt'].to_numpy()]].copy()
        att = coh.attempts[~coh.attempts['capped']].copy()
        qcode, qs = S.question_codes(kept, att)
        result['capped_excluded'] = {'n_questions': len(qs), 'n_pairs': len(kept),
            **D.describe_pairs(kept, qcode, len(qs), rng, 1000)}
        C.write_json(path, {'frozen_sha256': C.verify_json(binding)['sha256'], 'result': result})
        out['cohorts'][model + '/A'] = result
        C.log(f'{model}: clean descriptive complete ({len(coh.attempts)} attempts)')
    out['meta'] = {'stage': 'sparse-clean', 'seed': C.SEED,
        'frozen_sha256': C.verify_json(binding)['sha256'], 'slurm_job': os.environ['SLURM_JOB_ID'],
        'seconds': time.monotonic() - started, 'historical_confirm_exposure': True}
    C.write_json(results / 'sparse_descriptive.json', out)
    # Same top-level schema, with absent outcome/B stages explicitly declared.
    C.write_json(OUT / 'results.json', {'definition': {
        'population': 'A dev+tune 522 questions across every model; historical confirm exposure disclosed',
        'B': 'excluded', 'outcomes': 'OWNED_BY_R3-B; absent at this stage'},
        'input_sha256': inputs, 'parts': {'sparse_descriptive': C.verify_json(results / 'sparse_descriptive.json'),
        'dense': C.verify_json(results / 'dense.json'),
        'dense_sensitivity': C.verify_json(results / 'dense_sensitivity.json')},
        'provenance': out['meta']})
    historical = C.verify_json(HIST / 'results/sparse_descriptive.json')['cohorts']
    lines = ['# RK1 clean versus historical descriptive summaries', '',
        'Exact-ID-clean dev+tune across every model. Historical confirm exposure remains disclosed.', '',
        '| Model | Metric | Clean estimate [95% CI] | Historical estimate [95% CI] |',
        '|---|---|---|---|']
    for model in MODELS:
        for key in ('H', 'switch_rate', 'spectral_gap', 'EP', *('persist_' + c for c in C.CLASSES)):
            cells = []
            for item in (out['cohorts'][model + '/A'], historical[model + '/A']):
                metric = item['weighted']['metrics'].get(key)
                cells.append('unavailable' if metric is None else
                    f"{metric['est']:.5g} [{metric['lo']:.5g}, {metric['hi']:.5g}]")
            lines.append(f'| {model} | {key} | {cells[0]} | {cells[1]} |')
    lines += ['', 'Matrices, position profiles, flux loops and both reversible nulls are in results.json.',
              'Sparse labels are joined only when sentence indices and segments are contiguous.',
              'No outcome contrast or clean seven-model predictive claim follows from this stage.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        print(json.dumps(prepare(), indent=1))
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit('usage: rk1c.py [--prepare]')
