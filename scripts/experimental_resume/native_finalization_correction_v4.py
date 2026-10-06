"""Versioned correction of qualification-only failure; original seals retained."""
import argparse
import os
from pathlib import Path

import native_finalization_v1 as N

PARENT_DOC = N.DOC / 'correction-v3'
PARENT_ROOT = N.ROOT / 'correction-v3'
EXTRA = ('native_finalization_worker_v2.py', 'run_native_finalization_v4.py',
    'native_finalization_correction_v4.py', 'native_finalization_native_dispatch_v6.py',
    'native_finalization_cpu_v5.sbatch', 'run_native_finalization_v5.sbatch',
    'launch_native_finalization_v6.sh', 'native_finalization_health_v1.py', 'native_finalization_replay_v3.py', 'native_finalization_qualification_v4.py')


def activate():
    if N.DOC == PARENT_DOC.parent / 'correction-v4':
        return
    original_sources = N.sources
    N.DOC = PARENT_DOC.parent / 'correction-v4'
    N.ROOT = PARENT_ROOT.parent / 'correction-v4'

    def corrected_sources():
        paths = [Path(__file__).parent / name for name in EXTRA]
        paths += [N.U.REPO / 'tests/experimental_resume/test_native_finalization_v2.py',
                  PARENT_DOC / 'MANIFEST.json', N.DOC / 'CORRECTION.json']
        prior = N.U.sealed(PARENT_DOC / 'MANIFEST.json')['binding']['sources']
        N.require(all(N.U.file_sha(Path(p)) == h for p, h in prior.items()), 'prior sealed source changed')
        paths.extend([N.U.REPO / 'tests/experimental_resume/test_native_finalization_replay_v3.py', N.U.REPO / 'tests/experimental_resume/test_native_finalization_qualification_v4.py'])
        replay_path = PARENT_ROOT / 'allocations/59428024/ENGINEERING_REPLAY.json'
        replay = N.U.sealed(replay_path)
        paths += [replay_path, *[Path(r['path']) for r in replay['files'].values()],
                  *[Path(r['path']) for r in replay['captures']]]
        evidence = N.DOC / 'QUALIFICATION_CORRECTION_EVIDENCE.json'
        if evidence.exists():
            paths.append(evidence)
        return {**original_sources(), **prior, **{str(p): N.U.file_sha(p) for p in paths}}
    N.sources = corrected_sources
    import native_finalization_operations_v1 as Ops
    Ops.SUBMISSIONS = N.DOC / 'submissions'
    current_accounting = Ops.accounting

    def accounting_with_ancestor():
        report = current_accounting()
        evidence = N.U.sealed(N.DOC / 'CORRECTION.json')['parent_accounting']
        parent = N.U.sealed(Path(evidence['path']))
        N.require(parent['sha256'] == evidence['sha256'] and
                  all(not r['provisional'] for r in parent['allocations']), 'ancestor costs are unfinished')
        current_ids = {r['job_id'] for r in report['jobs']}
        for r in parent['allocations']:
            if r['job_id'] in current_ids:
                continue
            report['jobs'].append({'job_id': r['job_id'], 'name': 'preserved-parent:' + r['name'],
                'state': r['state'], 'exit_code': r['exit_code'], 'elapsed_seconds': r['elapsed_seconds'],
                'allocated_tres': r['allocation'], 'allocated_GPU_hours': r['GPU_hours'],
                'billing_core_hours': r['billing_core_hours'], 'provisional': False, 'raw': r['sacct']})
        report['allocated_GPU_hours_so_far'] = sum(r.get('allocated_GPU_hours', 0) for r in report['jobs'])
        report['billing_core_hours_so_far'] = sum(r.get('billing_core_hours') or 0 for r in report['jobs'])
        report['scope'] = 'This corrected cycle plus all preserved preparation and failed engineering allocations; no duplicate jobs.'
        report['ancestor_checkpoint_sha256'] = parent['sha256']
        return report
    Ops.accounting = accounting_with_ancestor


def correction():
    parent = N.U.sealed(PARENT_DOC / 'MANIFEST.json')
    index = N.U.sealed(PARENT_ROOT / 'INDEX.json')
    N.require(index['manifest_sha256'] == parent['sha256'] and
        all(r['status'] == 'MISSING' and r['attempt_path'] is None for r in index['records']),
        'recovery cannot replace a scientific execution or an uncertain attempt')
    for p, h in parent['binding']['sources'].items():
        N.require(N.U.file_sha(Path(p)) == h, 'parent sealed source changed')
    checkpoint_path = sorted(PARENT_DOC.glob('CHECKPOINT-*.json'))[-1]
    checkpoint = N.U.sealed(checkpoint_path)
    N.require(checkpoint['manifest_sha256'] == parent['sha256'] and
              all(not r['provisional'] for r in checkpoint['allocations']), 'parent costs not final')
    return N.save(N.DOC / 'CORRECTION.json', {
        'schema': 'native-finalization-execution-correction-v4',
        'parent_manifest_sha256': parent['sha256'], 'parent_index_sha256': index['sha256'],
        'failed_generation_job': '59428024', 'parent_measurement_job': '59428026',
        'failure': 'Cross-execution route-equality criterion rejected identical token replay even between two uninstrumented native runs.',
        'evidence': 'Sealed ENGINEERING_REPLAY: all three 64-token outputs identical; native route slots differed by 2829 and instrumented by 3455. Both direct TP captures exactly match same-execution returned routes, with zero ID or weight disagreement between ranks.',
        'correction': 'Keep exact token replay, complete aligned same-execution native ID capture, normalized executed weights and TP rank agreement. Remove cross-run route equality, which native controls demonstrably fail. Preserve every replay array and report route variation.',
        'affected_scientific_results': [], 'missing_original_assignments': 32,
        'parent_accounting': {'path': str(checkpoint_path), 'sha256': checkpoint['sha256'],
            'billing_core_hours': checkpoint['observed_billing_core_hours'],
            'GPU_hours': checkpoint['observed_GPU_hours']},
        'design': 'Same first eight frozen families, seeds 0/1, two prompts, sampler, native routing and cap; corrected execution bindings are separate.',
        'preserved_parent_directory': str(PARENT_DOC), 'automatic_downstream_experiments': False, 'parent_engineering_replay': str(PARENT_ROOT / 'allocations/59428024/ENGINEERING_REPLAY.json'), 'parent_engineering_replay_sha256': '3fc21b7968a7c4857a39049fea83176747291012ae9a037d3c59e5a254236193'})


def prepare():
    import native_finalization_operations_v1 as Ops
    correction()
    test_python = N.U.REPO / '.venv-native-finalization-tests/bin/python'
    test = 'tests/experimental_resume/test_native_finalization_qualification_v4.py'
    output = Ops.command([str(test_python), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--basetemp=' + os.environ['TMPDIR'] + '/pytest-correction', 'tests/experimental_resume/test_native_finalization_v2.py', 'tests/experimental_resume/test_native_finalization_replay_v3.py', test])
    N.save(N.DOC / 'CORRECTION_TESTS.json', {'schema': 'native-finalization-correction-tests-v4',
        'stdout': output, 'test_source_sha256': N.U.file_sha(N.U.REPO / test)})
    import numpy as np
    replay_path = PARENT_ROOT / 'allocations/59428024/ENGINEERING_REPLAY.json'
    replay = N.U.sealed(replay_path)
    N.require(replay['sha256'] == '3fc21b7968a7c4857a39049fea83176747291012ae9a037d3c59e5a254236193',
              'parent engineering evidence changed')
    for r in replay['files'].values():
        N.require(N.U.file_sha(Path(r['path'])) == r['file_sha256'], 'parent replay array changed')
    with np.load(replay['files']['instrumented']['path']) as data:
        api_routes = data['native_ids'].copy()
    from run_native_finalization_v1 import route_agreement
    agreement = route_agreement(replay['captures'], api_routes)
    from native_finalization_qualification_v4 import qualify
    checks = qualify(replay, agreement)
    N.save(N.DOC / 'QUALIFICATION_CORRECTION_EVIDENCE.json', {
        'schema': 'native-finalization-qualification-evidence-v4',
        'parent_engineering_replay_sha256': replay['sha256'], 'agreement': agreement, 'checks': checks,
        'native_repeat': replay['native_repeat'], 'instrumented_repeat': replay['instrumented_repeat'],
        'excluded_from_outcomes': True})
    Ops.prepare()
    current = N.U.sealed(N.DOC / 'MANIFEST.json'); parent = N.U.sealed(PARENT_DOC / 'MANIFEST.json')
    project = lambda rows: [(r['assignment']['family'], r['assignment']['question'],
        r['assignment']['seed'], r['assignment']['arm'], r['prompt_token_ids']) for r in rows]
    N.require(project(current['rows']) == project(parent['rows']) and
        current['binding']['sampler'] == parent['binding']['sampler'], 'experimental design drifted')
    N.save(N.DOC / 'DESIGN_RECONCILIATION.json', {'schema': 'native-finalization-corrected-design-v2',
        'parent_manifest_sha256': parent['sha256'], 'corrected_manifest_sha256': current['sha256'],
        'same_prompts_families_seeds_order_sampler': True, 'assignment_count': 32,
        'assignment_lineage': [{'parent_uid': a['assignment']['uid'], 'corrected_uid': b['assignment']['uid']}
            for a, b in zip(parent['rows'], current['rows'], strict=True)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'measure', 'finalize'))
    args = parser.parse_args(); activate()
    import native_finalization_operations_v1 as Ops
    {'prepare': prepare, 'measure': Ops.measure, 'finalize': Ops.finalize}[args.phase]()
