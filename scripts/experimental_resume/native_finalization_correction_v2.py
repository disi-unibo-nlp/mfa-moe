"""Versioned correction of qualification-only failure; original seals retained."""
import argparse
import os
from pathlib import Path

import native_finalization_v1 as N

PARENT_DOC = N.DOC
PARENT_ROOT = N.ROOT
EXTRA = ('native_finalization_worker_v2.py', 'run_native_finalization_v2.py',
    'native_finalization_correction_v2.py', 'native_finalization_native_dispatch_v4.py',
    'native_finalization_cpu_v3.sbatch', 'run_native_finalization_v3.sbatch',
    'launch_native_finalization_v4.sh', 'native_finalization_health_v1.py')


def activate():
    if N.DOC == PARENT_DOC / 'correction-v2':
        return
    original_sources = N.sources
    N.DOC = PARENT_DOC / 'correction-v2'
    N.ROOT = PARENT_ROOT / 'correction-v2'

    def corrected_sources():
        paths = [Path(__file__).parent / name for name in EXTRA]
        paths += [N.U.REPO / 'tests/experimental_resume/test_native_finalization_v2.py',
                  PARENT_DOC / 'MANIFEST.json', N.DOC / 'CORRECTION.json']
        return {**original_sources(), **{str(p): N.U.file_sha(p) for p in paths}}
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
        'schema': 'native-finalization-execution-correction-v2',
        'parent_manifest_sha256': parent['sha256'], 'parent_index_sha256': index['sha256'],
        'failed_generation_job': '59418248', 'parent_measurement_job': '59418250',
        'failure': 'Frozen validator import installed a class forward requiring an absent steering controller.',
        'evidence': 'Reference native fixture succeeded; captured fixture failed before qualification and before all scientific attempts.',
        'correction': 'Pin the native bound instance forward before importing the same frozen validators; retain all capture and engineering checks.',
        'affected_scientific_results': [], 'missing_original_assignments': 32,
        'parent_accounting': {'path': str(checkpoint_path), 'sha256': checkpoint['sha256'],
            'billing_core_hours': checkpoint['observed_billing_core_hours'],
            'GPU_hours': checkpoint['observed_GPU_hours']},
        'design': 'Same first eight frozen families, seeds 0/1, two prompts, sampler, native routing and cap; corrected execution bindings are separate.',
        'preserved_parent_directory': str(PARENT_DOC), 'automatic_downstream_experiments': False})


def prepare():
    import native_finalization_operations_v1 as Ops
    correction()
    test_python = N.U.REPO / '.venv-native-finalization-tests/bin/python'
    test = 'tests/experimental_resume/test_native_finalization_v2.py'
    output = Ops.command([str(test_python), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--basetemp=' + os.environ['TMPDIR'] + '/pytest-correction', test])
    N.save(N.DOC / 'CORRECTION_TESTS.json', {'schema': 'native-finalization-correction-tests-v2',
        'stdout': output, 'test_source_sha256': N.U.file_sha(N.U.REPO / test)})
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
