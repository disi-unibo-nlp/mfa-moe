"""Finite native login dispatch; CPU jobs produce sealed readiness only.

Corrects compute-node saldo absence observed in job 59417057. This follows
utility_native_orchestration_v3 and never relies on compute-to-login SSH.
"""
import argparse
import json
import os
import socket

import native_finalization_correction_v4 as C
C.activate()
import native_finalization_operations_v1 as Ops
N = Ops.N


def login_guard():
    import getpass
    N.require(getpass.getuser() == 'lmolfett' and socket.gethostname().startswith('login') and
        socket.gethostname().endswith('.leonardo.local') and not os.environ.get('SLURM_JOB_ID'),
        'finite dispatch requires native lmolfett login shell')


def ready(path):
    r = N.U.sealed(path)
    N.require(r['schema'] == 'native-finalization-readiness-v3' and
              r['status'] == 'READY_FOR_NATIVE_LOGIN_DISPATCH', 'readiness missing')
    raw = Ops.command(['sacct', '-X', '-nP', '-j', r['producer_job_id'], '--format=JobIDRaw,State,ExitCode'])
    rows = [s.split('|') for s in raw.splitlines() if s.split('|')[0] == r['producer_job_id']]
    N.require(len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0'], 'readiness producer not successful')
    return r


def preflight(required):
    snapshot = Ops.live()
    queue = Ops.command(['squeue', '-h', '-A', 'iscrc_miosr', '-o', '%100i|%T|%l|%L|%C|%b|%D'])
    from dispatch_utility_production_v1 import remaining_commitments
    budget = remaining_commitments(snapshot['balance'], queue)
    N.require(required <= budget['uncommitted_reported_billing_core_hours'], 'live account cannot fit immediate allocation')
    return N.save(N.DOC / ('BUDGET-' + snapshot['sha256'][:16] + '.json'), {
        'schema': 'native-finalization-live-budget-v3', 'snapshot_sha256': snapshot['sha256'],
        'queue': queue, 'budget': budget, 'required_immediate_billing_core_hours': required,
        'scope': 'Immediate authorized allocations only; no new quota subdivision or future campaign forecast.'})


def launch_preparation():
    C.correction()
    preflight(2.)
    return Ops.submit('preparation-qualification-v4', [str(Ops.SCRIPTS / 'native_finalization_cpu_v5.sbatch'), 'prepare'],
        {'scope': 'authorized cycle; repair demonstrated cross-run route-equality false rejection; retain same-execution capture gates', 'source_files': N.sources()})


def dispatch_generation():
    r = ready(N.DOC / 'PREPARED.json')
    manifest = N.U.sealed(N.DOC / 'MANIFEST.json'); N.validate(manifest)
    N.require(r['manifest_sha256'] == manifest['sha256'] and r['source_files'] == N.sources(), 'generation readiness changed')
    # Prior GPU allocations consumed 2006 seconds; this request keeps the
    # complete generation cycle below the original 28800-second allowance.
    preflight(16 * (7 + 20 / 60) + 2)
    binding = {'manifest_sha256': manifest['sha256'], 'source_files': N.sources(), 'readiness_sha256': r['sha256']}
    job = Ops.submit('generation', ['--time=07:20:00', str(Ops.SCRIPTS / 'run_native_finalization_v5.sbatch')], binding)
    cpu = Ops.submit('measurement', ['--dependency=afterany:' + job, '--job-name=native-finalization-measure',
        str(Ops.SCRIPTS / 'native_finalization_cpu_v5.sbatch'), 'measure'], binding)
    return N.save(Ops.SUBMISSIONS / 'GENERATION_CHAIN.json', {'schema': 'native-finalization-chain-v4',
        **binding, 'generation_job': job, 'measurement_job': cpu})


def dispatch_j1():
    r = ready(N.DOC / 'J1_READY.json')
    manifest = N.U.sealed(N.DOC / 'MANIFEST.json'); N.validate(manifest)
    prep = N.U.sealed(N.ROOT / 'grading/GRADE_PREP.json')
    price = N.U.sealed(N.ROOT / 'grading/J1_PRICE.json')
    N.require(r['manifest_sha256'] == manifest['sha256'] and r['grade_preparation_sha256'] == prep['sha256'] and
        r['price_sha256'] == price['sha256'] and price['items_file_sha256'] == N.U.file_sha(N.ROOT / 'grading/items.jsonl'),
        'J1 readiness changed')
    preflight(price['requested_allocation_GPU_hour_ceiling'] * 8 + 2)
    import utility_j1_entry_v2
    j1 = utility_j1_entry_v2.install(); plan = j1.validate_plan()
    N.require(price['frozen_j1_plan_sha256'] == plan['sha256'] and price['items'] <= 32 and len(price['shards']) <= 1,
              'J1 binding/resource shape changed')
    binding = {'manifest_sha256': manifest['sha256'], 'grade_preparation_sha256': prep['sha256'],
               'price_sha256': price['sha256'], 'readiness_sha256': r['sha256']}
    jobs = []
    if price['items']:
        jobs.append(Ops.submit('j1', ['--time=06:00:00', '--job-name=native-finalization-j1', str(j1.LEGACY),
            str(N.ROOT / 'grading/items.jsonl'), str(N.ROOT / 'grading/verdicts.jsonl')], binding,
            {'STEER_CODE': str(j1.BASE), 'STEER_EXPECT_TREE': N.U.sealed(j1.BASE / 'MANIFEST.json')['tree_sha256'],
             'STEER_PROVENANCE_DIR': str(N.ROOT / 'grading/provenance')}))
    final = Ops.submit('finalize', [*(['--dependency=afterany:' + ':'.join(jobs)] if jobs else []),
        '--job-name=native-finalization-finalize', str(Ops.SCRIPTS / 'native_finalization_cpu_v5.sbatch'), 'finalize'], binding)
    return N.save(Ops.SUBMISSIONS / 'GRADING_CHAIN.json', {'schema': 'native-finalization-grading-chain-v3',
        **binding, 'J1_jobs': jobs, 'finalize_job': final})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'generation', 'j1', 'account'))
    args = parser.parse_args(); login_guard()
    if args.phase == 'account':
        result = Ops.accounting()
    else:
        result = {'prepare': launch_preparation, 'generation': dispatch_generation, 'j1': dispatch_j1}[args.phase]()
    print(json.dumps({k: v for k, v in result.items() if k in (
        'schema', 'sha256', 'generation_job', 'measurement_job', 'J1_jobs', 'finalize_job',
        'billing_core_hours_so_far', 'allocated_GPU_hours_so_far', 'unknown_accounting_jobs')}
        if isinstance(result, dict) else result, indent=1))
