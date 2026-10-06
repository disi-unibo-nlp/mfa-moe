"""Finite native Slurm dispatch, exact J1 adaptation and additive study index."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess

import native_finalization_v1 as N

SCRIPTS = Path(__file__).parent
SUBMISSIONS = N.DOC / 'submissions'


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-12000:], flush=True)
        print(result.stderr[-12000:], flush=True)
        result.check_returncode()
    return result.stdout


def live():
    import getpass
    N.require(getpass.getuser() == 'lmolfett' and '.leonardo.' in socket.getfqdn(), 'wrong cluster identity')
    rows = {'hostname': socket.getfqdn(), 'user': getpass.getuser(),
        'observed_utc': datetime.now(timezone.utc).isoformat()}
    on_login = socket.gethostname().startswith('login')
    commands = {
        'association': ['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett', 'format=Account,Partition,QOS,DefaultQOS'],
        'queue': ['squeue', '-u', 'lmolfett', '-o', '%.18i %.32j %.10T %.10M %.30R'],
        'gpu_partition': ['scontrol', 'show', 'partition', 'boost_usr_prod'],
        'cpu_partition': ['scontrol', 'show', 'partition', 'lrd_all_viz']}
    if on_login:
        commands['balance'] = ['/cineca/bin/saldo', '-b', 'lmolfett']
    for name, args in commands.items():
        rows[name] = command(args)
    N.require('iscrc_miosr' in rows['association'] and 'normal' in rows['association'] and
              (not on_login or 'IscrC_MIOSR' in rows['balance']), 'account association absent')
    rows['balance_status'] = 'LIVE_NATIVE_LOGIN' if on_login else 'UNAVAILABLE_ON_COMPUTE; readiness only, no GPU dispatch'
    tag = os.environ.get('SLURM_JOB_ID') or ('login-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    return N.save(N.DOC / ('LIVE-' + tag + '.json'),
                  {'schema': 'native-finalization-live-account-v1', **rows})


def submit(name, args, binding, additions=None):
    import dispatch_overnight_readers_v1 as shared
    import dispatch_generated_dense_v1 as verify
    from native_finalization_transport_v2 import safe_environment
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    env = safe_environment(os.environ, additions)
    with (SUBMISSIONS / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = shared.submit(SUBMISSIONS, name, args, env, binding)
        verify.ensure_verified(SUBMISSIONS, name, job)
    print(json.dumps({'submitted_name': name, 'verified_job_id': job}), flush=True)
    return job


def accounting():
    jobs = {os.environ['SLURM_JOB_ID']: 'cpu-current'} if os.environ.get('SLURM_JOB_ID') else {}
    for p in SUBMISSIONS.glob('*.json'):
        r = N.U.sealed(p)
        if r.get('job_id') and r.get('schema') == 'overnight-submit-receipt-v1':
            jobs[str(r['job_id'])] = p.stem
    rows = []
    for job, name in sorted(jobs.items()):
        raw = command(['sacct', '-X', '-nP', '-j', job,
                       '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
        exact = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == job]
        if len(exact) != 1:
            rows.append({'job_id': job, 'name': name, 'status': 'ACCOUNTING_UNKNOWN', 'raw': raw}); continue
        r = exact[0]; tres = dict(x.split('=', 1) for x in r[4].split(',') if '=' in x)
        seconds = int(r[3]); gpu = int(tres.get('gres/gpu', 0)); billing = tres.get('billing')
        rows.append({'job_id': job, 'name': name, 'state': r[1], 'exit_code': r[2],
            'elapsed_seconds': seconds, 'allocated_tres': tres, 'allocated_GPU_hours': gpu * seconds / 3600,
            'billing_core_hours': float(billing) * seconds / 3600 if billing else None,
            'provisional': r[1] in ('RUNNING', 'PENDING', 'COMPLETING'), 'raw': raw})
    return {'jobs': rows, 'allocated_GPU_hours_so_far': sum(r.get('allocated_GPU_hours', 0) for r in rows),
        'billing_core_hours_so_far': sum(r.get('billing_core_hours') or 0 for r in rows),
        'unknown_accounting_jobs': [r['job_id'] for r in rows if r.get('status') == 'ACCOUNTING_UNKNOWN'],
        'scope': 'This diagnostic generation, engineering, strict/J1 and CPU allocations only; existing jobs preserved and not counted twice.'}


def study_index(status, analysis=None):
    previous = []
    for p in (N.U.REPO / 'report/experimental-resume-v1/routing-study-closeout-v2').glob('audit-*/STUDY_INDEX.json'):
        previous.append({'path': str(p), 'sha256': N.U.sealed(p)['sha256']})
    value = {'schema': 'native-finalization-study-index-v1', 'status': status,
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'previous_study_indices': previous,
        'manifest': str(N.DOC / 'MANIFEST.json'),
        'manifest_sha256': N.U.sealed(N.DOC / 'MANIFEST.json')['sha256'],
        'saved_audit_sha256': N.U.sealed(N.DOC / 'SAVED_AUDIT.json')['sha256'],
        'accounting': accounting(), 'analysis': analysis,
        'unresolved': ['Small discovery sample; no equivalence or accuracy-retention inference.',
            'Saved short continuations and original-prompt generations are separate.',
            'Missing execution and unfinished grades are unknown.',
            'External DeepSeek review is general methods advice; no private transfer or API use in this cycle.'],
        'next_step': analysis['recommendation'] if analysis else 'Wait for qualified native generation and offline grading; no downstream experiment authorized here.'}
    path = N.DOC / ('STUDY_INDEX-' + os.environ.get('SLURM_JOB_ID', 'login') + '.json')
    saved = N.save(path, value)
    print(json.dumps({'study_index': str(path), 'status': status, 'sha256': saved['sha256']}), flush=True)
    return saved


def prepare():
    live()
    test_python = N.U.REPO / '.venv-native-finalization-tests/bin/python'
    N.require(test_python.exists(), 'project test-only environment absent')
    tests = ['tests/experimental_resume/test_native_finalization_v1.py',
             'tests/experimental_resume/test_operator_panel_v1.py',
             'tests/experimental_resume/test_utility_controller_v2.py',
             'tests/experimental_resume/test_overnight_routing_dose_audit_v3.py']
    N.require(all((N.U.REPO / p).exists() for p in tests), 'required test file missing')
    # Legacy tests import the working package during collection. Isolate them
    # from tests that explicitly pin the immutable scientific namespace.
    log = ''
    for i, group in enumerate((tests[:1], tests[1:])):
        log += command([str(test_python), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            '--basetemp=' + os.environ['TMPDIR'] + '/pytest-' + str(i), *group])
    N.save(N.DOC / 'TESTS.json', {'schema': 'native-finalization-tests-v1', 'stdout': log,
        'python': str(test_python), 'test_sources': {p: N.U.file_sha(N.U.REPO / p) for p in tests}})
    manifest = N.prepare()
    from native_finalization_audit_v1 import saved_audit
    saved_audit()
    binding = {'manifest_sha256': manifest['sha256'], 'source_files': N.sources()}
    N.save(N.DOC / 'PREPARED.json', {'schema': 'native-finalization-readiness-v3',
        **binding, 'producer_job_id': os.environ['SLURM_JOB_ID'], 'phase': 'generation',
        'status': 'READY_FOR_NATIVE_LOGIN_DISPATCH'})
    study_index('PREPARED_AWAITING_NATIVE_LOGIN_DISPATCH')


def measure():
    live()
    manifest = N.U.sealed(N.DOC / 'MANIFEST.json'); N.validate(manifest)
    index = N.save(N.ROOT / 'INDEX.json', N.reconcile(manifest, N.ROOT))
    from native_finalization_outcomes_v1 import prepare as grade_prepare
    prep_path = N.ROOT / 'grading/GRADE_PREP.json'
    prep = N.U.sealed(prep_path) if prep_path.exists() else grade_prepare(index, prep_path.parent)
    N.require(prep['index_sha256'] == index['sha256'], 'prior grading index differs')
    import utility_j1_entry_v2
    j1 = utility_j1_entry_v2.install()
    plan = j1.validate_plan()
    import operator_panel_outcomes_v1 as O
    data = O.read_preparation(prep_path.parent, prep); items = data['items']
    from utility_outcomes_v3 import scoring_modules
    score, _ = scoring_modules()
    score.assert_blind(items, 'native finalization exact J1 price')
    rows = []
    if items:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(j1.MODEL, local_files_only=True)
        for item in items:
            ids = j1.exact_ids(tokenizer.apply_chat_template(j1.messages(item, score.answer_equivalence),
                tokenize=True, add_generation_prompt=True, enable_thinking=True))
            rows.append({'item_id': item['item_id'], 'prompt_tokens': len(ids), 'prompt_ids_sha256': N.U.digest(ids)})
    price = j1.price_rows(rows, plan['binding'])
    N.require(len(items) <= 32 and len(price['shards']) <= 1, 'grading exceeds this 32-assignment cycle')
    exact = N.save(N.ROOT / 'grading/J1_PRICE.json', {**price, 'grade_preparation_sha256': prep['sha256'],
        'frozen_j1_plan_sha256': plan['sha256'], 'prompt_records': rows,
        'items_file_sha256': N.U.file_sha(prep_path.parent / 'items.jsonl')})
    binding = {'manifest_sha256': manifest['sha256'], 'grade_preparation_sha256': prep['sha256'],
               'price_sha256': exact['sha256']}
    N.save(N.DOC / 'J1_READY.json', {'schema': 'native-finalization-readiness-v3',
        **binding, 'producer_job_id': os.environ['SLURM_JOB_ID'], 'phase': 'j1',
        'status': 'READY_FOR_NATIVE_LOGIN_DISPATCH'})
    study_index('J1_PRICED_AWAITING_NATIVE_LOGIN_DISPATCH')


def finalize():
    manifest = N.U.sealed(N.DOC / 'MANIFEST.json'); N.validate(manifest)
    prep = N.U.sealed(N.ROOT / 'grading/GRADE_PREP.json')
    chain = N.U.sealed(SUBMISSIONS / 'GRADING_CHAIN.json')
    verdicts = N.ROOT / 'grading/verdicts.jsonl'
    if chain['J1_jobs'] and verdicts.exists():
        import utility_j1_entry_v2
        j1 = utility_j1_entry_v2.install()
        price = N.U.sealed(N.ROOT / 'grading/J1_PRICE.json')
        job = chain['J1_jobs'][0]
        provenance = N.U.sealed(N.ROOT / 'grading/provenance' / (job + '.json'))
        N.require(provenance['items_sha256'] == price['items_file_sha256'] and
            provenance['launcher_sha256'] == N.U.file_sha(j1.LEGACY) and
            provenance['tree_sha256'] == N.U.sealed(j1.BASE / 'MANIFEST.json')['tree_sha256'] and
            provenance['mode'] == 'j1' and provenance['dry_run'] is False, 'J1 execution provenance differs')
        # Validate any completed votes; absent item IDs remain unknown.
        present = [json.loads(line)['item_id'] for line in verdicts.read_text().splitlines() if line.strip()]
        expected = {r['item_id'] for r in price['prompt_records']}
        N.require(set(present) <= expected, 'foreign J1 item')
        j1.valid_verdicts(verdicts, present)
    from native_finalization_outcomes_v1 import analyze
    result = analyze(prep, verdicts, N.DOC / 'analysis', accounting())
    study_index(result['status'], result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'measure', 'finalize', 'account'))
    args = parser.parse_args()
    if args.phase == 'account':
        print(json.dumps(accounting(), indent=2))
    else:
        N.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
                  os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz', 'finite CPU Slurm step required')
        N.require(str(Path(os.environ.get('TMPDIR', '/tmp')).resolve()).startswith(
            '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'), 'compute temporary directory is outside user staging')
        globals()[args.phase]()
