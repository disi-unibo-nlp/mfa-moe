"""Seal a correction to the post-parity NLL driver, with an eight-hour X2 budget audit."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run():
    old = json.loads((S/'runs/resume-v1/NLL_UID_AMENDMENT.json').read_text())['new_addendum']
    sources = {'native_nll.py': REPO/'scripts/experimental_resume/native_nll.py',
        'nll_helpers.py': REPO/'src/moe_exp/routing_control/nll.py',
        'receipts.py': REPO/'src/moe_exp/routing_control/receipts.py'}
    files = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    tree = digest(files)
    directory = S/'addenda/nll'/('nll-'+tree[:16])
    directory.mkdir(exist_ok=True)
    for name, source in sources.items():
        dest = directory/name
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError('immutable NLL recovery code differs')
        if not dest.exists():
            shutil.copy2(source, dest)
            dest.chmod(0o400)
    value = {'schema': 'native-nll-addendum-v3', 'tree_sha256': tree, 'files': files,
        'supersedes_measurement_code_only': old, 'method_change': False, 'sampler_change': False,
        'reason': 'Pass the results module explicitly into the post-parity function; load the same CPU-prepared branch prefixes.',
        'parity': 'Identical Q3 fixtures and frozen mean/p99 criteria rerun on the new model load.',
        'output_binding': 'new measurement tree; no old completed UID reuse'}
    value['sha256'] = digest(value)
    dest = directory/'MANIFEST.json'
    if dest.exists() and json.loads(dest.read_text()) != value:
        raise ValueError('NLL recovery inventory differs')
    if not dest.exists():
        dest.write_text(json.dumps(value, indent=1)+'\n')
        dest.chmod(0o400)
    directory.chmod(0o500)
    audit = json.loads((REPO/'report/experimental-resume-v1/JOB_AUDIT.json').read_text())
    jobs = [j for j in audit['jobs'] if j['task'] == 'x2' or j['task'].startswith('x2-nll')]
    if not all(j['final'] for j in jobs):
        raise ValueError('all previous X2 allocations must be final')
    spent = sum(j['actual_resource_hours'] for j in jobs)
    if spent + 1.5 + .5 > 8:
        raise ValueError('complete pessimistic recovery plus contingency exceeds X2 eight-hour ceiling')
    receipt = {'schema': 'native-nll-engineering-recovery-v1',
        'recorded_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'new_addendum': {'path': str(directory), 'tree_sha256': tree, 'seal': value['sha256']},
        'old_addendum': old, 'failure_job': 59095712, 'failure': 'NameError: RS after both parity checks passed; zero X2 NLL UIDs measured',
        'method_change': False, 'sampler_change': False, 'GPU_ceiling_change': False,
        'budget': {'X2_total_ceiling': 8., 'actual_generation_and_measurement_GPUh': spent,
            'pending_complete_measurement_pessimistic_GPUh': 1.5, 'contingency_GPUh': .5,
            'complete_commitment_GPUh': spent+2., 'wall_limit_minutes': 41,
            'observed_shutdown_allowance_seconds': 196,
            'allocation_plus_shutdown_GPUh': (41*60+196)*2/3600,
            'source': str(S/'runs/x2prep/cost_model.json'),
            'allocation_revision': 'Release unused generation reservation within the same total eight-hour X2 line; no transfer from another study.'}}
    dest = S/'runs/resume-v1/NLL_RECOVERY_AMENDMENT.json'
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous['new_addendum'] != receipt['new_addendum']:
            raise ValueError('different recovery amendment already exists')
        receipt = previous
    else:
        receipt['sha256'] = digest(receipt)
        dest.write_text(json.dumps(receipt, indent=1)+'\n')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    run()
