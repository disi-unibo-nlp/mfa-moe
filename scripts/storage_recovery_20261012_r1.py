#!/usr/bin/env python3
"""Retire an explicit set of terminal compiler caches with raw Slurm ID proofs.

Uses the preserved cleanup implementation; no campaign or installed runtime is
edited. The finite root allowlist comes from the October 12 read-only audit.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'scripts/research_campaign_v3_r3/reclaim_storage.py'
spec = importlib.util.spec_from_file_location('preserved_storage_cleanup', BASE)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
ROOTS = {
    **{str(j): 'mechanism-semantic-runtime-' + str(j) for j in (
        59342880, 59342881, 59342882, 59342883, 59342884, 59342885,
        59342909, 59342910, 59342983)},
    **{str(j): 'mechanism-cap-recovery-runtime-' + str(j) for j in (
        59347891, 59349963, 59349964, 59349965, 59350008)},
}
_accounting = {}
_save = cleanup.save
_is_live = cleanup.is_live


def files():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (BASE, Path(__file__).resolve())}


def save(path, value):
    return _save(path, {**value, 'implementation_files': files()})


def candidates():
    result = []
    for jid, name in ROOTS.items():
        path = cleanup.STAGING / name
        cleanup.require(path.is_dir() and not path.is_symlink()
                        and path.stat().st_uid == os.getuid(),
                        'audited allocation root is missing, linked or not owned')
        result.append((path, jid))
    return result


def protected_paths():
    protected = set()
    keys = ('code_files', 'source_files', 'runtime_files',
            'scientific_files_preserved', 'readiness_files_preserved',
            'preserved_scientific_files', 'preserved_operational_files')
    for path in (REPO / 'report').glob('*/CAMPAIGN.json'):
        value = json.loads(path.read_text())
        for key in keys:
            protected.update(str(Path(p).resolve()) for p in value.get(key, {}))
    return protected


def accounting(ids):
    cleanup.require(set(ids) <= set(ROOTS), 'job outside audited cleanup roster')
    raw = cleanup.command(['sacct', '-X', '-nP', '-S', '2026-09-01',
                           '-j', ','.join(ids),
                           '--format=JobIDRaw%40,JobID%40,State%40,ElapsedRaw,Start'])
    result = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 5 or parts[0] not in ids:
            continue
        jid, display, state, elapsed, start = parts[:5]
        value = {'raw_job_id': jid, 'display_job_id': display,
                 'state': state.split()[0], 'elapsed_seconds': int(elapsed or 0),
                 'start': start}
        cleanup.require(jid not in result or result[jid] == value,
                        'ambiguous raw/display Slurm identity')
        result[jid] = value
    _accounting.update(result)
    return result


def is_live(jid, live):
    display = _accounting.get(jid, {}).get('display_job_id', jid)
    return _is_live(jid, live) or _is_live(display, live)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('plan', 'apply'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    cleanup.require(cleanup.command(['whoami']) == 'lmolfett'
                    and 'leonardo' in cleanup.command(['hostname', '-f']),
                    'wrong cluster identity')
    cleanup.require(args.plan.resolve().parent == REPO / 'report/storage-recovery-20261012-r1',
                    'cleanup receipt outside approved recovery root')
    cleanup.candidates = candidates
    cleanup.accounting = accounting
    cleanup.protected_paths = protected_paths
    cleanup.is_live = is_live
    cleanup.save = save
    cleanup.CACHE_NAMES = frozenset(('torch', 'triton', 'tmp', 'xdg'))
    with (cleanup.STAGING / 'campaign-storage-recovery.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == 'apply':
            value = json.loads(args.plan.read_text())
            cleanup.require(value['implementation_files'] == files(),
                            'cleanup implementation changed after plan')
            result = cleanup.apply(args.plan)
        else:
            result = cleanup.plan(args.plan)
    print(json.dumps({k: result[k] for k in ('status', 'kind', 'sha256',
                    'estimated_allocated_bytes') if k in result}, sort_keys=True))


if __name__ == '__main__':
    main()
