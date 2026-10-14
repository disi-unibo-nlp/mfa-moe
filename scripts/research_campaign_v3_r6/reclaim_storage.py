#!/usr/bin/env python3
"""Plan and retire only terminal-allocation compiler caches, with exact receipts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

WORK = Path('/leonardo_work/IscrC_MIOSR/lmolfett')
STAGING = WORK/'tools/tmp'
REPO = WORK/'mfa-moe/repo'
CACHE_NAMES = frozenset(('tmp','torch','triton','xdg','cuda','inductor','vllm-cache','vllm_cache','vllm'))
TERMINAL = frozenset(('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY',
                      'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=60).stdout.strip()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    value = {**value, 'sha256': digest(value)}
    if path.exists():
        require(json.loads(path.read_text()) == value, 'cleanup receipt conflict')
        return value
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return value


def protected_paths():
    protected = set()
    for manifest in (REPO/'report').glob('*/CAMPAIGN.json'):
        value = json.loads(manifest.read_text())
        for key in ('code_files','source_files','runtime_files','scientific_files_preserved','readiness_files_preserved'):
            protected.update(str(Path(p).resolve()) for p in value.get(key,{}))
    return protected


def safe_cache(path, protected):
    path = Path(path)
    resolved = str(path.resolve())
    return (path.name in CACHE_NAMES and not path.is_symlink() and path.is_dir()
            and path.stat().st_uid == os.getuid() and resolved.startswith(str(WORK)+'/')
            and not any(p == resolved or p.startswith(resolved+'/') for p in protected))


def candidates():
    roots = []
    flat = re.compile(r'(?:rd|rs|nf|cv3-runtime|overnight-semantic-runtime|mechanism-semantic-runtime|mechanism-cap-recovery-runtime)-(?P<jid>[0-9]{8}(?:_[0-9]+)?)')
    for p in STAGING.iterdir():
        match = flat.fullmatch(p.name)
        if match and p.is_dir() and not p.is_symlink():
            roots.append((p,match['jid']))
    for base in (STAGING/'claude-analysis-2026-09-24/campaign-v3/runtime',
                 STAGING/'claude-analysis-2026-09-24/steering-v1/runtime', WORK/'mfa-moe/r', STAGING/'t3'):
        if not base.is_dir() or base.is_symlink():
            continue
        for p in base.iterdir():
            match = re.fullmatch(r'[a-zA-Z0-9-]*?([0-9]{8}(?:_[0-9]+)?)',p.name)
            if match and p.is_dir() and not p.is_symlink():
                roots.append((p,match[1]))
    return roots


def accounting(ids):
    values = {}
    for offset in range(0,len(ids),150):
        text = command(['sacct','-X','-nP','-S','2026-09-01','-j',','.join(ids[offset:offset+150]),
                        '--format=JobID%40,State%40,ElapsedRaw,Start'])
        for line in text.splitlines():
            fields = line.split('|')
            if len(fields) >= 4:
                jid,state,elapsed,start = (f.strip() for f in fields[:4])
                values[jid] = {'state':state.split()[0], 'elapsed_seconds':int(elapsed or 0),'start':start}
    return values


def live_ids():
    return set(command(['squeue','-h','-u','lmolfett','-o','%i']).splitlines())


def is_live(jid, live):
    parent = jid.split('_')[0]
    return any(j.split('_')[0] == parent for j in live)


def identity(path):
    stat = Path(path).lstat()
    return {'device':stat.st_dev,'inode':stat.st_ino,'uid':stat.st_uid,'mtime_ns':stat.st_mtime_ns}


def plan(output):
    require(not output.exists(), 'use a fresh cleanup plan')
    protected, roots, live = protected_paths(), candidates(), live_ids()
    observed = accounting(sorted({jid for _,jid in roots}))
    eligible = [(p,jid) for p,jid in roots if jid in observed and observed[jid]['state'] in TERMINAL
                and not is_live(jid,live)]
    entries = []
    for root,jid in eligible:
        for path in root.iterdir():
            if safe_cache(path,protected):
                entries.append({'path':str(path),'job_id':jid,'identity':identity(path),'accounting':observed[jid]})
    sizes = {}
    for offset in range(0,len(entries),100):
        text = command(['du','-sxB1','--',*[e['path'] for e in entries[offset:offset+100]]])
        for line in text.splitlines():
            size,path = line.split('\t',1)
            sizes[path] = int(size)
    for entry in entries:
        entry['allocated_bytes'] = sizes[entry['path']]
    return save(output,{'kind':'terminal-allocation compiler caches only','entries':entries,
        'live_jobs_preserved':sorted(live),'protected_source_file_count':len(protected),
        'estimated_allocated_bytes':sum(e['allocated_bytes'] for e in entries),
        'created_epoch':time.time(),'original_captures_and_scientific_outputs_preserved':True,
        'quota_before':command(['lfs','quota','-p','20149688','/leonardo_work'])})


def apply(plan_path):
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and os.environ.get('SLURM_STEP_ID') is not None,
            'cleanup requires an allocated Slurm step')
    require(os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_serial', 'cleanup requires serial allocation')
    value = json.loads(plan_path.read_text())
    require(value['sha256'] == digest({k:v for k,v in value.items() if k != 'sha256'}), 'cleanup plan changed')
    protected, live = protected_paths(), live_ids()
    allowed = {(str(root / name), jid) for root,jid in candidates() for name in CACHE_NAMES}
    observed = accounting(sorted({e['job_id'] for e in value['entries']}))
    removed, skipped = [], []
    for entry in value['entries']:
        path,jid = Path(entry['path']),entry['job_id']
        if (str(path),jid) not in allowed:
            # Removed directories retain their allocation root for restart.
            raise ValueError('cleanup path is outside registered allocation cache roots')
        receipt = plan_path.parent/'removed'/(digest(entry['path'])+'.json')
        if receipt.exists():
            require(not path.exists(), 'retired cache reappeared')
            removed.append(json.loads(receipt.read_text()))
            continue
        if is_live(jid,live) or observed.get(jid,{}).get('state') not in TERMINAL:
            skipped.append({'path':str(path),'reason':'allocation not proved terminal'}); continue
        if not path.exists():
            removed.append(save(receipt,{'path':str(path),'job_id':jid,'already_absent':True})); continue
        require(safe_cache(path,protected), 'cache is symlinked, bound, or outside the allowlist')
        require(identity(path) == entry['identity'], 'cache changed after inspection')
        shutil.rmtree(path)
        require(not path.exists(), 'cache retirement not verified')
        removed.append(save(receipt,{'path':str(path),'job_id':jid,'accounting':observed[jid],
                                     'estimated_allocated_bytes':entry['allocated_bytes']}))
    return save(plan_path.parent/'COMPLETE.json',{'status':'complete','plan_sha256':value['sha256'],
        'removed':removed,'skipped':skipped,'job_id':os.environ['SLURM_JOB_ID'],
        'quota_after':command(['lfs','quota','-p','20149688','/leonardo_work']),
        'scientific_artifacts_preserved':True,'installed_environments_and_agent_runtime_unchanged':True})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('plan','apply'))
    parser.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args()
    require(command(['whoami']) == 'lmolfett' and 'leonardo' in command(['hostname','-f']), 'wrong cluster identity')
    require(str(args.plan.resolve()).startswith(str(REPO/'report')+'/'), 'receipt must be private project report')
    with (STAGING/'campaign-storage-recovery.lock').open('a') as handle:
        os.chmod(handle.name,0o600)
        fcntl.flock(handle,fcntl.LOCK_EX)
        result = plan(args.plan) if args.command == 'plan' else apply(args.plan)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','sha256','estimated_allocated_bytes','kind')},sort_keys=True))


if __name__ == '__main__':
    main()
