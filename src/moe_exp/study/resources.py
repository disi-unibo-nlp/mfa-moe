"""Live entitlement, exact separate quotas and allocation reservations."""
import getpass
import re
import socket
import time
from pathlib import Path
from .common import REPO, submission_paths, reservation_paths, digest, load, publish, require, verified
from .scheduler import command, duration_minutes, job_fields


def quota_bytes(raw, mount):
    lines=raw.splitlines()
    line = next((s.strip()+(' '+lines[i+1].strip() if s.strip()==mount and i+1<len(lines) else '')
                 for i,s in enumerate(lines) if s.strip()==mount or s.strip().startswith(mount+' ')), None)
    require(line is not None, 'missing exact quota for '+mount)
    fields = line.split()
    used = int(fields[1].rstrip('*'))
    limits = [int(v) for v in fields[2:4] if v.isdigit() and int(v) > 0]
    require(limits, 'finite project quota unavailable')
    return (min(limits)-used)*1024


def balance_query(run=command):
    if Path('/cineca/bin/saldo').is_file():
        return run(['/cineca/bin/saldo', '-b'])
    # Compute nodes lack /cineca; the installed native executable hard-codes it.
    permission = verified(REPO / 'report/final-routing-20261015-r1/STUDY_ACCOUNTING_SSH_AUTHORIZATION.json')
    host = 'lmolfett@login01.leonardo.local'
    require(permission['status'] == 'approved' and permission['host'] == host and
            permission['command'] == '/cineca/bin/saldo -b', 'accounting SSH exception is not authorized')
    return run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=10', '-o', 'ConnectionAttempts=1',
                '-o', 'ControlMaster=no', '-o', 'ClearAllForwardings=yes',
                '-o', 'ForwardAgent=no', host, '/cineca/bin/saldo', '-b'])


def snapshot(run=command):
    require(getpass.getuser() == 'lmolfett' and 'leonardo' in socket.getfqdn(), 'wrong cluster identity')
    balance = balance_query(run)
    fields = next(s.split() for s in balance.splitlines() if s.strip().startswith('IscrC_MIOSR '))
    total, consumed = float(fields[3]), float(fields[4])
    association = run(['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett',
                       'format=Account,Partition,QOS,DefaultQOS'])
    require(any(s.startswith('iscrc_miosr|') and 'normal' in s.split('|')[2].split(',')
                for s in association.splitlines()), 'normal QoS entitlement unavailable')
    partition = run(['scontrol', 'show', 'partition', 'boost_usr_prod'])
    require('State=UP' in partition, 'Booster partition is down')
    qos = run(['sacctmgr', '-nP', 'show', 'qos', 'where', 'name=boost_usr_prod,normal',
               'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES,MaxWall'])
    work = run(['lfs', 'quota', '-p', '20149688', '/leonardo_work'])
    fast = run(['lfs', 'quota', '-p', '22149688', '/leonardo_scratch/fast'])
    ids = run(['squeue', '--array', '-h', '-u', 'lmolfett', '-o', '%i']).splitlines()
    jobs = []
    for jid in sorted(set(ids)):
        require(re.fullmatch(r'\d+(?:_\d+)?', jid), 'unexpanded Slurm array')
        try:
            raw = run(['scontrol', 'show', 'job', '-o', jid])
        except Exception:
            # A disappearing queue item must be reconciled by the next snapshot.
            raise ValueError('queue changed during accounting; take another snapshot')
        f = job_fields(raw)
        if f.get('Account') != 'iscrc_miosr':
            continue
        tres = f.get('AllocTRES')
        if not tres or tres == '(null)':
            tres = f.get('ReqTRES', '')
        bill = re.search(r'(?:^|,)billing=([\d.]+)', tres)
        require(bill, 'unverifiable outstanding billing reservation')
        maximum = duration_minutes(f['TimeLimit'])/60*float(bill[1])
        elapsed = duration_minutes(f.get('RunTime', '00:00:00'))/60*float(bill[1])
        jobs.append({'job_id': jid, 'name': f['JobName'], 'state': f['JobState'],
                     'remaining_hours': max(0, maximum-elapsed), 'fields': f})
    return {'epoch': time.time(), 'total_hours': total, 'consumed_hours': consumed,
            'remaining_hours': total-consumed, 'outstanding_hours': sum(j['remaining_hours'] for j in jobs),
            'free_bytes': {'work': quota_bytes(work, '/leonardo_work'),
                           'fast': quota_bytes(fast, '/leonardo_scratch/fast')},
            'jobs': jobs, 'balance': balance, 'association': association, 'partition': partition,
            'qos': qos, 'quota': {'work': work, 'fast': fast}}


def availability(config, snap, *, campaign_actual=0):
    expended = max(campaign_actual, snap['consumed_hours']-config['budget']['baseline_consumed'])
    available = min(config['budget']['ceiling_hours']-expended, snap['remaining_hours'])
    return {'expended_hours': expended, 'available_hours': available-snap['outstanding_hours'],
            'outstanding_hours': snap['outstanding_hours'], 'free_bytes': snap['free_bytes']}


def retained_usage(binding):
    """Allocated filesystem blocks, excluding already present weights and disposable caches."""
    result={}
    for fs,path in [('work',binding['root']),('fast',binding['config']['bulk_root'])]:
        output=command(['du','-sx','--block-size=1','--exclude=runtime','--exclude=checkpoints',
                        '--exclude=cpu-pricing',path],timeout=180)
        result[fs]=int(output.split()[0])
    return result


def preparation_admission(config, snap):
    from moe_exp.final_routing.budget import preparation
    needed = preparation(config)['margin_inclusive_hours']
    available = availability(config, snap)
    reserve = config['budget']['reserve_gib_per_filesystem']*1024**3
    # The checkpoint is already on FAST. Preparation output ceilings are
    # enforced again by workers; no prospective production storage is assumed.
    output = config['budget']['preparation_output_gib']
    reasons = []
    if needed+config['budget']['closeout_hours'] > available['available_hours']:
        reasons.append('authorized preparation does not fit remaining entitlement')
    for fs in ('work', 'fast'):
        if output[fs]*1024**3+reserve > snap['free_bytes'][fs]:
            reasons.append(fs+' preparation storage and reserve do not fit')
    return {'status': 'admitted' if not reasons else 'not_admitted', 'reasons': reasons,
            'reserved_hours': needed, 'available': available, 'snapshot_sha256': digest(snap),
            'output_limits_gib': output}


def accounting(root, run=command):
    accepted = [verified(p) for p in submission_paths(root)]
    ids = [r['job_id'] for r in accepted]
    if not ids:
        return {}
    raw = run(['sacct', '-nP', '-X', '-j', ','.join(ids),
               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
    output = {}
    for line in raw.splitlines():
        jid, state, code, elapsed, tres = line.split('|')[:5]
        billing = re.search(r'(?:^|,)billing=([\d.]+)', tres)
        output[jid] = {'state': state.split()[0], 'exit_code': code, 'seconds': int(elapsed or 0),
                       'billing_hours': float(billing[1])*int(elapsed or 0)/3600 if billing else 0,
                       'allocated_tres': tres}
    return output
