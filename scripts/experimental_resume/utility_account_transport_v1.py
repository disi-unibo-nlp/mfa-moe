"""Finite verified compute-to-login reads of native CINECA accounting.

No credentials, trust entries, agents, sockets or services are created. A probe
qualifies actual compute-node access; every later balance read connects again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import pwd
import socket
import subprocess

import utility_production_v1 as P
import dispatch_utility_production_v1 as original

TARGET = 'login05.leonardo.local'
OPTIONS = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
    '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
    '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ForwardAgent=no',
    '-o', 'UpdateHostKeys=no']
REMOTE_COMMANDS = ('/bin/hostname', '/usr/bin/id -un', '/cineca/bin/saldo -b lmolfett')


def read_balance():
    """Read identity first, then balance over separate bounded verified calls."""
    records = []
    expected = (TARGET, 'lmolfett')
    for index, command in enumerate(REMOTE_COMMANDS):
        arguments = ['/usr/bin/ssh', *OPTIONS, 'lmolfett@' + TARGET, command]
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        records.append({'arguments': arguments, 'exit_code': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
        P.require(result.returncode == 0,
            'verified accounting SSH failed at ' + command + ': ' + result.stderr.strip())
        if index < 2:
            P.require(result.stdout.strip() == expected[index], 'wrong remote accounting host or user')
    balance = records[-1]['stdout']
    original.remaining_commitments(balance, '')
    return balance, records


def source_files():
    files = (Path(__file__).resolve(), P.SCRIPTS / 'probe_utility_account_transport_v1.sbatch')
    return {str(path): P.U.file_sha(path) for path in files}


def validate_probe(path):
    probe = P.U.sealed(Path(path))
    P.require(probe['schema'] == 'utility-account-transport-probe-v1' and
        probe['status'] == 'PASS_NATIVE_LIVE_ACCOUNT_READ' and
        probe['source_files'] == source_files() and
        probe['ssh_target'] == TARGET and probe['ssh_options'] == OPTIONS and
        probe['user'] == 'lmolfett' and probe['partition'] == 'lrd_all_viz' and
        not probe['compute_hostname'].startswith('login'), 'actual compute accounting transport is not qualified')
    record = subprocess.run(['sacct', '-X', '-nP', '-j', probe['probe_job_id'],
        '--format=JobIDRaw,State%40,ExitCode'], check=True, capture_output=True, text=True)
    rows = [line.split('|') for line in record.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == probe['probe_job_id']]
    P.require(len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0'],
              'qualified transport lacks exact successful Slurm accounting')
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
        pwd.getpwuid(os.getuid()).pw_name == 'lmolfett' and not socket.gethostname().startswith('login'),
        'transport probe must run in the authorized actual CPU allocation')
    body = {'schema': 'utility-account-transport-probe-v1',
        'probe_job_id': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
        'compute_hostname': socket.gethostname(), 'user': 'lmolfett', 'ssh_target': TARGET,
        'ssh_options': OPTIONS, 'source_files': source_files(),
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Read-only on-demand verified SSH; no persistent connection, agent forwarding, trust-file changes or cached production balance.'}
    try:
        balance, records = read_balance()
        body.update(status='PASS_NATIVE_LIVE_ACCOUNT_READ', balance=balance, records=records)
    except (ValueError, subprocess.TimeoutExpired, OSError) as error:
        body.update(status='FAIL_NATIVE_ACCOUNT_TRANSPORT', error=str(error))
    saved = P.save(args.out, body)
    print(saved['status'], str(args.out), saved['sha256'], flush=True)
    if body['status'] != 'PASS_NATIVE_LIVE_ACCOUNT_READ':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
