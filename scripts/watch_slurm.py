#!/usr/bin/env python3
"""Bounded, foreground Slurm watcher; delivers to an existing Codex tmux pane.
No daemon, server, scheduler mutation, GPU allocation, or automatic resubmission.
Runtime state is private in HOME. Run with python -B to avoid bytecode writes.
"""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}
STATE_ROOT = Path.home() / ".local/state/mfa-moe-watches"


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=15).strip()


def process(pid):
    root = Path('/proc') / str(pid)
    tail = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'parent': int(tail[1]), 'start': tail[19],
            'name': (root / 'comm').read_text().strip(), 'uid': root.stat().st_uid}


def binding():
    pid = os.getppid()
    while pid > 1:
        info = process(pid)
        if info['name'] == 'codex':
            info.update(pane=os.environ['TMUX_PANE'], thread=os.environ['CODEX_THREAD_ID'])
            return info
        pid = info['parent']
    raise RuntimeError('Run from the existing Codex tmux session')


def verify_target(target):
    info = process(target['pid'])
    if any(info[k] != target[k] for k in ('start', 'name', 'uid')):
        raise RuntimeError('Original Codex process changed')
    pane = command(['tmux', 'display-message', '-p', '-t', target['pane'],
                    '#{pane_current_command}|#{pane_pid}|#{pane_in_mode}']).split('|')
    if pane[0] != 'codex' or pane[2] != '0':
        raise RuntimeError('Pane is not an active Codex terminal')
    pid = target['pid']
    while pid > 1 and pid != int(pane[1]):
        pid = process(pid)['parent']
    if pid != int(pane[1]):
        raise RuntimeError('Codex no longer belongs to this pane')


def empty_prompt(screen):
    lines = screen.rstrip().splitlines()
    # Exact placeholder plus no wrapped input; fail closed on UI changes.
    return len(lines) >= 3 and lines[-3].strip() == '› Ask Codex to do anything'


def capture(target):
    return command(['tmux', 'capture-pane', '-p', '-t', target['pane']])


def notify(target, message):
    # Serialize all watchers; only notify an idle, empty session. Never queue
    # a second agent turn while this one is already processing job events.
    with (STATE_ROOT / 'delivery.lock').open('a') as delivery_lock:
        fcntl.flock(delivery_lock, fcntl.LOCK_EX)
        verify_target(target)
        for _ in range(2):
            screen = capture(target)
            if ('esc to interrupt' in screen or 'Queued follow-up inputs' in screen):
                return 'deferred_until_idle'
            if not empty_prompt(screen):
                return 'busy_prompt'
            verify_target(target)
        subprocess.run(['tmux', 'send-keys', '-t', target['pane'], '-l', '--', message],
                       check=True, timeout=15)
        time.sleep(0.3)
        verify_target(target)
        subprocess.run(['tmux', 'send-keys', '-t', target['pane'], 'Enter'], check=True, timeout=15)
        return 'sent_to_tui'


def parse_accounting(raw, job):
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) >= 5 and fields[0] == job:
            if fields[1] != 'lmolfett':
                raise RuntimeError('Job belongs to another user')
            return {'state': fields[2].split()[0].rstrip('+'),
                    'exit_code': fields[3], 'elapsed': fields[4]}
    return None  # Accounting lag is never interpreted as completion.


def poll(job):
    # Check live queue first: PREEMPTED accounting can precede an automatic requeue.
    try:
        live = command(['squeue', '-h', '-j', job, '-o', '%i|%u|%T'])
    except subprocess.CalledProcessError as exc:
        if 'Invalid job id specified' not in (exc.stderr or ''):
            raise
        live = ''  # Completed jobs disappear from squeue; consult accounting.
    for line in live.splitlines():
        fields = line.split('|')
        if fields[0] == job:
            if fields[1] != 'lmolfett':
                raise RuntimeError('Job belongs to another user')
            return {'state': fields[2], 'exit_code': '', 'elapsed': ''}
    raw = command(['sacct', '-n', '-P', '-X', '-j', job,
                   '--format=JobIDRaw,User,State%30,ExitCode,Elapsed'])
    return parse_accounting(raw, job)


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=str)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--ack', action='store_true')
    parser.add_argument('--session-scope', action='store_true', help='Keep a separate receipt/lock for this session during verified handover')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9]+', args.job) or not 1 <= args.max_seconds <= 86400:
        parser.error('Numeric individual job ID and 1..86400 second lifetime required')
    if getpass.getuser() != 'lmolfett' or not socket.gethostname().endswith('.leonardo.local'):
        parser.error('This watcher is restricted to lmolfett on LEONARDO')
    os.umask(0o077)
    STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    STATE_ROOT.chmod(0o700)
    state_root = STATE_ROOT
    if args.session_scope:
        import hashlib
        thread = os.environ['CODEX_THREAD_ID']
        state_root = STATE_ROOT / ('session-' + hashlib.sha256(thread.encode()).hexdigest()[:16])
        state_root.mkdir(mode=0o700, exist_ok=True)
    path = state_root / (args.job + '.json')
    if args.ack:
        data = json.loads(path.read_text())
        data['acknowledged_at'] = time.time()
        save(path, data)
        print('Agent acknowledgement recorded:', args.job)
        return
    with (state_root / (args.job + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and (json.loads(path.read_text()).get('delivery') == 'sent_to_tui' or json.loads(path.read_text()).get('acknowledged_at')):
            print('Notification already sent for job', args.job)
            return
        target = binding()
        verify_target(target)
        data = dict(job=args.job, watcher_pid=os.getpid(), target=target,
                    hostname=socket.gethostname(), watcher_start=process(os.getpid())['start'], started_at=time.time(), deadline=time.time()+args.max_seconds)
        save(path, data)
        print('ARMED', args.job, 'until', data['deadline'], flush=True)
        end = time.monotonic() + args.max_seconds
        event = None
        while True:
            if path.exists() and json.loads(path.read_text()).get('acknowledged_at'):
                print('Acknowledged job; watcher finished', args.job, flush=True)
                return
            expired = time.monotonic() >= end
            if event is None and not expired:
                try:
                    status = poll(args.job)
                    if status:
                        data['observed'] = status
                        if status['state'] in TERMINAL:
                            event = status['state']
                except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                    data['last_error'] = str(exc)
            if expired and event is None:
                event = 'WATCH_EXPIRED'
            if event:
                data['event'] = event
                message = (f'[SLURM_EVENT job={args.job} state={event}] '
                           'Automatic notification requested by the user. Inspect scheduler status, '
                           'logs and artifacts; report the result and continue only already-authorized work. '
                           'No new job or budget is authorized by this signal. '
                           f'Record receipt using scripts/watch_slurm.py {args.job} --ack' + (' --session-scope.' if args.session_scope else '.'))
                try:
                    data['delivery'] = notify(target, message)
                except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                    data['delivery'] = 'undelivered'
                    data['delivery_error'] = str(exc)
                save(path, data)
                if data['delivery'] == 'sent_to_tui' or expired:
                    print(json.dumps({k: data[k] for k in ('job', 'event', 'delivery')}), flush=True)
                    return
            save(path, data)
            time.sleep(min(30, max(0, end-time.monotonic())))


if __name__ == '__main__':
    main()
