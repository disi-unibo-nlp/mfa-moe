"""Accepted Slurm owners and nonblocking exclusive locks for cohort stores.

Ownership is established from exact native readbacks before a worker touches
state. A successor retains its predecessor's receipt and terminal accounting.
The store lock spans inspection, engine lifetime and terminal publication.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
import time
from moe_exp.io import digest, publish, require, verified
from .cohort import writable
from moe_exp.final_routing.scheduler import job_fields, command, duration_minutes

TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
                   'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def accounting(job: str, run=command) -> dict:
    """Read one exact parent job from native accounting.

    Args:
        job: Accepted numeric job ID.
        run: Native command runner, injectable for contract tests.

    Returns:
        Actual state, elapsed time and allocated resources.
    """
    require(job.isdigit(), 'invalid owner job ID')
    raw = run(['sacct', '-nPX', '-j', job,
               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%160'])
    rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == job]
    require(len(rows) == 1, 'owner has no unique native accounting row')
    fields = rows[0]
    return {'job': job, 'state': fields[1].split()[0].split('+')[0],
            'exit_code': fields[2], 'elapsed_seconds': int(fields[3]), 'allocated_tres': fields[4], 'raw': raw}


def allocation(job: str, shape: dict, run=command) -> dict:
    """Check the accepted owner and every allocated resource before model loading.

    Args:
        job: Exact expected job ID.
        shape: Frozen CPUs, GPUs, memory, minutes, partition, QoS and job name.
        run: Native read-only command runner.

    Returns:
        Native scontrol fields after identity and resource checks.
    """
    fields = job_fields(run(['scontrol', 'show', 'job', '-o', job]))
    expected = {'JobId': job, 'Account': 'iscrc_miosr', 'Partition': shape['partition'],
                'QOS': shape['qos'], 'NumCPUs': str(shape['cpus']), 'NumTasks': '1',
                'MinMemoryNode': str(shape['memory_gib'])+'G', 'JobName': shape['job_name']}
    require(all(fields.get(key) == value for key, value in expected.items()), 'accepted owner allocation differs')
    require(fields.get('UserId', '').startswith('lmolfett('), 'accepted owner user differs')
    require(fields.get('NumNodes') in ('1', '1-1'), 'owner requires one node')
    require(fields.get('Requeue') == '0', 'automatic owner requeue is forbidden')
    require(duration_minutes(fields['TimeLimit']) == shape['minutes'], 'owner walltime differs')
    tres = dict(item.split('=', 1) for item in fields['ReqTRES'].split(','))
    require(int(tres.get('gres/gpu', 0)) == shape['gpus'], 'owner GPU allocation differs')
    return fields


def task_staging() -> Path:
    """Pin Python temporary files to the task's explicit HOME/WORK directory.

    Returns:
        Existing private staging directory; invalid task environments fail before
        tests or model loading rather than falling back to a system directory.
    """
    import tempfile
    import zmq
    from uuid import UUID
    require(bool(os.environ.get('TMPDIR')), 'worker needs explicit private TMPDIR')
    root = writable(Path(os.environ['TMPDIR']))
    require(root.is_dir(), 'worker private TMPDIR does not exist')
    # vLLM appends a UUID to this directory before binding an AF_UNIX socket.
    # Validate its native byte limit before any engine initialization.
    socket_bytes = len(str(root/str(UUID(int=0))).encode())
    require(socket_bytes <= zmq.IPC_PATH_MAX_LEN, 'private TMPDIR exceeds the native IPC socket path limit')
    tempfile.tempdir = str(root)
    os.environ['VLLM_RPC_BASE_PATH'] = str(root)
    return root


@contextmanager
def store_lock(root: Path):
    """Reject a concurrent owner immediately, without waiting for another engine."""
    root = writable(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root/'.store.lock').open('a') as stream:
        os.chmod(stream.name, 0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'concurrent store ownership: {root}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def accept(root: Path, execution: dict, job: str, shape: dict, *, run=command) -> dict:
    """Accept one new store owner after its predecessor is natively terminal.

    Args:
        root: Canonical writable store.
        execution: Stable cohort/stage/roster identity.
        job: Newly accepted Slurm job ID.
        shape: Verified immutable resource card.
        run: Native command runner.

    Returns:
        Immutable owner receipt including any terminal predecessor proof.
    """
    root = writable(root)
    with store_lock(root):
        path = root/'owners'/(job+'.json')
        if path.exists():
            receipt = verified(path)
            require(receipt['execution'] == execution and receipt['shape'] == shape, 'owner identity changed')
            return receipt
        require(not (root/'SEALED.json').exists(), 'terminal generation index sealed this store; no new owner may write')
        owners = sorted((root/'owners').glob('*.json'))
        if not owners:
            require(all(child.name in ('.store.lock', 'owners') for child in root.iterdir()),
                    'store contains unbound prior state; preserve it and select a fresh cohort destination')
        predecessors = []
        for owner in owners:
            previous = verified(owner)
            require(previous['execution'] == execution, 'store execution identity changed')
            terminal = accounting(previous['job'], run)
            require(terminal['state'] in TERMINAL_STATES, 'predecessor owner is not terminal')
            predecessors.append(terminal)
        fields = allocation(job, shape, run)
        return publish(path, {
            'store': str(root),
            'execution': execution,
            'job': job,
            'shape': shape,
            'accepted_fields': fields,
            'predecessors': predecessors,
        })


@contextmanager
def worker(root: Path, execution: dict, *, run=command):
    """Hold exclusive ownership through terminal publication.

    Args:
        root: Exact canonical writable store.
        execution: Cohort/stage/frozen-roster identity.
        run: Native scheduler runner.

    Returns:
        Yields the verified owner, with its active native allocation.
    """
    root = writable(root)
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit(), 'cohort worker requires an accepted Slurm job owner')
    task_staging()
    with store_lock(root):
        require(not (root/'SEALED.json').exists(), 'terminal generation index sealed this store against new execution')
        owner = verified(root/'owners'/(job+'.json'))
        require(owner['job'] == job and owner['execution'] == execution and owner['store'] == str(root),
                'worker job/store/cohort/frozen roster differs from accepted owner')
        for other in (root/'owners').glob('*.json'):
            if other.stem != job:
                require(accounting(other.stem, run)['state'] in TERMINAL_STATES, 'another accepted owner is live')
        require(allocation(job, owner['shape'], run)['JobState'] == 'RUNNING', 'owner allocation is not running')
        require(os.environ.get('SLURM_CPUS_PER_TASK') == str(owner['shape']['cpus']), 'worker CPU environment differs')
        require(bool(os.environ.get('SLURM_JOB_GPUS')) == bool(owner['shape']['gpus']), 'worker GPU environment differs')
        yield owner


def check_time(value: dict, *, seconds: float = 0, reporting: bool = False) -> float:
    """Reject work whose full admitted duration cannot fit before the frozen cutoff.

    Args:
        value: Cohort descriptor.
        seconds: Remaining full resource limit or model-load reserve.
        reporting: Use delivery deadline for final CPU reconciliation.

    Returns:
        Absolute frozen deadline.
    """
    cutoff = datetime.fromisoformat(value['report_deadline' if reporting else 'science_cutoff']).timestamp()
    require(time.time()+seconds < cutoff, 'cohort deadline expired or full work limit does not fit')
    return cutoff
