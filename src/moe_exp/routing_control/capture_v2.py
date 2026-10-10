"""Binary capture provenance and durable, bounded whole-assignment attempts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .study_v1 import require, digest, file_sha, sealed, save
from .study_v2 import seal


def array_binding(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': value.dtype.str, 'bytes': value.nbytes,
            'sha256': hashlib.sha256(memoryview(value).cast('B') if value.size else b'').hexdigest()}


class CaptureStore:
    """Two ranks × native/executed × emitted rows, without JSON trace expansion."""
    def __init__(self, directory, cap):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows, self.cap = 0, cap
        self.ids = np.lib.format.open_memmap(self.directory / 'ids.npy', mode='w+', dtype=np.uint16, shape=(2, 2, cap, 40, 8))
        self.weights = np.lib.format.open_memmap(self.directory / 'weights.npy', mode='w+', dtype=np.float32, shape=(2, 2, cap, 40, 8))
        self.dose = np.lib.format.open_memmap(self.directory / 'dose.npy', mode='w+', dtype=np.float32, shape=(cap, 40, 2))
        self.tokens = np.lib.format.open_memmap(self.directory / 'tokens.npy', mode='w+', dtype=np.int32, shape=(cap,))
        self.journal = (self.directory / 'rows.jsonl').open('x')
        self.chain = digest(['routing-capture-v2', cap])

    def append(self, receipt, turnover, weight_l1_half):
        require(self.rows < self.cap, 'capture cap exceeded')
        ranks = receipt.get('rank_captures')
        require(ranks and len(ranks) == 2 and {r['rank'] for r in ranks} == {0, 1}, 'both-rank raw captures required')
        for rank in ranks:
            self.ids[rank['rank'], :, self.rows] = rank['ids']
            self.weights[rank['rank'], :, self.rows] = rank['weights']
        self.tokens[self.rows] = receipt['token_id']
        self.dose[self.rows, :, 0] = turnover
        self.dose[self.rows, :, 1] = weight_l1_half
        row = {'offset': self.rows, 'position': receipt['row_position'], 'token_id': receipt['token_id'],
               'policy': receipt['policy_name'], 'previous_sha256': self.chain,
               'capture_sha256': digest([array_binding(self.ids[:, :, self.rows]), array_binding(self.weights[:, :, self.rows])])}
        row = seal(row)
        self.chain = row['sha256']
        self.rows += 1
        # Data precedes its durable cursor. Process death can leave only an
        # uncommitted tail; readers accept exactly the journal's valid rows.
        self.journal.write(json.dumps(row, separators=(',', ':')) + '\n')
        if self.rows % 256 == 0:
            self.flush()

    def flush(self):
        self.ids.flush()
        self.weights.flush()
        self.dose.flush()
        self.tokens.flush()
        self.journal.flush()

    def close(self):
        self.flush()
        self.journal.close()
        arrays = {'ids': array_binding(self.ids[:, :, :self.rows]),
                  'weights': array_binding(self.weights[:, :, :self.rows]),
                  'dose': array_binding(self.dose[:self.rows]), 'tokens': array_binding(self.tokens[:self.rows])}
        return save(self.directory / 'BINDING.json', {'schema': 'routing-binary-capture-v2', 'rows': self.rows,
            'capacity': self.cap, 'rank_axis': [0, 1], 'routing_axis': ['native', 'executed'], 'arrays': arrays,
            'cursor_sha256': self.chain, 'files': {n: file_sha(self.directory / n) for n in
                ('ids.npy', 'weights.npy', 'dose.npy', 'tokens.npy', 'rows.jsonl')},
            'allocated_storage_bytes': sum((self.directory / n).stat().st_size for n in
                ('ids.npy', 'weights.npy', 'dose.npy', 'tokens.npy', 'rows.jsonl'))})


def begin_attempt(root, assignment, manifest_sha256):
    """After process death the sole retry starts from the original frozen input."""
    root = Path(root)
    assignment = json.loads(json.dumps(assignment, allow_nan=False))
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / 'SUCCESS.json').exists(), 'assignment already has a successful result')
    attempts = sorted(root.glob('attempt-*/ASSIGNMENT.json'))
    require(len(attempts) < 2, 'one whole-assignment retry exhausted')
    for path in attempts:
        prior = sealed(path)
        require(prior['assignment'] == assignment and prior['manifest_sha256'] == manifest_sha256,
                'retry source or manifest changed')
        require(not (path.parent / 'SUCCESS.json').exists(), 'successful attempt cannot be retried')
        if not (path.parent / 'FAILURE.json').exists():
            save(path.parent / 'FAILURE.json', {'schema': 'routing-interrupted-attempt-v2',
                'status': 'PROCESS_DEATH_OR_UNCOMMITTED_RESULT', 'actual_cost': 'PARTIAL_CAPTURE_AND_SLURM_ACCOUNTING_REQUIRED',
                'continuation': False})
    directory = root / f'attempt-{len(attempts)}'
    directory.mkdir(exist_ok=False)
    save(directory / 'ASSIGNMENT.json', {'schema': 'routing-assignment-attempt-v2', 'assignment': assignment,
        'manifest_sha256': manifest_sha256, 'attempt': len(attempts), 'retry_scope': 'WHOLE_ASSIGNMENT_FROM_ORIGINAL_PREFIX',
        'continued_execution': False, 'maximum_attempts': 2})
    return directory


def commit_attempt(root, attempt, receipt):
    root, attempt = Path(root), Path(attempt)
    require(not (root / 'SUCCESS.json').exists(), 'duplicate successful result')
    source = sealed(attempt / 'ASSIGNMENT.json')
    require(receipt['assignment_uid'] == source['assignment']['uid'], 'result belongs to another assignment')
    result = save(attempt / 'SUCCESS.json', {'schema': 'routing-assignment-success-v2', 'receipt': receipt,
        'attempt_sha256': source['sha256']})
    return save(root / 'SUCCESS.json', {'schema': 'routing-assignment-result-v2', 'result_path': str(attempt / 'SUCCESS.json'),
        'result_sha256': result['sha256'], 'assignment_uid': receipt['assignment_uid'], 'accepted_successes': 1,
        'attempt': source['attempt'], 'failed_attempts_retained': True})
