"""Stable receipt primitives and durable study execution states."""
from pathlib import Path
import time
from moe_exp.final_routing.common import (
    REPO, WORK, HOME, FAST, CLASSES, BENCHMARKS, ARMS, MODELS, TERMINAL,
    require, clean, canonical, digest, file_sha, load, private, locked, publish,
    verified, check_files,
)


class Units:
    """Untouched -> intent -> admitted -> output -> result; no ambiguous redraw."""
    def __init__(self, root):
        self.root = private(root)

    def folder(self, uid):
        require(len(uid) == 64 and all(c in '0123456789abcdef' for c in uid), 'invalid execution ID')
        return self.root / uid

    def state(self, uid):
        folder = self.folder(uid)
        if (folder / 'RESULT.json').exists():
            return verified(folder / 'RESULT.json')
        if (folder / 'OUTPUT.json').exists():
            return {'status': 'output', 'path': str(folder / 'OUTPUT.json')}
        if (folder / 'INTENT.json').exists() or (folder / 'ADMITTED.json').exists():
            return {'status': 'unknown', 'ambiguous_admission': True,
                    'admitted_receipt': (folder / 'ADMITTED.json').exists()}
        return None

    def intent(self, row, job_id):
        with locked(self.root, row['uid']):
            if self.state(row['uid']) is not None:
                return False
            publish(self.folder(row['uid']) / 'INTENT.json',
                    {'uid': row['uid'], 'assignment_sha256': digest(row),
                     'job_id': str(job_id), 'time': time.time()})
            return True

    def admitted(self, row, job_id, internal_id=None):
        identity = {"internal_request_id": internal_id} if internal_id is not None else {}
        return publish(self.folder(row['uid']) / 'ADMITTED.json',
                       {'uid': row['uid'], 'assignment_sha256': digest(row), 'job_id': str(job_id), **identity})

    def output(self, row, value):
        return publish(self.folder(row['uid']) / 'OUTPUT.json',
                       {'uid': row['uid'], 'assignment_sha256': digest(row), **value})

    def commit(self, row, value):
        require(self.state(row['uid']) is not None, 'cannot commit an untouched execution')
        return publish(self.folder(row['uid']) / 'RESULT.json',
                       {'uid': row['uid'], 'assignment_sha256': digest(row), **value})


def control_root(binding):
    return Path(binding.get('control_root', binding['root']))


def engineering_namespace(binding, role):
    return 'engineering/' + binding['source_commit'] + '/' + role


def submission_paths(root):
    root = Path(root)
    return [*(root / 'submissions').glob('*/ACCEPTED.json'),
            *root.glob('study/*/submissions/*/ACCEPTED.json')]


def reservation_paths(root):
    root = Path(root)
    return [*(root / 'reservations').glob('*.json'),
            *root.glob('study/*/reservations/*.json')]
