"""Exclusive, digest-bound measurement receipts with atomic crash recovery."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    pending = path.with_name('.' + path.name + '.pending')
    with pending.open('w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


class ReceiptStore:
    """A failed batch can be retried; published UIDs cannot be duplicated or replaced."""
    def __init__(self, out, binding, assigned_uids):
        self.out = Path(out)
        self.binding = binding
        self.assigned = set(assigned_uids)
        if len(self.assigned) != len(assigned_uids):
            raise ValueError('duplicate measurement assignment')
        length = 32 if binding.get('uid_format') == 'legacy-sha256-prefix32' else 64
        if any(len(uid) != length or set(uid) - set('0123456789abcdef') for uid in self.assigned):
            raise ValueError('measurement UIDs must match the explicitly bound lowercase SHA256 format')
        self.lock = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / '.measurement.lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            binding = self.out / 'INPUT_BINDING.json'
            if binding.exists():
                if json.loads(binding.read_text()) != self.binding:
                    raise ValueError('measurement directory has a different input/code binding')
            else:
                if any(p.name not in ('.measurement.lock', '.INPUT_BINDING.json.pending')
                       for p in self.out.iterdir()):
                    raise ValueError('refusing an unbound nonempty measurement directory')
                atomic_json(binding, self.binding)
            (self.out / 'records').mkdir(exist_ok=True)
            (self.out / 'attempts').mkdir(exist_ok=True)
            self.read()
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *_):
        self.lock.close()
        self.lock = None

    def read(self):
        rows = {}
        for path in sorted((self.out / 'records').glob('*.json')):
            row = json.loads(path.read_text())
            uid = row['uid']
            if uid not in self.assigned or path.stem != uid:
                raise ValueError('unknown or misnamed measurement receipt')
            rows[uid] = row
        return rows

    def put(self, row):
        uid = row['uid']
        if uid not in self.assigned:
            raise ValueError('unassigned measurement UID')
        path = self.out / 'records' / (uid + '.json')
        if path.exists():
            if json.loads(path.read_text()) != row:
                raise ValueError('refusing to replace a completed measurement')
            return
        atomic_json(path, row)

    def attempt(self):
        existing = [int(p.name) for p in (self.out / 'attempts').iterdir() if p.is_dir()]
        path = self.out / 'attempts' / str(max(existing, default=0) + 1)
        path.mkdir()
        return path
