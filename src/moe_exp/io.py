"""Stable receipt primitives and durable per-unit execution states.

The primitives (digest, publish, verified, private, locked, ...) are vendored verbatim from the former
`final_routing.common`; `Units` is the former `study.common.Units`. `read_source` is vendored from
`research_campaign_v3_r6.inputs`. `iter_jsonl` and `write_jsonl` are the former `jsonl.iter_jsonl` and
`utils.write_jsonl`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
WORK = Path('/leonardo_work/IscrC_MIOSR/lmolfett')
HOME = Path('/leonardo/home/userexternal/lmolfett')
FAST = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs')
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')
BENCHMARKS = ('aime24', 'aime25', 'amc23', 'math500', 'minerva', 'olympiad')
ARMS = ('native', 'target_activate', 'target_deactivate', 'random_activate', 'random_deactivate')
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')
TERMINAL = {'complete', 'unknown', 'execution_failed', 'upstream_unavailable',
            'insufficient_support', 'resource_limited'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def private(path):
    path = Path(path).resolve()
    require(any(path.is_relative_to(p) for p in (WORK, HOME, FAST)),
            f'path outside authorized private storage: {path}')
    return path


@contextmanager
def locked(root, name='journal'):
    root = private(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / (digest(name) + '.lock')).open('a') as stream:
        os.chmod(stream.name, 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def publish(path, value):
    """Atomic, exclusive publication. An existing receipt must match exactly."""
    path = private(path)
    payload = clean({k: v for k, v in value.items() if k != 'sha256'})
    payload['sha256'] = digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        require(load(path) == payload, f'immutable receipt conflict: {path}')
        return payload
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(payload) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(load(path) == payload, f'concurrent receipt conflict: {path}')
        require(load(path) == payload, f'publication readback differs: {path}')
    finally:
        Path(tmp).unlink(missing_ok=True)
    return payload


def verified(path):
    value = load(path)
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            f'receipt hash differs: {path}')
    return value


def check_files(binding):
    for path, expected in binding.items():
        require(Path(path).is_file() and file_sha(path) == expected, f'bound input changed: {path}')


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


def read_source(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location["path"]).open("rb") as f:
        f.seek(location["byte_offset"])
        raw = f.read(location["line_bytes"])
    require(hashlib.sha256(raw).hexdigest() == location["line_sha256"], "saved trace line changed")
    return json.loads(raw)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL objects without treating Unicode line separators as delimiters."""
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                error.add_note(f"Invalid JSONL record {line_number} in {path}")
                raise


def write_jsonl(records: list, path: Path) -> None:
    """Write JSONL atomically so an interrupted run is never a completion marker."""
    import jsonlines

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with jsonlines.open(tmp_path, mode="w") as writer:
            for rec in records:
                writer.write(rec.model_dump() if hasattr(rec, "model_dump") else rec)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
