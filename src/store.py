"""Shared basics: sealed receipts, per-unit execution states, trace-line reading and JSONL files.

A receipt is a JSON file with its own content hash under "sha256". `publish` writes one atomically and
refuses to change an existing one; `verified` reads one back and checks the hash. `digest` hashes any
JSON-like value after `clean` has made it JSON-ready (numpy to Python, tuples to lists, missing and
non-finite numbers to null).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
WORK = Path('/leonardo_work/IscrC_MIOSR/lmolfett')
HOME = Path('/leonardo/home/userexternal/lmolfett')
FAST = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs')
# Saved analyses outside the repo that the seven-model readout, next-state and benchmark code read.
ANALYSIS = WORK/'tools/tmp/claude-analysis-2026-09-24'
STUDY = Path(os.environ.get('STUDY_DATA_ROOT', REPO/'report/final-routing-20261015-r1'))
DRIVER = STUDY/'driver'
SEED = 20261008
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')
BENCHMARKS = ('aime24', 'aime25', 'amc23', 'math500', 'minerva', 'olympiad')
# The same six benchmarks in the order the sentence samplers and the labeller walk them. The order
# is hashed into every sampling plan and labelling key, so it must never change (it is not
# BENCHMARKS' order).
DATASETS = ('math500', 'aime24', 'aime25', 'olympiad', 'amc23', 'minerva')
ARMS = ('native', 'target_activate', 'target_deactivate', 'random_activate', 'random_deactivate')
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')
TERMINAL = {'complete', 'unknown', 'execution_failed', 'upstream_unavailable',
            'insufficient_support', 'resource_limited'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value, reasons=None, path=''):
    """JSON-ready copy: string keys, lists for tuples and arrays, Python numbers, null for missing or non-finite.

    If `reasons` is a dict, it gets one entry per null, keyed by its path, saying why."""
    kind = type(value)
    if kind in (str, int, bool):
        return value
    if kind is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        return {str(k): clean(v, reasons, f'{path}/{k}') for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, reasons, f'{path}/{i}') for i, v in enumerate(value)]
    if value is None or kind.__name__ in ('NAType', 'NaTType'):
        if reasons is not None:
            reasons[path] = 'missing_input'
        return None
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            if reasons is not None:
                reasons[path] = 'non_finite_input'
            return None
        return value.item() if hasattr(value, 'item') else value
    if hasattr(value, 'tolist'):
        return clean(value.tolist(), reasons, path)
    return value


def _dump(cleaned):
    return json.dumps(cleaned, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def canonical(value):
    return _dump(clean(value))


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    """Clean copy of a dict with its content hash under "sha256"."""
    payload = clean({k: v for k, v in value.items() if k != 'sha256'})
    return {**payload, 'sha256': hashlib.sha256(_dump(payload)).hexdigest()}


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
    payload = seal(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        require(load(path) == payload, f'immutable receipt conflict: {path}')
        return payload
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_dump(payload) + b'\n')
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


def write_once(path, write):
    """Write a data file atomically through `write(stream)`; a file already there must have the same bytes.

    Returns {'path', 'sha256'} of the file."""
    path = private(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(file_sha(tmp) == file_sha(path), f'conflicting immutable data file: {path}')
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(path):
    value = load(path)
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            f'receipt hash differs: {path}')
    return value


def binding():
    """The eight-model study's BINDING.json: its design, models and the bulk-data root."""
    return load(STUDY/'BINDING.json')


def check_files(files):
    """Every bound file must still exist with the recorded sha256."""
    for path, expected in files.items():
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
