"""Memoize repeated ancestry only within one inherited-unit lookup."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

from moe_exp.research_campaign_v3_r6.contracts import require, file_sha

_scope = ContextVar('r6_inherited_lookup_scope', default=None)
_epoch = ContextVar('r6_read_only_ancestry_epoch', default=None)


@contextmanager
def read_only_epoch():
    if _epoch.get() is not None:
        yield
        return
    indexes = {}
    token = _epoch.set(indexes)
    try:
        yield
    finally:
        changed = []
        for key, expected in indexes.items():
            directory = Path(key)
            current = tuple(sorted(p.name for p in directory.iterdir())) if directory.is_dir() else ()
            if current != expected:
                changed.append(key)
        _epoch.reset(token)
        require(not changed, 'Inherited unit directory changed during read-only lookup: ' + ', '.join(changed))


def starts(directory, uid):
    directory = Path(directory)
    indexes = _epoch.get()
    if indexes is None:
        return sorted(directory.glob(uid + '.start-*.json'))
    key = str(directory)
    if key not in indexes:
        indexes[key] = tuple(sorted(p.name for p in directory.iterdir())) if directory.is_dir() else ()
    names = indexes[key]
    return sorted(directory / name for name in names
                  if name.startswith(uid + '.start-') and name.endswith('.json'))


def implementation(root, uid, seen=None):
    """Same depth-first contract, with verified snapshots only for read-only checks."""
    from moe_exp.research_campaign_v3_r6 import resume
    root = Path(root)
    seen = set() if seen is None else seen
    identity = str(root.resolve())
    require(identity not in seen, 'inherited unit ancestry cycle')
    seen.add(identity)
    path = root.parent / 'INHERITED_UNITS.json'
    if not path.is_file():
        return None
    inherited = resume.load(path)
    directories = inherited['directories'].get(root.name, [])
    if isinstance(directories, str):
        directories = [directories]
    for directory in directories:
        result = Path(directory) / (uid + '.result.json')
        if result.is_file():
            value = resume.load(result)
            if value.get('status') in inherited['compatible_statuses']:
                return {**value, 'inherited_receipt': str(result),
                        'inherited_receipt_sha256': file_sha(result)}
        original_starts = starts(directory, uid)
        if original_starts:
            return {'status': 'started', 'starts': [resume.load(p) for p in original_starts],
                    'inherited_directory': directory}
        value = resume.inherited_state(directory, uid, set(seen))
        if value is not None:
            return value
    return None


def scoped_lookup(original):
    def lookup(root, uid, seen=None):
        memo = _scope.get()
        top = memo is None
        token = _scope.set({'identities': {}, 'values': {}}) if top else None
        if top:
            memo = _scope.get()
        try:
            lexical = str(root)
            identities = memo['identities']
            if lexical not in identities:
                identities[lexical] = str(Path(root).resolve())
            identity = identities[lexical]
            require(identity not in (seen or set()), 'inherited unit ancestry cycle')
            key = (identity, uid)
            values = memo['values']
            if key not in values:
                values[key] = original(root, uid, seen)
            return values[key]
        finally:
            if top:
                _scope.reset(token)
    return lookup


def install():
    from moe_exp.research_campaign_v3_r6 import resume
    if getattr(resume.inherited_state, '_scoped_lookup_r1', False):
        return
    wrapped = scoped_lookup(implementation)
    wrapped._scoped_lookup_r1 = True
    resume.inherited_state = wrapped


def install_forecasts():
    """End the snapshot and verify entries before a caller can submit its wave."""
    install()
    from functools import wraps
    from moe_exp.research_campaign_v3_r6 import controller
    for name in ('_measurement_forecast', '_production_plan', 'ready'):
        original = getattr(controller, name)
        if getattr(original, '_scoped_lookup_r1', False):
            continue
        def bind(function):
            @wraps(function)
            def run(*args, **kwargs):
                with read_only_epoch():
                    return function(*args, **kwargs)
            run._scoped_lookup_r1 = True
            return run
        setattr(controller, name, bind(original))
