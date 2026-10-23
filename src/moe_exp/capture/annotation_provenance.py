"""Validate the external annotation inputs bound to a capture plan."""
from pathlib import Path
from moe_exp.capture.bundle import file_sha


def verify_annotation_inputs(binding):
    if binding is None:
        return {}
    inputs = binding.get('inputs')
    if not isinstance(inputs, list) or not inputs or not binding.get('annotations_sha256'):
        raise ValueError('Missing annotation input binding')
    checked = {}
    for item in inputs:
        path = str(Path(item['path']).resolve())
        expected = item['sha256']
        if path in checked:
            if checked[path] != expected:
                raise ValueError('Conflicting annotation input hashes')
            continue
        if file_sha(path) != expected:
            raise ValueError('Annotation input changed since capture planning: ' + path)
        checked[path] = expected
    return checked
