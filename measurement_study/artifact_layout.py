"""Separate approved sensitivity results and bind analyses to observed captures."""
import os
from pathlib import Path
from measurement_study.production_plan import ROOT, sha
from measurement_study.gemma_cache_policy import VARIANT


def result_dir(source, stage, variant=None):
    if stage not in ('production-analysis', 'production-supplement'):
        raise ValueError('Unknown result stage')
    variant = os.environ.get('MEASUREMENT_RESULT_VARIANT') if variant is None else variant
    if variant and (source != 'gemma' or variant != VARIANT):
        raise ValueError('Unverified result variant')
    slug = 'gemma-bf16-kv-sensitivity' if variant else source
    return ROOT / 'results/measurement-study' / stage / slug


def require_result_variant(source, manifests):
    variants = {m['spec'].get('replay_variant', '') for m in manifests}
    expected = os.environ.get('MEASUREMENT_RESULT_VARIANT', '')
    if variants != {expected}:
        raise ValueError('Capture and analysis variants disagree')
    result_dir(source, 'production-analysis', expected)
    return expected


def capture_binding(storage, source, manifests):
    return dict(manifest_bindings=[m['binding'] for m in manifests],
                capture_result_sha256={str(i): sha(storage/'capture'/source/str(i)/'result.json') for i in range(4)})
