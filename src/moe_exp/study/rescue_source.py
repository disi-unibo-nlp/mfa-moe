"""Bind rescue sources and prevent reuse across incompatible executions.

Historical generation receipts remain immutable. A future release needs an
explicit new source receipt and a fresh configured HOME or WORK store; this
module never adopts or migrates the existing unbound rescue outputs.
"""
from __future__ import annotations
from pathlib import Path
from .common import HOME, REPO, WORK, digest, file_sha, publish, require, verified

REPORT = REPO/'report'/'final-routing-rescue-20261019-r1'
RECEIPT = REPORT/'SOURCE_BINDING-r8.json'
STUDY_BINDING = REPO/'report/final-routing-20261015-r1/BINDING.json'


def _scientific_binding(study: dict) -> dict:
    """Remove only the allowed output and control location differences."""
    return {
        **{key: value for key, value in study.items() if key not in ('sha256', 'control_root', 'config')},
        'config': {key: value for key, value in study['config'].items() if key not in ('bulk_root', 'control_root')},
    }


def load_study_binding(path: Path | None = None) -> dict:
    """Verify a selected binding while preserving the original scientific design.

    Args:
        path: Fresh binding; only bulk_root and control_root locations may differ.

    Returns:
        The verified binding, without changing the historical file.
    """
    original = verified(STUDY_BINDING)
    selected = verified(path or STUDY_BINDING)
    require(_scientific_binding(selected) == _scientific_binding(original),
            'fresh rescue binding may change only config.bulk_root and control_root locations')
    return selected


def paths(study_binding_path: Path | None = None) -> dict[str, Path]:
    """Include the repository Python closure, entrypoints and frozen input joins."""
    rel = (
        'scripts/slurm/rescue_generate.sbatch',
        'scripts/slurm/rescue_outcomes.sbatch',
        'scripts/slurm/rescue_analyze.sbatch',
        'scripts/slurm/rescue_reader_plan.sbatch',
        'scripts/slurm/rescue_reader.sbatch',
        'scripts/slurm/rescue_stage.sbatch',
        'scripts/research_campaign_v3_r6/renderer.py',
        'configs/final_routing.yaml',
    )
    binding_path = Path(study_binding_path or STUDY_BINDING).resolve()
    study = load_study_binding(binding_path)
    required = set((REPO/'src/moe_exp').rglob('*.py')) | {REPO/path for path in rel}
    required.update((REPO/'tests/study').glob('*.py'))
    required.add(binding_path)
    required.add(STUDY_BINDING)
    required.add(Path(study['source_campaign'])/'CAMPAIGN.json')
    for model in ('gpt', 'qwen36'):
        required.add(STUDY_BINDING.parent/'driver'/model/'RESCUE.json')
        required.add(Path(study['source_campaign'])/model/'SOURCES.json')
    require(all(path.is_file() for path in required), 'rescue source or frozen input closure is incomplete')
    return {str(path): path for path in sorted(required)}


def runtime_binding() -> dict:
    """Reuse the installed selector, kernel and package fingerprint without model loading."""
    from moe_exp.verification_study.prepare import runtime_binding as native_runtime_binding
    import importlib.metadata
    native = native_runtime_binding()
    root = Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    engine = set((root/'v1/engine').glob('*.py'))
    engine.update(root/name for name in ('sampling_params.py', 'entrypoints/llm.py', 'envs.py',
                                       'utils/network_utils.py', 'distributed/device_communicators/shm_broadcast.py'))
    return {
        'native': native,
        'request_engine_files': {str(path): file_sha(path) for path in sorted(engine)},
        'analysis_packages': {name: importlib.metadata.version(name)
                              for name in ('scipy', 'pandas', 'pyarrow', 'tokenizers', 'matplotlib', 'pyzmq')},
    }


def build(receipt_path: Path | None = None, *, study_binding_path: Path | None = None) -> dict:
    """Publish a new immutable source receipt at an explicitly chosen path.

    Args:
        receipt_path: New release receipt; the historical default is never overwritten.
        study_binding_path: Explicit fresh storage binding with the original science.

    Returns:
        The verified source payload, including installed execution dependencies.
    """
    binding_path = Path(study_binding_path or STUDY_BINDING).resolve()
    load_study_binding(binding_path)
    import subprocess
    marker = REPO/'RELEASE_COMMIT'
    release_commit = (marker.read_text().strip() if marker.is_file() else
                      subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    require(len(release_commit) == 40 and all(c in '0123456789abcdef' for c in release_commit),
            'invalid source release commit')
    files = {str(path): file_sha(path) for path in paths(binding_path).values()}
    value = {'schema': 'final-routing-rescue-source-v2',
             'purpose': 'score-override rescue source receipt',
             'release_commit': release_commit,
             'study_binding_path': str(binding_path),
             'files': files,
             'runtime': runtime_binding()}
    return publish(receipt_path or RECEIPT, value)


def verify(receipt_path: Path | None = None) -> dict:
    """Check an explicit source receipt against current code, inputs and runtime.

    Args:
        receipt_path: Selected release receipt, defaulting to the historical r8 path.

    Returns:
        The unchanged receipt after every dependency matches.
    """
    value = verified(receipt_path or RECEIPT)
    binding_path = Path(value.get('study_binding_path', STUDY_BINDING))
    load_study_binding(binding_path)
    current = {str(path): file_sha(path) for path in paths(binding_path).values()}
    require(value.get('files') == current, 'rescue source receipt differs from current files')
    require(value.get('runtime') == runtime_binding(), 'rescue installed runtime differs from source receipt')
    return value


def execution_identity(study: dict, model: str, manifest: dict, source: dict) -> dict:
    """Identify the exact source, manifest, sampling profile and storage binding.

    Args:
        study: Configured study binding, including the private output location.
        model: Model whose entire rescue manifest is being executed.
        manifest: Verified full manifest, before shard selection.
        source: Source receipt already verified against the installed runtime.

    Returns:
        Identity payload shared by every partial and terminal unit in this store.
    """
    require(manifest.get('model') == model, 'rescue manifest belongs to another model')
    require(source.get('schema') == 'final-routing-rescue-source-v2',
            'rescue execution needs a new source receipt; historical receipts remain read-only')
    return {
        'schema': 'final-routing-rescue-execution-v1',
        'model': model,
        'source_sha256': source['sha256'],
        'manifest_sha256': digest(manifest),
        'profile_sha256': digest(study['models'][model]),
        'study_binding_sha256': digest(study),
    }


def bind_execution(study: dict, model: str, manifest: dict, source: dict) -> dict:
    """Reject an unbound or incompatible store before recovery or engine creation.

    Args:
        study: Configured study with a fresh user-owned HOME or WORK bulk root.
        model: Rescue model being dispatched.
        manifest: Verified full manifest, including token limits and capture layers.
        source: Verified source and installed runtime receipt.

    Returns:
        Immutable execution receipt for the exact configured unit store.
    """
    from .driver import STORES, bulk

    units = (bulk(study)/model/STORES['rescue']).resolve()
    require(any(units.is_relative_to(root) for root in (HOME, WORK)),
            'rescue execution requires a fresh HOME/WORK binding; existing FAST outputs are read-only')
    receipt_path = units.parent/'EXECUTION_IDENTITY.json'
    identity = execution_identity(study, model, manifest, source)
    if receipt_path.exists():
        existing = verified(receipt_path)
        require({key: value for key, value in existing.items() if key != 'sha256'} == identity,
                'rescue execution identity differs; preserve this store and use a fresh HOME/WORK binding')
        return existing
    require(not units.exists() or not any(units.iterdir()),
            'rescue store contains unbound prior outputs or partials; preserve them and use a fresh HOME/WORK binding')
    return publish(receipt_path, identity)
