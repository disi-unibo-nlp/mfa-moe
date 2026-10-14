"""Bind this amendment without changing CAMPAIGN.json or the inference environment."""
from importlib.metadata import distribution
from pathlib import Path
import subprocess

from moe_exp.research_campaign_v3_r6.contracts import REPO, check_files, file_sha, load, publish, require

PAPER = REPO / 'paper/6a510c6db0d9b08aa438c142'


def git_state(directory):
    result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True, timeout=30)
    index = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', '--git-path', 'index'],
                                    text=True, timeout=30).strip()
    path = Path(index)
    if not path.is_absolute():
        path = directory / path
    return {'head': result.stdout.strip(), 'index_sha256': file_sha(path) if path.exists() else None}


def manuscript_state():
    return {'files': {str(p): file_sha(p) for p in sorted(PAPER.rglob('*'))
                      if p.is_file() and '.git' not in p.relative_to(PAPER).parts},
            'repository': git_state(REPO), 'overleaf': git_state(PAPER)}


def prepare(campaign):
    root = Path(campaign['root'])
    check_files(campaign['code_files']); check_files(campaign['source_files'])
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((REPO / 'scripts/research_campaign_v3_r6_ops_r1').glob('*'))
    paths += list((REPO / 'tests/research_campaign_v3_r6_ops_r1').glob('*.py'))
    paths += [REPO / ('scripts/research_campaign_v3_r6_validation_candidate_' + name + '.py') for name in ('r4', 'r5')]
    paths += list((REPO / 'tests/research_campaign_v3_r6_validation_r2').glob('*.py'))
    names = ('v1/engine/output_processor.py', 'v1/engine/core_client.py', 'sampling_params.py',
             'envs.py', 'v1/engine/__init__.py')
    runtime = {str(distribution('vllm').locate_file('vllm/' + name)):
               file_sha(distribution('vllm').locate_file('vllm/' + name)) for name in names}
    binding = publish(root / 'OPS_R1_BINDING.json', {
        'campaign_sha256': campaign['sha256'], 'files': {str(p): file_sha(p) for p in sorted(paths) if p.is_file()},
        'runtime_files': runtime, 'remote_writes_allowed': False, 'manuscript_writes_allowed': False,
        'scientific_identity_fields_unchanged': True, 'original_seal_file_count': len(campaign['code_files']),
        'validation_attempts': 1, 'validation_cpus': 4, 'validation_memory_gib': 24,
        'validation_minutes': 180, 'validation_reservation_hours': 12, 'validation_hours_with_margin': 15,
        'authorization': 'User: PLEASE IMPLEMENT THIS PLAN / Implement the plan, 2026-10-14',
        'worker_entrypoint': str(REPO / 'scripts/research_campaign_v3_r6_ops_r1/job.sbatch'),
        'reader_amendment': 'external request IDs for interrupted cleanup only',
        'preserved_failed_jobs': ['60481682', '60484614', '60488160'],
    })
    return binding
