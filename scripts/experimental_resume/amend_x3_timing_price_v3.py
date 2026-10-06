"""Seal the ready, timing-only X3 pilot resource amendment and launcher binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPORT = REPO / 'report/experimental-resume-v1'
SNAPSHOT = S / 'code/s1-9a61e32f48c04c24'
OUT = REPORT / 'X3_TIMING_PILOT_PRICE_v3.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def main():
    v2 = sealed(REPORT / 'X3_TIMING_PILOT_PRICE_v2.json')
    manifest = sealed(Path(v2['manifest_path']))
    pilot = sealed(REPORT / 'CAUSAL_MICROSCREEN_FULL_STAGE_PRICE_v1.json')
    snapshot = json.loads((SNAPSHOT / 'MANIFEST.json').read_text())
    if (v2['status'] != 'PROPOSED_GPU_HOLD_NEW_STUDY_PRIORITY' or
        manifest['sha256'] != v2['manifest_sha256'] or
        v2['proposed_job_walltime_seconds'] != 7200 or
        v2['proposed_stage_ceiling_GPUh'] != 4.0 or
        pilot['pilot_job_id'] != '59180099' or pilot['pilot_job_state'] != 'COMPLETED' or
        pilot['pilot_exit_code'] != '0:0' or
        snapshot['tree_sha256'] != manifest['code_tree'] or
        len(manifest['requests']) != 40 or manifest['n_shards'] != 1):
        raise ValueError('X3 manifest, bounded price, or completed causal pilot differs')
    out = Path(v2['manifest_path']).parent / ('results-' + manifest['sha256'][:16])
    if out.exists() and any(out.iterdir()):
        raise ValueError('fresh digest-bound result directory is already populated')
    body = {
        'schema': 'x3-32k-timing-pilot-price-v3',
        'status': 'READY_TIMING_ONLY',
        'source_price_sha256': v2['sha256'],
        'causal_priority_gate': {
            'scope': 'legacy cost-timing pilot only; new full-prefix causal audit remains independent and retains priority',
            'completed_qualification_job_id': '59180099',
            'completed_qualification_state': 'COMPLETED',
            'completed_qualification_exit_code': '0:0',
            'sealed_qualification_price_sha256': pilot['sha256'],
            'completed_v22_scout_job_id': '59180589',
            'completed_v22_scout_state': 'COMPLETED',
            'completed_v22_scout_exit_code': '0:0',
        },
        'scope': v2['scope'],
        'manifest_path': str(Path(v2['manifest_path']).resolve()),
        'manifest_sha256': manifest['sha256'],
        'request_count': 40,
        'max_decode_tokens': v2['max_decode_tokens'],
        'prefill_tokens_without_cache_reuse': v2['prefill_tokens_without_cache_reuse'],
        'builder_sha256': sha(REPO / 'scripts/experimental_resume/x3_build_v2.py'),
        'preparation_sha256': sha(REPO / 'scripts/experimental_resume/prepare_x3_timing_pilot.py'),
        'launcher_sha256': sha(REPO / 'scripts/experimental_resume/launch_x3_timing_pilot_v3.py'),
        'planner_sha256': sha(SNAPSHOT / 'scripts/steer_submit.py'),
        'snapshot_path': str(SNAPSHOT),
        'snapshot_manifest_file_sha256': sha(SNAPSHOT / 'MANIFEST.json'),
        'snapshot_tree_sha256': snapshot['tree_sha256'],
        'output_path': str(out),
        'output_binding_rule': 'fresh results-<manifest seal first16>; launcher writes immutable digest/price/code binding before one Slurm submission',
        'proposed_job_walltime_seconds': 7200,
        'proposed_GPUs': 2,
        'proposed_stage_ceiling_GPUh': 4.0,
        'projected_wall_seconds': v2['projected_wall_seconds'],
        'projected_walltime_margin_seconds': v2['projected_walltime_margin_seconds'],
        'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal',
                  'nodes': 1, 'gpus': 2, 'cpus': 16, 'memory': '120G', 'time': '02:00:00'},
        'resubmission_rule': 'No second allocation under this four-GPU-hour receipt; checkpoint can be resumed only after a separately priced amendment.',
        'post_run_allowed_fields': v2['post_run_allowed_fields'],
        'post_run_embargoed_fields': v2['post_run_embargoed_fields'],
        'full_X3': v2['full_X3'],
    }
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing v3 timing amendment differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'], 'output': str(out)}))


if __name__ == '__main__':
    main()
