"""Seal the timing-only X3 recovery after the v3 runner admission-margin failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPORT = REPO / 'report/experimental-resume-v1'
SNAPSHOT = S / 'code/s1-9a61e32f48c04c24'
OUT = REPORT / 'X3_TIMING_PILOT_PRICE_v4.json'


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
    v3 = sealed(REPORT / 'X3_TIMING_PILOT_PRICE_v3.json')
    manifest = sealed(Path(v3['manifest_path']))
    previous = Path(v3['output_path']) / 'shard-0.status.json'
    prior = json.loads(previous.read_text())
    snapshot = json.loads((SNAPSHOT / 'MANIFEST.json').read_text())
    runner = SNAPSHOT / 'moe_steer/runner.py'
    runner_text = runner.read_text()
    if (v3['status'] != 'READY_TIMING_ONLY' or manifest['sha256'] != v3['manifest_sha256'] or
        len(manifest['requests']) != 40 or manifest['n_shards'] != 1 or
        snapshot['tree_sha256'] != manifest['code_tree'] or
        prior.get('manifest_sha256') != manifest['sha256'] or prior.get('n_done') != 0 or
        prior.get('tokens_this_run') != 0 or prior.get('status') != 'deadline' or
        'DEFAULT_MARGIN_S = 3 * 3600' not in runner_text or
        'parser.add_argument("--margin-seconds"' not in runner_text):
        raise ValueError('pilot failure cause or frozen manifest differs')
    output = Path(v3['manifest_path']).parent / ('results-' + manifest['sha256'][:16] + '-v4')
    if output.exists() and any(output.iterdir()):
        raise ValueError('recovery output already populated')
    observed_build = float(prior['engine_build_seconds'])
    projected_generation = float(v3['projected_wall_seconds'])
    proposed_wall = 10800
    admission_margin = 900
    reserve = 600
    projected_total = observed_build + projected_generation + reserve
    if projected_total >= proposed_wall - admission_margin:
        raise ValueError('complete pilot does not fit guarded request-admission window')
    body = {
        'schema': 'x3-32k-timing-pilot-price-v4',
        'status': 'READY_TIMING_ONLY_RECOVERY',
        'source_price_sha256': v3['sha256'],
        'previous_job_id': '59185118',
        'previous_job_state': 'COMPLETED',
        'previous_job_exit_code': '0:0',
        'previous_status_sha256': sha(previous),
        'previous_result': {'n_done': 0, 'tokens_this_run': 0, 'status': 'deadline',
                            'engine_build_seconds': observed_build,
                            'cause': 'snapshot runner default three-hour admission margin exceeded two-hour allocation'},
        'scope': v3['scope'],
        'manifest_path': v3['manifest_path'],
        'manifest_sha256': manifest['sha256'],
        'request_count': 40,
        'max_decode_tokens': v3['max_decode_tokens'],
        'prefill_tokens_without_cache_reuse': v3['prefill_tokens_without_cache_reuse'],
        'builder_sha256': v3['builder_sha256'],
        'preparation_sha256': v3['preparation_sha256'],
        'launcher_sha256': sha(REPO / 'scripts/experimental_resume/launch_x3_timing_pilot_v4.py'),
        'planner_sha256': v3['planner_sha256'],
        'snapshot_path': v3['snapshot_path'],
        'snapshot_manifest_file_sha256': v3['snapshot_manifest_file_sha256'],
        'snapshot_tree_sha256': v3['snapshot_tree_sha256'],
        'runner_sha256': sha(runner),
        'runner_flags': ['--margin-seconds', '900', '--abort-margin-seconds', '600'],
        'output_path': str(output),
        'output_binding_rule': 'fresh results-<manifest seal first16>-v4; launcher writes immutable digest/price/code binding',
        'proposed_job_walltime_seconds': proposed_wall,
        'proposed_GPUs': 2,
        'proposed_stage_ceiling_GPUh': 6.0,
        'projected_components_seconds': {'observed_engine_build': observed_build,
                                         'prior_generation_model_for_40_capped_requests': projected_generation,
                                         'retry_and_shutdown_reserve': reserve,
                                         'guarded_total': projected_total},
        'projected_admission_window_seconds': proposed_wall - admission_margin,
        'projected_admission_slack_seconds': proposed_wall - admission_margin - projected_total,
        'uncertainty': 'Previous pilot yielded no decode throughput; generation projection remains historical, not measured on 32k batch. Stop/resume with a new receipt if the 40 requests do not finish.',
        'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal',
                  'nodes': 1, 'gpus': 2, 'cpus': 16, 'memory': '120G', 'time': '03:00:00'},
        'resubmission_rule': 'No second allocation under this six-GPU-hour receipt; checkpoint resume requires separately priced amendment.',
        'post_run_allowed_fields': v3['post_run_allowed_fields'],
        'post_run_embargoed_fields': v3['post_run_embargoed_fields'],
        'full_X3': v3['full_X3'],
    }
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing v4 timing amendment differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'], 'output': str(output),
                      'guarded_total_seconds': projected_total}))


if __name__ == '__main__':
    main()
