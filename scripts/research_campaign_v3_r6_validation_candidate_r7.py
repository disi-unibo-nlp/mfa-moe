#!/usr/bin/env python3
"""Unsubmitted completion of preserved validation; requires a new extension."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import (
    MODELS, allocated, check_files, file_sha, load, publish, require, STAGING,
)
from moe_exp.research_campaign_v3_r6_ops_r1 import policy
from moe_exp.research_campaign_v3_r6_ops_r1.binding import PAPER, git_state

spec = importlib.util.spec_from_file_location('active_source_builder', REPO / 'scripts/paper/build_campaign_paper_r4.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


def integrity_state():
    return {'active_manuscript_files': builder.source_state(),
            'repository': git_state(REPO), 'overleaf': git_state(PAPER)}


def validate(campaign):
    root = Path(campaign['root']); binding = load(root / 'OPS_R1_BUILD_RECOVERY_BINDING.json')
    require(binding['explicit_extension_authorized'] is True, 'Another allocation requires an explicit user extension')
    check_files(binding['files']); operations = policy.check_binding(campaign)
    require(binding['ops_binding_sha256'] == operations['sha256'], 'Operations ancestry differs')
    partial = load(root / 'OPS_R1_LOADER_PARTIAL_VALIDATION.json')
    require(partial['sha256'] == binding['partial_validation_sha256'] and
            partial['ops_binding_sha256'] == operations['sha256'] and
            partial['original_campaign_sha256'] == campaign['sha256'], 'Preserved validation source differs')
    require(file_sha(Path(partial['stderr_path'])) == partial['stderr_sha256'] and
            partial['operations_tests_passed'] == 20 and partial['inherited_tests_passed'] == 111,
            'Preserved actual contract evidence differs')
    allocated(); job = os.environ['SLURM_JOB_ID']; started = time.monotonic()
    terminal = root / 'OPS_R1_BUILD_RECOVERY_TERMINAL.json'
    require(not terminal.exists(), 'One corrected completion attempt is exhausted')
    start = root / 'OPS_R1_BUILD_RECOVERY_START.json'
    if start.exists(): require(load(start)['job_id'] == job, 'No automatic allocation retry is authorized')
    else: publish(start, {'job_id': job, 'binding_sha256': binding['sha256']})
    before = integrity_state()
    before_receipt = publish(root / 'ops_r1/BUILD_RECOVERY_MANUSCRIPT_BEFORE.json', before)
    try:
        policy.install(campaign)
        reader = load(root / 'READER_INPUT_VALIDATION.json')
        require(reader['sha256'] == partial['reader_validation_sha256'], 'Actual reader proof differs')
        directory = Path(partial['full_export_directory'])
        report = load(directory / 'REPORT.json'); manifest = load(directory / 'EXPORT_FILES.json')
        require(report['sha256'] == partial['full_export_report_sha256'] and
                manifest['sha256'] == partial['export_files_sha256'], 'Committed export dependencies differ')
        check_files(manifest['files'])
        require(file_sha(directory / 'EXPORT_STATE.json') == partial['export_state_sha256'], 'Export checkpoint differs')
        from moe_exp.research_campaign_v3_r6.export import closeout
        value = {**campaign, 'root': str(directory.parent), 'validation_only': True}
        replay = closeout(value)
        require(replay['REPORT_COMPLETE'] and not replay['SCIENCE_COMPLETE'] and
                file_sha(directory / 'EXPORT_STATE.json') == partial['export_state_sha256'], 'Actual checkpoint replay differs')
        build = STAGING / ('research-r6-active-validation-' + job)
        subprocess.run([sys.executable, '-B', str(REPO / 'scripts/paper/build_campaign_paper_r4.py'),
                        '--only-cached', '--output', str(build)], check=True, timeout=180)
        built = json.loads((build / 'BUILD.json').read_text())
        require(built['state'] == 'COMPILED_FULL_CLASS' and not built['changed_sources'] and
                file_sha(Path(built['pdf_path'])) == built['pdf_sha256'], 'Isolated active-source build differs')
        full = publish(root / 'FULL_EXPORT_VALIDATION.json', {'status': 'complete', 'cells': report['cells'],
            'report_sha256': report['sha256'], 'export_state_sha256': partial['export_state_sha256'],
            'full_size_recovery_verified': True, 'build_receipt': str(build / 'BUILD.json'),
            'pdf_sha256': built['pdf_sha256'], 'isolated_scientific_outputs': str(directory),
            'accepted_scientific_journals_not_simulated': True, 'publication_attempts': 0,
            'preserved_actual_export_job': '60520129', 'partial_validation_sha256': partial['sha256']})
        from moe_exp.research_campaign_v3_r6.validation import traverse
        graph = traverse(campaign)
        graph_proof = publish(root / 'IMPORT_GRAPH_VALIDATION.json', {'status': 'complete',
            'models': list(MODELS), 'non_reap_generations': 69459, 'compact_shapes': 346,
            'reader_validation_sha256': reader['sha256'], 'full_export_validation_sha256': full['sha256'],
            'graph': graph, 'job_id': job, 'partial_validation_sha256': partial['sha256'],
            'scientific_draws': 0, 'preserved_actual_import_job': '60520129'})
        after = integrity_state()
        require(before == after, 'Active canonical manuscript, HEAD or raw Git index changed during validation')
        policy.check_binding(campaign); check_files(binding['files'])
        ready = publish(root / 'READINESS.json', {'passed': True, 'tests_run': 111,
            'files': campaign['code_files'], 'source_unchanged_during_tests': True,
            'actual_import_graph_validated': True, 'actual_validation_sha256': graph_proof['sha256'],
            'preserved_actual_contract_job': '60520129', 'partial_validation_sha256': partial['sha256'],
            'allocated_cpus': 4, 'allocated_memory_gib': 24, 'allocation_minutes': 30,
            'production_released': False, 'seconds': time.monotonic() - started})
        return publish(root / 'OPS_R1_VALIDATION_COMPLETED.json', {'status': 'complete', 'job_id': job,
            'binding_sha256': operations['sha256'], 'readiness_sha256': ready['sha256'],
            'operations_tests': 20, 'inherited_tests': 111, 'preserved_actual_contract_job': '60520129',
            'manuscript_unchanged': True, 'integrity_scope': 'active canonical manuscript sources and raw HEAD/index state',
            'manuscript_state_sha256': before_receipt['sha256'], 'scientific_draws': 0,
            'validation_runner_binding_sha256': binding['sha256'], 'seconds': time.monotonic() - started})
    except BaseException as exc:
        publish(terminal, {'status': 'execution_failed', 'job_id': job, 'error_type': type(exc).__name__,
            'error': str(exc)[:4000], 'binding_sha256': binding['sha256'], 'automatic_retries': 0,
            'scientific_release': False, 'active_source_and_git_unchanged': before == integrity_state()})
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args(); campaign = load(args.campaign / 'CAMPAIGN.json')
    result = validate(campaign)
    from moe_exp.research_campaign_v3_r6.controller import launch, stage_key
    publish(args.campaign / 'stages' / stage_key('validate') / 'RESULT.json', {
        'stage': 'validate', 'model': 'all', 'shard_index': 0, 'threshold_index': 0,
        'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'], 'failure_scope': None,
        'detail': {'readiness_sha256': result['readiness_sha256'],
                   'validation_runner_binding_sha256': result['validation_runner_binding_sha256']}})
    print(json.dumps({'validation': result, 'launch': launch(campaign)}, sort_keys=True))


if __name__ == '__main__': main()
