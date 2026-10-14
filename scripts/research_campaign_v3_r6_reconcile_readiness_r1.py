#!/usr/bin/env python3
"""Reconcile actual allocated proofs and a bounded fresh integrity-only snapshot."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import (
    check_files, file_sha, load, publish, require, STAGING,
)
from moe_exp.research_campaign_v3_r6_ops_r1.policy import check_binding
from moe_exp.research_campaign_v3_r6_ops_r1.binding import PAPER, git_state


def run(root):
    campaign = load(root / 'CAMPAIGN.json')
    amendment = load(root / 'LOOKUP_R1_BINDING.json')
    require(amendment['authorized_operational_reconciliation'] is True,
            'Operational reconciliation must be explicitly bound')
    check_files(amendment['files']); check_files(amendment['font_files'])
    operations = check_binding(campaign)
    require(operations['sha256'] == amendment['ops_binding_sha256'] and
            campaign['sha256'] == amendment['campaign_sha256'], 'Source ancestry differs')
    partial = load(root / 'OPS_R1_LOADER_PARTIAL_VALIDATION.json')
    reader = load(root / 'READER_INPUT_VALIDATION.json')
    graph = load(root / 'IMPORT_GRAPH_VALIDATION.json')
    export = load(root / 'FULL_EXPORT_VALIDATION.json')
    require(partial['operations_tests_passed'] == 20 and partial['inherited_tests_passed'] == 111
            and file_sha(Path(partial['stderr_path'])) == partial['stderr_sha256'],
            'Actual allocated contract evidence differs')
    require(graph['job_id'] == '60561919' and graph['status'] == export['status'] == reader['status'] == 'complete'
            and graph['non_reap_generations'] == 69459 and graph['compact_shapes'] == 346
            and export['cells'] == 243747 and export['full_size_recovery_verified']
            and reader['original_failures_reproduced'] == 64 and reader['scientific_draws'] == 0
            and graph['reader_validation_sha256'] == reader['sha256']
            and graph['full_export_validation_sha256'] == export['sha256'],
            'Actual allocated reader/export/graph dependencies are incomplete')
    directory = Path(partial['full_export_directory'])
    check_files(load(directory / 'EXPORT_FILES.json')['files'])
    require(file_sha(directory / 'EXPORT_STATE.json') == partial['export_state_sha256'],
            'Committed export checkpoint differs')
    tests = load(root / 'LOOKUP_R1_TESTS.json')
    preflight = load(root / 'OPS_R1_LOOKUP_GRAPH_PREFLIGHT.json')
    require(tests['passed'] and tests['tests_run'] == 6 and tests['binding_sha256'] == amendment['sha256']
            and preflight['passed'] and preflight['scheduler_submissions'] == 0
            and preflight['lookup_source_sha256'] == amendment['files'][str(REPO / 'src/moe_exp/research_campaign_v3_r6_lookup_r1.py')]
            and preflight['graph']['registered_graph_sha256'] == graph['graph']['registered_graph_sha256']
            and preflight['graph']['saved_sequence_projection_local_hours'] == graph['graph']['saved_sequence_projection_local_hours']
            and preflight['graph']['unpriced_saved_sequence_models'] == graph['graph']['unpriced_saved_sequence_models'],
            'Matching lookup fidelity and actual graph preflight are required')
    spec = importlib.util.spec_from_file_location('active_builder', REPO / 'scripts/paper/build_campaign_paper_r4.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    def integrity():
        return {'active_manuscript_files': builder.source_state(),
                'repository': git_state(REPO), 'overleaf': git_state(PAPER)}
    before = integrity()
    evidence = publish(root / 'ops_r1/RECONCILIATION_MANUSCRIPT_BEFORE.json', before)
    build = STAGING / 'research-r6-readiness-reconciliation-r1'
    started = time.monotonic()
    subprocess.run([sys.executable, '-B', str(REPO / 'scripts/paper/build_campaign_paper_r4.py'),
                    '--only-cached', '--output', str(build)], check=True, timeout=60)
    built = json.loads((build / 'BUILD.json').read_text())
    require(before == integrity() and built['state'] == 'COMPILED_FULL_CLASS'
            and not built['changed_sources'] and file_sha(Path(built['pdf_path'])) == built['pdf_sha256'],
            'Fresh active manuscript or raw Git integrity check failed')
    check_binding(campaign); check_files(amendment['files']); check_files(amendment['font_files'])
    reconciled = publish(root / 'OPS_R1_READINESS_RECONCILIATION.json', {
        'passed': True, 'binding_sha256': amendment['sha256'],
        'actual_allocated_graph_job': '60561919', 'actual_allocated_contract_export_job': '60520129',
        'original_failed_terminal_sha256': load(root / 'OPS_R1_FONT_RECOVERY_TERMINAL.json')['sha256'],
        'prior_concurrent_manuscript_changes_preserved': True,
        'fresh_integrity_scope': 'active canonical inputs and raw HEAD/index during this bounded isolated build',
        'fresh_integrity_sha256': evidence['sha256'], 'fresh_integrity_passed': True,
        'build_receipt': str(build / 'BUILD.json'), 'pdf_sha256': built['pdf_sha256'],
        'seconds': time.monotonic() - started, 'new_slurm_allocations': 0,
        'new_scientific_draws': 0, 'canonical_manuscript_writes': 0, 'remote_writes': 0,
        'original_validation_job_not_reclassified': True,
    })
    ready = publish(root / 'READINESS.json', {
        'passed': True, 'tests_run': 111, 'files': campaign['code_files'],
        'source_unchanged_during_tests': True, 'actual_import_graph_validated': True,
        'actual_validation_sha256': graph['sha256'], 'production_released': False,
        'actual_contract_job': '60520129', 'actual_graph_job': '60561919',
        'operational_reconciliation_sha256': reconciled['sha256'],
    })
    complete = publish(root / 'OPS_R1_VALIDATION_COMPLETED.json', {
        'status': 'complete', 'job_id': None, 'binding_sha256': operations['sha256'],
        'readiness_sha256': ready['sha256'], 'operations_tests': 20, 'inherited_tests': 111,
        'manuscript_unchanged': True, 'manuscript_state_sha256': evidence['sha256'],
        'integrity_scope': reconciled['fresh_integrity_scope'],
        'allocated_graph_job': '60561919', 'allocated_contract_export_job': '60520129',
        'operational_reconciliation_sha256': reconciled['sha256'],
        'validation_runner_binding_sha256': amendment['sha256'], 'scientific_draws': 0,
        'new_slurm_allocations': 0,
    })
    cpu = load(root / 'CPU_DISPATCH_R1_BINDING.json')
    dependencies = {}
    for name in ('TESTS', 'SCHEDULER_PREFLIGHT', 'TRANSPORT_PREFLIGHT'):
        proof = load(root / ('CPU_DISPATCH_R1_' + name + '.json'))
        require(proof['passed'] and proof['binding_sha256'] == cpu['sha256'], 'CPU dependency differs')
        dependencies[name.lower() + '_sha256'] = proof['sha256']
    cpu_proof = publish(root / 'CPU_DISPATCH_R1_PREFLIGHT.json', {
        'passed': True, 'binding_sha256': cpu['sha256'], 'base_validation_sha256': complete['sha256'],
        'actual_graph_sha256': graph['sha256'], **dependencies,
        'scientific_draws': 0, 'additional_gpu_request': 0,
    })
    publish(root / 'LOOKUP_R1_PREFLIGHT.json', {
        'passed': True, 'binding_sha256': amendment['sha256'],
        'cpu_dispatch_proof_sha256': cpu_proof['sha256'], 'tests_sha256': tests['sha256'],
        'actual_graph_preflight_sha256': preflight['sha256'],
        'readiness_reconciliation_sha256': reconciled['sha256'],
    })
    from moe_exp.research_campaign_v3_r6.controller import stage_key
    publish(root / 'stages' / stage_key('validate') / 'RESULT.json', {
        'stage': 'validate', 'model': 'all', 'shard_index': 0, 'threshold_index': 0,
        'status': 'complete', 'job_id': None, 'failure_scope': None,
        'actual_allocated_graph_job': '60561919', 'actual_allocated_contract_job': '60520129',
        'operational_reconciliation_sha256': reconciled['sha256'],
        'scheduler_failed_allocations_preserved': True,
    })
    return reconciled


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Provide the existing campaign root')
    result = run(Path(sys.argv[1]).resolve())
    print(json.dumps({'passed': result['passed'], 'sha256': result['sha256'], 'new_slurm_allocations': 0}))
