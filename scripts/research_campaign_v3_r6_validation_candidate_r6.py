#!/usr/bin/env python3
"""Unsubmitted correction: independent unittest loaders for two non-package suites."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))


def discover_suite(path):
    path = str(Path(path).resolve())
    import unittest
    return unittest.TestLoader().discover(path, top_level_dir=path)


"""One actual full CPU validation, with durable terminal failure and no retry."""
from pathlib import Path
import importlib.util
import os
import time
import unittest

from moe_exp.research_campaign_v3_r6.contracts import REPO, allocated, load, publish, require
from moe_exp.research_campaign_v3_r6_ops_r1.binding import manuscript_state
from moe_exp.research_campaign_v3_r6_ops_r1.policy import check_binding, install, require_readiness


def validate(campaign):
    runner_binding = load(Path(campaign['root']) / 'OPS_R1_LOADER_BINDING.json')
    require(runner_binding.get('explicit_extension_authorized') is True, 'Another allocation requires explicit user authorization')
    from moe_exp.research_campaign_v3_r6.contracts import check_files
    check_files(runner_binding['files'])
    require(runner_binding['ops_binding_sha256'] == check_binding(campaign)['sha256'], 'Corrected runner binding differs')
    allocated()
    root = Path(campaign['root']); binding = check_binding(campaign)
    done = root / 'OPS_R1_VALIDATION_COMPLETED.json'
    if done.exists():
        return require_readiness(campaign)
    require(not (root / 'OPS_R1_LOADER_VALIDATION_TERMINAL.json').exists(), 'Single operations validation attempt is exhausted')
    start = time.monotonic()
    job = os.environ['SLURM_JOB_ID']
    intent = root / 'OPS_R1_LOADER_VALIDATION_START.json'
    if intent.exists():
        require(load(intent)['job_id'] == job, 'Validation belongs to another allocation; no retry authorized')
    else:
        publish(intent, {'job_id': job, 'binding_sha256': binding['sha256'], 'started_epoch': time.time()})
    before = manuscript_state()
    publish(root / 'ops_r1/LOADER_VALIDATION_MANUSCRIPT_BEFORE.json', before)
    try:
        tests = discover_suite(REPO / 'tests/research_campaign_v3_r6_ops_r1')
        result = unittest.TextTestRunner(verbosity=1).run(tests)
        require(result.wasSuccessful(), 'Operations contracts failed')
        # Run inherited contracts before installing hooks: their sealed contracts
        # stay intact. Full real export/graph below exercise the installed adapter.
        spec = importlib.util.spec_from_file_location('r6_cpu_fixture_candidate_r4',
                    REPO / 'scripts/research_campaign_v3_r6_validation_candidate_r4.py')
        candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
        inherited = discover_suite(REPO / 'tests/research_campaign_v3_r6_validation_r2')
        base_result = unittest.TextTestRunner(verbosity=1).run(inherited)
        require(base_result.wasSuccessful(), 'Inherited regression contracts failed')
        # Candidate uses its own unittest loader. Prevent a second identical
        # regression run; the already executed TestResult remains authoritative.
        original_loader = unittest.defaultTestLoader.discover
        original_runner = unittest.TextTestRunner.run
        install(campaign)
        candidate.unittest.defaultTestLoader.discover = lambda *a, **k: inherited
        candidate.unittest.TextTestRunner.run = lambda self, suite: base_result
        try:
            readiness = candidate.validate(campaign)
        finally:
            unittest.defaultTestLoader.discover = original_loader
            unittest.TextTestRunner.run = original_runner
        after = manuscript_state()
        require(before == after, 'CPU validation changed manuscript, HEAD or index')
        check_binding(campaign)
        return publish(done, {'status': 'complete', 'job_id': job, 'binding_sha256': binding['sha256'],
            'readiness_sha256': readiness['sha256'], 'operations_tests': result.testsRun,
            'inherited_tests': base_result.testsRun, 'manuscript_unchanged': True,
            'manuscript_state_sha256': load(root / 'ops_r1/LOADER_VALIDATION_MANUSCRIPT_BEFORE.json')['sha256'],
            'scientific_draws': 0, 'validation_runner_binding_sha256': runner_binding['sha256'], 'original_failed_job': '60510557', 'seconds': time.monotonic() - start})
    except BaseException as exc:
        import traceback
        after = manuscript_state()
        publish(root / 'OPS_R1_LOADER_VALIDATION_TERMINAL.json', {
            'status': 'execution_failed', 'job_id': job, 'binding_sha256': binding['sha256'],
            'error_type': type(exc).__name__, 'error': str(exc)[:4000], 'traceback': traceback.format_exc()[-10000:],
            'seconds': time.monotonic() - start, 'manuscript_unchanged': before == after,
            'scientific_release': False, 'automatic_retries': 0, 'original_attempts_preserved': True,
        })
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    campaign = load(args.campaign / 'CAMPAIGN.json')
    if args.preflight:
        binding = check_binding(campaign)
        ops = discover_suite(REPO / 'tests/research_campaign_v3_r6_ops_r1')
        inherited = discover_suite(REPO / 'tests/research_campaign_v3_r6_validation_r2')
        require(ops.countTestCases() == 20 and inherited.countTestCases() == 111, 'Full original contract roster differs')
        value = {'passed': True, 'ops_tests_discovered': 20, 'inherited_tests_discovered': 111,
                 'test_execution_in_this_preflight': False, 'full_validation_executed': False,
                 'job_submissions': 0, 'scientific_draws': 0, 'ops_binding_sha256': binding['sha256']}
        print(json.dumps(publish(args.campaign / 'OPS_R1_LOADER_PREFLIGHT.json', value), sort_keys=True))
        return
    result = validate(campaign)
    from moe_exp.research_campaign_v3_r6.controller import stage_key
    root = Path(campaign['root'])
    directory = root / 'stages' / stage_key('validate')
    start = directory / 'START-5.json'
    if not start.exists():
        publish(start, {'job_id': os.environ['SLURM_JOB_ID'],
                        'runner_binding_sha256': result['validation_runner_binding_sha256'],
                        'original_failed_job': '60510557'})
    completed = directory / 'RESULT.json'
    if not completed.exists():
        publish(completed, {'stage': 'validate', 'model': 'all', 'shard_index': 0, 'threshold_index': 0,
            'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'], 'failure_scope': None,
            'detail': {'readiness_sha256': result['readiness_sha256'],
                      'validation_runner_binding_sha256': result['validation_runner_binding_sha256']}})
    from moe_exp.research_campaign_v3_r6.controller import launch
    print(json.dumps({'validation': result, 'launch': launch(campaign)}, sort_keys=True))


if __name__ == '__main__':
    main()
