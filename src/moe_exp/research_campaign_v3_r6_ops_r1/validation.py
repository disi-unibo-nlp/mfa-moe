"""One actual full CPU validation, with durable terminal failure and no retry."""
from pathlib import Path
import importlib.util
import os
import time
import unittest

from moe_exp.research_campaign_v3_r6.contracts import REPO, allocated, load, publish, require
from .binding import manuscript_state
from .policy import check_binding, install, require_readiness


def validate(campaign):
    allocated()
    root = Path(campaign['root']); binding = check_binding(campaign)
    done = root / 'OPS_R1_VALIDATION_COMPLETED.json'
    if done.exists():
        return require_readiness(campaign)
    require(not (root / 'OPS_R1_VALIDATION_TERMINAL.json').exists(), 'Single operations validation attempt is exhausted')
    start = time.monotonic()
    job = os.environ['SLURM_JOB_ID']
    intent = root / 'OPS_R1_VALIDATION_START.json'
    if intent.exists():
        require(load(intent)['job_id'] == job, 'Validation belongs to another allocation; no retry authorized')
    else:
        publish(intent, {'job_id': job, 'binding_sha256': binding['sha256'], 'started_epoch': time.time()})
    before = manuscript_state()
    publish(root / 'ops_r1/VALIDATION_MANUSCRIPT_BEFORE.json', before)
    try:
        tests = unittest.defaultTestLoader.discover(str(REPO / 'tests/research_campaign_v3_r6_ops_r1'))
        result = unittest.TextTestRunner(verbosity=1).run(tests)
        require(result.wasSuccessful(), 'Operations contracts failed')
        # Run inherited contracts before installing hooks: their sealed contracts
        # stay intact. Full real export/graph below exercise the installed adapter.
        spec = importlib.util.spec_from_file_location('r6_cpu_fixture_candidate_r4',
                    REPO / 'scripts/research_campaign_v3_r6_validation_candidate_r4.py')
        candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
        inherited = unittest.defaultTestLoader.discover(str(REPO / 'tests/research_campaign_v3_r6_validation_r2'))
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
            'manuscript_state_sha256': load(root / 'ops_r1/VALIDATION_MANUSCRIPT_BEFORE.json')['sha256'],
            'scientific_draws': 0, 'seconds': time.monotonic() - start})
    except BaseException as exc:
        import traceback
        after = manuscript_state()
        publish(root / 'OPS_R1_VALIDATION_TERMINAL.json', {
            'status': 'execution_failed', 'job_id': job, 'binding_sha256': binding['sha256'],
            'error_type': type(exc).__name__, 'error': str(exc)[:4000], 'traceback': traceback.format_exc()[-10000:],
            'seconds': time.monotonic() - start, 'manuscript_unchanged': before == after,
            'scientific_release': False, 'automatic_retries': 0, 'original_attempts_preserved': True,
        })
        raise
