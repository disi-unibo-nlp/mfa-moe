"""Process-local execution adapter for CPU-adjudicated qualification and seals.

Preserves the frozen v2 generation algorithm, worker, schema, designs and UIDs'
construction. New manifests bind this adapter and all operational entry files.
"""
from pathlib import Path

import adjudicate_overnight_operator_qualification_v3 as adjudication
import overnight_routing_dose_audit_v3 as corrected

INSTALLED = False


def execution_files():
    names = ('overnight_routing_adjudicated_v3.py', 'overnight_routing_dose_audit_v3.py',
             'adjudicate_overnight_operator_qualification_v3.py', 'overnight_routing_entry_v3.py',
             'overnight_routing_run_v3.sbatch', 'prepare_overnight_operator_stage_v3.py',
             'prepare_overnight_operator_stage_v3.sbatch', 'dispatch_overnight_stage_v3.py',
             'dispatch_overnight_stage_v3.sbatch', 'build_price_overnight_semantics_v3.sbatch')
    return [Path(__file__).with_name(name) for name in names]


def install():
    global INSTALLED
    if INSTALLED: return
    import overnight_routing_runner_v2 as runner
    base, common = runner.base, runner.common
    original_write = common.write_once
    original_validate = runner.validate

    def provenance():
        receipt = adjudication.qualified_result()
        return {'schema': 'overnight-routing-cpu-checker-adjudication-v3',
                'result_path': str(adjudication.RESULT), 'result_sha256': receipt['sha256'],
                'original_failed_result_path': receipt['original_result_path'],
                'original_failed_result_sha256': receipt['original_result_sha256'],
                'adapter_sha256': base.file_sha(__file__), 'checker_sha256': base.file_sha(corrected.__file__),
                'engine_changed': False, 'gpu_qualification_reexecuted': False}

    def write(path, body):
        if body.get('schema') == 'overnight-routing-manifest-v2':
            body = {**body, 'code_files': {**body['code_files'],
                    **{str(p.resolve()): base.file_sha(p) for p in execution_files()}},
                    'cpu_checker_adjudication': provenance()}
        elif body.get('schema') in ('overnight-routing-completion-v1', 'overnight-routing-stage-completion-v2'):
            body = {**body, 'cpu_checker_adjudication': provenance()}
        return original_write(path, body)

    def validate(manifest, driver_path):
        result = original_validate(manifest, driver_path)
        common.require(manifest.get('cpu_checker_adjudication') == provenance() and
            all(manifest['code_files'].get(str(p.resolve())) == base.file_sha(p) for p in execution_files()),
            'manifest lacks exact adjudicated checker and operational sources')
        return result

    runner.qualified_result = adjudication.qualified_result
    runner.validate = validate
    common.audit_output_dose = corrected.audit_output_dose
    common.write_once = write
    INSTALLED = True
